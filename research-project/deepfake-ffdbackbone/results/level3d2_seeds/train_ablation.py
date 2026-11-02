
import os, sys, json, math, time, random, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.distributed as dist
import cv2, albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from sklearn.metrics import roc_auc_score
import importlib.util

ap = argparse.ArgumentParser(); ap.add_argument('--cfg', required=True)
A = json.load(open(ap.parse_args().cfg))
CFG, REPO_DIR = A['cfg'], Path(A['repo_dir'])
RUN_DIR = Path(A['run_dir']); RUN_DIR.mkdir(parents=True, exist_ok=True)
FL = A.get('flags', dict(dual=True, dec=True, ufm=True, edu=True))
VARIANT = A.get('variant', 'cfb_full')
sys.path.insert(0, str(REPO_DIR))

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

MF  = load_mod('beitv2_mf',  REPO_DIR/'models'/'lib'/'BEiT_v2'/'modeling_finetune.py')
EDL = load_mod('ffd_edl',    REPO_DIR/'common'/'losses'/'edl_loss.py')

LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
WORLD      = int(os.environ.get('WORLD_SIZE', 1))
DDP        = WORLD > 1
if DDP:
    dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(LOCAL_RANK)
DEV = torch.device('cuda', LOCAL_RANK)
IS_MAIN = LOCAL_RANK == 0

seed = CFG['seed'] + LOCAL_RANK
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def log(*a):
    if IS_MAIN:
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        with open(RUN_DIR/'train.log', 'a') as f: f.write(msg + '\n')

# ---------------- data ----------------
def _jpeg(lo, hi, p):
    # albumentations >=1.4.21 เปลี่ยนชื่อ arg เป็น quality_range
    try:
        return alb.ImageCompression(quality_range=(lo, hi), p=p)
    except TypeError:
        return alb.ImageCompression(quality_lower=lo, quality_upper=hi, p=p)

def tf_train(cfg):
    # ตรงกับ create_base_transforms(split='train') ของ repo ทุกตัว
    return alb.Compose([
        alb.RGBShift((-20,20),(-20,20),(-20,20), p=0.3),
        alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3),
                               val_shift_limit=(-0.3,0.3), p=0.3),
        alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
        _jpeg(40, 100, 0.5),
        alb.HorizontalFlip(),
        alb.ToGray(p=0.01),
        alb.Resize(cfg['image_size'], cfg['image_size']),
        alb.Normalize(mean=cfg['mean'], std=cfg['std']),
        ToTensorV2(),
    ])

def tf_eval(cfg):
    return alb.Compose([alb.Resize(cfg['image_size'], cfg['image_size']),
                        alb.Normalize(mean=cfg['mean'], std=cfg['std']), ToTensorV2()])

class Cutout:
    # map_util.Cutout ของ repo (คัดลอกตรง ๆ): 1-3 หลุม ขนาดสุ่ม 2%-60% ของด้าน
    def __call__(self, imgs):
        h, w = imgs[0].size(1), imgs[0].size(2)
        n_holes = np.random.randint(1, 4)
        length = int(h * np.random.randint(1, 30) * 0.01) * 2
        mask = np.ones((h, w), np.float32)
        for _ in range(n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip(y-length//2, 0, h), np.clip(y+length//2, 0, h)
            x1, x2 = np.clip(x-length//2, 0, w), np.clip(x+length//2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        return imgs * torch.from_numpy(mask).expand_as(imgs)

class FrameDS(Dataset):
    # train: 1 frame = 1 sample (เหมือน Image_dataset ของ repo) -> [1,3,H,W]
    def __init__(self, items, cfg):
        self.items, self.tf = items, tf_train(cfg)
        self.cut = Cutout() if cfg['cutout'] else None
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        img = np.asarray(Image.open(p).convert('RGB'))
        x = self.tf(image=img)['image'].unsqueeze(0)       # [1,3,H,W]
        if self.cut is not None: x = self.cut(x)
        return x, y

class VideoDS(Dataset):
    # val/test: 8 frame = 1 sample -> [8,3,H,W]
    def __init__(self, videos, cfg):
        self.videos, self.tf, self.k = videos, tf_eval(cfg), cfg['num_segments']
    def __len__(self): return len(self.videos)
    def __getitem__(self, i):
        fs, y = self.videos[i]['frames'], self.videos[i]['label']
        xs = [self.tf(image=np.asarray(Image.open(p).convert('RGB')))['image'] for p in fs]
        return torch.stack(xs), y, i

# ---------------- model (= DDBF_BEiT_v2 ของ repo) ----------------
BEiT_Config = {'drop_path_rate': CFG['drop_path_rate'], 'use_mean_pooling': False, 'init_values': 0.1,
               'qkv_bias': True, 'use_abs_pos_emb': False, 'use_rel_pos_bias': True,
               'use_shared_rel_pos_bias': False}

def enable_grad_ckpt(vit):
    # BEiT ของ repo ไม่มี gradient checkpointing -> patch forward_features ให้ใช้
    from torch.utils.checkpoint import checkpoint
    def ff(self, x, return_patch_tokens=False, return_all_tokens=False, **kw):
        x = self.patch_embed(x)
        B = x.size(0)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.pos_embed is not None: x = x + self.pos_embed
        x = self.pos_drop(x)
        rpb = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.training:
                x = checkpoint(lambda t, b=blk: b(t, rel_pos_bias=rpb), x, use_reentrant=False)
            else:
                x = blk(x, rel_pos_bias=rpb)
        x = self.norm(x)
        return x[:, 0]
    vit.forward_features = ff.__get__(vit, type(vit))
    return vit

def pearson_correlation_loss(x, y):          # models/custom.py ของ repo
    xm = x.sub(x.mean(-1, keepdim=True)); ym = y.sub(y.mean(-1, keepdim=True))
    cov = torch.bmm(xm.unsqueeze(1), ym.unsqueeze(2)).squeeze() / x.size(-1)
    r = (cov / (x.std(-1) * y.std(-1))).mean()
    return torch.clamp(r, -1.0, 1.0)

def uncertainty(out):                        # Eq. 3-4 (K=2)
    alpha = torch.exp(torch.clamp(out, -10, 10)) + 1
    return 2 / torch.sum(alpha, dim=1, keepdim=True)

class DDBF_BEiT_v2(nn.Module):
    # = DDBF_BEiT_v2 ของ repo เมื่อ flags ครบทุกตัว · ปิดบางส่วนได้เพื่อทำ ablation (Table VI)
    def __init__(self, cfg, pretrained_path=None, grad_ckpt=False, flags=None):
        super().__init__()
        self.fl = dict(flags or FL)
        self.backbone_model = MF.beit_base_patch16_224(**BEiT_Config)
        self.backbone_model.head = nn.Identity()
        if self.fl['dual']:
            self.backbone_model2 = MF.beit_base_patch16_224(**BEiT_Config)
            self.backbone_model2.head = nn.Identity()
        if pretrained_path:
            ck = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
            m1 = self.backbone_model.load_state_dict(sd, strict=False)
            if self.fl['dual']:
                self.backbone_model2.load_state_dict(sd, strict=False)
            log('load pretrain | missing:', len(m1.missing_keys), 'unexpected:', len(m1.unexpected_keys))
        if grad_ckpt:
            enable_grad_ckpt(self.backbone_model)
            if self.fl['dual']: enable_grad_ckpt(self.backbone_model2)
        self.norm_for_cls = nn.LayerNorm(cfg['feature_dim'])
        self.header = nn.Sequential(nn.Linear(cfg['feature_dim'], cfg['num_classes']))
    def forward(self, x):
        B, T = x.size(0), x.size(1)
        x = x.flatten(0, 1)
        f1 = self.backbone_model(x)
        p1 = self.header(self.norm_for_cls(f1))
        if not self.fl['dual']:                                   # FB: backbone เดียว + FC
            return p1.view(B, T, -1), p1.view(B, T, -1), None, f1.new_zeros(())
        f2 = self.backbone_model2(x)
        ploss = pearson_correlation_loss(f1.detach().float(), f2.float())   # gradient-stop ที่ main
        p2 = self.header(self.norm_for_cls(f2))
        if self.fl['ufm']:                                        # UFM (Eq. 3-5)
            u = torch.cat([uncertainty(p1.float()), uncertainty(p2.float())], dim=-1)
            w = torch.softmax(-10 * u, dim=-1).to(f1.dtype)
            fused = torch.matmul(w.unsqueeze(1), torch.stack((f1, f2), dim=1)).squeeze(1)
        else:                                                     # ไม่มี UFM → บวก feature (caption Table VI)
            fused = f1 + f2
        p = self.header(self.norm_for_cls(fused))
        return p.view(B, T, -1), p1.view(B, T, -1), p2.view(B, T, -1), ploss

# ---------------- optimizer: layer-wise lr decay ----------------
NO_DECAY = ('bias', 'norm', 'cls_token', 'pos_embed', 'relative_position_bias_table', 'gamma_')

def layer_id_of(name, n_layers=12):
    for pre in ('backbone_model2.', 'backbone_model.'):
        if name.startswith(pre): name = name[len(pre):]; break
    if name.startswith(('cls_token', 'pos_embed', 'patch_embed', 'rel_pos_bias')): return 0
    if name.startswith('blocks.'): return int(name.split('.')[1]) + 1
    return n_layers + 1

def build_param_groups(model, cfg, n_layers=12):
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        lid = layer_id_of(n, n_layers)
        wd = 0.0 if (p.ndim == 1 or any(k in n for k in NO_DECAY)) else cfg['weight_decay']
        key = (lid, wd)
        if key not in groups:
            groups[key] = {'params': [], 'weight_decay': wd,
                           'lr_scale': cfg['layer_decay'] ** (n_layers + 1 - lid)}
        groups[key]['params'].append(p)
    return list(groups.values())

def lr_at(step, total_steps, warm_steps, cfg):
    if step < warm_steps:
        return cfg['warmup_lr'] + (cfg['lr'] - cfg['warmup_lr']) * step / max(1, warm_steps)
    t = (step - warm_steps) / max(1, total_steps - warm_steps)
    return cfg['min_lr'] + (cfg['lr'] - cfg['min_lr']) * 0.5 * (1 + math.cos(math.pi * t))

# ---------------- eval ----------------
@torch.no_grad()
def evaluate(model, loader, n_videos):
    model.eval()
    scores = torch.zeros(n_videos, device=DEV); labels = torch.zeros(n_videos, device=DEV)
    seen   = torch.zeros(n_videos, device=DEV)
    for x, y, idx in loader:
        x = x.to(DEV, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=CFG['amp']):
            out, _, _, _ = model(x)
        p = torch.softmax(out.float(), dim=-1)[..., 1].mean(dim=1)     # เฉลี่ย 8 frame = score วิดีโอ
        idx = idx.to(DEV)
        scores[idx] = p; labels[idx] = y.to(DEV).float(); seen[idx] = 1
    if DDP:
        for t in (scores, labels, seen): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    m = seen > 0
    # DistributedSampler เติม sample ซ้ำให้หารลงตัว -> index ที่ซ้ำถูกบวก 2 ครั้ง จึงหารด้วย seen
    scores = torch.where(m, scores / seen.clamp(min=1), scores)
    labels = torch.where(m, labels / seen.clamp(min=1), labels)
    s, l = scores[m].cpu().numpy(), labels[m].cpu().numpy().round().astype(int)
    auc = roc_auc_score(l, s) if len(set(l.tolist())) > 1 else float('nan')
    acc = float(((s >= 0.5).astype(int) == l).mean())
    return auc, acc, s, l

# ---------------- main ----------------
def main():
    train_items = [tuple(x) for x in A['train_items']]
    val_videos, test_videos = A['val_videos'], A['test_videos']

    tr_ds = FrameDS(train_items, CFG)
    tr_sampler = DistributedSampler(tr_ds, shuffle=True, drop_last=True) if DDP else None
    tr_ld = DataLoader(tr_ds, batch_size=CFG['batch_size'], shuffle=tr_sampler is None,
                       sampler=tr_sampler, num_workers=CFG['num_workers'], pin_memory=True, drop_last=True)
    def vloader(vids):
        ds = VideoDS(vids, CFG)
        sp = DistributedSampler(ds, shuffle=False) if DDP else None
        return DataLoader(ds, batch_size=max(1, CFG['batch_size'] // CFG['num_segments']),
                          shuffle=False, sampler=sp, num_workers=CFG['num_workers'], pin_memory=True)
    va_ld, te_ld = vloader(val_videos), vloader(test_videos)

    model = DDBF_BEiT_v2(CFG, A['pretrain_path'], CFG['grad_ckpt'], FL).to(DEV)
    log(f'variant {VARIANT} | flags {FL}')
    if DDP:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK],
                                                    find_unused_parameters=False)
    core = model.module if DDP else model

    import timm.optim as topt
    pg = build_param_groups(core, CFG)
    for g in pg: g['lr'] = CFG['lr'] * g['lr_scale']
    optimizer = topt.Lamb(pg, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=CFG['amp'])
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=CFG['amp'])

    crit_ce  = nn.CrossEntropyLoss().to(DEV)
    crit_edl = EDL.EvidenceLoss(**CFG['edl']).to(DEV)

    steps_per_epoch = len(tr_ld) // CFG['grad_accum']
    total_steps = steps_per_epoch * CFG['epochs']
    warm_steps  = steps_per_epoch * CFG['warmup_epochs']

    start_epoch, best_auc, history = 1, 0.0, []
    last = RUN_DIR / 'last.pth'
    if last.exists():
        ck = torch.load(last, map_location='cpu', weights_only=False)
        core.load_state_dict(ck['state_dict'])
        optimizer.load_state_dict(ck['optimizer'])
        if ck.get('scaler'): scaler.load_state_dict(ck['scaler'])
        start_epoch, best_auc, history = ck['epoch'] + 1, ck.get('best_auc', 0.0), ck.get('history', [])
        log(f'resume จาก epoch {ck["epoch"]} (best val AUC {100*best_auc:.2f})')

    def sum_edl(out, y, ep):
        d = crit_edl(out, y, epoch=ep, total_epoch=CFG['epochs'])
        return sum(v.mean() for k, v in d.items() if 'loss' in k and torch.is_tensor(v))

    t_start = time.time()
    for epoch in range(start_epoch, CFG['epochs'] + 1):
        if DDP: tr_sampler.set_epoch(epoch)
        model.train()
        agg = {k: 0.0 for k in ['loss', 'ce', 'edl1', 'edl2', 'pear', 'acc']}
        n = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for it, (x, y) in enumerate(tr_ld):
            gstep = (epoch - 1) * steps_per_epoch + it // CFG['grad_accum']
            lr_now = lr_at(gstep, total_steps, warm_steps, CFG)
            for g in optimizer.param_groups: g['lr'] = lr_now * g['lr_scale']

            x = x.to(DEV, non_blocking=True); y = y.to(DEV, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=CFG['amp']):
                out, out_m, out_a, ploss = model(x)
            T = out.size(1)
            yr = y.unsqueeze(1).repeat(1, T).flatten(0, 1)
            o_f, o_m = out.flatten(0,1).float(), out_m.flatten(0,1).float()
            zero = o_f.new_zeros(())
            if FL['dual']:
                o_a = out_a.flatten(0,1).float()
                branch = (lambda o: sum_edl(o, yr, epoch)) if FL['edu'] else (lambda o: crit_ce(o, yr))
                l_ce   = crit_ce(o_f, yr)                        # L^F_ce (fused head)
                l_edl1 = branch(o_m)                             # L^M_EDU หรือ L^M_ce
                l_edl2 = branch(o_a)                             # L^A_EDU หรือ L^A_ce
                loss   = l_ce + l_edl1 + l_edl2 + (ploss if FL['dec'] else zero)   # Eq. 9 เมื่อครบทุกตัว
            else:                                                # FB: loss เดียวที่ head เดียว
                l_ce   = zero if FL['edu'] else crit_ce(o_m, yr)
                l_edl1 = sum_edl(o_m, yr, epoch) if FL['edu'] else zero
                l_edl2 = zero
                loss   = l_edl1 if FL['edu'] else l_ce
            scaler.scale(loss / CFG['grad_accum']).backward()
            if (it + 1) % CFG['grad_accum'] == 0:
                if CFG['clip_grad']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CFG['clip_grad'])
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = y.size(0); n += bs
            agg['loss'] += loss.item()*bs; agg['ce'] += l_ce.item()*bs
            agg['edl1'] += l_edl1.item()*bs; agg['edl2'] += l_edl2.item()*bs
            agg['pear'] += ploss.item()*bs
            agg['acc'] += (o_f.argmax(1) == yr).float().mean().item()*bs
            if IS_MAIN and it % 50 == 0:
                log(f'E{epoch} [{it}/{len(tr_ld)}] loss {loss.item():.4f} '
                    f'ce {l_ce.item():.4f} edlM {l_edl1.item():.4f} edlA {l_edl2.item():.4f} '
                    f'pear {ploss.item():+.4f} acc {agg["acc"]/n:.4f} lr {lr_now:.2e} '
                    f'| {(time.time()-t0)/60:.1f} min')

        tr = {k: v/max(n,1) for k, v in agg.items()}
        va_auc, va_acc, _, _ = evaluate(model, va_ld, len(val_videos))
        log(f'[EPOCH {epoch}] train loss {tr["loss"]:.4f} acc {tr["acc"]:.4f} pearson {tr["pear"]:+.4f} '
            f'| val video-AUC {100*va_auc:.2f} acc {100*va_acc:.2f} | {(time.time()-t0)/60:.1f} min')
        history.append({'epoch': epoch, **tr, 'val_auc': va_auc, 'val_acc': va_acc})

        if IS_MAIN:
            ck = {'epoch': epoch, 'state_dict': core.state_dict(), 'optimizer': optimizer.state_dict(),
                  'scaler': scaler.state_dict() if CFG['amp'] else None,
                  'best_auc': max(best_auc, va_auc), 'history': history, 'cfg': CFG}
            torch.save(ck, RUN_DIR/'last.pth')
            if va_auc > best_auc:
                torch.save({'epoch': epoch, 'state_dict': core.state_dict(),
                            'val_auc': va_auc, 'cfg': CFG, 'variant': VARIANT, 'flags': FL},
                           RUN_DIR/'best.pth')
                log(f'  -> best (val AUC {100*va_auc:.2f})')
            json.dump(history, open(RUN_DIR/'history.json', 'w'), indent=1)
        best_auc = max(best_auc, va_auc)
        if DDP: dist.barrier()

        if (time.time() - t_start)/3600 > CFG['max_train_hours']:
            log(f'ใกล้ชน time limit ที่ epoch {epoch} — หยุดไว้ก่อน รันใหม่เพื่อ resume')
            break

    # ประเมิน test ด้วย best checkpoint
    if (RUN_DIR/'best.pth').exists():
        core.load_state_dict(torch.load(RUN_DIR/'best.pth', map_location='cpu', weights_only=False)['state_dict'])
    te_auc, te_acc, s, l = evaluate(model, te_ld, len(test_videos))
    log(f'[FINAL] FF++ {A["compression"]} test video-AUC {100*te_auc:.2f} acc {100*te_acc:.2f}')
    if IS_MAIN:
        np.save(RUN_DIR/'test_scores.npy', np.stack([s, l]))
        json.dump({'test_auc': te_auc, 'test_acc': te_acc, 'best_val_auc': best_auc,
                   'variant': VARIANT, 'flags': FL},
                  open(RUN_DIR/'final.json', 'w'), indent=1)
    if DDP: dist.destroy_process_group()

main()
