"""สร้าง FFD_level3g_fewshot.ipynb จาก FFD_level3f_significance.ipynb (ใช้ cell ที่ทดสอบแล้ว: repo, Detector, ข้อมูล, inference)

ใช้:  python make_3g.py <path ของ FFD_level3f_significance.ipynb> <path ของ notebook ที่จะสร้าง>
"""
import json, sys

src_path, out_path = sys.argv[1], sys.argv[2]
SRC = json.load(open(src_path, encoding="utf-8"))
C = lambda i: "".join(SRC["cells"][i]["source"])

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                           "source": t.strip("\n").splitlines(keepends=True)})
def rep(s, a, b, count=1):
    assert s.count(a) == count, (s.count(a), a[:80])
    return s.replace(a, b)

# ------------------------------------------------------------------ 0. หัวเรื่อง
md(r"""
# ระดับ 3g — Few-shot adaptation บน Celeb-DF (ทำซ้ำ protocol "adaptation" ของบทความ, Table V)

**บทความ (Sec. IV-C2):** นำโมเดลที่ฝึกบน FF++ แล้วมา fine-tune ต่อด้วยภาพจากชุดข้อมูลที่ไม่เคยเห็นเพียง **10 หรือ 20 ภาพ (real : fake เท่ากัน)**
**2 epoch, lr 2e-5** แล้ววัดผลทั้งบนชุดเป้าหมาย, บนชุดเดิม (FF++) และบนชุดอื่นที่ไม่เคยเห็น

**ความต่างจากบทความ (ต้องระบุในรายงาน):**
- ชุดเป้าหมายของบทความคือ DFDC / FFIW (เราไม่มีข้อมูล) → ใช้ **Celeb-DF** เป็นเป้าหมาย, **WildDeepfake** เป็นชุดที่ไม่เคยเห็น
- repo ไม่มีโค้ด few-shot และบทความไม่ระบุวิธีเลือกภาพ, batch size, optimizer, การทำซ้ำ → กำหนดเองดังนี้
  - เลือก **วิดีโอ k วิดีโอ (k/2 real + k/2 fake) วิดีโอละ 1 frame** แบบสุ่ม · สุ่ม **3 รอบ** (ชุดภาพเดียวกันสำหรับทุกโมเดล → เปรียบเทียบแบบ paired)
  - **กัน leakage:** ตัดวิดีโอที่ใช้ปรับ และทุกวิดีโอที่มี identity ร่วมกัน ออกจากชุดวัดผล Celeb-DF · AUC "ก่อน" และ "หลัง" วัดบนชุดที่เหลือชุดเดียวกัน
  - fine-tune **ทั้งโมเดล** ด้วย loss เดิมของแต่ละ variant (FB·CE: L_ce · CFB full: Eq. 9) · augmentation เดิม · Lamb, lr คงที่ 2e-5, batch 8 · inference ใช้ main branch
- โมเดล: checkpoint FB·CE และ CFB full จากระดับ 3d2 (seed 2025, 3407)

**Add Input:** output ของ **3d2** (`ablation_seeds/`) · **3a** (FF++ test) · **3c** (`cdf_crops`) · 3e ที่มี `wdf_crops/` · GPU **T4 ×1** · Internet On
**เวลา:** ประมาณ 40–60 นาที
""")

# ------------------------------------------------------------------ 1. ตั้งค่า
md("## 0. ตั้งค่า")
cfg = C(2)
cfg = rep(cfg, 'OUT_DIR   = WORK_DIR / "results_level3f"', 'OUT_DIR   = WORK_DIR / "results_level3g"')
i = cfg.find("# --- ส่วน 1: paired test ---")
assert i > 0
cfg = cfg[:i] + r'''# --- few-shot (บทความ Sec. IV-C2: 10/20-shot, real:fake เท่ากัน, 2 epoch, lr 2e-5) ---
SHOTS     = [int(x) for x in os.environ.get("FFD_SHOTS", "10,20").split(",")]
N_DRAWS   = int(os.environ.get("FFD_DRAWS", "3"))      # สุ่มชุดภาพกี่รอบ
FT_EPOCHS = 2
FT_LR     = 2e-5
FT_BS     = int(os.environ.get("FFD_FT_BS", "8"))   # batch ต่อ step (T4 16 GB รับ 8 ได้สบาย)
FT_WD     = 0.05
FT_CLIP   = 1.0
FS_VARIANTS = ["fb_ce", "cfb_full"]
EDL_CFG = dict(num_classes=2, evidence="exp", loss_type="log",
               with_kldiv=False, with_avuloss=True, annealing_method="exp")   # = DBBF.yaml
PAPER_TABLE5 = {"DFDC 10-shot": (84.90, 87.72), "DFDC 20-shot": (84.90, 88.07), "FFIW 20-shot": (90.97, 96.65)}

SEED_CKPTS = sorted(p for p in (INPUT_DIR.rglob("ablation_seeds/*_s*/best.pth") if INPUT_DIR.exists() else [])
                    if p.parent.name.rsplit("_s", 1)[0] in FS_VARIANTS)
_seen = {}
for p in SEED_CKPTS: _seen.setdefault(p.parent.name, p)          # Add Input ซ้ำหลายชุด → ใช้ตัวแรก
SEED_CKPTS = list(_seen.values())
assert SEED_CKPTS, "ไม่พบ ablation_seeds/fb_ce_s*/best.pth หรือ cfb_full_s*/best.pth — Add Input เป็น output ของระดับ 3d2"
DATASETS = ["FF++", "CDF", "WDF"]
print("checkpoint:", [p.parent.name for p in SEED_CKPTS], "| shots:", SHOTS, "| draws:", N_DRAWS)
'''
code(cfg)

# ------------------------------------------------------------------ 2. repo
md("## 1. ติดตั้ง + clone repo")
r = C(4)
r = rep(r, "from scipy.stats import rankdata, norm",
        "from scipy.stats import rankdata, norm\n"
        "pip_install(\"albumentations\")\n"
        "import albumentations as alb\n"
        "from albumentations.pytorch.transforms import ToTensorV2\n"
        "import timm.optim as topt\n"
        "EDL = load_module_from_file(\"ffd_edl\", REPO_DIR / \"common\" / \"losses\" / \"edl_loss.py\")   # L_EDU ของผู้เขียน")
code(r)

# ------------------------------------------------------------------ 3. โมเดล
md("## 2. โมเดล (โครงสร้างเดียวกับระดับ 3e/3f) + forward สำหรับการฝึก")
m = C(11)
m = rep(m, 'MODELS = [(f"ours_{p.parent.name}", p) for p in SEED_CKPTS]',
        'MODELS = [(p.parent.name, p) for p in SEED_CKPTS]')
m += r'''

def pearson_correlation_loss(x, y):          # models/custom.py ของ repo (เหมือน 3b/3d2)
    xm = x.sub(x.mean(-1, keepdim=True)); ym = y.sub(y.mean(-1, keepdim=True))
    cov = torch.bmm(xm.unsqueeze(1), ym.unsqueeze(2)).squeeze() / x.size(-1)
    r = (cov / (x.std(-1) * y.std(-1))).mean()
    return torch.clamp(r, -1.0, 1.0)

def train_forward(model, x):
    """คืน logits ของ fused / main / aux และ Pearson loss · ตรงกับ DDBF_BEiT_v2.forward ของ 3b/3d2"""
    f1 = model.backbone_model(x)
    p1 = model.classify(f1)
    if not model.dual:
        return p1, p1, None, f1.new_zeros(())
    f2 = model.backbone_model2(x)
    ploss = pearson_correlation_loss(f1.detach().float(), f2.float())   # stop-gradient ที่ main
    p2 = model.classify(f2)
    if model.ufm:
        u = torch.cat([evidential_uncertainty(p1.float()), evidential_uncertainty(p2.float())], dim=-1)
        w = torch.softmax(-10 * u, dim=-1).to(f1.dtype)
        fused = w[:, :1] * f1 + w[:, 1:] * f2
    else:
        fused = f1 + f2
    return model.classify(fused), p1, p2, ploss
'''
code(m)

# ------------------------------------------------------------------ 4. ข้อมูล
md("## 3. ชุดข้อมูล (FF++ test จาก 3a · Celeb-DF จาก 3c · WildDeepfake จาก 3e)")
code(C(12))
code(C(13))

# ------------------------------------------------------------------ 5. inference
md("## 4. Inference → score ระดับวิดีโอ")
s = C(14)
i = s.find("SCORES, INFO = {}, {}")
assert i > 0
code(s[:i].rstrip())

# ------------------------------------------------------------------ 6. few-shot
md(r"""
## 5. Few-shot: เลือกภาพ · กัน identity leakage · fine-tune

ชื่อวิดีโอ Celeb-DF: `Celeb-real/id0_0000` (identity id0) · `Celeb-synthesis/id0_id16_0000` (id0 และ id16) · `YouTube-real/00000` (ไม่ใช่ identity ใน Celeb)
""")
code(r'''
def cdf_ids(uid):
    sub, stem = uid.split("__", 1) if "__" in uid else ("", uid)
    ids = set(re.findall(r"id\d+", stem))
    return ids if ids else {f"{sub}:{stem}"}

CDF_IDS = {u: cdf_ids(u) for u in DATA["CDF"].uid}

def draw_shots(k, draw):
    """k วิดีโอ (k/2 real + k/2 fake) วิดีโอละ 1 frame · ชุดเดียวกันสำหรับทุกโมเดล"""
    rng = np.random.default_rng(10_000 * k + draw)
    df = DATA["CDF"]
    pick = []
    for lab in (0, 1):
        pool = df[df.label == lab]
        rows = pool.iloc[rng.choice(len(pool), k // 2, replace=False)]
        for _, r_ in rows.iterrows():
            pick.append((r_.uid, r_.frames[rng.integers(len(r_.frames))], int(lab)))
    used = set().union(*[CDF_IDS[u] for u, _, _ in pick])
    rest = df[[not (CDF_IDS[u] & used) for u in df.uid]].reset_index(drop=True)
    return pick, rest, sorted(used)

TF_TRAIN = alb.Compose([
    alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
    alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3), val_shift_limit=(-0.3, 0.3), p=0.3),
    alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
    (alb.ImageCompression(quality_range=(40, 100), p=0.5) if "quality_range" in alb.ImageCompression.__init__.__code__.co_varnames
     else alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5)),
    alb.HorizontalFlip(), alb.ToGray(p=0.01), alb.Resize(IMG_SIZE, IMG_SIZE),
    alb.Normalize(mean=MEAN, std=STD), ToTensorV2()])                    # = create_base_transforms(train) ของ repo

def cutout(img):                                                         # = map_util.Cutout ของ repo
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    length = int(h * np.random.randint(1, 30) * 0.01) * 2
    for _ in range(np.random.randint(1, 4)):
        y, x = np.random.randint(h), np.random.randint(w)
        mask[np.clip(y - length // 2, 0, h):np.clip(y + length // 2, 0, h),
             np.clip(x - length // 2, 0, w):np.clip(x + length // 2, 0, w)] = 0.
    return img * torch.from_numpy(mask)

def load_shot(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return cutout(TF_TRAIN(image=rgb)["image"])

def finetune(model, info, pick, seed):
    """fine-tune ทั้งโมเดลด้วย loss เดิมของ variant · คืนค่า loss ราย step"""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    flags = info["flags"] or {"dual": info["dual"], "dec": True, "ufm": info["ufm"], "edu": True}
    model.train()
    opt = topt.Lamb([p for p in model.parameters() if p.requires_grad], lr=FT_LR, weight_decay=FT_WD)
    try: scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    except (AttributeError, TypeError): scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    ce = nn.CrossEntropyLoss(); edl = EDL.EvidenceLoss(**EDL_CFG).to(DEVICE)
    def edu(o, y, ep):
        d = edl(o, y, epoch=ep, total_epoch=FT_EPOCHS)
        return sum(v.mean() for k_, v in d.items() if "loss" in k_ and torch.is_tensor(v))
    losses = []
    for ep in range(1, FT_EPOCHS + 1):
        order = np.random.permutation(len(pick))
        for b in range(0, len(order), FT_BS):
            idx = order[b:b + FT_BS]
            x = torch.stack([load_shot(pick[j][1]) for j in idx]).to(DEVICE)
            y = torch.tensor([pick[j][2] for j in idx], device=DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                pf, pm, pa, ploss = train_forward(model, x)
            pf, pm = pf.float(), pm.float()
            if info["dual"]:
                branch = (lambda o: edu(o, y, ep)) if flags.get("edu", True) else (lambda o: ce(o, y))
                loss = ce(pf, y) + branch(pm) + branch(pa.float()) + (ploss.float() if flags.get("dec", True) else 0.0)
            else:
                loss = edu(pm, y, ep) if flags.get("edu", False) else ce(pm, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), FT_CLIP)
            scaler.step(opt); scaler.update()
            losses.append(float(loss))
    model.eval()
    return losses

def auc(v): return 100 * roc_auc_score(v.label, v.p_main)
''')

# ------------------------------------------------------------------ 7. main loop
md("## 6. รัน: baseline (ก่อนปรับ) → fine-tune → วัดผลหลังปรับ")
code(r'''
ROWS, BASE = [], {}
SHOT_LOG = []
for k in SHOTS:
    for d in range(N_DRAWS):
        pick, rest, used = draw_shots(k, d)
        SHOT_LOG.append({"k": k, "draw": d, "shots": [u for u, _, _ in pick], "excluded_ids": used,
                         "cdf_rest_n": len(rest), "cdf_rest_real": int((rest.label == 0).sum()),
                         "cdf_rest_fake": int((rest.label == 1).sum())})
json.dump(SHOT_LOG, open(OUT_DIR / "fewshot_draws.json", "w"), indent=1, ensure_ascii=False)
print(pd.DataFrame(SHOT_LOG)[["k", "draw", "cdf_rest_n", "cdf_rest_real", "cdf_rest_fake"]].to_string(index=False))

for name, path in MODELS:
    t0 = time.time()
    model, info = load_detector(path)
    base_sd = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
    BASE[name] = {ds: score_dataset(model, df) for ds, df in DATA.items()}
    for ds, v in BASE[name].items(): v.to_csv(OUT_DIR / f"scores_{name}_base_{ds}.csv", index=False)
    print(f"\n{name}: baseline " + " · ".join(f"{ds} {auc(v):.2f}" for ds, v in BASE[name].items()))
    for k in SHOTS:
        for d in range(N_DRAWS):
            pick, rest, used = draw_shots(k, d)
            model.load_state_dict(base_sd)
            losses = finetune(model, info, pick, seed=1000 * k + d)
            if k == SHOTS[0] and d == 0:                     # ตรวจว่า weight เปลี่ยนจริง (ทุก branch)
                sd_now = model.state_dict()
                for key in ["header.0.weight", "backbone_model.blocks.11.mlp.fc2.weight", "backbone_model2.blocks.11.mlp.fc2.weight"]:
                    if key in sd_now:
                        rel = ((sd_now[key].float().cpu() - base_sd[key].float()).norm() / base_sd[key].float().norm()).item()
                        print(f"    weight เปลี่ยน {key}: {rel:.2e}")
            after = {"CDF": score_dataset(model, rest), "FF++": score_dataset(model, DATA["FF++"]),
                     "WDF": score_dataset(model, DATA["WDF"])}
            b_cdf = BASE[name]["CDF"][BASE[name]["CDF"].uid.isin(set(rest.uid))]
            before = {"CDF": b_cdf, "FF++": BASE[name]["FF++"], "WDF": BASE[name]["WDF"]}
            for ds in DATASETS:
                ROWS.append({"model": name, "variant": name.rsplit("_s", 1)[0], "train_seed": int(name.rsplit("_s", 1)[1]),
                             "k": k, "draw": d, "dataset": ds, "n": len(after[ds]),
                             "AUC_before": auc(before[ds]), "AUC_after": auc(after[ds]),
                             "loss_first": losses[0], "loss_last": losses[-1]})
            r_ = [x for x in ROWS[-3:]]
            print(f"  {k}-shot draw {d}: " + " · ".join(f"{x['dataset']} {x['AUC_before']:.2f}→{x['AUC_after']:.2f}" for x in r_)
                  + f"  (loss {losses[0]:.3f}→{losses[-1]:.3f})")
    del model; torch.cuda.empty_cache()
    print(f"  {(time.time() - t0) / 60:.1f} นาที")

R = pd.DataFrame(ROWS); R["ΔAUC"] = R.AUC_after - R.AUC_before
R.to_csv(OUT_DIR / "fewshot_runs.csv", index=False)
''')

# ------------------------------------------------------------------ 8. สรุป
md(r"""
## 7. สรุป: mean ± SD (ข้าม train seed × การสุ่มชุดภาพ)

- **CDF** = ชุดเป้าหมาย (วัดบนวิดีโอที่เหลือ · identity ไม่ซ้ำกับภาพที่ใช้ปรับ) → ปรับตัวได้ไหม
- **FF++** = ชุดเดิม → ลืมของเดิมไหม (บทความ: FB/CFB "maintains stable performance in the source dataset")
- **WDF** = ไม่เคยเห็นทั้งตอนฝึกและตอนปรับ
""")
code(r'''
S_ = (R.groupby(["variant", "k", "dataset"])
       .agg(before=("AUC_before", "mean"), after=("AUC_after", "mean"),
            delta=("ΔAUC", "mean"), delta_sd=("ΔAUC", "std"), runs=("ΔAUC", "size"))
       .round(2).reset_index())
S_.to_csv(OUT_DIR / "fewshot_summary.csv", index=False)
print("ΔAUC = หลังปรับ − ก่อนปรับ (จุด) · mean ± SD ข้าม train seed × draw")
for ds in ["CDF", "FF++", "WDF"]:
    print(f"\n=== {ds} ===")
    print(S_[S_.dataset == ds].to_string(index=False))

# paired: CFB full vs FB·CE ใช้ชุดภาพเดียวกัน (k, draw) และ train seed เดียวกัน
P = R.pivot_table(index=["train_seed", "k", "draw", "dataset"], columns="variant", values="ΔAUC").dropna()
if {"fb_ce", "cfb_full"} <= set(P.columns):
    P["diff"] = P["cfb_full"] - P["fb_ce"]
    G = P.groupby(["k", "dataset"])["diff"].agg(["mean", "std", "count"]).round(2)
    print("\nΔ(CFB full) − Δ(FB·CE) · paired ตามชุดภาพและ train seed (> 0 = CFB ปรับตัวได้ดีกว่า)")
    print(G.to_string())
    G.to_csv(OUT_DIR / "fewshot_cfb_vs_fb.csv")
print("\nบทความ Table V (CFB, AUC ก่อน → หลัง):", PAPER_TABLE5)
''')
code(r'''
INK2, GRID = "#52514e", "#e6e5e0"
COL = {"fb_ce": "#b7b0a7", "cfb_full": "#7a1f22"}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
for ax, ds in zip(axes, ["CDF", "FF++", "WDF"]):
    for j, v in enumerate(["fb_ce", "cfb_full"]):
        for i_, k in enumerate(SHOTS):
            d = R[(R.dataset == ds) & (R.variant == v) & (R.k == k)]["ΔAUC"].values
            if not len(d): continue
            x = i_ + (j - 0.5) * 0.28
            ax.scatter(np.full(len(d), x) + np.random.uniform(-0.04, 0.04, len(d)), d, s=22, color=COL[v], alpha=0.7)
            ax.hlines(d.mean(), x - 0.11, x + 0.11, color=COL[v], lw=3)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(SHOTS))); ax.set_xticklabels([f"{k}-shot" for k in SHOTS])
    ax.set_title({"CDF": "Celeb-DF (เป้าหมาย)", "FF++": "FF++ (ชุดเดิม)", "WDF": "WildDeepfake (ไม่เคยเห็น)"}[ds], fontsize=11)
    ax.grid(axis="y", color=GRID); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("ΔAUC หลัง − ก่อน (จุด)")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=COL[v], marker="o", lw=3) for v in COL], ["FB · CE", "CFB full"],
           loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(OUT_DIR / "fewshot_delta.png", dpi=160); plt.show()
print("ไฟล์ผลลัพธ์:", sorted(p.name for p in OUT_DIR.iterdir() if not p.name.startswith("scores_")))
print(f"เวลารวม: {(time.time() - T_START) / 60:.1f} นาที")
''')

nb = {"cells": cells, "metadata": SRC.get("metadata", {}), "nbformat": 4, "nbformat_minor": SRC.get("nbformat_minor", 5)}
json.dump(nb, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", out_path, len(cells), "cells")
