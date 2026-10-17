"""สร้าง FFD_level3d_ablation_train.ipynb จาก 3b โดยเพิ่มการเปิด/ปิดองค์ประกอบตาม Table VI"""
import json, copy, re

src = json.load(open("FFD_level3b_CFB_finetune.ipynb", encoding="utf-8"))
nb = copy.deepcopy(src)
C = nb["cells"]

def set_src(i, text):
    C[i]["source"] = [l for l in text.splitlines(keepends=True)]
    if C[i]["cell_type"] == "code":
        C[i]["outputs"] = []; C[i]["execution_count"] = None

def get(i): return "".join(C[i]["source"])

def must_replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)

# ---------------------------------------------------------------- [0] title
set_src(0, r"""# ระดับ 3d — Ablation study (Table VI) : เทรนโมเดลที่ปิดบางองค์ประกอบ

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026

**เป้าหมาย:** ทำซ้ำ Table VI — วัดว่าแต่ละองค์ประกอบของ CFB (`L_dec`, UFM, `L_EDU`) ช่วยเพิ่มผลจริงหรือไม่
โดยเทรนทุก variant ด้วย **ข้อมูล, hyperparameter และ pre-trained backbone เดียวกับระดับ 3b ทุกอย่าง** ต่างกันเฉพาะองค์ประกอบที่ปิด

| variant | branch | `L_dec` | UFM | `L_EDU` | paper CDF | paper Avg AUC (CDF/DFDC/FFIW) |
|---|---|---|---|---|---|---|
| `fb_ce`        | 1 | | | | 89.18 | 86.56 |
| `fb_edu`       | 1 | | | ✓ | 89.34 | 87.43 |
| `cfb_base`     | 2 | | | | 89.77 | 87.04 |
| `cfb_dec`      | 2 | ✓ | | | 90.31 | 88.14 |
| `cfb_dec_ufm`  | 2 | ✓ | ✓ | | 91.16 | 88.64 |
| `cfb_full`     | 2 | ✓ | ✓ | ✓ | 90.46 | 88.78 ← **ระดับ 3b แล้ว ไม่ต้องเทรนซ้ำ** |

กติกาตาม caption ของ Table VI: **ไม่มี UFM → รวม feature 2 branch ด้วยการบวก** · **ไม่มี `L_EDU` → ใช้ `L_ce` แทน**

**ก่อนรัน:** GPU **T4 ×2** · Internet **On** · Add Input = output ของระดับ 3a (เหมือน 3b)
เวลา: branch เดียว ~50 นาที · 2 branch ~100 นาที → ครบ 5 variant ~6.5 ชม. (อยู่ใน 12 ชม. ของ Kaggle)

ถ้าอยากแบ่งรันหลาย session ให้แก้ `VARIANTS` ในหัวข้อ 0 เช่น `["fb_ce", "cfb_base"]`
ผลลัพธ์อยู่ที่ `ablation_runs/<variant>/best.pth` → นำไปประเมินใน **ระดับ 3e**
""")

# ---------------------------------------------------------------- [2] settings
s = get(2)
s = must_replace(s, 'RUN_DIR   = WORK_DIR / "cfb_run"                # checkpoint + log (ต้องอยู่ใน /kaggle/working เพื่อให้ commit เก็บ)',
                 'RUN_ROOT  = WORK_DIR / "ablation_runs"          # ablation_runs/<variant>/ (ต้องอยู่ใน /kaggle/working)\nRUN_DIR   = RUN_ROOT                            # จะถูกตั้งใหม่ต่อ variant ในหัวข้อ 6')
s = must_replace(s, 'OUT_DIR   = WORK_DIR / "results_level3b"', 'OUT_DIR   = WORK_DIR / "results_level3d"')
s = must_replace(s, "for d in [TMP_DIR, DATA_ROOT, RUN_DIR, OUT_DIR]:", "for d in [TMP_DIR, DATA_ROOT, RUN_ROOT, OUT_DIR]:")
s = must_replace(s, "PAPER_FFPP_AUC = 99.36", r'''PAPER_FFPP_AUC = 99.36

# --- ablation (Table VI) ---
# dual = 2 branch · dec = L_dec · ufm = uncertainty-based fusion (ไม่มี = บวก feature) · edu = L_EDL+L_EUC (ไม่มี = L_ce)
VARIANT_FLAGS = {
    "fb_ce":       dict(dual=False, dec=False, ufm=False, edu=False),
    "fb_edu":      dict(dual=False, dec=False, ufm=False, edu=True),
    "cfb_base":    dict(dual=True,  dec=False, ufm=False, edu=False),
    "cfb_dec":     dict(dual=True,  dec=True,  ufm=False, edu=False),
    "cfb_dec_ufm": dict(dual=True,  dec=True,  ufm=True,  edu=False),
    "cfb_full":    dict(dual=True,  dec=True,  ufm=True,  edu=True),     # = ระดับ 3b
}
PAPER_TABLE6 = {   # Table VI: CDF, DFDC, FFIW, Avg AUC, Avg ACC
    "fb_ce": (89.18, 84.02, 86.48, 86.56, 79.37), "fb_edu": (89.34, 84.19, 88.76, 87.43, 80.96),
    "cfb_base": (89.77, 83.67, 87.69, 87.04, 79.26), "cfb_dec": (90.31, 84.89, 89.21, 88.14, 79.42),
    "cfb_dec_ufm": (91.16, 84.54, 90.22, 88.64, 80.84), "cfb_full": (90.46, 84.90, 90.97, 88.78, 82.29),
}
# ลำดับความสำคัญ: 2 ตัวแรกคือ baseline หลัก (ไม่มีองค์ประกอบใหม่ของ paper เลย)
VARIANTS = ["fb_ce", "cfb_base", "cfb_dec", "cfb_dec_ufm", "fb_edu"]
NOTEBOOK_HOURS = 11.3          # เผื่อเวลาให้ commit ก่อนชน 12 ชม. — variant ที่เริ่มไม่ทันจะถูกข้าม (รันต่อ session หน้าได้)''')
s = must_replace(s, 'print("ON_KAGGLE:", ON_KAGGLE, "| RUN_DIR:", RUN_DIR)',
                 'assert all(v in VARIANT_FLAGS for v in VARIANTS), "ชื่อ variant ไม่ถูกต้อง"\nprint("ON_KAGGLE:", ON_KAGGLE, "| RUN_ROOT:", RUN_ROOT, "| VARIANTS:", VARIANTS)')
set_src(2, s)

# ---------------------------------------------------------------- [13] markdown before script
set_src(13, r"""## 5. เขียนสคริปต์เทรน (DDP) — เหมือนระดับ 3b ทุกบรรทัด ยกเว้นส่วนที่เปิด/ปิดได้

สิ่งที่เปลี่ยนจาก 3b มีแค่ 2 จุด (ควบคุมด้วย `flags` ใน config):

1. **โมเดล** — `dual=False` ใช้ backbone เดียว (= FB) · `ufm=False` รวม feature แบบ `f_M + f_A`
2. **loss** — `edu=False` ใช้ `L_ce` แทน `L_EDU` ใน branch · `dec=False` ไม่บวก `L_dec` (ยังคำนวณ Pearson ไว้ดูอย่างเดียว)

การเลือก checkpoint และ val/test AUC ใช้ output ตัวเดียวกับ 3b (fused head สำหรับ 2 branch, main สำหรับ 1 branch)
ส่วน **main branch** ตาม paper จะประเมินใหม่ทุกตัวในระดับ 3e
""")

# ---------------------------------------------------------------- [14] training script
s = get(14)
s = must_replace(s, 'SCRIPT = RUN_DIR / "train_cfb.py"', 'SCRIPT = RUN_ROOT / "train_ablation.py"')
s = must_replace(s, "RUN_DIR = Path(A['run_dir']); RUN_DIR.mkdir(parents=True, exist_ok=True)",
                 "RUN_DIR = Path(A['run_dir']); RUN_DIR.mkdir(parents=True, exist_ok=True)\n"
                 "FL = A.get('flags', dict(dual=True, dec=True, ufm=True, edu=True))\n"
                 "VARIANT = A.get('variant', 'cfb_full')")

old_model = s[s.index("class DDBF_BEiT_v2(nn.Module):"): s.index("# ---------------- optimizer")]
new_model = r'''class DDBF_BEiT_v2(nn.Module):
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

'''
s = s.replace(old_model, new_model)

s = must_replace(s, "    model = DDBF_BEiT_v2(CFG, A['pretrain_path'], CFG['grad_ckpt']).to(DEV)",
                 "    model = DDBF_BEiT_v2(CFG, A['pretrain_path'], CFG['grad_ckpt'], FL).to(DEV)\n"
                 "    log(f'variant {VARIANT} | flags {FL}')")

old_loss = s[s.index("            T = out.size(1)"): s.index("            scaler.scale(loss")]
new_loss = r'''            T = out.size(1)
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
'''
s = s.replace(old_loss, new_loss)
s = must_replace(s, "        json.dump({'test_auc': te_auc, 'test_acc': te_acc, 'best_val_auc': best_auc},",
                 "        json.dump({'test_auc': te_auc, 'test_acc': te_acc, 'best_val_auc': best_auc,\n"
                 "                   'variant': VARIANT, 'flags': FL},")
s = must_replace(s, "                            'val_auc': va_auc, 'cfg': CFG}, RUN_DIR/'best.pth')",
                 "                            'val_auc': va_auc, 'cfg': CFG, 'variant': VARIANT, 'flags': FL},\n"
                 "                           RUN_DIR/'best.pth')")
set_src(14, s)

# ---------------------------------------------------------------- [15]-[19] run loop
set_src(15, "### 5.1 ข้อมูลที่ทุก variant ใช้ร่วมกัน (เหมือน 3b)\n")
set_src(16, r'''base_payload = {
    "cfg": CFG,
    "repo_dir": str(REPO_DIR),
    "pretrain_path": str(pretrain_file),
    "compression": COMPRESSION,
    "train_items": train_frames,
    "val_videos":  [{"frames": r["frames"], "label": int(r["label"])} for _, r in index["val"].iterrows()],
    "test_videos": [{"frames": r["frames"], "label": int(r["label"])} for _, r in index["test"].iterrows()],
}
print(f"train frames {len(base_payload['train_items']):,} | val {len(base_payload['val_videos'])} | test {len(base_payload['test_videos'])} วิดีโอ")
eff_bs = CFG["batch_size"] * max(1, N_GPU) * CFG["grad_accum"]
print(f"effective batch size = {CFG['batch_size']} x {N_GPU} GPU x {CFG['grad_accum']} accum = {eff_bs}  (paper: 64)")
if eff_bs != 64:
    print("[WARN] ไม่เท่ากับ 64 ของ paper — ปรับ batch_size/grad_accum ใน CFG ให้ได้ 64")
''')
set_src(17, r"""## 6. เทรนทีละ variant

- variant ที่มี `final.json` อยู่แล้ว (จาก session ก่อนที่ Add Input ไว้) จะ**ข้าม**
- variant ที่ค้างกลางทาง (มี `last.pth`) จะ **resume**
- เทรนเสร็จแล้วจะลบ `last.pth` (2 GB) เหลือแค่ `best.pth` เพื่อไม่ให้ output เกิน 20 GB ของ Kaggle
- ถ้าเวลาไม่พอ variant ที่เหลือจะถูกข้าม → Save Version อีกรอบโดย Add Input เป็น output ของรอบนี้
""")
set_src(18, r'''def restore_from_input(variant, run_dir):
    """คัดลอกผลของ variant นี้จาก session ก่อน (ถ้า Add Input ไว้)"""
    if not INPUT_DIR.exists():
        return "none"
    done = sorted(INPUT_DIR.rglob(f"ablation_runs/{variant}/final.json"))
    if done:
        src = done[0].parent
        for f in ["best.pth", "final.json", "history.json", "train.log", "test_scores.npy", "config_meta.json"]:
            if (src / f).exists() and not (run_dir / f).exists():
                shutil.copyfile(src / f, run_dir / f)
        return "done"
    part = sorted(INPUT_DIR.rglob(f"ablation_runs/{variant}/last.pth"))
    if part and not (run_dir / "last.pth").exists():
        src = part[0].parent
        for f in ["last.pth", "best.pth", "history.json", "train.log"]:
            if (src / f).exists():
                shutil.copyfile(src / f, run_dir / f)
        return "resume"
    return "none"

def run_variant(variant):
    run_dir = RUN_ROOT / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    state = restore_from_input(variant, run_dir)
    if (run_dir / "final.json").exists():
        print(f"[{variant}] เสร็จแล้วจาก session ก่อน → ข้าม"); return 0
    left_h = NOTEBOOK_HOURS - (time.time() - T_START) / 3600
    need_h = 1.8 if VARIANT_FLAGS[variant]["dual"] else 1.0
    if left_h < need_h:
        print(f"[{variant}] เวลาเหลือ {left_h:.1f} ชม. ไม่พอ (ต้องการ ~{need_h} ชม.) → ข้าม รันต่อ session หน้า"); return None
    cfg = dict(CFG, max_train_hours=max(0.5, left_h - 0.4), variant=variant, flags=VARIANT_FLAGS[variant])
    payload = dict(base_payload, cfg=cfg, run_dir=str(run_dir), variant=variant, flags=VARIANT_FLAGS[variant])
    cfg_path = run_dir / "config.json"
    json.dump(payload, open(cfg_path, "w"))
    json.dump({"variant": variant, "flags": VARIANT_FLAGS[variant], "pretrain_sha256": PRETRAIN_SHA,
               "repo_commit": REPO_COMMIT, "env": ENV}, open(run_dir / "config_meta.json", "w"), indent=1)
    print("=" * 70, f"\n[{variant}] flags={VARIANT_FLAGS[variant]} | state={state}", flush=True)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(N_GPU),
           "--master_port", "11023", str(SCRIPT), "--cfg", str(cfg_path)]
    env = {**os.environ, "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
           "PYTHONWARNINGS": "ignore", "NCCL_P2P_DISABLE": "1"}
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    print(f"[{variant}] exit code {rc} · {(time.time()-t0)/60:.1f} นาที")
    if rc == 0 and (run_dir / "final.json").exists():
        (run_dir / "last.pth").unlink(missing_ok=True)          # เก็บแค่ best.pth
        (run_dir / "config.json").unlink(missing_ok=True)       # config ใหญ่ (รายชื่อ frame) ไม่ต้องเก็บ
    elif rc != 0:
        print("[ERROR] ดู traceback ด้านบน · ถ้า CUDA out of memory: CFG['batch_size']=16, CFG['grad_accum']=2")
    return rc
''')
set_src(19, r'''status = {}
for v in VARIANTS:
    status[v] = run_variant(v)
print("\nสถานะ:", status)
''')

# ---------------------------------------------------------------- [21] results
set_src(21, r'''rows, hists = [], {}
for v in ["cfb_full"] + VARIANTS:
    rd = RUN_ROOT / v
    fj = rd / "final.json"
    if v == "cfb_full":
        hits = sorted(INPUT_DIR.rglob("cfb_run/final.json")) if INPUT_DIR.exists() else []
        if not hits:
            continue
        fj = hits[0]; rd = fj.parent
    if not fj.exists():
        rows.append({"variant": v, **VARIANT_FLAGS[v], "status": "ยังไม่เสร็จ"}); continue
    fin = json.load(open(fj))
    if (rd / "history.json").exists():
        hists[v] = pd.DataFrame(json.load(open(rd / "history.json")))
    rows.append({"variant": v, **VARIANT_FLAGS[v], "status": "เสร็จ",
                 "best val AUC": round(100 * fin["best_val_auc"], 2),
                 "FF++ test AUC": round(100 * fin["test_auc"], 2), "FF++ test ACC": round(100 * fin["test_acc"], 2),
                 "paper CDF AUC": PAPER_TABLE6[v][0]})
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "ablation_ffpp_summary.csv", index=False)
print(summary.to_string(index=False))
print("\n* FF++ test AUC ในตารางนี้ใช้ fused head (2 branch) / main (1 branch) เหมือน 3b — ตัวเลข main branch และ cross-dataset ดูระดับ 3e")

if hists:
    INK2, GRID = "#52514e", "#e6e5e0"
    plt.rcParams.update({"axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for v, h in hists.items():
        axes[0].plot(h["epoch"], h["loss"], marker=".", label=v)
        axes[1].plot(h["epoch"], 100 * h["val_auc"], marker="o", label=v)
    axes[0].set(xlabel="epoch", ylabel="training loss", title="Training loss")
    axes[1].set(xlabel="epoch", ylabel="val video-AUC (%)", title="Validation (FF++ c23)")
    for ax in axes: ax.grid(color=GRID, lw=0.8); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT_DIR / "ablation_curves.png", dpi=150); plt.show()
print(f"\nเวลารวม notebook: {(time.time()-T_START)/60:.1f} นาที")
''')
set_src(22, r"""## 8. ขั้นต่อไป

1. **Save Version → Save & Run All (Commit)** เพื่อเก็บ `ablation_runs/<variant>/best.pth`
2. ถ้ายังไม่ครบทุก variant → Add Input เป็น output ของรอบนี้ แล้ว Save Version อีกรอบ (variant ที่เสร็จแล้วจะถูกข้าม)
3. ไปที่ **ระดับ 3e** — Add Input เป็น output ของ 3a, 3b, 3c และ 3d แล้วประเมินทุกโมเดลบน FF++ / Celeb-DF / WildDeepfake
""")

for c in C:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
json.dump(nb, open("FFD_level3d_ablation_train.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote FFD_level3d_ablation_train.ipynb", len(C), "cells")
