"""สร้าง FFD_level3d2_ablation_seeds.ipynb — ฝึก variant เดิมซ้ำด้วย seed ใหม่ (จาก notebook 3d)"""
import json, copy

nb = json.load(open("FFD_level3d_ablation_train.ipynb", encoding="utf-8"))
C = nb["cells"]
src = lambda i: "".join(C[i]["source"])
def put(i, s): C[i]["source"] = s.splitlines(keepends=True); C[i]["outputs"] = [] if C[i]["cell_type"] == "code" else C[i].get("outputs")
def rep(s, a, b):
    assert a in s, a[:60]; return s.replace(a, b)

put(0, r"""# ระดับ 3d2 — Ablation ซ้ำด้วย seed ใหม่ (ตรวจความแปรปรวนจากการฝึก)

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026

**เป้าหมาย:** ระดับ 3d ฝึกทุก variant ด้วย seed เดียว (1234 ตาม `DBBF.yaml`) ผลกลับด้านจาก Table VI แต่ต่างกันไม่เกิน 3 จุด
notebook นี้ฝึก **variant เดิมซ้ำด้วย seed อื่น** เพื่อดูว่า AUC แกว่งตาม seed แค่ไหน และแนวโน้มยังกลับด้านอยู่หรือไม่

โค้ดเหมือน 3d ทุกบรรทัด ต่างกันเฉพาะ `CFG["seed"]` (มีผลกับ head ที่สุ่ม init, ลำดับ batch, Cutout, drop path)
backbone เริ่มจาก FB weight เดียวกันทุก seed

| ค่าเริ่มต้นใน `JOBS` | เวลา (T4 ×2) |
|---|---|
| `fb_ce` × seed 2025, 3407 | ~50 นาที/run |
| `cfb_full` × seed 2025, 3407 | ~100 นาที/run |
| **รวม 4 run** | **~5 ชม.** |

เลือก 2 variant ที่ผลต่างกันมากที่สุดใน 3d (fb_ce สูงสุด, cfb_full ต่ำสุด)
ถ้ามีหลาย Kaggle account แบ่งรันได้ เช่น account A: `JOBS = [("fb_ce", 2025), ("cfb_full", 2025)]` · account B: seed 3407

**ก่อนรัน:** GPU **T4 ×2** · Internet **On** · Add Input = output ของระดับ 3a (เหมือน 3b, 3d)
ผลลัพธ์อยู่ที่ `ablation_seeds/<variant>_s<seed>/best.pth` → ประเมินใน **ระดับ 3f**
""")

s = src(2)
s = rep(s, 'RUN_ROOT  = WORK_DIR / "ablation_runs"          # ablation_runs/<variant>/ (ต้องอยู่ใน /kaggle/working)',
           'RUN_ROOT  = WORK_DIR / "ablation_seeds"         # ablation_seeds/<variant>_s<seed>/ (ต้องอยู่ใน /kaggle/working)')
s = rep(s, 'OUT_DIR   = WORK_DIR / "results_level3d"', 'OUT_DIR   = WORK_DIR / "results_level3d2"')
s = rep(s, 'VARIANTS = ["fb_ce", "cfb_base", "cfb_dec", "cfb_dec_ufm", "fb_edu"]',
           '# (variant, seed) ที่จะฝึก · seed 1234 ฝึกไปแล้วใน 3b / 3d\n'
           'JOBS = [("fb_ce", 2025), ("cfb_full", 2025), ("fb_ce", 3407), ("cfb_full", 3407)]\n'
           'VARIANTS = sorted({v for v, _ in JOBS})')
s = rep(s, 'print("ON_KAGGLE:", ON_KAGGLE, "| RUN_ROOT:", RUN_ROOT, "| VARIANTS:", VARIANTS)',
           'assert all(sd != 1234 for _, sd in JOBS), "seed 1234 ฝึกไปแล้วใน 3b/3d"\n'
           'print("ON_KAGGLE:", ON_KAGGLE, "| RUN_ROOT:", RUN_ROOT, "| JOBS:", JOBS)')
put(2, s)

s = src(18)
s = rep(s, 'def run_variant(variant):\n    run_dir = RUN_ROOT / variant',
           'def run_variant(variant, seed):\n    tag = f"{variant}_s{seed}"\n    run_dir = RUN_ROOT / tag')
s = rep(s, 'done = sorted(INPUT_DIR.rglob(f"ablation_runs/{variant}/final.json"))',
           'done = sorted(INPUT_DIR.rglob(f"ablation_seeds/{variant}/final.json"))')
s = rep(s, 'part = sorted(INPUT_DIR.rglob(f"ablation_runs/{variant}/last.pth"))',
           'part = sorted(INPUT_DIR.rglob(f"ablation_seeds/{variant}/last.pth"))')
s = rep(s, 'state = restore_from_input(variant, run_dir)', 'state = restore_from_input(tag, run_dir)')
s = rep(s, 'print(f"[{variant}] เสร็จแล้วจาก session ก่อน → ข้าม"); return 0',
           'print(f"[{tag}] เสร็จแล้วจาก session ก่อน → ข้าม"); return 0')
s = rep(s, 'cfg = dict(CFG, max_train_hours=max(0.5, left_h - 0.4), variant=variant, flags=VARIANT_FLAGS[variant])',
           'cfg = dict(CFG, seed=seed, max_train_hours=max(0.5, left_h - 0.4), variant=variant, flags=VARIANT_FLAGS[variant])')
s = rep(s, 'json.dump({"variant": variant, "flags": VARIANT_FLAGS[variant],',
           'json.dump({"variant": variant, "seed": seed, "flags": VARIANT_FLAGS[variant],')
s = rep(s, 'print("=" * 70, f"\\n[{variant}] flags={VARIANT_FLAGS[variant]} | state={state}", flush=True)',
           'print("=" * 70, f"\\n[{tag}] seed={seed} flags={VARIANT_FLAGS[variant]} | state={state}", flush=True)')
put(18, s)

put(19, r'''status = {}
for v, sd in JOBS:
    status[f"{v}_s{sd}"] = run_variant(v, sd)
print("\nสถานะ:", status)
''')

put(21, r'''rows = []
for v, sd in JOBS:
    rd = RUN_ROOT / f"{v}_s{sd}"; fj = rd / "final.json"
    if not fj.exists():
        rows.append({"variant": v, "seed": sd, "status": "ยังไม่เสร็จ"}); continue
    fin = json.load(open(fj))
    rows.append({"variant": v, "seed": sd, "status": "เสร็จ",
                 "best val AUC": round(100 * fin["best_val_auc"], 2),
                 "FF++ test AUC": round(100 * fin["test_auc"], 2), "FF++ test ACC": round(100 * fin["test_acc"], 2)})
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "seed_ffpp_summary.csv", index=False)
print(summary.to_string(index=False))
print("\nseed 1234 (จาก 3b/3d): fb_ce val 97.31 / test 97.37 · cfb_full val 96.45 / test 96.72  (fused head สำหรับ 2 branch)")
print("ผล cross-dataset (main branch) และการทดสอบทางสถิติ → ระดับ 3f")
print(f"\nเวลารวม notebook: {(time.time()-T_START)/60:.1f} นาที")
''')

put(22, r"""## 8. ขั้นต่อไป

1. **Save Version → Save & Run All (Commit)** เพื่อเก็บ `ablation_seeds/<variant>_s<seed>/best.pth`
2. ถ้ายังไม่ครบทุก job → Add Input เป็น output ของรอบนี้ แล้ว Save Version อีกรอบ (job ที่เสร็จแล้วจะถูกข้าม)
3. ไปที่ **ระดับ 3f** — Add Input เป็น output ของ 3a, 3c, 3e และ 3d2 (ทุก account) แล้วรันการทดสอบทางสถิติ
""")

json.dump(nb, open("FFD_level3d2_ablation_seeds.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
