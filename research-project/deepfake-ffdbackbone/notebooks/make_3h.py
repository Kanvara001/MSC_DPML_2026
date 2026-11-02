"""สร้าง FFD_level3h_wdf_dupcheck.ipynb — ตรวจภาพซ้ำ WDF test vs FF++ (ทุก split) และ Celeb-DF · CPU อย่างเดียว"""
import json, sys
exec(open("dup_cell.py", encoding="utf-8").read())          # DUP_MD, DUP_CODE (ชุดเดียวกับหัวข้อ 8 ของ 3g)
DUP_CODE = DUP_CODE.replace(
    "    rng_ = np.random.default_rng(0); a_, b_ = rng_.integers(len(hs), size=(2, 5000))\n    base_d = popcount64(hs[a_] ^ hs[b_])",
    "    us = np.array([wdf_uid[f] for f, _ in wdf_h])\n    rng_ = np.random.default_rng(0); a_, b_ = rng_.integers(len(hs), size=(2, 20000))\n"
    "    keep = us[a_] != us[b_]                                   # คนละ sequence เท่านั้น\n    base_d = popcount64(hs[a_[keep]] ^ hs[b_[keep]])")
DUP_CODE = DUP_CODE.replace("ระยะ Hamming ระหว่าง frame สุ่ม 2 ภาพใน WDF ·", "ระยะ Hamming ระหว่าง frame สุ่มจาก 2 sequence ต่างกันใน WDF ·")
assert "keep = us[a_]" in DUP_CODE
cells = []
md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(keepends=True)})
md(r"""
# ระดับ 3h — ตรวจภาพซ้ำ: WildDeepfake test vs ข้อมูลที่ใช้ฝึก/ปรับ

WildDeepfake ใช้เป็นชุดทดสอบที่ "ไม่เคยเห็น" ใน 3e, 3f, 3g → ตรวจว่ามี frame ซ้ำหรือเกือบซ้ำกับ
**FF++ (ทุก split, ข้อมูล fine-tune)** หรือ **Celeb-DF (ข้อมูล few-shot)** หรือไม่

**Add Input:** **3a** (tar ของ FF++) · **3c** (`cdf_crops`) · 3e ที่มี `wdf_crops/` · **ไม่ต้องใช้ GPU / Internet** · ประมาณ 5–10 นาที
""")
code(r'''
import os, re, json, time, tarfile
from pathlib import Path
import numpy as np, pandas as pd, cv2
T_START = time.time()
ON_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR  = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FFD_WORK", "./work")).resolve()
INPUT_DIR = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("FFD_INPUT", str(WORK_DIR / "input")))
OUT_DIR   = WORK_DIR / "results_level3h"; OUT_DIR.mkdir(parents=True, exist_ok=True)
FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
N_FRAMES = 8

# WildDeepfake test (crop จากระดับ 3e)
metas = sorted(INPUT_DIR.rglob("wdf_crops/wdf_meta.csv"))
assert metas, "ไม่พบ wdf_crops/wdf_meta.csv — Add Input เป็น output ของ 3e ที่มี wdf_crops"
w = pd.read_csv(metas[0]); root = metas[0].parent
w["frames"] = w["frames"].map(lambda s: [str(root / x) for x in json.loads(s)])
# Celeb-DF (crop จากระดับ 3c)
roots = sorted(p for p in INPUT_DIR.rglob("cdf_crops") if p.is_dir())
assert roots, "ไม่พบ cdf_crops — Add Input เป็น output ของ 3c"
cdf = pd.DataFrame([{"uid": d.name, "frames": [str(p) for p in sorted(d.glob("*.png"))]}
                    for d in sorted(p for p in roots[0].iterdir() if p.is_dir())])
DATA = {"WDF": w, "CDF": cdf}
print(f"WDF test {len(w)} sequence · Celeb-DF {len(cdf)} วิดีโอ · tar ของ FF++:",
      sorted(t.name for t in INPUT_DIR.rglob("*.tar") if t.stem in ["original"] + FF_METHODS))
''')
md(DUP_MD.replace("## 8. ", "## 1. "))
code(DUP_CODE)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1])
