"""สร้าง FFD_level3i_DF40_eval.ipynb — ประเมินบน DF40 (7 วิธีของ Table IV) โดยดาวน์โหลดใน Kaggle ด้วย gdown

ใช้โค้ดโหลดโมเดล / inference / realign ชุดเดียวกับระดับ 3e (ดึง cell มาจาก FFD_level3e_eval_utom_wdf.ipynb)
"""
import json

base = json.load(open("FFD_level3e_eval_utom_wdf.ipynb", encoding="utf-8"))
src = lambda i: "".join(base["cells"][i]["source"])

S_SETUP   = src(4)                                   # pip + clone repo + patch + imports
S_MODEL   = src(6)                                   # Detector / load_detector
S_AUTHOR  = src(7)                                   # หา checkpoint ผู้เขียน + ของเรา -> MODELS
S_REALIGN = src(11).split("IMG_EXT =")[0]            # FaceDetector + crop_face_test + realign
S_INFER   = src(13).split("SCORES, INFO = {}, {}")[0]  # to_tensor / FrameList / score_dataset
assert "class Detector" in S_MODEL and "def realign" in S_REALIGN and "def score_dataset" in S_INFER

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                            "source": t.splitlines(keepends=True)})

md(r"""# ระดับ 3i — ทดสอบบน DF40 (Table IV ของ paper)

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026
**DF40:** Yan et al., *DF40: Toward Next-Generation Deepfake Detection*, NeurIPS 2024 (CC BY-NC 4.0 · ใช้เพื่อการศึกษาเท่านั้น)

Table IV ของ paper ใช้ DF40 แค่ **7 วิธี**: SadTalker · FOMM · FaceDancer · MobileSwap · SimSwap · InSwapper · UniFace
(CFB: Avg AUC 96.22) — notebook นี้ทำซ้ำเฉพาะ 7 วิธีนี้

| หัวข้อ | สิ่งที่ทำ |
|---|---|
| 2 | ดาวน์โหลด JSON + zip ทีละไฟล์จาก Google Drive (gdown) → แตกเฉพาะ 8 frame ของวิดีโอ test → ลบ zip |
| 3 | (ถ้าเปิด) RetinaFace + align ซ้ำแบบเดียวกับ FF++/CDF ของเรา |
| 4 | inference ทุกโมเดล (main branch ตาม paper) |
| 5 | AUC ระดับวิดีโอ แยกวิธี × โดเมน เทียบกับ Table IV |

**โดเมนของ DF40:** แต่ละวิธีมีภาพปลอม 2 ชุด
- `ff`  = สลับ/ขับใบหน้าจากวิดีโอ FF++ (test 140 วิดีโอ) · real = FF++ test 140 วิดีโอ
- `cdf` = ใช้ใบหน้าจาก Celeb-DF · real = Celeb-DF test 178 วิดีโอ → **ข้ามโดเมนจริง** (โมเดลไม่เคยเห็นใบหน้า CDF)

paper ไม่ได้ระบุว่า Table IV ใช้โดเมนไหน → รายงานทั้ง `ff`, `cdf` และ `all` (รวม 2 โดเมน)

**ภาพจริงต้องมาจาก DF40 เอง** (crop ด้วย pipeline เดียวกับภาพปลอม) — ถ้าเอาภาพจริงจาก pipeline ของเรามาผสม
โมเดลอาจแยกจริง/ปลอมจากวิธี crop แทนร่องรอยการปลอม → AUC สูงเกินจริง

**ก่อนรัน:** GPU **T4 ×1** · Internet **On** · Add Input: output ของระดับ **3b** (`cfb_run`) · (ถ้ามี) **3d**
ถ้า gdown ติดโควตา Google Drive: ดาวน์โหลด zip เองแล้วอัปเป็น Kaggle Dataset → Add Input (notebook หาเจอเองจากชื่อไฟล์)
**ต้องเป็น zip ชุด test** (โฟลเดอร์ Drive ชื่อ `DF40`) ไม่ใช่ `DF40_train` — notebook ตรวจให้และจะหยุดถ้าผิดชุด
""")

md("## 0. ตั้งค่า")
code(r'''import os, sys, re, json, time, glob, random, shutil, zipfile, tarfile, subprocess, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
T_START = time.time()

ON_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR  = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FFD_WORK", "./work")).resolve()
TMP_DIR   = Path("/tmp/ffd") if ON_KAGGLE else WORK_DIR / "tmp"
INPUT_DIR = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("FFD_INPUT", str(WORK_DIR / "input")))
REPO_DIR  = TMP_DIR / "FFDBackbone"
DL_DIR    = TMP_DIR / "df40_dl"                  # zip ชั่วคราว (ลบทิ้งหลังแตก)
DF40_DIR  = WORK_DIR / "df40_crops"              # 8 frame/วิดีโอ ที่แตกแล้ว — เก็บใน working ใช้ซ้ำรอบหน้าได้ (Add Input)
WEIGHT_DIR = TMP_DIR / "weights"
OUT_DIR   = WORK_DIR / "results_level3i"
for d in [TMP_DIR, DL_DIR, DF40_DIR, WEIGHT_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/zhenglab/FFDBackbone.git"
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G"   # checkpoint ผู้เขียน

# --- โมเดล ---
USE_AUTHOR_CKPT   = True
AUTHOR_CKPT_PATH  = None
AUTHOR_HINTS      = ["ours", "final", "ddbf", "dbbf", "mainbranch"]
AUTHOR_EXCLUDE    = ["beit-1k-face-55w"]
INCLUDE_ABLATION  = ["fb_ce"]                    # variant จากระดับ 3d ที่จะประเมินด้วย ([] = ไม่เอา, None = ทุกตัว)

# --- DF40: 7 วิธีของ Table IV (ชื่อไฟล์ใน Drive → ชื่อใน paper) ---
METHODS = ["sadtalker", "fomm", "facedancer", "mobileswap", "simswap", "inswap", "uniface"]
PAPER_NAME = {"sadtalker": "SadTalker", "fomm": "FOMM", "facedancer": "FaceDancer", "mobileswap": "MobileSwap",
              "simswap": "SimSwap", "inswap": "InSwapper", "uniface": "UniFace"}
DOMAINS = ["ff", "cdf"]
PREPS   = ["raw"]                                # ["raw", "realign"] ถ้าต้องการเทียบ crop แบบเรา (+~40 นาที) · raw = ภาพ DF40 resize เป็น 224 · realign = RetinaFace + align แบบของเรา

# Google Drive file ID — copy ของโฟลเดอร์ "DF40" (ชุด test) ใน Drive ของเรา (DF40_copy_for_kaggle) เพราะไฟล์ต้นฉบับติดโควตา
# ต้นฉบับ: https://drive.google.com/drive/folders/1980LCMAutfWvV6zvdxhoeIa67TmzKLQ_ · ลบ copy หลังใช้ (DF40: CC BY-NC, ห้ามแจกจ่ายต่อ)
FAKE_ZIP_ID = {
    "sadtalker":  "1JNjcKq1F-GW-7M-Kbn8DaR2We6rbsca7",
    "fomm":       "1ihDPVIjT-7S6tpLZDWLqAAg5fTJmwSRK",
    "facedancer": "1utlUtqt2x3CWcOa5VzOruhZFSXne4X6S",
    "mobileswap": "1xixQ4tIfwt6uR8W99jnV9Z5ocJxu7N91",
    "simswap":    "1AgBMoUuhYw5j0rKdPgqzvGpNuHlIo8cz",
    "inswap":     "18copihk3sO6yRJeSTOM5W-pyDgikLKMU",
    "uniface":    "1ZLmMUf3IZXAP8O0cCE4xwBzKwnaRcgLn",
}
REAL_ZIP = {   # domain → (ชื่อไฟล์, file ID) — ลิงก์ CDF ในหน้าฟอร์มผิด (ชี้ไป check.py) ใช้ลิงก์จาก README แทน
    "ff":  ("FaceForensics++_real_data_for_DF40.zip", "1mBoYVjZgMzn5sF5hGFtZWH_E4qJy1uol"),
    "cdf": ("Celeb-DF-v2_real_data_for_DF40.zip",     "1e33Vs8tdhDPihMBWKg_bX2SA_eebKt7k"),
}
JSON_ID = {   # dataset_json/<method>_<domain>.json
    ("sadtalker", "ff"): "1NXm1zGmPVxJKI3MT71JmzftgLFu1rIu6",  ("sadtalker", "cdf"): "1NH_9eun0X7THJPBlNILbkt5V-K1CaqNm",
    ("fomm", "ff"):      "1-Adz58aUMFanQLeu1IvwaKHgL0FsVbOJ",  ("fomm", "cdf"):      "1x7VF-A_1IWURHU6JZ60rIDaqWE14IvzO",
    ("facedancer", "ff"): "1IFQBDF781qF_BkD1XpEXTmxRRxlqHZs5", ("facedancer", "cdf"): "1_BJb5GXLPAiejv4sNw8tymiB_wYfjYEz",
    ("mobileswap", "ff"): "1bKTc6-bIXHJk9tGVATLfdtKtiikBl_GY", ("mobileswap", "cdf"): "1odjJBtax-bLvY7-2RBjtFU9tGurSrNWO",
    ("simswap", "ff"):   "1ZscgrprUT6Tx3JibiF6JhqeqWz9ZgMWw",  ("simswap", "cdf"):   "1isStEussymA6ToZZHl87TqWJR0feobpA",
    ("inswap", "ff"):    "1mMe6FYKoDqTz55Sheb5HLIBVnJzowYn1",  ("inswap", "cdf"):    "1JfjfvDpwzAxRPaYnJ15K73HmrD37m34M",
    ("uniface", "ff"):   "1wIhQ56jguKKbOSW1owNd-GHO5zssiWUS",  ("uniface", "cdf"):   "1xj2kDbBPPDEM3x88nVW98nrY-fLMKGoa",
}
KEEP_ZIPS = False                                # True = ไม่ลบ zip หลังแตก (ต้องมีพื้นที่ ~20 GB)

# --- preprocessing / inference (ค่าเดียวกับ 3e) ---
N_FRAMES, IMG_SIZE = 8, 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DET_THRES = 0.5
BATCH_SIZE, NUM_WORKERS, USE_AMP = 64, 2, True
N_BOOT, SEED = 1000, 42

# Table IV ของ paper (AUC %)
PAPER_TABLE4 = {
    "CFB":   {"sadtalker": 87.94, "fomm": 99.15, "facedancer": 95.23, "mobileswap": 99.18,
              "simswap": 95.92, "inswap": 97.22, "uniface": 98.93, "Avg": 96.22},
    "CFB++": {"sadtalker": 88.98, "fomm": 99.39, "facedancer": 94.93, "mobileswap": 99.52,
              "simswap": 95.76, "inswap": 97.14, "uniface": 98.91, "Avg": 96.38},
}
random.seed(SEED)
print("ON_KAGGLE:", ON_KAGGLE, "| methods:", len(METHODS), "| domains:", DOMAINS, "| preps:", PREPS)
''')

md("## 1. ติดตั้ง + clone repo + patch (เหมือนระดับ 3c/3e)")
code(S_SETUP)

md("## 2. โหลดโมเดล (โค้ดเดียวกับ 3e)")
code(S_MODEL)
code(S_AUTHOR + r'''
# เลือก ablation ที่จะประเมิน (ประหยัดเวลา)
if INCLUDE_ABLATION is not None:
    keep = {"author", "ours_cfb_full"} | {f"ours_{v}" for v in INCLUDE_ABLATION}
    MODELS = [(n, p) for n, p in MODELS if n in keep]
print("ประเมินจริง:", [n for n, _ in MODELS])
''')

md(r"""## 3. ดาวน์โหลด DF40

ลำดับการหาไฟล์: (1) Add Input ที่มีชื่อไฟล์ตรงกัน → (2) ดาวน์โหลดด้วย gdown (ลองซ้ำ 3 ครั้ง)
JSON บอกว่าวิดีโอไหนอยู่ split `test` และ path ของทุก frame (32 frame/วิดีโอ) → เลือก 8 frame แบบเว้นระยะเท่ากัน (เหมือน 3e)
จับคู่ path ใน JSON กับไฟล์ใน zip ด้วย **ส่วนท้ายของ path** (เช่น `cdf/frames/id9_id0_0006/374.png`) เพราะ zip ไม่มี prefix `deepfakes_detection_datasets/`
""")
code(r'''import gdown
IMG_EXT = (".png", ".jpg", ".jpeg")

def find_in_input(name):
    if not INPUT_DIR.exists(): return None
    hits = sorted(INPUT_DIR.rglob(name))
    return hits[0] if hits else None

def _looks_ok(path, name, min_bytes):
    if not path.exists() or path.stat().st_size < min_bytes: return False
    head = open(path, "rb").read(512).lstrip().lower()
    if head.startswith((b"<!doctype", b"<html")): return False          # ได้หน้า HTML แทนไฟล์
    if name.endswith(".json") and not head.startswith(b"{"): return False
    if name.endswith(".zip") and not head.startswith(b"pk"): return False
    return True

def _gdrive_direct(fid, out):
    """ดาวน์โหลดตรงจาก drive.usercontent.google.com (ผ่านหน้ายืนยันไฟล์ใหญ่ให้เอง)"""
    import requests
    s = requests.Session(); url = "https://drive.usercontent.google.com/download"
    r = s.get(url, params={"id": fid, "export": "download", "confirm": "t"}, stream=True, timeout=120)
    if "text/html" in r.headers.get("content-type", ""):
        html = r.text
        form = dict(re.findall(r'name="(\w+)" value="([^"]*)"', html))
        if "id" in form:
            r = s.get(url, params=form, stream=True, timeout=120)
        if "text/html" in r.headers.get("content-type", ""):
            msg = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))[:400]
            raise RuntimeError(f"Google Drive ตอบเป็นหน้าเว็บแทนไฟล์ (HTTP {r.status_code}): {msg}")
    r.raise_for_status()
    tmp = out.with_suffix(out.suffix + ".part")
    with open(tmp, "wb") as f:
        for ch in r.iter_content(1 << 20): f.write(ch)
    tmp.rename(out)

def fetch(fid, name, min_bytes=1000):
    """คืน (path, from_input) · ลำดับ: Add Input → gdown → ดาวน์โหลดตรง"""
    p = find_in_input(name)
    if p is not None:
        return p, True
    out = DL_DIR / name
    if _looks_ok(out, name, min_bytes):
        return out, False
    errs = []
    for attempt in range(2):
        for how in ("gdown", "direct"):
            try:
                out.unlink(missing_ok=True)
                if how == "gdown":
                    gdown.download(id=fid, output=str(out), quiet=True)
                else:
                    _gdrive_direct(fid, out)
                if _looks_ok(out, name, min_bytes):
                    return out, False
                errs.append(f"{how}: ได้ไฟล์ที่ไม่ใช่ {name} ({out.stat().st_size if out.exists() else 0} bytes)")
            except Exception as e:
                errs.append(f"{how}: {type(e).__name__}: {str(e)[:300]}")
        time.sleep(10)
    raise RuntimeError(
        f"ดาวน์โหลด {name} ไม่สำเร็จ\n  สาเหตุ:\n    " + "\n    ".join(errs[-2:]) + "\n"
        f"  ทางแก้: ดาวน์โหลดเองจาก https://drive.google.com/file/d/{fid}/view\n"
        f"  แล้วอัปเป็น Kaggle Dataset (ชื่อไฟล์เดิม) → Add Input → รัน cell นี้ใหม่")

num_key = lambda p: (int(re.sub(r"\D", "", Path(p).stem) or 0), p)
def pick_k(frames, k=N_FRAMES):
    fr = sorted(frames, key=num_key)
    if len(fr) >= k:
        return fr[::len(fr) // k][:k]
    return sorted(np.resize(np.array(fr), k).tolist(), key=num_key)

def tail(p, k):
    return "/".join(str(p).replace("\\", "/").lower().split("/")[-k:])

# ---------- 3.1 JSON → รายชื่อวิดีโอ test ----------
TEST = {}          # (method, domain) → {"fake": {vid: [8 path]}, "real": {vid: [8 path]}}
NFR  = []          # จำนวน frame ต้นฉบับต่อวิดีโอใน JSON (ใช้ในหัวข้อ 3.4)
for m in METHODS:
    for d in DOMAINS:
        p, _ = fetch(JSON_ID[(m, d)], f"{m}_{d}.json")
        J = json.load(open(p))
        J = J[next(iter(J))]                                  # {"<m>_Real": {...}, "<m>_Fake": {...}}
        ent = {"fake": {}, "real": {}}
        for lab, splits in J.items():
            kind = "real" if lab.lower().endswith("real") else "fake"
            for vid, info in splits.get("test", {}).items():
                ent[kind][vid] = pick_k(info["frames"])
                NFR.append({"method": m, "domain": d, "kind": kind, "vid": vid, "n_frames": len(info["frames"]),
                            "n_train_listed": len(splits.get("train", {}))})
        TEST[(m, d)] = ent
summary = pd.DataFrame([{"method": m, "domain": d, "fake": len(e["fake"]), "real": len(e["real"])}
                        for (m, d), e in TEST.items()])
print(summary.pivot(index="method", columns="domain", values=["fake", "real"]).to_string())
''')
code(r'''# ---------- 3.2 แตกเฉพาะ frame ที่ต้องใช้ ----------
ARCH_EXT = (".zip", ".tar", ".tar.gz", ".tgz")

def list_images(path, prefix="", depth=0):
    """คืน dict ชื่อภาพ → ฟังก์ชันอ่าน bytes · รองรับ zip/tar ที่ซ้อนอยู่ข้างใน (แตก archive ชั้นในลงดิสก์ก่อน)"""
    out, others = {}, []
    def add_inner(fname, opener):
        dst = DL_DIR / "_inner" / path.stem / fname
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with opener() as src_, open(dst, "wb") as f: shutil.copyfileobj(src_, f, 1 << 24)
        stem = re.sub(r"(\.tar\.gz|\.tgz|\.tar|\.zip)$", "", fname, flags=re.I)
        sub, oth = list_images(dst, prefix + stem + "/", depth + 1)
        out.update(sub); others.extend(oth)
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            nl = n.lower()
            if nl.endswith(IMG_EXT): out[prefix + n] = (lambda z=z, n=n: z.read(n))
            elif nl.endswith(ARCH_EXT) and depth < 2: add_inner(n, lambda z=z, n=n: z.open(n))
            elif not n.endswith("/"): others.append(prefix + n)
    elif tarfile.is_tarfile(path):
        tf = tarfile.open(path, "r:*")
        for m_ in tf.getmembers():
            if not m_.isfile(): continue
            nl = m_.name.lower()
            if nl.endswith(IMG_EXT): out[prefix + m_.name] = (lambda tf=tf, m_=m_: tf.extractfile(m_).read())
            elif nl.endswith(ARCH_EXT) and depth < 2: add_inner(m_.name, lambda tf=tf, m_=m_: tf.extractfile(m_))
            else: others.append(prefix + m_.name)
    else:
        others.append(prefix + path.name + " (ไม่ใช่ zip/tar)")
    return out, others

def extract_needed(arc_path, wanted, tag, expect_domain_dirs=False):
    """wanted = list ของ path ใน JSON → คืน dict path_json → path ภาพบนดิสก์"""
    imgs, others = list_images(arc_path)
    names = list(imgs)
    if not names:
        from collections import Counter
        ext = Counter(Path(o).suffix.lower() or "(ไม่มีนามสกุล)" for o in others).most_common(8)
        raise RuntimeError(f"{arc_path.name}: ไม่พบไฟล์ภาพเลย · ไฟล์ทั้งหมด {len(others):,} ไฟล์ · นามสกุล {ext}\n"
                           f"  ตัวอย่าง: {others[:10]}")
    if expect_domain_dirs and not any(("/ff/frames/" in "/" + n.lower() or "/cdf/frames/" in "/" + n.lower()) for n in names[:5000]):
        raise RuntimeError(
            f"{arc_path.name} ไม่มีโฟลเดอร์ ff/ หรือ cdf/ — น่าจะเป็น zip ชุด **train** (DF40_train)\n"
            f"  ตัวอย่าง: {names[:3]}\n  ให้ใช้ zip จากโฟลเดอร์ Drive ชื่อ 'DF40' (ชุด test)")
    idx4 = {tail(n, 4): n for n in names}
    idx3 = {tail(n, 3): n for n in names}
    print(f"  {arc_path.name}: {len(names):,} ภาพ · ตัวอย่าง {names[0]}" + (f" · ไฟล์อื่น {len(others)}" if others else ""))
    got, miss = {}, []
    for wp in wanted:
        n = idx4.get(tail(wp, 4)) or idx3.get(tail(wp, 3))
        if n is None:
            miss.append(wp); continue
        out = DF40_DIR / "raw" / tag / tail(wp, 4)
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(imgs[n]())
        got[wp] = out
    if miss:
        print(f"  [WARN] หาไม่เจอ {len(miss)}/{len(wanted)} frame เช่น {miss[:2]}")
    shutil.rmtree(DL_DIR / "_inner" / arc_path.stem, ignore_errors=True)
    return got

def done_marker(tag): return DF40_DIR / "raw" / f".done_{tag}.json"

PATHMAP, FAILED = {}, {}
# ภาพปลอม: zip ละวิธี (มีทั้ง ff/ และ cdf/ ในไฟล์เดียว) · วิธีที่แตกไม่ได้จะถูกข้าม ไม่หยุดทั้ง notebook
for m in METHODS:
    mk = done_marker(m)
    if mk.exists():
        PATHMAP.update({k: Path(v) for k, v in json.load(open(mk)).items()}); print(f"{m}: ใช้ที่แตกไว้แล้ว"); continue
    t0 = time.time()
    try:
        zp, from_input = fetch(FAKE_ZIP_ID[m], f"{m}.zip", min_bytes=100_000_000)
        wanted = [p for d in DOMAINS for fs in TEST[(m, d)]["fake"].values() for p in fs]
        got = extract_needed(zp, wanted, m, expect_domain_dirs=True)
        if not got: raise RuntimeError("จับคู่ frame กับ JSON ไม่ได้เลย")
    except Exception as e:
        FAILED[m] = str(e); print(f"[ข้าม {m}] {e}"); continue
    json.dump({k: str(v) for k, v in got.items()}, open(mk, "w")); PATHMAP.update(got)
    if not from_input and not KEEP_ZIPS: zp.unlink(missing_ok=True)
    print(f"{m}: {len(got):,} frame · {(time.time()-t0)/60:.1f} นาที")

if FAILED:
    METHODS = [m for m in METHODS if m not in FAILED]
    print(f"\n[WARN] ข้าม {list(FAILED)} → ประเมินเฉพาะ {METHODS} (Avg จะไม่ใช่ค่าเฉลี่ย 7 วิธีเหมือน Table IV)")
    json.dump(FAILED, open(OUT_DIR / "df40_failed_methods.json", "w"), ensure_ascii=False, indent=1)
assert METHODS, "แตกภาพปลอมไม่ได้เลยสักวิธี"

# ภาพจริง: zip ละโดเมน (ใช้ร่วมกันทุกวิธี)
for d in DOMAINS:
    tag = f"real_{d}"; mk = done_marker(tag)
    if mk.exists():
        PATHMAP.update({k: Path(v) for k, v in json.load(open(mk)).items()}); print(f"{tag}: ใช้ที่แตกไว้แล้ว"); continue
    t0 = time.time()
    name, fid = REAL_ZIP[d]
    zp, from_input = fetch(fid, name, min_bytes=100_000_000)
    wanted = sorted({p for m in METHODS for fs in TEST[(m, d)]["real"].values() for p in fs})
    got = extract_needed(zp, wanted, tag)
    json.dump({k: str(v) for k, v in got.items()}, open(mk, "w")); PATHMAP.update(got)
    if not from_input and not KEEP_ZIPS: zp.unlink(missing_ok=True)
    print(f"{tag}: {len(got):,} frame · {(time.time()-t0)/60:.1f} นาที")

print(f"รวม {len(PATHMAP):,} frame · ใช้พื้นที่ {sum(p.stat().st_size for p in DF40_DIR.rglob('*.png'))/1e9:.2f} GB")
''')
code(r'''# ---------- 3.3 สร้างตารางวิดีโอ ----------
def to_rows(ent, kind, label, group):
    rows = []
    for vid, fs in ent.items():
        loc = [PATHMAP[p] for p in fs if p in PATHMAP]
        if not loc: continue
        loc = (loc * N_FRAMES)[:N_FRAMES]
        rows.append({"uid": f"{group}/{vid}", "label": label, "group": group, "frames": [str(p) for p in loc]})
    return rows

REAL_DF, FAKE_DF = {}, {}
for d in DOMAINS:
    real_ent = {}
    for m in METHODS: real_ent.update(TEST[(m, d)]["real"])      # real ของทุกวิธีในโดเมนเดียวกันเป็นชุดเดียวกัน
    REAL_DF[d] = pd.DataFrame(to_rows(real_ent, "real", 0, f"real_{d}"))
    for m in METHODS:
        FAKE_DF[(m, d)] = pd.DataFrame(to_rows(TEST[(m, d)]["fake"], "fake", 1, f"{m}_{d}"))
cnt = pd.DataFrame([{"method": m, "domain": d, "fake": len(FAKE_DF[(m, d)]), "real": len(REAL_DF[d])}
                    for m in METHODS for d in DOMAINS])
print(cnt.pivot(index="method", columns="domain", values=["fake", "real"]).to_string())
cnt.to_csv(OUT_DIR / "df40_counts.csv", index=False)
''')

md(r"""### 3.4 Data understanding ของ DF40 (สำหรับสไลด์ — แบบเดียวกับ FF++ c23)

ข้อมูลเป็นไปตามที่ paper ใช้ (DF40, 7 วิธีของ Table IV) จึงสรุปแค่ลักษณะข้อมูล:
จำนวน/สัดส่วน real–fake · วิธีปลอมแปลงแต่ละวิธีและประเภท (FS / FR) · สัดส่วนของแต่ละวิธี · frame ต่อวิดีโอ · ภาพตัวอย่าง
""")
code(r'''from PIL import Image as _PIL
CATEGORY = {"sadtalker": "FR", "fomm": "FR", "facedancer": "FS", "mobileswap": "FS",
            "simswap": "FS", "inswap": "FS", "uniface": "FS"}   # FS = face swapping · FR = face reenactment (ตาม DF40)

rows = []
for d in DOMAINS:
    rows.append({"domain": d, "class": "Real", "category": "-", "videos": len(REAL_DF[d])})
    for m in METHODS:
        rows.append({"domain": d, "class": PAPER_NAME[m], "category": CATEGORY[m], "videos": len(FAKE_DF[(m, d)])})
DU = pd.DataFrame(rows)
DU["% ในโดเมน"] = (100 * DU["videos"] / DU.groupby("domain")["videos"].transform("sum")).round(1)
DU.to_csv(OUT_DIR / "df40_data_understanding.csv", index=False)
print("จำนวนวิดีโอ test ต่อ class (real ใช้ชุดเดียวกันทุกวิธีในโดเมนนั้น)")
print(DU.to_string(index=False))

summ = []
for d in DOMAINS:
    x = DU[DU.domain == d]; nr = int(x[x["class"] == "Real"].videos.sum()); nf_ = int(x[x["class"] != "Real"].videos.sum())
    summ.append({"domain": d, "real": nr, "fake": nf_, "real %": round(100 * nr / (nr + nf_), 1),
                 "FS videos": int(x[x.category == "FS"].videos.sum()), "FR videos": int(x[x.category == "FR"].videos.sum()),
                 "frame/วิดีโอ (JSON)": int(pd.DataFrame(NFR).query("domain == @d").n_frames.median()),
                 "frame ที่ใช้": N_FRAMES})
SUMM_DU = pd.DataFrame(summ); SUMM_DU.to_csv(OUT_DIR / "df40_data_summary.csv", index=False)
print("\nสรุปต่อโดเมน"); print(SUMM_DU.to_string(index=False))
print("หมายเหตุ: AUC ต่อวิธีคำนวณจาก real ของโดเมน vs fake ของวิธีนั้น (real 1 ชุด เทียบกับ 7 วิธี)")

# แท่งสัดส่วน (แบบเดียวกับสไลด์ FF++ c23)
PAL = ["#9cc3ec", "#f3b496", "#e98d6b", "#d9663f", "#b84a2a", "#8f3a20", "#c7a3d9", "#9a73b5"]
fig, axes = plt.subplots(len(DOMAINS), 1, figsize=(11, 1.3 * len(DOMAINS) + 0.6))
for ax, d in zip(np.atleast_1d(axes), DOMAINS):
    x = DU[DU.domain == d]; left = 0
    for k, r in enumerate(x.itertuples()):
        ax.barh(0, r.videos, left=left, color=PAL[k], edgecolor="white")
        if r._5 >= 3: ax.text(left + r.videos / 2, 0, f"{r._2}\n{r.videos}", ha="center", va="center", fontsize=7)
        left += r.videos
    ax.set_xlim(0, left); ax.set_yticks([]); ax.set_title(f"DF40 test · domain = {d} · {left} videos", fontsize=9, loc="left")
    for s in ("left", "top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT_DIR / "df40_class_bar.png", dpi=150); plt.show()

# ภาพตัวอย่าง: แถว = โดเมน · คอลัมน์ = real + 7 วิธี
cols = ["real"] + METHODS
fig, axes = plt.subplots(len(DOMAINS), len(cols), figsize=(1.9 * len(cols), 2.1 * len(DOMAINS)))
for i, d in enumerate(DOMAINS):
    for j, c in enumerate(cols):
        df = REAL_DF[d] if c == "real" else FAKE_DF[(c, d)]
        ax = axes[i, j]; ax.axis("off")
        if len(df): ax.imshow(_PIL.open(df["frames"].iloc[0][0]).convert("RGB"))
        ax.set_title(("Real" if c == "real" else f"{PAPER_NAME[c]} ({CATEGORY[c]})") + f"\n{d}", fontsize=8)
plt.tight_layout(); plt.savefig(OUT_DIR / "df40_samples.png", dpi=130); plt.show()
''')

md(r"""## 4. (ทางเลือก) realign ด้วย RetinaFace แบบเดียวกับ FF++/CDF ของเรา

ภาพ DF40 ถูก crop ด้วย pipeline ของ DF40 (คนละแบบกับที่ใช้ตอนเทรน) — จึงประเมิน 2 แบบ:
- **raw** — ใช้ภาพ DF40 ตรง ๆ แค่ resize เป็น 224 (เหมือนที่ DF40/DeepfakeBench ใช้)
- **realign** — ขยายภาพ → RetinaFace → align 5 จุด → crop แบบของเรา (หาหน้าไม่เจอ → ใช้ภาพเดิม)

ทั้ง real และ fake ผ่านขั้นตอนเดียวกันเสมอ จึงไม่มี bias จากการ crop ต่างกัน
""")
code(S_REALIGN + r'''
def realign_all():
    srcs = sorted({p for df in list(REAL_DF.values()) + list(FAKE_DF.values()) for fs in df["frames"] for p in fs})
    n_ok, out_map = 0, {}
    for s in tqdm(srcs, desc="realign"):
        dst = Path(str(s).replace(str(DF40_DIR / "raw"), str(DF40_DIR / "realign")))
        if not dst.exists():
            im = cv2.imread(s)
            im2, ok = realign(im); n_ok += ok
            dst.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(dst), im2)
        out_map[s] = str(dst)
    print(f"realign สำเร็จ {n_ok}/{len(srcs)} ภาพ (ที่แตกไว้แล้วไม่นับซ้ำ)")
    json.dump({"realign_ok": int(n_ok), "n_images": len(srcs)}, open(OUT_DIR / "df40_realign_rate.json", "w"))
    return out_map

if "realign" in PREPS:
    t0 = time.time()
    RMAP = realign_all()
    print(f"{(time.time()-t0)/60:.1f} นาที")

def frames_for(df, prep):
    if prep == "raw": return df
    df = df.copy(); df["frames"] = df["frames"].map(lambda fs: [RMAP[p] for p in fs]); return df
''')

md("## 5. Inference — main branch (ตาม paper) · score ระดับวิดีโอ = เฉลี่ย 8 frame")
code(S_INFER + r'''
SCORES = {}         # (model, prep, domain, "real" | method) → DataFrame
for name, path in MODELS:
    t0 = time.time()
    model, info = load_detector(path)
    for prep in PREPS:
        for d in DOMAINS:
            SCORES[(name, prep, d, "real")] = score_dataset(model, frames_for(REAL_DF[d], prep))
            for m in METHODS:
                SCORES[(name, prep, d, m)] = score_dataset(model, frames_for(FAKE_DF[(m, d)], prep))
    print(f"{name:<18} dual={info['dual']!s:<5} · {(time.time()-t0)/60:.1f} นาที")
    del model; torch.cuda.empty_cache()

pd.concat([v.assign(model=k[0], prep=k[1], domain=k[2], subset=k[3]) for k, v in SCORES.items()]) \
  .to_csv(OUT_DIR / "scores_df40_all.csv", index=False)
''')

md(r"""## 6. AUC ต่อวิธี (real ของโดเมนนั้น vs fake ของวิธีนั้น) — เทียบ Table IV

- `ff` / `cdf` = AUC ภายในโดเมน · `all` = รวม real และ fake ของทั้ง 2 โดเมน
- ช่วงความเชื่อมั่น 95% ของค่าเฉลี่ย 7 วิธี: bootstrap แยก real/fake ของแต่ละวิธี
""")
code(r'''def auc_pair(real, fake, head="main"):
    y = np.r_[np.zeros(len(real)), np.ones(len(fake))]
    s = np.r_[real[f"p_{head}"].values, fake[f"p_{head}"].values]
    return 100 * roc_auc_score(y, s)

def get(name, prep, dom, sub):
    if dom != "all": return SCORES[(name, prep, dom, sub)]
    return pd.concat([SCORES[(name, prep, d, sub)] for d in DOMAINS], ignore_index=True)

rows = []
for name, _ in MODELS:
    for prep in PREPS:
        for dom in DOMAINS + ["all"]:
            r = {"model": name, "prep": prep, "domain": dom}
            for m in METHODS:
                r[m] = auc_pair(get(name, prep, dom, "real"), get(name, prep, dom, m))
            r["Avg"] = np.mean([r[m] for m in METHODS])
            rows.append(r)
TAB = pd.DataFrame(rows)

def boot_avg(name, prep, dom, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    real = get(name, prep, dom, "real")["p_main"].values
    fakes = [get(name, prep, dom, m)["p_main"].values for m in METHODS]
    out = []
    for _ in range(n):
        rr = rng.choice(real, len(real))
        out.append(np.mean([100 * roc_auc_score(np.r_[np.zeros(len(rr)), np.ones(len(f))],
                                                np.r_[rr, rng.choice(f, len(f))]) for f in fakes]))
    return np.percentile(out, [2.5, 97.5])

TAB["Avg 95% CI"] = [("[%.2f, %.2f]" % tuple(boot_avg(r.model, r.prep, r.domain, n=300)))
                     for r in TAB.itertuples()]
TAB.to_csv(OUT_DIR / "table4_df40_auc.csv", index=False)

show = TAB.rename(columns=PAPER_NAME).round(2)
paper = pd.DataFrame([{"model": f"paper {k}", "prep": "-", "domain": "?", **{PAPER_NAME[m]: v[m] for m in METHODS},
                       "Avg": v["Avg"]} for k, v in PAPER_TABLE4.items()])
print("AUC (%) — main branch, video-level")
print(pd.concat([paper, show], ignore_index=True).to_string(index=False))
''')
code(r'''# ผลต่างจาก paper (CFB) — เฉพาะ checkpoint ผู้เขียน และ CFB ของเรา
rows = []
for r in TAB.itertuples():
    if r.model not in ("author", "ours_cfb_full"): continue
    d = {"model": r.model, "prep": r.prep, "domain": r.domain}
    for m in METHODS + ["Avg"]:
        d[PAPER_NAME.get(m, m)] = round(getattr(r, m) - PAPER_TABLE4["CFB"][m], 2)
    rows.append(d)
DIFF = pd.DataFrame(rows)
DIFF.to_csv(OUT_DIR / "table4_diff_vs_paper.csv", index=False)
print("Δ AUC = เรา − paper CFB (จุด %)"); print(DIFF.to_string(index=False))
''')

md("## 7. กราฟ")
code(r'''C = {"author": "#eb6834", "ours_cfb_full": "#2a78d6", "ours_fb_ce": "#1e9e6a"}
INK2, GRID = "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
prep0 = "raw"
fig, axes = plt.subplots(1, len(DOMAINS) + 1, figsize=(5.2 * (len(DOMAINS) + 1), 4.2), sharey=True)
x = np.arange(len(METHODS))
names = [n for n, _ in MODELS]
w = 0.8 / (len(names) + 1)
for ax, dom in zip(axes, DOMAINS + ["all"]):
    ax.bar(x - 0.4 + w / 2, [PAPER_TABLE4["CFB"][m] for m in METHODS], w, color="#b9b8b3", label="paper CFB")
    for i, n in enumerate(names):
        r = TAB[(TAB.model == n) & (TAB.prep == prep0) & (TAB.domain == dom)].iloc[0]
        ax.bar(x - 0.4 + w * (i + 1.5), [r[m] for m in METHODS], w, color=C.get(n, "#888"), label=n)
    ax.set_xticks(x, [PAPER_NAME[m] for m in METHODS], rotation=35, ha="right")
    ax.set_ylim(40, 100); ax.set_title(f"DF40 · domain={dom} · {prep0}"); ax.grid(axis="y", color=GRID)
axes[0].set_ylabel("AUC (%)"); axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(OUT_DIR / "df40_auc_bars.png", dpi=150); plt.show()

# การกระจายของ score: real vs แต่ละวิธี (ดูว่าวิธีไหนโมเดลแยกไม่ออก เช่น score ของ fake ต่ำใกล้ real)
for n in [x for x in ("author", "ours_cfb_full") if x in names]:
    fig, axes = plt.subplots(1, len(DOMAINS), figsize=(6.5 * len(DOMAINS), 3.8), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), DOMAINS):
        data = [SCORES[(n, prep0, d, "real")]["p_main"].values] + [SCORES[(n, prep0, d, m)]["p_main"].values for m in METHODS]
        bp = ax.boxplot(data, labels=["real"] + [PAPER_NAME[m] for m in METHODS], showfliers=False, patch_artist=True)
        for k, b in enumerate(bp["boxes"]): b.set_facecolor("#9cc3ec" if k == 0 else "#f3b496")
        ax.axhline(0.5, color=INK2, ls="--", lw=1); ax.tick_params(axis="x", rotation=35)
        ax.set_title(f"P(fake) · {n} · domain={d} · {prep0}"); ax.grid(axis="y", color=GRID)
    plt.tight_layout(); plt.savefig(OUT_DIR / f"df40_score_dist_{n}.png", dpi=150); plt.show()
''')

md("## 8. สรุป")
code(r'''print("=" * 72); print("สรุประดับ 3i — DF40"); print("=" * 72)
print(TAB[["model", "prep", "domain", "Avg", "Avg 95% CI"]].round(2).to_string(index=False))
print(f"""
paper (Table IV) CFB Avg = {PAPER_TABLE4['CFB']['Avg']}

ข้อควรระบุในรายงาน
  1. ใช้ DF40 เฉพาะ 7 วิธีของ Table IV · split = test ตาม JSON ของ DF40 · 8 frame/วิดีโอ · main branch
  2. paper ไม่ได้ระบุโดเมน (ff / cdf) และวิธี preprocess ของ DF40 → รายงานทุกแบบ
  3. ภาพจริงมาจากไฟล์ real ของ DF40 (crop แบบเดียวกับภาพปลอม) ไม่ได้ผสมกับ crop ของเรา
  4. DF40 ใช้ภายใต้ CC BY-NC 4.0 เพื่อการศึกษา
เวลารวม: {(time.time()-T_START)/60:.1f} นาที""")
print("ไฟล์ผลลัพธ์:", *[p.name for p in sorted(OUT_DIR.iterdir())], sep="\n  ")
''')

nb = {"cells": cells, "metadata": base["metadata"], "nbformat": base["nbformat"], "nbformat_minor": base["nbformat_minor"]}
json.dump(nb, open("FFD_level3i_DF40_eval.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote FFD_level3i_DF40_eval.ipynb", len(cells), "cells")
