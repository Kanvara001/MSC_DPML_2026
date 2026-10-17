"""สร้าง FFD_level3e_eval_utom_wdf.ipynb — ประเมินทุกโมเดลบน FF++ / Celeb-DF / WildDeepfake + UTOM เต็มรูปแบบ"""
import json, copy

base = json.load(open("FFD_level3c_eval_crossdataset.ipynb", encoding="utf-8"))
c3 = lambda i: "".join(base["cells"][i]["source"])

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                            "source": t.splitlines(keepends=True)})

md(r"""# ระดับ 3e — ประเมินทุกโมเดล (FF++ · Celeb-DF · WildDeepfake) + UTOM เต็มรูปแบบ + ตาราง ablation

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026

notebook นี้รวมการประเมินทั้งหมดไว้ที่เดียว ใช้ **preprocessing และวิธีคำนวณเดียวกันทุกโมเดล** จึงเทียบกันได้ตรง

| หัวข้อ | ทำซ้ำส่วนไหนของ paper | หมายเหตุ |
|---|---|---|
| 5 | ผลหลัก (Table II คอลัมน์ CDF) | ใช้ **main branch** ตาม paper (3b รายงาน FF++ จาก fused head) |
| 6 | Ablation (Table VI) | ใช้ checkpoint จากระดับ 3d · paper มีคอลัมน์ CDF ให้เทียบตรง |
| 7 | UTOM (Sec. III-D, Table IX) | บันทึก uncertainty `u` แล้ว → คำนวณ `τ_ot^UTOM` ได้ครบ |
| 8 | ทดสอบเพิ่มเติม: **WildDeepfake** | **ไม่อยู่ในตารางหลักของ paper** (ใช้แค่ใน Table X) → เทียบ checkpoint ผู้เขียน vs ของเรา |

**โมเดลที่ประเมิน** (ค้นหาอัตโนมัติจาก Add Input):
1. checkpoint ของผู้เขียน (ดาวน์โหลดจาก Google Drive ของ repo)
2. CFB ที่เราเทรนเอง — `cfb_run/best.pth` จากระดับ 3b
3. ablation ทุก variant — `ablation_runs/<variant>/best.pth` จากระดับ 3d (ถ้ามี)

**ก่อนรัน:** GPU **T4 ×1** · Internet **On** · Add Input:
- output ของระดับ **3a** (ภาพใบหน้า FF++ ที่ crop แล้ว) · **3b** (`cfb_run`) · **3c** (`cdf_crops`) · **3d** (ถ้ามี)

WildDeepfake test (~10 GB) ดาวน์โหลดจาก Hugging Face (`xingjunm/WildDeepfake`, Apache-2.0) ทีละไฟล์แล้วลบทิ้ง
เก็บไว้แค่ 8 frame ต่อ sequence · ใช้เวลารวมประมาณ 1–1.5 ชม.
""")

md("## 0. ตั้งค่า")
code(r'''import os, sys, re, json, time, glob, random, shutil, tarfile, hashlib, subprocess, warnings, io
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"   # ไม่พิมพ์ progress bar ของการดาวน์โหลด 157 ไฟล์ (ทำให้ notebook ใหญ่เกิน 1 MB ตอน commit)
T_START = time.time()

ON_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR  = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FFD_WORK", "./work")).resolve()
TMP_DIR   = Path("/tmp/ffd") if ON_KAGGLE else WORK_DIR / "tmp"
INPUT_DIR = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("FFD_INPUT", str(WORK_DIR / "input")))
REPO_DIR  = TMP_DIR / "FFDBackbone"
DATA_ROOT = TMP_DIR / "data"                     # แตก tar ของระดับ 3a (เฉพาะ test)
WDF_DIR   = WORK_DIR / "wdf_crops"               # เก็บไว้ใน working เพื่อใช้ซ้ำรอบหน้า (Add Input)
WEIGHT_DIR = TMP_DIR / "weights"
OUT_DIR   = WORK_DIR / "results_level3e"
for d in [TMP_DIR, DATA_ROOT, WDF_DIR, WEIGHT_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/zhenglab/FFDBackbone.git"
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G"

# --- โมเดล ---
USE_AUTHOR_CKPT   = True                         # ดาวน์โหลด checkpoint "Ours" ของผู้เขียนมาเทียบ
AUTHOR_CKPT_PATH  = None                         # ถ้าอัปเป็น Kaggle Dataset ไว้แล้วให้ใส่ path
AUTHOR_HINTS      = ["ours", "final", "ddbf", "dbbf", "mainbranch"]
AUTHOR_EXCLUDE    = ["beit-1k-face-55w"]

# --- dataset ---
DATASETS   = ["FF++", "CDF", "WDF"]              # ลบ "WDF" ออกได้ถ้าไม่ต้องการ
WDF_REPO   = "xingjunm/WildDeepfake"
WDF_SPLITS = {"real_test": 0, "fake_test": 1}    # label: 0 = real, 1 = fake
WDF_REALIGN = True                               # True = RetinaFace + align ซ้ำแบบเดียวกับ FF++/CDF (หาไม่เจอ → ใช้ภาพเดิม)
WDF_MAX_FILES = None                             # จำกัดจำนวนไฟล์ tar ต่อ split (None = ทั้งหมด) — ใช้ทดสอบเร็ว ๆ

# --- preprocessing / inference (ค่าตาม repo) ---
N_FRAMES, IMG_SIZE = 8, 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DET_THRES = 0.5
BATCH_SIZE, NUM_WORKERS, USE_AMP = 64, 2, True

# --- UTOM (Sec. III-D + Supplementary 2.1, 2.3) ---
UTOM_LAMBDA, UTOM_EPS = 0.02, 1e-7
N_BOOT, SEED = 1000, 42

PAPER_MAIN = {"FF++": 99.36, "CDF": 90.46}
PAPER_TABLE6 = {   # Table VI: CDF, DFDC, FFIW, Avg AUC, Avg ACC
    "fb_ce": (89.18, 84.02, 86.48, 86.56, 79.37), "fb_edu": (89.34, 84.19, 88.76, 87.43, 80.96),
    "cfb_base": (89.77, 83.67, 87.69, 87.04, 79.26), "cfb_dec": (90.31, 84.89, 89.21, 88.14, 79.42),
    "cfb_dec_ufm": (91.16, 84.54, 90.22, 88.64, 80.84), "cfb_full": (90.46, 84.90, 90.97, 88.78, 82.29),
}
PAPER_TABLE9_CDF = {   # Table IX, CFB + L_EDU, source = CDF → target CDF (ACC %)
    "tau=0.5": 83.20, "tau_ot": 83.78, "tau_ot^UTOM": 85.33}

random.seed(SEED)
print("ON_KAGGLE:", ON_KAGGLE, "| DATASETS:", DATASETS)
''')

md("## 1. ติดตั้ง + clone repo + patch (เหมือนระดับ 3c)")
s4 = c3(4)
s4 = s4.replace('pip_install("gdown")', 'pip_install("gdown")\ntry: import huggingface_hub\nexcept ImportError: pip_install("huggingface_hub")')
code(s4)

md(r"""## 2. โหลดโมเดล

โครงสร้างเดียวกับ `DDBF_BEiT_v2` ของ repo · รองรับ checkpoint ทั้ง 1 branch (FB) และ 2 branch
ตัวที่ไม่มี UFM (`cfb_base`, `cfb_dec`) รวม feature ด้วยการบวก ตามที่เทรนมาในระดับ 3d

ทุกโมเดลให้ output 3 head: **main** (ใช้ตอน inference ตาม paper), **branch2** (auxiliary), **fused** (รวม 2 branch)
พร้อม uncertainty `u = K/S` ของแต่ละ head (Eq. 3–4)
""")
code(r'''BEIT_CONFIG = dict(drop_path_rate=0.1, use_mean_pooling=False, init_values=0.1, qkv_bias=True,
                   use_abs_pos_emb=False, use_rel_pos_bias=True, use_shared_rel_pos_bias=False)

def make_beit():
    m = modeling_finetune.beit_base_patch16_224(**BEIT_CONFIG)
    m.head = nn.Identity()
    return m

def evidential_uncertainty(logits, K=2):
    alpha = torch.exp(torch.clamp(logits, -10, 10)) + 1
    return K / alpha.sum(dim=1, keepdim=True)

class Detector(nn.Module):
    def __init__(self, dual: bool, ufm: bool = True):
        super().__init__()
        self.dual, self.ufm = dual, ufm
        self.backbone_model = make_beit()
        if dual: self.backbone_model2 = make_beit()
        self.norm_for_cls = nn.LayerNorm(768)
        self.header = nn.Sequential(nn.Linear(768, 2))
    def classify(self, f): return self.header(self.norm_for_cls(f))
    def forward(self, x):
        f1 = self.backbone_model(x)
        out = {"main": self.classify(f1)}
        if self.dual:
            f2 = self.backbone_model2(x)
            l2 = self.classify(f2)
            if self.ufm:
                u = torch.cat([evidential_uncertainty(out["main"].float()),
                               evidential_uncertainty(l2.float())], dim=-1)
                w = torch.softmax(-10 * u, dim=-1).to(f1.dtype)
                fused = w[:, :1] * f1 + w[:, 1:] * f2
            else:
                fused = f1 + f2
            out["branch2"] = l2
            out["fused"] = self.classify(fused)
        return out

def load_state_dict_any(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ck.items() if not isinstance(v, (dict, torch.Tensor))} if isinstance(ck, dict) else {}
    flags = ck.get("flags") if isinstance(ck, dict) else None
    if flags is None and isinstance(ck, dict) and isinstance(ck.get("cfg"), dict):
        flags = ck["cfg"].get("flags")
    sd = ck
    for key in ("state_dict", "model", "model_state_dict", "net"):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]; break
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items() if torch.is_tensor(v)}
    return sd, meta, flags

def load_detector(path):
    sd, meta, flags = load_state_dict_any(path)
    dual = any(k.startswith("backbone_model2.") for k in sd)
    ufm = True if flags is None else bool(flags.get("ufm", True))
    m = Detector(dual=dual, ufm=ufm)
    expected = set(m.state_dict())
    loadable = {k: v for k, v in sd.items() if k in expected}
    missing = sorted(expected - set(loadable))
    bad = [k for k, v in loadable.items() if v.shape != m.state_dict()[k].shape]
    if missing or bad:
        raise RuntimeError(f"{path}: checkpoint ไม่ตรงกับโมเดล missing={missing[:5]} shape={bad[:5]}")
    m.load_state_dict(loadable, strict=True)
    info = {"dual": dual, "ufm": ufm, "flags": flags, "epoch": meta.get("epoch"), "variant": meta.get("variant")}
    return m.eval().to(DEVICE), info
''')
code(r'''def find_or_download_author():
    if AUTHOR_CKPT_PATH and Path(AUTHOR_CKPT_PATH).exists():
        return Path(AUTHOR_CKPT_PATH)
    def ok(name):
        low = name.lower()
        return low.endswith((".tar", ".pth", ".pt", ".ckpt", ".bin")) and not any(x in low for x in AUTHOR_EXCLUDE)
    rank = lambda name: sum(h in name.lower() for h in AUTHOR_HINTS)
    local = [p for root in [INPUT_DIR, WEIGHT_DIR] if root.exists() for p in root.rglob("*")
             if p.is_file() and ok(p.name) and rank(str(p)) > 0
             and "cfb_run" not in str(p) and "ablation_runs" not in str(p)]
    if local:
        local.sort(key=lambda p: (-rank(str(p)), str(p)))
        return local[0]
    import gdown, inspect
    kw = dict(url=GDRIVE_FOLDER, skip_download=True, quiet=True)
    if "remaining_ok" in inspect.signature(gdown.download_folder).parameters:
        kw["remaining_ok"] = True
    files = gdown.download_folder(**kw) or []
    entries = [(getattr(f, "id", None) or f[0], str(getattr(f, "path", None) or f[1])) for f in files]
    print("ไฟล์ใน Drive:", *[p for _, p in entries], sep="\n  ")
    cands = sorted([(i, p) for i, p in entries if ok(p) and rank(p) > 0], key=lambda x: (-rank(x[1]), x[1]))
    if not cands:
        raise RuntimeError("ไม่พบ checkpoint ของผู้เขียนใน Drive — ตั้ง AUTHOR_CKPT_PATH เอง")
    fid, rel = cands[0]
    out = WEIGHT_DIR / Path(rel).name
    gdown.download(id=fid, output=str(out), quiet=False)
    if not out.exists() or out.stat().st_size < 10_000_000:
        raise RuntimeError("ดาวน์โหลดไม่สำเร็จ (อาจติดโควตา Google Drive) — ดาวน์โหลดเองแล้วตั้ง AUTHOR_CKPT_PATH")
    return out

MODELS = []                     # (ชื่อ, path)
if USE_AUTHOR_CKPT:
    try:
        MODELS.append(("author", find_or_download_author()))
    except Exception as e:
        print("[WARN] ข้าม checkpoint ผู้เขียน:", e)
ours = sorted(INPUT_DIR.rglob("cfb_run/best.pth")) if INPUT_DIR.exists() else []
if ours:
    MODELS.append(("ours_cfb_full", ours[0]))
else:
    print("[WARN] ไม่พบ cfb_run/best.pth — Add Input เป็น output ของระดับ 3b")
for p in (sorted(INPUT_DIR.rglob("ablation_runs/*/best.pth")) if INPUT_DIR.exists() else []):
    MODELS.append((f"ours_{p.parent.name}", p))
assert MODELS, "ไม่พบโมเดลเลย"
print("โมเดลที่จะประเมิน:")
for n, p in MODELS:
    print(f"  {n:<20} {p}  ({p.stat().st_size/1e6:.0f} MB)")
''')

md(r"""## 3. ชุดข้อมูลทดสอบ

| ชุด | ที่มา | หน่วย | label |
|---|---|---|---|
| **FF++ test** | ภาพที่ crop แล้วจากระดับ 3a · official split 140 pairs | 700 วิดีโอ | real 140 / fake 560 |
| **Celeb-DF v2** | `cdf_crops` จากระดับ 3c | 518 วิดีโอ | real 178 / fake 340 |
| **WildDeepfake** | Hugging Face `xingjunm/WildDeepfake` (test) | 1 sequence = 1 หน่วย | `real_test` = 0, `fake_test` = 1 |

ทุกชุดใช้ **8 frame ต่อหน่วย** และ score = ค่าเฉลี่ยของ 8 frame (video-level) เหมือนกัน
""")
code(r'''# ---------- FF++ test (จากระดับ 3a) ----------
FF_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
REAL_REL = "original_sequences/youtube/c23/videos"

def frames_of(d, k=N_FRAMES):
    """เหมือน get_sampled_idx() ของ repo: step = N // k แล้วเอา k ตัวแรก"""
    if not d.is_dir(): return []
    idxs = sorted(int(p.stem) for p in d.glob("*.png") if p.stem.isdigit())
    if not idxs: return []
    sel = idxs[::len(idxs) // k][:k] if len(idxs) >= k else sorted(np.resize(np.array(idxs), k).tolist())
    return [str(d / f"{i}.png") for i in sorted(sel)]

def build_ffpp_test():
    tars = sorted(INPUT_DIR.rglob("*.tar")) if INPUT_DIR.exists() else []
    tars = {t.stem: t for t in tars if t.stem in ["meta", "original"] + FF_METHODS}
    face_root = None
    if "meta" in tars:
        with tarfile.open(tars["meta"]) as tf: tf.extractall(DATA_ROOT)
        face_root = DATA_ROOT / "FF++" / "face_v1"
    else:
        hits = sorted(INPUT_DIR.rglob("face_v1/splits/test.json")) if INPUT_DIR.exists() else []
        if hits: face_root = hits[0].parent.parent
    if face_root is None or not (face_root / "splits" / "test.json").exists():
        raise RuntimeError("ไม่พบข้อมูล FF++ จากระดับ 3a — Add Input เป็น output ของ notebook FFD_level3a")
    pairs = json.load(open(face_root / "splits" / "test.json"))
    rows = []
    for t, s in pairs:
        rows += [{"rel": f"{REAL_REL}/{x}", "label": 0, "method": "original", "video": x} for x in (t, s)]
        for m in FF_METHODS:
            rows += [{"rel": f"manipulated_sequences/{m}/c23/videos/{x}", "label": 1, "method": m, "video": x}
                     for x in (f"{t}_{s}", f"{s}_{t}")]
    # แตกเฉพาะไฟล์ของ test split จาก tar (ประหยัดเวลา/พื้นที่)
    want = tuple(f"FF++/face_v1/{r['rel']}/" for r in rows)
    for stem in ["original"] + FF_METHODS:
        if stem not in tars: continue
        marker = DATA_ROOT / f".test_{stem}"
        if marker.exists(): continue
        with tarfile.open(tars[stem]) as tf:
            tf.extractall(DATA_ROOT, members=[m for m in tf.getmembers() if m.name.startswith(want)])
        marker.touch()
    out = []
    for r in rows:
        fr = frames_of(face_root / r["rel"] / "frame")
        if len(fr) == N_FRAMES:
            out.append({"uid": f"{r['method']}/{r['video']}", "label": r["label"], "group": r["method"], "frames": fr})
    df = pd.DataFrame(out)
    print(f"FF++ test: {len(df)} วิดีโอ (คาด 700) · real {int((df.label==0).sum())} / fake {int((df.label==1).sum())}")
    return df

# ---------- Celeb-DF (crop จากระดับ 3c) ----------
def build_cdf():
    roots = sorted(p for p in INPUT_DIR.rglob("cdf_crops") if p.is_dir()) if INPUT_DIR.exists() else []
    if not roots:
        raise RuntimeError("ไม่พบ cdf_crops — Add Input เป็น output ของระดับ 3c")
    root, out = roots[0], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        subset = d.name.split("__")[0]
        fr = sorted(d.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        if not fr: continue
        fr = [str(p) for p in (fr[:N_FRAMES] if len(fr) >= N_FRAMES else np.resize(np.array(fr), N_FRAMES))]
        out.append({"uid": d.name, "label": int("synthesis" in subset.lower()), "group": subset, "frames": fr})
    df = pd.DataFrame(out)
    print(f"Celeb-DF: {len(df)} วิดีโอ (คาด 518) · real {int((df.label==0).sum())} / fake {int((df.label==1).sum())}  ← {root}")
    return df

DATA = {}
if "FF++" in DATASETS: DATA["FF++"] = build_ffpp_test()
if "CDF"  in DATASETS: DATA["CDF"]  = build_cdf()
''')

md(r"""### 3.1 WildDeepfake — ดาวน์โหลดทีละไฟล์ · เลือก 8 frame ต่อ sequence · (ถ้าเปิด) RetinaFace + align ซ้ำ

โครงสร้างไฟล์: `deepfake_in_the_wild/{real_test,fake_test}/<n>.tar.gz` — แต่ละ tar คือ 1 วิดีโอต้นทาง
ภายในมีหลาย **sequence** (โฟลเดอร์ของภาพใบหน้า) → ใช้ sequence เป็นหน่วยประเมิน เหมือน "วิดีโอ" ใน FF++/CDF
cell นี้พิมพ์โครงสร้างของ tar แรกออกมาให้ตรวจก่อนเสมอ
""")
code(r'''# ---------- face detector + align (เหมือนระดับ 3c) ----------
detector = FaceDetector(0 if DEVICE.type == "cuda" else -1)

def crop_face_test(img, bbox):
    H, W = img.shape[:2]
    (x0, y0), (x1, y1) = bbox
    w, h = x1 - x0, y1 - y0
    w0 = w / 4 * 0.5; h0 = h / 4 * 0.5
    return img[max(0, int(y0-h0)):min(H, int(y1+h0)+1), max(0, int(x0-w0)):min(W, int(x1+w0)+1)]

def realign(img_bgr):
    """คืน (ภาพ 224x224, aligned?) — ขยายภาพก่อนตรวจ เพราะภาพ WDF เป็นหน้าที่ crop มาแล้วขนาดเล็ก"""
    big = cv2.resize(img_bgr, (448, 448), interpolation=cv2.INTER_LINEAR)
    try:
        faces = [f for f in detector.detect(big) if f[2] >= DET_THRES]
    except Exception:
        faces = []
    if faces:
        box, lm, _ = max(faces, key=lambda f: (f[0][2]-f[0][0]) * (f[0][3]-f[0][1]))
        try:
            aligned, _, bb = face_utils.crop_aligned(big, lm.astype(np.float64), landmarks_68=None,
                                                    bboxes=box.astype(np.float64), aligned_image_size=IMG_SIZE)
            crop = crop_face_test(aligned, np.array([[bb[0], bb[1]], [bb[2], bb[3]]]))
            if crop.size:
                return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR), True
        except Exception:
            pass
    return cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR), False

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
num_key = lambda n: (int(re.sub(r"\D", "", Path(n).stem) or 0), n)

def pick_k(names, k=N_FRAMES):
    names = sorted(names, key=num_key)
    if len(names) >= k:
        return names[::len(names) // k][:k]
    return sorted(np.resize(np.array(names), k).tolist(), key=num_key)

def build_wdf():
    meta_csv = WDF_DIR / "wdf_meta.csv"
    prev = sorted(INPUT_DIR.rglob("wdf_crops/wdf_meta.csv")) if INPUT_DIR.exists() else []
    if prev and not meta_csv.exists():
        print("ใช้ wdf_crops จาก Add Input:", prev[0].parent)
        m = pd.read_csv(prev[0]); root = prev[0].parent
        m["frames"] = m["frames"].map(lambda s: [str(root / x) for x in json.loads(s)])
        return m
    from huggingface_hub import HfApi, hf_hub_download
    files = HfApi().list_repo_files(WDF_REPO, repo_type="dataset")
    todo = []
    for split, lab in WDF_SPLITS.items():
        fs = sorted([f for f in files if f"/{split}/" in f and f.endswith(".tar.gz")], key=num_key)
        todo += [(f, split, lab) for f in fs[:WDF_MAX_FILES]]
    print(f"WildDeepfake: {len(todo)} ไฟล์ tar · " +
          ", ".join(f"{s} {sum(t[1]==s for t in todo)}" for s in WDF_SPLITS))
    rows, n_aligned, n_img, shown = [], 0, 0, False
    dl_dir = TMP_DIR / "wdf_dl"; dl_dir.mkdir(parents=True, exist_ok=True)
    for f, split, lab in tqdm(todo, desc="WDF"):
        local = Path(hf_hub_download(WDF_REPO, f, repo_type="dataset", local_dir=str(dl_dir)))
        vid = Path(f).name.replace(".tar.gz", "")
        with tarfile.open(local, "r:*") as tf:
            names = [m.name for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(IMG_EXT)]
        seqs = {}
        for n in names: seqs.setdefault(str(Path(n).parent), []).append(n)
        if not shown:
            print(f"\nตัวอย่างโครงสร้าง {f}: {len(names)} ภาพ, {len(seqs)} sequence")
            for s_, ns in list(seqs.items())[:3]: print("  ", s_, "→", len(ns), "ภาพ เช่น", ns[:2])
            shown = True
        want = {}
        for s_, ns in seqs.items():
            for j, n in enumerate(pick_k(ns)): want.setdefault(n, []).append((s_, j))
        imgs = {}
        with tarfile.open(local, "r|*") as tf:               # อ่านรอบเดียวแบบ stream (r|* = ตรวจเองว่าบีบอัดหรือไม่ — ไฟล์ .tar.gz ของ WDF เป็น tar ธรรมดา)
            for m in tf:
                if m.name in want:
                    buf = np.frombuffer(tf.extractfile(m).read(), np.uint8)
                    imgs[m.name] = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        for s_, ns in seqs.items():
            sid = re.sub(r"[^\w.-]", "_", f"{vid}__{s_}")
            out_dir = WDF_DIR / split / sid; out_dir.mkdir(parents=True, exist_ok=True)
            rel = []
            for j, n in enumerate(pick_k(ns)):
                im = imgs.get(n)
                if im is None: continue
                if WDF_REALIGN:
                    im, ok = realign(im); n_aligned += ok
                else:
                    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
                n_img += 1
                p = out_dir / f"{j}.png"; cv2.imwrite(str(p), im); rel.append(str(p.relative_to(WDF_DIR)))
            if rel:
                rel = (rel * N_FRAMES)[:N_FRAMES]
                rows.append({"uid": f"{split}/{sid}", "label": lab, "group": split, "source_video": vid,
                             "n_frames_seq": len(ns), "frames": json.dumps(rel)})
        local.unlink(missing_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(meta_csv, index=False)
    print(f"\nrealign สำเร็จ {n_aligned}/{n_img} ภาพ ({100*n_aligned/max(n_img,1):.1f}%) — ที่เหลือใช้ภาพเดิม resize เป็น 224")
    m["frames"] = m["frames"].map(lambda s: [str(WDF_DIR / x) for x in json.loads(s)])
    return m

if "WDF" in DATASETS:
    t0 = time.time()
    DATA["WDF"] = build_wdf()
    w = DATA["WDF"]
    print(f"WildDeepfake: {len(w)} sequence จาก {w['source_video'].nunique()} วิดีโอ · "
          f"real {int((w.label==0).sum())} / fake {int((w.label==1).sum())} · {(time.time()-t0)/60:.1f} นาที")
''')

md("## 4. Inference ทุกโมเดล × ทุกชุดข้อมูล → score ระดับวิดีโอ (p และ u ของทุก head)")
code(r'''def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())

class FrameList(Dataset):
    def __init__(self, paths): self.paths = list(paths)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        rgb = np.asarray(Image.open(self.paths[i]).convert("RGB"))
        if rgb.shape[:2] != (IMG_SIZE, IMG_SIZE):
            rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return to_tensor(rgb)

@torch.no_grad()
def score_dataset(model, df):
    paths = [p for fs in df["frames"] for p in fs]
    owner = np.repeat(np.arange(len(df)), [len(fs) for fs in df["frames"]])
    loader = DataLoader(FrameList(paths), batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    acc = {}
    for x in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP and DEVICE.type == "cuda"):
            out = model(x)
        for h, lg in out.items():
            lg = lg.float()
            acc.setdefault(f"p_{h}", []).append(torch.softmax(lg, 1)[:, 1].cpu().numpy())
            acc.setdefault(f"u_{h}", []).append(evidential_uncertainty(lg)[:, 0].cpu().numpy())
    fr = pd.DataFrame({k: np.concatenate(v) for k, v in acc.items()}); fr["row"] = owner
    vid = fr.groupby("row").mean()                                  # เฉลี่ย 8 frame = score วิดีโอ
    return pd.concat([df[["uid", "label", "group"]].reset_index(drop=True), vid.reset_index(drop=True)], axis=1)

SCORES, INFO = {}, {}
for name, path in MODELS:
    t0 = time.time()
    model, info = load_detector(path); INFO[name] = info
    for ds, df in DATA.items():
        v = score_dataset(model, df)
        v.to_csv(OUT_DIR / f"scores_{name}_{ds}.csv", index=False)
        SCORES[(name, ds)] = v
    print(f"{name:<20} dual={info['dual']!s:<5} ufm={info['ufm']!s:<5} · {(time.time()-t0)/60:.1f} นาที")
    del model; torch.cuda.empty_cache()
''')

md("## 5. ผลหลัก — main branch (ตาม paper) · AUC, ACC และช่วงความเชื่อมั่น 95%")
code(r'''def eer_score(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2)

def boot_ci(y, s, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); y, s = np.asarray(y), np.asarray(s)
    i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    a = [roc_auc_score(y[idx], s[idx]) for idx in
         (np.concatenate([rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]) for _ in range(n))]
    return np.percentile(a, [2.5, 97.5])

def metrics(v, head="main", ci=False):
    y, s = v["label"].values.astype(int), v[f"p_{head}"].values
    pred = (s >= 0.5).astype(int)
    r = {"n": len(y), "AUC": 100 * roc_auc_score(y, s), "ACC": 100 * (pred == y).mean(),
         "EER": 100 * eer_score(y, s), "TNR": 100 * ((pred == 0) & (y == 0)).sum() / max((y == 0).sum(), 1),
         "TPR": 100 * ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)}
    if ci:
        lo, hi = boot_ci(y, s); r["CI"] = f"[{100*lo:.2f}, {100*hi:.2f}]"
    return r

rows = []
for (name, ds), v in SCORES.items():
    for h in ["main", "fused", "branch2"]:
        if f"p_{h}" in v:
            rows.append({"model": name, "dataset": ds, "head": h, **metrics(v, h, ci=(h == "main"))})
ALL = pd.DataFrame(rows)
ALL.to_csv(OUT_DIR / "metrics_all.csv", index=False)

main = ALL[ALL["head"] == "main"]
tab = main.pivot(index="model", columns="dataset", values="AUC").round(2)
tab_acc = main.pivot(index="model", columns="dataset", values="ACC").round(2)
cross = [d for d in ["CDF", "WDF"] if d in tab.columns]
if cross:
    tab["Avg cross AUC"] = tab[cross].mean(1).round(2); tab["Avg cross ACC"] = tab_acc[cross].mean(1).round(2)
print("AUC (%) — main branch, video-level")
print(tab.to_string())
print("\npaper (CFB): FF++ 99.36 · CDF 90.46")
print("\nช่วงความเชื่อมั่น 95% ของ AUC (bootstrap):")
print(main[["model", "dataset", "AUC", "CI"]].round(2).to_string(index=False))
tab.to_csv(OUT_DIR / "table_main_auc.csv")
''')
code(r'''# FF++ แยกตามวิธีปลอม (real vs method) — main branch
rows = []
for (name, ds), v in SCORES.items():
    if ds != "FF++": continue
    real = v[v["group"] == "original"]
    for m in FF_METHODS:
        sub = pd.concat([real, v[v["group"] == m]])
        rows.append({"model": name, "method": m, "AUC": 100 * roc_auc_score(sub["label"], sub["p_main"])})
if rows:
    per = pd.DataFrame(rows).pivot(index="model", columns="method", values="AUC").round(2)
    per.to_csv(OUT_DIR / "ffpp_per_method.csv"); print(per.to_string())
''')

md(r"""## 6. Ablation (Table VI) — เทียบคอลัมน์ CDF ของ paper ได้ตรง

paper รายงาน Avg ของ CDF/DFDC/FFIW ซึ่งเราไม่มี DFDC/FFIW → เทียบได้ตรงเฉพาะ **CDF**
คอลัมน์ WDF เป็นการทดสอบเพิ่มเติม (paper ไม่มีตัวเลข)
""")
code(r'''order = ["fb_ce", "fb_edu", "cfb_base", "cfb_dec", "cfb_dec_ufm", "cfb_full"]
rows = []
for v in order:
    name = "ours_cfb_full" if v == "cfb_full" else f"ours_{v}"
    r = {"variant": v, "paper CDF": PAPER_TABLE6[v][0], "paper Avg AUC*": PAPER_TABLE6[v][3]}
    for ds in DATA:
        m = main[(main.model == name) & (main.dataset == ds)]
        r[f"ours {ds}"] = round(float(m["AUC"].iloc[0]), 2) if len(m) else np.nan
    rows.append(r)
ABL = pd.DataFrame(rows)
if "ours CDF" in ABL:
    ABL["Δ CDF (ours − paper)"] = (ABL["ours CDF"] - ABL["paper CDF"]).round(2)
ABL.to_csv(OUT_DIR / "table6_ablation.csv", index=False)
print(ABL.to_string(index=False))
print("\n* paper Avg AUC = ค่าเฉลี่ยของ CDF, DFDC, FFIW (อ้างอิงเท่านั้น)")
print("  ทุกค่าของเราคือ main branch · NaN = ยังไม่ได้เทรน variant นั้นในระดับ 3d")
''')

md(r"""## 7. UTOM เต็มรูปแบบ (Sec. III-D, Eq. 10–11, Table IX)

- **τ = 0.5** — threshold มาตรฐาน
- **τ_ot** — ลองทุกค่าของ p บน *source* แล้วเลือกตัวที่ ACC สูงสุด → นำไปใช้กับ *target*
- **τ_ot^UTOM** — เหมือน τ_ot แต่ปรับ score ก่อนด้วย `p′ = tanh(λ·p / 2(u+ε))` · λ = 0.02 (paper) ทั้ง source และ target

ทำครบทุกคู่ source → target (รวม source = target แบบคอลัมน์ "CDF to CDF" ใน Table IX)
และทำ 2 แบบตาม Supplementary 2.3: **original** (สัดส่วน real/fake ตามจริง) กับ **balanced** (ถ่วงน้ำหนักให้ real = fake ก่อนหา τ)
""")
code(r'''def utom_adjust(p, u, lam=UTOM_LAMBDA, eps=UTOM_EPS):
    return np.tanh(lam * np.asarray(p) / (2 * (np.asarray(u) + eps)))

def best_threshold(y, s, balanced=False):
    """Eq. 10-11 — ทุกค่า score เป็นผู้สมัคร threshold (p >= τ → fake) · balanced = ถ่วงน้ำหนักให้ 2 class เท่ากัน"""
    y, s = np.asarray(y).astype(int), np.asarray(s, dtype=float)
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)) if balanced else np.full(len(y), 1 / len(y))
    o = np.argsort(s); ss, ys, ws = s[o], y[o], w[o]
    real_below = np.concatenate([[0], np.cumsum(ws * (1 - ys))])[:-1]     # real ที่อยู่ต่ำกว่า τ (ถูก)
    fake_above = (ws * ys).sum() - np.concatenate([[0], np.cumsum(ws * ys)])[:-1]
    acc = real_below + fake_above
    first = np.r_[True, ss[1:] != ss[:-1]]
    i = int(np.argmax(np.where(first, acc, -1)))
    return float(ss[i])

acc_at = lambda y, s, t: 100 * float(((np.asarray(s) >= t).astype(int) == np.asarray(y)).mean())

rows = []
for name, _ in MODELS:
    dss = [d for d in DATA if (name, d) in SCORES]
    for src in dss:
        S = SCORES[(name, src)]
        lam_auto = 2.0 / np.median(S["p_main"] / (S["u_main"] + UTOM_EPS))
        for bal in [False, True]:
            t05 = 0.5
            t_ot = best_threshold(S["label"], S["p_main"], bal)
            t_ut = best_threshold(S["label"], utom_adjust(S["p_main"], S["u_main"]), bal)
            for tgt in dss:
                T = SCORES[(name, tgt)]
                pu = utom_adjust(T["p_main"], T["u_main"])
                rows += [
                    {"model": name, "source": src, "target": tgt, "balance": "balanced" if bal else "original",
                     "threshold": "tau=0.5", "tau": t05, "ACC": acc_at(T["label"], T["p_main"], t05)},
                    {"model": name, "source": src, "target": tgt, "balance": "balanced" if bal else "original",
                     "threshold": "tau_ot", "tau": t_ot, "ACC": acc_at(T["label"], T["p_main"], t_ot)},
                    {"model": name, "source": src, "target": tgt, "balance": "balanced" if bal else "original",
                     "threshold": "tau_ot^UTOM", "tau": t_ut, "ACC": acc_at(T["label"], pu, t_ut), "lambda_auto": lam_auto},
                ]
UT = pd.DataFrame(rows)
UT.to_csv(OUT_DIR / "table9_utom_long.csv", index=False)

def show_table9(model, balance="original"):
    d = UT[(UT.model == model) & (UT.balance == balance)]
    if d.empty: return
    pv = d.pivot_table(index="threshold", columns=["source", "target"], values="ACC").reindex(["tau=0.5", "tau_ot", "tau_ot^UTOM"])
    print(f"\n[{model}] ACC (%) · source→target · {balance}")
    print(pv.round(2).to_string())

for name, _ in MODELS:
    if name in ("author", "ours_cfb_full"):
        show_table9(name, "original"); show_table9(name, "balanced")
print("\npaper Table IX (CFB + L_EDU, source = CDF → target CDF):", PAPER_TABLE9_CDF)
lam = UT.dropna(subset=["lambda_auto"]).groupby(["model", "source"])["lambda_auto"].first().round(4)
print("\nλ ที่คำนวณจาก 2/median(p/(u+ε)) ตาม Supplementary 2.1 (paper ใช้ 0.02):"); print(lam.to_string())
''')
code(r'''# สรุป UTOM แบบสั้น: ACC เฉลี่ยของ target ที่ไม่ใช่ source (ไม่เคยเห็นตอนหา threshold)
d = UT[UT.source != UT.target]
SUMM = (d.groupby(["model", "balance", "threshold"])["ACC"].mean().unstack("threshold")
          .reindex(columns=["tau=0.5", "tau_ot", "tau_ot^UTOM"]).round(2))
SUMM.to_csv(OUT_DIR / "utom_summary_unseen.csv")
print("ACC เฉลี่ยบน target ที่ไม่ใช่ source (%)"); print(SUMM.to_string())
''')

md("## 8. กราฟ — ROC และการกระจายของ score (main branch)")
code(r'''C = {"author": "#eb6834", "ours_cfb_full": "#2a78d6"}
INK2, GRID = "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
dss = list(DATA)
fig, axes = plt.subplots(2, len(dss), figsize=(4.4 * len(dss), 8))
axes = np.array(axes).reshape(2, -1)
for j, ds in enumerate(dss):
    ax = axes[0, j]
    for name in ["author", "ours_cfb_full"]:
        if (name, ds) not in SCORES: continue
        v = SCORES[(name, ds)]
        fpr, tpr, _ = roc_curve(v["label"], v["p_main"])
        ax.plot(fpr, tpr, lw=2, color=C[name], label=f"{name} ({100*roc_auc_score(v['label'], v['p_main']):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    ax.set(title=f"ROC — {ds}", xlabel="FPR", ylabel="TPR"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1, j]
    if ("ours_cfb_full", ds) in SCORES:
        v = SCORES[("ours_cfb_full", ds)]
        for lab, col, nm in [(0, "#2a78d6", "real"), (1, "#eb6834", "fake")]:
            ax.hist(v[v.label == lab]["p_main"], bins=np.linspace(0, 1, 31), density=True, alpha=0.35, color=col, label=nm)
        ax.axvline(0.5, color=INK2, ls="--", lw=1)
        ax.set(title=f"score (ours) — {ds}", xlabel="P(fake)", ylabel="density"); ax.legend(frameon=False)
for ax in axes.ravel(): ax.grid(color=GRID, lw=0.8)
plt.tight_layout(); plt.savefig(OUT_DIR / "roc_hist_all.png", dpi=150); plt.show()
''')

md("## 9. สรุป")
code(r'''print("=" * 72)
print("สรุประดับ 3e")
print("=" * 72)
print(tab.to_string())
if not ABL.empty: print("\nAblation (main branch):\n" + ABL.to_string(index=False))
print("\nUTOM (ACC เฉลี่ย target ที่ไม่เคยเห็น):\n" + SUMM.to_string())
print(f"""
ข้อควรระบุในรายงาน
  1. main branch    ทุกตัวเลขในหัวข้อ 5–7 ใช้ main branch ตาม paper (ระดับ 3b รายงาน FF++ จาก fused head)
  2. WildDeepfake   ไม่อยู่ใน Table II/VI/IX ของ paper — ใช้เป็นการทดสอบเพิ่มเติม เทียบผู้เขียน vs เรา
                    ใบหน้าถูก crop มาแล้วโดยผู้สร้าง dataset → realign ด้วย RetinaFace={WDF_REALIGN}
  3. UTOM           u ระดับวิดีโอ = ค่าเฉลี่ย u ของ 8 frame (paper ไม่ได้ระบุวิธีรวม) · λ = {UTOM_LAMBDA}
  4. DFDC / FFIW    ไม่ได้ประเมิน (ไม่สามารถเข้าถึง test set ได้) → เทียบกับ paper ได้เฉพาะคอลัมน์ CDF
เวลารวม: {(time.time()-T_START)/60:.1f} นาที""")
print("ไฟล์ผลลัพธ์:", *[p.name for p in sorted(OUT_DIR.iterdir())], sep="\n  ")
''')

nb = {"cells": cells, "metadata": base["metadata"], "nbformat": base["nbformat"], "nbformat_minor": base["nbformat_minor"]}
json.dump(nb, open("FFD_level3e_eval_utom_wdf.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote FFD_level3e_eval_utom_wdf.ipynb", len(cells), "cells")
