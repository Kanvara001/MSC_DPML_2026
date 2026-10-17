"""สร้าง FFD_level4_live_demo.ipynb — demo สดตอนนำเสนอ: รันโค้ดจริงของ repo ให้เห็น algorithm ภายในทีละขั้น"""
import json

base = json.load(open("FFD_level3c_eval_crossdataset.ipynb", encoding="utf-8"))
c3 = lambda i: "".join(base["cells"][i]["source"])
e3 = open("make_3e.py", encoding="utf-8").read()

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                            "source": t.strip("\n").splitlines(keepends=True)})

# ---------------------------------------------------------------- 0
md(r"""
# ระดับ 4 — Live demo: algorithm ภายใน FFDBackbone ทีละขั้น

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026
**ผู้นำเสนอ:** นางสาวกัญญ์วรา ถาวโรฤทธิ์ · 695020018-8

notebook นี้ใช้ตอนนำเสนอ รันจบในประมาณ **3–5 นาที** แต่ละหัวข้อตรงกับสไลด์ส่วน Algorithm

| หัวข้อ | สิ่งที่แสดง | โค้ดต้นทาง |
|---|---|---|
| 1 | โค้ดจริงของ CFB, L_dec, u, L_EDU ที่ดึงจาก repo | `models/custom.py`, `common/losses/edl_loss.py` |
| 2 | End-to-end: วิดีโอ → RetinaFace → crop 224 → real / fake | `face_process/`, `inference.py` |
| 3 | ภายในโมเดลทีละขั้นบน 1 ภาพ: patch → ViT 12 block → CLS → logit → evidence → *u* | `models/lib/BEiT_v2/modeling_finetune.py` |
| 4 | CFB: 2 branch, Pearson, UFM, fused head และ loss Eq. 9 | `models/custom.py`, `engine_finetune.py` |
| 5 | Video-level: เฉลี่ย 8 frame บนหลายวิดีโอ เทียบ checkpoint ผู้เขียนกับของเรา | `common/evaluations/` |
| 6 | UTOM: ปรับ score ด้วย *u* แล้วหา threshold | (ไม่มีใน repo — เขียนเองตาม Eq. 10–11) |

**ก่อนรัน:** GPU **T4 ×1** · Internet **On** · Add Input:
- output ของระดับ **3b** (`cfb_run/best.pth`) · **3c** (`cdf_crops`) · **3e_V2** (`results_level3e`)
- (ไม่บังคับ) Kaggle Dataset ที่มีไฟล์ `.mp4` สั้น ๆ เช่นวิดีโอเซลฟี่ 5–10 วินาที สำหรับหัวข้อ 2
""")

# ---------------------------------------------------------------- config
md("## 0. ตั้งค่า")
code(r'''
import os, sys, re, json, time, glob, random, shutil, subprocess, warnings, ast, textwrap
from pathlib import Path
warnings.filterwarnings("ignore")
T_START = time.time()

ON_KAGGLE = Path("/kaggle/working").exists()
WORK_DIR  = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("FFD_WORK", "./work")).resolve()
TMP_DIR   = Path("/tmp/ffd") if ON_KAGGLE else WORK_DIR / "tmp"
INPUT_DIR = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("FFD_INPUT", str(WORK_DIR / "input")))
REPO_DIR  = TMP_DIR / "FFDBackbone"
WEIGHT_DIR = TMP_DIR / "weights"
OUT_DIR   = WORK_DIR / "demo_level4"
for d in [TMP_DIR, WEIGHT_DIR, OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/zhenglab/FFDBackbone.git"
GDRIVE_FOLDER = "https://drive.google.com/drive/folders/1jFIpb4TftJiL82h69c3Roqy8uIzGQP4G"
AUTHOR_CKPT_PATH = None                  # ใส่ path ถ้าอัป Final-mainbranch.tar เป็น Kaggle Dataset ไว้แล้ว
AUTHOR_HINTS   = ["ours", "final", "ddbf", "dbbf", "mainbranch"]
AUTHOR_EXCLUDE = ["beit-1k-face-55w"]

DEMO_VIDEO = None                        # path ของ .mp4 ที่ต้องการ (None = ค้นหาใน Add Input)
N_DEMO_VIDEOS = 6                        # จำนวนวิดีโอ Celeb-DF ในหัวข้อ 5

N_FRAMES, IMG_SIZE = 8, 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
DET_THRES, MAX_SHIFT = 0.5, 10
UTOM_LAMBDA, UTOM_EPS = 0.02, 1e-7
SEED = 7
random.seed(SEED)

def tick(msg=""):
    print(f"[{time.time() - T_START:5.0f} s] {msg}")
tick("ตั้งค่าเสร็จ")
''')

md("## 0.1 ติดตั้ง + clone repo ของผู้เขียน + patch (เหมือนระดับ 3c)")
s4 = c3(4)
code(s4 + r'''

import matplotlib
matplotlib.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
torch.set_grad_enabled(False)
tick("repo พร้อม")
''')

# ---------------------------------------------------------------- 1 repo code
md(r"""
## 1. โค้ดจริงของ algorithm (ดึงจาก repo ของผู้เขียน)

cell นี้อ่านไฟล์ใน repo แล้วพิมพ์ฟังก์ชันหลักออกมา จากนั้นนำฟังก์ชัน**ตัวเดียวกันนี้**ไปใช้ในหัวข้อถัดไป
ไม่ได้เขียนใหม่ ตัวเลขที่เห็นจึงมาจากโค้ดต้นฉบับโดยตรง

| ฟังก์ชัน | ในบทความ |
|---|---|
| `DDBF_BEiT_v2.forward` | CFB: 2 branch + UFM (Sec. III-C, Fig. 4) |
| `pearson_correlation_loss` | *L*<sub>dec</sub> (Eq. 5) |
| `uncertainty` | *u* = K / S (Eq. 3–4) |
| `EvidenceLoss.edl_loss` | *L*<sub>EDU</sub> = *L*<sub>EDL</sub> + *L*<sub>EUC</sub> (Eq. 6–8) |
""")
code(r'''
def show_source(path, names, cls=None):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    nodes = tree.body
    if cls:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = {}
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            seg = " " * n.col_offset + ast.get_source_segment(src, n)
            out[n.name] = seg
            title = f"{Path(path).relative_to(REPO_DIR)}  ::  {(cls + '.') if cls else ''}{n.name}   (บรรทัด {n.lineno}-{n.end_lineno})"
            print("=" * len(title)); print(title); print("=" * len(title))
            print(textwrap.dedent(seg)); print()
    return out

CUSTOM = REPO_DIR / "models" / "custom.py"
EDL    = REPO_DIR / "common" / "losses" / "edl_loss.py"

show_source(CUSTOM, ["forward"], cls="DDBF_BEiT_v2")
repo_fns = show_source(CUSTOM, ["pearson_correlation_loss", "exp_evidence", "uncertainty"])
show_source(EDL, ["edl_loss"], cls="EvidenceLoss")

# นำฟังก์ชันของ repo มาใช้จริง (exec จาก source ของไฟล์ ไม่ได้พิมพ์ใหม่)
REPO = {"torch": torch}
for seg in repo_fns.values():
    exec(textwrap.dedent(seg), REPO)
pearson_correlation_loss, repo_uncertainty = REPO["pearson_correlation_loss"], REPO["uncertainty"]
edl_mod = load_module_from_file("repo_edl_loss", EDL)
# ค่าเดียวกับ checkpoints/Ours/DBBF.yaml ของผู้เขียน
criterion_e = edl_mod.EvidenceLoss(num_classes=2, evidence="exp", loss_type="log",
                                   with_kldiv=False, with_avuloss=True, annealing_method="exp")
criterion_b = nn.CrossEntropyLoss()
print("โหลดฟังก์ชันจาก repo แล้ว:", list(repo_fns), "+ EvidenceLoss")
''')

# ---------------------------------------------------------------- models
md(r"""
## 1.1 โหลดโมเดล 2 ตัว

- **checkpoint ของผู้เขียน** `Final-mainbranch.tar` — มีเฉพาะ main branch (ใช้ตอน inference ตามบทความ)
- **CFB ที่เราฝึกเอง** `cfb_run/best.pth` จากระดับ 3b — มีครบ 2 branch จึงใช้แสดง *L*<sub>dec</sub> และ UFM
""")
i0 = e3.index("code(r'''BEIT_CONFIG")
i1 = e3.index("''')", e3.index("def load_detector"))
code(e3[i0 + len("code(r'''"):i1] + r'''

''' + e3[e3.index("def find_or_download_author"):e3.index("MODELS = []")].rstrip() + r'''

def find_one(pattern, depth=5):
    if not INPUT_DIR.exists(): return None
    for k in range(depth + 1):
        hits = sorted(INPUT_DIR.glob("/".join(["*"] * k + [pattern])))
        if hits: return hits[0]
    return None

author, info_a = load_detector(find_or_download_author())
ours_path = find_one("cfb_run/best.pth")
assert ours_path is not None, "ไม่พบ cfb_run/best.pth — Add Input เป็น output ของระดับ 3b"
ours, info_o = load_detector(ours_path)
for n, m, i in [("author", author, info_a), ("ours (CFB)", ours, info_o)]:
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{n:<11} dual={i['dual']!s:<5} ufm={i['ufm']!s:<5} params={n_par:6.1f} M")
tick("โหลดโมเดลเสร็จ")
''')

# ---------------------------------------------------------------- 2 end-to-end
md(r"""
## 2. End-to-end: วิดีโอ → ใบหน้า → Real / Fake

ขั้นตอนเดียวกับที่ใช้เตรียม FF++ และ Celeb-DF (ระดับ 3a, 3c)

1. เลือก 8 frame กระจายเท่า ๆ กันทั้งวิดีโอ
2. **RetinaFace** หาใบหน้า + landmark 5 จุด (เลือกหน้าที่ใหญ่สุด แล้วติดตามด้วย IoU)
3. Align ตาม landmark → crop ขยายขอบ 12.5% → resize 224 × 224
4. ส่งเข้า **main branch** → *p*(fake) ของแต่ละ frame → เฉลี่ย 8 frame = score ของวิดีโอ

ถ้าไม่มีไฟล์ `.mp4` ใน Add Input จะใช้ภาพใบหน้าจาก Celeb-DF แทนเพื่อแสดงขั้น RetinaFace + align
""")
code(c3(8).replace("def process_video(path, out_dir):", "def _unused_process_video(path, out_dir):") + r'''

def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())

def read_rgb(p):
    rgb = np.asarray(Image.open(p).convert("RGB"))
    return rgb if rgb.shape[:2] == (IMG_SIZE, IMG_SIZE) else cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

@torch.no_grad()
def score_frames(model, rgbs):
    x = torch.stack([to_tensor(r) for r in rgbs]).to(DEVICE)
    out = model(x)
    return {h: (torch.softmax(lg.float(), 1)[:, 1].cpu().numpy(),
                evidential_uncertainty(lg.float())[:, 0].cpu().numpy()) for h, lg in out.items()}

def video_to_faces(path):
    """คืน (ภาพใบหน้า RGB 8 ภาพ, frame index, ภาพ frame แรกที่วาด box + landmark)"""
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faces_rgb, idxs, vis, prev = [], [], None, None
    for t in sample_indices(n):
        for shift in range(MAX_SHIFT + 1):
            fi = int(t) + shift
            if fi >= n: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, frame = cap.read()
            if not ok: continue
            faces = valid_faces(detector.detect(frame))
            if not faces: continue
            face = pick_face(faces, prev); crop = align_and_crop(frame, face)
            if crop is None: continue
            if vis is None:
                vis = frame.copy(); (x0, y0, x1, y1) = face[0].astype(int)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 40, 200), 3)
                for (lx, ly) in face[1].astype(int): cv2.circle(vis, (lx, ly), 5, (60, 200, 60), -1)
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            faces_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); idxs.append(fi); prev = face[0]
            break
    cap.release()
    return faces_rgb, idxs, vis, n

def image_to_face(rgb):
    """ใช้เมื่อไม่มีวิดีโอ: ขยายภาพใบหน้าแล้วตรวจซ้ำด้วย RetinaFace (แบบเดียวกับ WildDeepfake ใน 3e)"""
    big = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (448, 448))
    faces = valid_faces(detector.detect(big))
    if not faces:
        print("RetinaFace ไม่พบใบหน้าในภาพนี้ → ใช้ภาพเดิม"); return rgb, rgb
    face = pick_face(faces, None); crop = align_and_crop(big, face)
    vis = big.copy(); (x0, y0, x1, y1) = face[0].astype(int)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 40, 200), 3)
    for (lx, ly) in face[1].astype(int): cv2.circle(vis, (lx, ly), 5, (60, 200, 60), -1)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

CDF_CROPS = find_one("cdf_crops")
assert CDF_CROPS is not None, "ไม่พบ cdf_crops — Add Input เป็น output ของระดับ 3c"
def cdf_frames(uid):
    fr = sorted((CDF_CROPS / uid).glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    return [str(p) for p in (fr[:N_FRAMES] if len(fr) >= N_FRAMES else np.resize(np.array(fr), N_FRAMES))]

video = Path(DEMO_VIDEO) if DEMO_VIDEO else find_one("*.mp4", depth=4)
t0 = time.time()
if video is not None and video.exists():
    faces, idxs, vis, n_total = video_to_faces(video)
    src_label = f"{video.name} ({n_total} frames)"
else:
    uid0 = sorted(p.name for p in CDF_CROPS.iterdir() if p.is_dir() and "synthesis" in p.name.lower())[0]
    f0, vis = image_to_face(read_rgb(cdf_frames(uid0)[0]))
    faces, idxs = [f0], [0]
    src_label = f"Celeb-DF crop {uid0} (no .mp4 in input)"
print(f"แหล่งข้อมูล: {src_label} · ได้ใบหน้า {len(faces)} ภาพ · RetinaFace + align {time.time()-t0:.1f} s")

res = score_frames(author, faces)
p, u = res["main"]
fig = plt.figure(figsize=(14, 3.6))
ax = fig.add_subplot(1, len(faces) + 2, (1, 2)); ax.imshow(vis); ax.axis("off"); ax.set_title("RetinaFace: box + 5 landmarks", fontsize=9)
for k, im in enumerate(faces):
    a = fig.add_subplot(1, len(faces) + 2, k + 3); a.imshow(im); a.axis("off")
    a.set_title(f"f{idxs[k]}\np={p[k]:.3f} u={u[k]:.3f}", fontsize=8)
plt.tight_layout(); plt.show()
vp = float(p.mean())
print(f"score ของวิดีโอ = mean(p) = {vp:.4f} → ที่ τ = 0.5 ทำนายว่า {'FAKE' if vp >= 0.5 else 'REAL'}  (checkpoint ผู้เขียน, main branch)")
tick("หัวข้อ 2 เสร็จ")
''')

# ---------------------------------------------------------------- 3 inside the backbone
md(r"""
## 3. ภายในโมเดลทีละขั้น (1 ภาพ · checkpoint ผู้เขียน · main branch)

เดิน forward ของ `VisionTransformer.forward_features` ด้วยมือทีละบรรทัด แล้วตรวจว่าได้ค่าเท่ากับเรียก `model(x)` ตรง ๆ

```
x (1,3,224,224) ─patch_embed (Conv 16×16, stride 16)→ (1,196,768)
               ─concat [CLS]→ (1,197,768)
               ─12 × Block: x += γ₁·Attn(LN(x)) + rel_pos_bias ; x += γ₂·MLP(LN(x))→ (1,197,768)
               ─LayerNorm → เลือก token [CLS] → f (1,768)
f ─LayerNorm → Linear(768→2)→ logit z ─exp(clamp)→ evidence e ─+1→ α ─→ p = α/S , u = K/S
```
""")
code(r'''
pick = None
sc_a = find_one("results_level3e/scores_author_CDF.csv") or find_one("scores_author_CDF.csv", depth=4)
sc_o = find_one("results_level3e/scores_ours_cfb_full_CDF.csv") or find_one("scores_ours_cfb_full_CDF.csv", depth=4)
if sc_a is not None and sc_o is not None:
    A, O = pd.read_csv(sc_a), pd.read_csv(sc_o)
    M = A[["uid", "label", "p_main", "u_main"]].merge(O[["uid", "p_main", "u_main"]], on="uid", suffixes=("_a", "_o"))
    M = M[M.uid.map(lambda d: (CDF_CROPS / d).is_dir())]
    # วิดีโอปลอมที่ผู้เขียนตอบถูก แต่โมเดลของเราพลาด → เห็นความต่างชัด
    cand = M[(M.label == 1) & (M.p_main_a > 0.8) & (M.p_main_o < 0.5)]
    pick = (cand if len(cand) else M[M.label == 1]).sample(1, random_state=SEED).iloc[0]
    UID = pick.uid
else:
    UID = sorted(p.name for p in CDF_CROPS.iterdir() if p.is_dir() and "synthesis" in p.name.lower())[0]
FR = cdf_frames(UID)
img = read_rgb(FR[0])
x = to_tensor(img)[None].to(DEVICE)
print("วิดีโอ:", UID, "| label = FAKE" if "synthesis" in UID.lower() else "| label = REAL")
print("x  :", tuple(x.shape), "(normalize ด้วย mean/std ของ ImageNet)")

bb = author.backbone_model
t = bb.patch_embed(x);                       print("① patch_embed :", tuple(t.shape), "= 14×14 patch × 768 มิติ")
h = torch.cat([bb.cls_token.expand(1, -1, -1), t], dim=1); print("② + [CLS]     :", tuple(h.shape))
print("   pos_embed   :", bb.pos_embed, "→ ใช้ relative position bias ในแต่ละ block แทน (use_rel_pos_bias=True)")
h = bb.pos_drop(h)
cls_norm = []
for i, blk in enumerate(bb.blocks):
    if i == len(bb.blocks) - 1:
        attn = blk(h, rel_pos_bias=None, return_attention=True)        # (1,12,197,197)
    h = blk(h, rel_pos_bias=None)
    cls_norm.append(float(h[0, 0].norm()))
print("③ 12 blocks   :", tuple(h.shape), "· ‖CLS‖ ต่อ block =", " ".join(f"{v:.1f}" for v in cls_norm))
f = bb.norm(h)[:, 0];                         print("④ CLS feature :", tuple(f.shape))
print("   ตรวจ: ‖f − backbone(x)‖ =", f"{float((f - bb(x)).abs().max()):.2e}", "(ต้องเป็น ~0)")

z = author.classify(f)[0].float()
e = torch.exp(torch.clamp(z, -10, 10)); alpha = e + 1; S = alpha.sum()
print(f"\n⑤ logit z      = [{z[0]:+.3f}, {z[1]:+.3f}]   (real, fake)")
print(f"⑥ evidence e   = exp(clamp(z)) = [{e[0]:.3f}, {e[1]:.3f}]")
print(f"⑦ α = e + 1    = [{alpha[0]:.3f}, {alpha[1]:.3f}] · S = Σα = {S:.3f}")
print(f"⑧ p(fake)      = softmax(z)[1] = {float(torch.softmax(z, 0)[1]):.4f}   (ที่ inference.py ใช้)")
print(f"   α/S          = {float(alpha[1] / S):.4f}   (Dirichlet mean)")
print(f"⑨ uncertainty  u = K/S = 2/{S:.3f} = {float(2 / S):.4f}   ← repo: {float(repo_uncertainty(z[None]))  :.4f}")

att = attn[0, :, 0, 1:].mean(0).reshape(14, 14).cpu().numpy()
att = cv2.resize((att - att.min()) / (att.max() - att.min() + 1e-9), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
axs[0].imshow(img); axs[0].set_title("input 224x224")
g = img.copy(); g[::16] = 255; g[:, ::16] = 255
axs[1].imshow(g); axs[1].set_title("14x14 patches (16 px)")
axs[2].imshow(img); axs[2].imshow(att, cmap="magma", alpha=.55); axs[2].set_title("block 12: CLS -> patch attention")
for a in axs: a.axis("off")
plt.tight_layout(); plt.show()
tick("หัวข้อ 3 เสร็จ")
''')

# ---------------------------------------------------------------- 4 CFB
md(r"""
## 4. CFB: 2 branch แข่งกัน + UFM + loss Eq. 9 (โมเดลของเรา, 8 frame ของวิดีโอเดียวกัน)

ทำตาม `DDBF_BEiT_v2.forward` ทุกบรรทัด โดยเรียกฟังก์ชันของ repo ที่โหลดไว้ในหัวข้อ 1

- **main** `backbone_model` กับ **auxiliary** `backbone_model2` เริ่มจาก weight เดียวกัน (FB) แต่ใช้ `norm_for_cls` + `header` **ร่วมกัน**
- *L*<sub>dec</sub> = Pearson(**f<sub>M</sub>.detach()**, f<sub>A</sub>) → gradient ไปที่ auxiliary เท่านั้น ดันให้หา cue อื่น
- UFM: w = softmax(−10·[u<sub>M</sub>, u<sub>A</sub>]) → branch ที่มั่นใจกว่าได้น้ำหนักมากกว่า
- loss รวม (Eq. 9) = CE(fused) + L<sub>EDU</sub>(main) + L<sub>EDU</sub>(aux) + L<sub>dec</sub>
""")
code(r'''
X = torch.stack([to_tensor(read_rgb(p)) for p in FR]).to(DEVICE)           # (8,3,224,224) = B·T
label = int("synthesis" in UID.lower())

fM = ours.backbone_model(X); fA = ours.backbone_model2(X)                  # (8,768) ทั้งคู่
r  = pearson_correlation_loss(fM.detach(), fA)                             # L_dec  (ฟังก์ชันของ repo)
zM = ours.classify(fM); zA = ours.classify(fA)                             # header ร่วมกัน
uM, uA = repo_uncertainty(zM.float()), repo_uncertainty(zA.float())        # (8,1)
w  = torch.softmax(-10 * torch.cat([uM, uA], -1), -1)                     # UFM
fF = torch.matmul(w.unsqueeze(1), torch.stack((fM, fA), 1)).squeeze(1)    # Σ w_i f_i
zF = ours.classify(fF)

chk = ours(X)
print("ตรวจกับ Detector.forward: |Δ main| =", f"{float((chk['main'] - zM).abs().max()):.1e}",
      "· |Δ fused| =", f"{float((chk['fused'] - zF).abs().max()):.1e}")
print(f"\nPearson(f_M, f_A) เฉลี่ย 8 frame = {float(r):+.3f}   (ค่าลบ = 2 branch เข้ารหัสต่างกัน)")
pr = lambda z: torch.softmax(z.float(), 1)[:, 1].cpu().numpy()
tab = pd.DataFrame({"frame": [Path(p).stem for p in FR],
                    "p_main": pr(zM), "u_main": uM[:, 0].cpu().numpy(),
                    "p_aux": pr(zA), "u_aux": uA[:, 0].cpu().numpy(),
                    "w_main": w[:, 0].cpu().numpy(), "w_aux": w[:, 1].cpu().numpy(), "p_fused": pr(zF)})
print(tab.round(4).to_string(index=False))
print(f"\nvideo score: main {tab.p_main.mean():.4f} · aux {tab.p_aux.mean():.4f} · fused {tab.p_fused.mean():.4f} · label = {'FAKE' if label else 'REAL'}")

# --- loss Eq. 9 แบบเดียวกับ engine_finetune.train_dual_one_epoch ---
y = torch.full((len(FR),), label, device=DEVICE)
def edu(z, epoch, total=10):
    d = criterion_e(z.float(), y, epoch=epoch, total_epoch=total)
    return sum(v.mean() for k, v in d.items() if "loss" in k), {k: float(v.mean()) for k, v in d.items() if "loss" in k}
print("\nLoss Eq. 9 บน batch นี้ (8 frame):")
for ep in [0, 9]:
    lM, dM = edu(zM, ep); lA, _ = edu(zA, ep); lc = criterion_b(zF.float(), y)
    lam = float(0.01 * np.exp(-np.log(0.01) / 10 * ep))
    print(f"  epoch {ep}: λ_anneal={lam:.3f} | L_ce(fused)={float(lc):.4f}  L_EDU(main)={float(lM):.4f} {dM}"
          f"  L_EDU(aux)={float(lA):.4f}  L_dec={float(r):+.4f}  →  รวม {float(lc + lM + lA + r):.4f}")

# --- stop-gradient: L_dec ไม่ส่ง gradient กลับไปที่ main branch ---
with torch.enable_grad():
    a = fM.detach().clone().requires_grad_(True); b = fA.detach().clone().requires_grad_(True)
    pearson_correlation_loss(a.detach(), b).backward()
print(f"\nstop-gradient: ∂L_dec/∂f_M = {a.grad}  ·  ‖∂L_dec/∂f_A‖ = {float(b.grad.norm()):.4f}")

def cls_attn(bbm, x1):
    h = torch.cat([bbm.cls_token.expand(1, -1, -1), bbm.patch_embed(x1)], 1)
    for blk in bbm.blocks[:-1]: h = blk(h)
    a = bbm.blocks[-1](h, return_attention=True)[0, :, 0, 1:].mean(0).reshape(14, 14).cpu().numpy()
    return cv2.resize((a - a.min()) / (a.max() - a.min() + 1e-9), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
axs[0].imshow(img); axs[0].set_title(UID[:28], fontsize=8)
for a_, bbm, nm in [(axs[1], ours.backbone_model, "main"), (axs[2], ours.backbone_model2, "auxiliary")]:
    a_.imshow(img); a_.imshow(cls_attn(bbm, X[:1]), cmap="magma", alpha=.55); a_.set_title(f"{nm} branch attention")
for a_ in axs: a_.axis("off")
plt.tight_layout(); plt.show()
tick("หัวข้อ 4 เสร็จ")
''')

# ---------------------------------------------------------------- 5 video-level
md(r"""
## 5. Video-level บน Celeb-DF หลายวิดีโอ (รันสด แล้วเทียบกับผลระดับ 3e_V2)

score ของวิดีโอ = ค่าเฉลี่ย *p* ของ 8 frame (main branch) · ค่าที่รันสดควรเท่ากับไฟล์ score ของ 3e_V2
(คอลัมน์ Δ ต่างได้ราว 0.001 เพราะ 3e ใช้ AMP fp16 ส่วน demo นี้ใช้ fp32)
""")
code(r'''
if pick is not None:
    rs = np.random.RandomState(SEED)
    sel = pd.concat([M[M.label == 0].sample(N_DEMO_VIDEOS // 2, random_state=rs),
                     M[M.label == 1].sample(N_DEMO_VIDEOS - N_DEMO_VIDEOS // 2, random_state=rs)])
else:
    dirs = sorted(p.name for p in CDF_CROPS.iterdir() if p.is_dir())
    sel = pd.DataFrame({"uid": random.sample(dirs, N_DEMO_VIDEOS)})
    sel["label"] = sel.uid.str.lower().str.contains("synthesis").astype(int)

rows, t0 = [], time.time()
for _, s in sel.iterrows():
    rgbs = [read_rgb(p) for p in cdf_frames(s.uid)]
    pa, ua = score_frames(author, rgbs)["main"]; po, uo = score_frames(ours, rgbs)["main"]
    rows.append({"uid": s.uid, "label": "FAKE" if s.label else "REAL",
                 "author_p": pa.mean(), "author_u": ua.mean(), "ours_p": po.mean(), "ours_u": uo.mean(),
                 "Δ vs 3e (author)": abs(pa.mean() - s.get("p_main_a", np.nan)),
                 "Δ vs 3e (ours)": abs(po.mean() - s.get("p_main_o", np.nan)), "_img": rgbs[0]})
V = pd.DataFrame(rows)
for m in ["author", "ours"]:
    V[f"{m}_pred"] = np.where(V[f"{m}_p"] >= 0.5, "FAKE", "REAL")
print(f"รัน {len(V)} วิดีโอ × 8 frame × 2 โมเดล ใน {time.time()-t0:.1f} s")
print(V.drop(columns="_img").round(4).to_string(index=False))

fig, axs = plt.subplots(1, len(V), figsize=(2.2 * len(V), 3))
for a, (_, v) in zip(np.atleast_1d(axs), V.iterrows()):
    a.imshow(v._img); a.axis("off")
    a.set_title(f"GT {v.label}\nauthor {v.author_p:.2f} | ours {v.ours_p:.2f}", fontsize=8,
                color="black" if v.author_pred == v.label else "crimson")
plt.tight_layout(); plt.show()
tick("หัวข้อ 5 เสร็จ")
''')

# ---------------------------------------------------------------- 6 UTOM
md(r"""
## 6. UTOM: ปรับ score ด้วย uncertainty แล้วหา threshold (Eq. 10–11)

repo ไม่มีโค้ดส่วนนี้ จึงเขียนตามบทความ (เหมือนระดับ 3e):

```
p′ = tanh( λ·p / 2(u + ε) ) ,  λ = 0.02
τ  = argmax_τ ACC(source, τ)  → นำ τ ไปใช้กับ target ที่ไม่มี label
```
ใช้ไฟล์ score ของทุกวิดีโอจาก 3e_V2 (FF++ 700 · Celeb-DF 518 · WildDeepfake 806) — คำนวณบน CPU ไม่ถึง 1 วินาที
""")
utom = e3[e3.index("def utom_adjust"):e3.index("acc_at = lambda")]
code(utom + r'''acc_at = lambda y, s, t: 100 * float(((np.asarray(s) >= t).astype(int) == np.asarray(y)).mean())

SC = {}
for m in ["author", "ours_cfb_full"]:
    for ds in ["FF++", "CDF", "WDF"]:
        f = find_one(f"results_level3e/scores_{m}_{ds}.csv") or find_one(f"scores_{m}_{ds}.csv", depth=4)
        if f is not None: SC[(m, ds)] = pd.read_csv(f)
print("โหลด score:", {k: len(v) for k, v in SC.items()})

rows = []
for m in ["author", "ours_cfb_full"]:
    if (m, "CDF") not in SC: continue
    S = SC[(m, "CDF")]
    pu = utom_adjust(S.p_main, S.u_main)
    t_ot, t_ut = best_threshold(S.label, S.p_main), best_threshold(S.label, pu)
    rows.append({"model": m, "τ=0.5": acc_at(S.label, S.p_main, .5), "τ_ot": acc_at(S.label, S.p_main, t_ot),
                 "τ_UTOM": acc_at(S.label, pu, t_ut), "τ_ot value": t_ot, "τ_UTOM value": t_ut,
                 "λ_auto": 2 / np.median(S.p_main / (S.u_main + UTOM_EPS))})
print("\nTable IX (source = target = Celeb-DF) · ACC (%) · บทความ: 83.20 / 83.78 / 85.33")
print(pd.DataFrame(rows).round(4).to_string(index=False))

if ("author", "CDF") in SC:
    S = SC[("author", "CDF")]; pu = utom_adjust(S.p_main, S.u_main); t_ut = best_threshold(S.label, pu)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for lab, c, nm in [(0, "#6aa58e", "real"), (1, "#c9773f", "fake")]:
        k = S.label == lab
        axs[0].scatter(S.p_main[k], S.u_main[k], s=10, c=c, alpha=.6, label=nm)
        axs[1].hist(pu[k], bins=40, color=c, alpha=.6, label=nm)
    axs[0].set_xlabel("p (fake)"); axs[0].set_ylabel("uncertainty u"); axs[0].legend(); axs[0].set_title("author · Celeb-DF videos")
    axs[1].axvline(t_ut, color="#7a1f22", ls="--", label=f"tau_UTOM={t_ut:.4f}")
    axs[1].set_xlabel("p' = tanh(lambda p / 2(u+eps))"); axs[1].legend(); axs[1].set_title("UTOM-adjusted score")
    plt.tight_layout(); plt.show()
tick("หัวข้อ 6 เสร็จ")
''')

md("## 7. สรุป")
code(r'''
print(f"""
สิ่งที่ demo นี้แสดง (รันจบใน {time.time() - T_START:.0f} วินาที)
  1. โค้ดจริงของ repo: CFB forward, L_dec (Pearson), u = K/S, L_EDU
  2. วิดีโอ → RetinaFace → align/crop 224 → p(fake) ราย frame → เฉลี่ยเป็นของวิดีโอ
  3. ViT-B/16: 196 patch + CLS → 12 block → CLS 768 มิติ → logit → evidence → α → p, u
  4. CFB: Pearson ระหว่าง 2 branch, น้ำหนัก UFM, fused head, loss Eq. 9 และ stop-gradient ของ L_dec
  5. รันสดได้ score เท่ากับผลระดับ 3e_V2
  6. UTOM: ปรับ score ด้วย u แล้วหา threshold (Table IX)
""")
''')

nb = {"cells": cells, "metadata": base["metadata"], "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("FFD_level4_live_demo.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cells:", len(cells))
