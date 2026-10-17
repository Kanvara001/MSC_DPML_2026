"""สร้าง FFD_level3f_significance.ipynb — ทดสอบทางสถิติของ ablation (paired test) + วิเคราะห์ seed"""
import json

e = json.load(open("FFD_level3e_eval_utom_wdf.ipynb", encoding="utf-8"))
ec = lambda i: "".join(e["cells"][i]["source"])
def rep(s, a, b):
    assert a in s, a[:70]; return s.replace(a, b)

cells = []
def md(t): cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(keepends=True)})
def code(t): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                            "source": t.strip("\n").splitlines(keepends=True)})

md(r"""
# ระดับ 3f — ablation ต่างกันจริงหรือไม่: paired test + ความแปรปรวนจาก seed

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026

**คำถาม:** องค์ประกอบของ CFB (2 branch, `L_dec`, UFM, `L_EDU`) ช่วยเพิ่ม AUC บน dataset ที่ไม่เคยเห็นจริงหรือไม่
ใน 3e ผลของเรากลับด้านจาก Table VI แต่ต่างกันไม่เกิน 3 จุด และช่วงความเชื่อมั่นของแต่ละโมเดลซ้อนกัน

ความบังเอิญมีที่มา 2 แบบ notebook นี้ตรวจทั้งคู่:

| ส่วน | ตรวจความบังเอิญจาก | วิธี | ต้องใช้ |
|---|---|---|---|
| **1** | ชุดทดสอบมีจำกัด (Celeb-DF 518 วิดีโอ) | **Paired test** บนวิดีโอชุดเดียวกัน: DeLong test + paired bootstrap ของ ΔAUC · แก้ multiple comparison ด้วย Holm | score จาก 3e (CPU, < 2 นาที) |
| **2** | การฝึก (seed) | ประเมิน checkpoint seed ใหม่จาก 3d2 แล้วดู mean ± SD ข้าม seed | GPU T4 ×1 (ข้ามอัตโนมัติถ้าไม่มี checkpoint จาก 3d2) |

paired test มีความไวกว่าการดูว่าช่วงความเชื่อมั่นซ้อนกันหรือไม่ เพราะทุกโมเดลถูกทดสอบบนวิดีโอเดียวกัน ความยากง่ายของวิดีโอจึงหักล้างกัน

**ก่อนรัน:** Add Input:
- output ของ **3e_V2** (`results_level3e/scores_*.csv`) — จำเป็น
- (ส่วน 2) output ของ **3d2** (`ablation_seeds/`) ทุก account · **3a** (FF++ test) · **3c** (`cdf_crops`) · 3e รุ่นที่มี `wdf_crops/` · GPU T4 ×1 · Internet On
""")

md("## 0. ตั้งค่า")
s = ec(2)
s = rep(s, 'OUT_DIR   = WORK_DIR / "results_level3e"', 'OUT_DIR   = WORK_DIR / "results_level3f"')
s = rep(s, 'USE_AUTHOR_CKPT   = True', 'USE_AUTHOR_CKPT   = False')
s = rep(s, 'DATASETS   = ["FF++", "CDF", "WDF"]              # ลบ "WDF" ออกได้ถ้าไม่ต้องการ',
           'DATASETS_P2 = [d for d in os.environ.get("FFD_DATASETS", "FF++,CDF,WDF").split(",") if d]  # ชุดที่ใช้ในส่วน 2')
s = rep(s, 'print("ON_KAGGLE:", ON_KAGGLE, "| DATASETS:", DATASETS)', 'print("ON_KAGGLE:", ON_KAGGLE)')
s += r'''

# --- ส่วน 1: paired test ---
N_BOOT_PAIR = 2000
ALPHA = 0.05
# ลำดับเดียวกับ Table VI · Δ = (เพิ่มองค์ประกอบ) − (ก่อนเพิ่ม)
COMPS = [("fb_ce", "fb_edu", "+ L_EDU (1 branch)"),
         ("fb_ce", "cfb_base", "+ branch ที่ 2"),
         ("cfb_base", "cfb_dec", "+ L_dec"),
         ("cfb_dec", "cfb_dec_ufm", "+ UFM"),
         ("cfb_dec_ufm", "cfb_full", "+ L_EDU (2 branch)"),
         ("fb_ce", "cfb_full", "FB+CE → CFB ครบ")]
PAPER_CDF = {k: v[0] for k, v in PAPER_TABLE6.items()}

# --- ส่วน 2: checkpoint จาก 3d2 ---
SEED_CKPTS = sorted(INPUT_DIR.rglob("ablation_seeds/*_s*/best.pth")) if INPUT_DIR.exists() else []
DATASETS = DATASETS_P2 if SEED_CKPTS else []      # ไม่มี checkpoint ใหม่ → ส่วน 2 ไม่โหลดข้อมูลและไม่รันโมเดล
print("checkpoint seed ใหม่:", [p.parent.name for p in SEED_CKPTS] or "ไม่มี → รันเฉพาะส่วน 1")
'''
code(s)

md("## 0.1 ติดตั้ง + clone repo (ใช้ในส่วน 2)")
code(ec(4) + "\nfrom scipy.stats import rankdata, norm\n")

# ------------------------------------------------------------------ part 1
md(r"""
## 1. Paired test บน score จาก 3e_V2

- **DeLong test** (DeLong et al., 1988; fast version ของ Sun & Xu, 2014): ทดสอบว่า AUC ของ 2 โมเดลที่ใช้ชุดทดสอบเดียวกันต่างกันหรือไม่
- **Paired bootstrap:** สุ่มวิดีโอใหม่ (แยก real / fake) 2,000 รอบ ใช้ index เดียวกันกับทุกโมเดล → ช่วงความเชื่อมั่น 95% ของ ΔAUC
- **Holm correction** ต่อ dataset (6 การเปรียบเทียบ) เพื่อคุม family-wise error ที่ α = 0.05
- **MDE<sub>80</sub>** = 2.8 × SD(ΔAUC) ≈ ความต่างที่เล็กที่สุดที่ชุดทดสอบนี้ตรวจพบได้ด้วย power 80%
- เทียบ checkpoint ผู้เขียนกับโมเดลของเราด้วย เพื่อยืนยันว่า test นี้ตรวจความต่างที่มีจริงได้
""")
code(r'''
def load_scores():
    """เลือกโฟลเดอร์ results_level3e ที่มีไฟล์ score มากที่สุด (กรณี Add Input หลายรุ่น)"""
    dirs = {}
    for f in (INPUT_DIR.rglob("scores_*_*.csv") if INPUT_DIR.exists() else []):
        if "results_level3f" in str(f): continue
        dirs.setdefault(f.parent, []).append(f)
    assert dirs, "ไม่พบ scores_*.csv — Add Input เป็น output ของระดับ 3e_V2"
    best = sorted(dirs, key=lambda d: (len(dirs[d]), str(d)))[-1]
    out = {}
    for f in dirs[best]:
        m = re.match(r"scores_(.+)_(FF\+\+|CDF|WDF)\.csv$", f.name)
        if m: out[(m.group(1), m.group(2))] = pd.read_csv(f)
    print("ใช้ score จาก:", best, "·", len(out), "ไฟล์")
    return out

S = load_scores()
MODELS_P1 = sorted({m for m, _ in S})
DS_P1 = [d for d in ["CDF", "WDF", "FF++"] if any(k[1] == d for k in S)]
print("โมเดล:", MODELS_P1); print("dataset:", DS_P1)
vname = lambda v: "ours_cfb_full" if v == "cfb_full" else f"ours_{v}"
''')
code(r'''
def auc_rank(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def delong(y, s1, s2):
    """fast DeLong (Sun & Xu 2014) · คืน (AUC1, AUC2, z, p) สำหรับ 2 score บนตัวอย่างชุดเดียวกัน"""
    y = np.asarray(y).astype(int); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        X, Y = s[pos], s[neg]
        tz = rankdata(np.r_[X, Y]); tx = rankdata(X); ty = rankdata(Y)
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n); v01.append(1 - (tz[m:] - ty) / m)
    S10, S01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    var = (S10[0, 0] + S10[1, 1] - 2 * S10[0, 1]) / m + (S01[0, 0] + S01[1, 1] - 2 * S01[0, 1]) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return aucs[0], aucs[1], z, 2 * norm.sf(abs(z))

def aligned(ds, names):
    """รวม score ของหลายโมเดลบน uid เดียวกัน · ตรวจว่า label ตรงกัน"""
    base = None
    for nm in names:
        d = S[(nm, ds)][["uid", "label", "p_main"]].rename(columns={"p_main": nm, "label": f"y_{nm}"})
        base = d if base is None else base.merge(d, on="uid", how="inner")
    ys = base[[f"y_{nm}" for nm in names]].values
    assert (ys == ys[:, :1]).all(), "label ไม่ตรงกันระหว่างไฟล์ score"
    return base["label" if "label" in base else f"y_{names[0]}"].values.astype(int), base

def boot_idx(y, B, seed):
    rng = np.random.default_rng(seed); i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
    return [np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))] for _ in range(B)]

def holm(p):
    p = np.asarray(p); o = np.argsort(p); k = len(p); adj = np.empty(k); run = 0
    for r, i in enumerate(o):
        run = max(run, (k - r) * p[i]); adj[i] = min(1.0, run)
    return adj

# ตรวจ DeLong กับ sklearn
_y = np.r_[np.zeros(50), np.ones(70)].astype(int); _r = np.random.default_rng(0)
_a, _b = _r.random(120) + _y * .3, _r.random(120) + _y * .2
assert abs(delong(_y, _a, _b)[0] - roc_auc_score(_y, _a)) < 1e-12 and abs(auc_rank(_y, _a) - roc_auc_score(_y, _a)) < 1e-12
print("ตรวจสอบ: AUC จาก DeLong / rank = sklearn ✓")
''')
code(r'''
rows, BOOT = [], {}
for ds in DS_P1:
    need = sorted({vname(a) for a, b, _ in COMPS} | {vname(b) for a, b, _ in COMPS} | {"author"})
    have = [nm for nm in need if (nm, ds) in S]
    y, T = aligned(ds, have)
    idx = boot_idx(y, N_BOOT_PAIR, SEED)
    BA = {nm: np.array([auc_rank(y[i], T[nm].values[i]) for i in idx]) for nm in have}
    BOOT[ds] = BA
    pairs = [(vname(a), vname(b), lab, a, b) for a, b, lab in COMPS] + [("ours_cfb_full", "author", "ours → checkpoint ผู้เขียน (ตรวจ power)", None, None)]
    for A_, B_, lab, a, b in pairs:
        if A_ not in have or B_ not in have: continue
        aA, aB, z, p = delong(y, T[A_].values, T[B_].values)
        d = BA[B_] - BA[A_]
        rows.append({"dataset": ds, "comparison": lab, "A": A_, "B": B_, "n": len(y),
                     "AUC_A": 100 * aA, "AUC_B": 100 * aB, "ΔAUC": 100 * (aB - aA),
                     "CI_lo": 100 * np.percentile(d, 2.5), "CI_hi": 100 * np.percentile(d, 97.5),
                     "p_boot": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), "p_DeLong": p,
                     "MDE80": 100 * 2.8 * d.std(ddof=1),
                     "paper_ΔCDF": (PAPER_CDF[b] - PAPER_CDF[a]) if a else np.nan, "ablation": a is not None})
SIG = pd.DataFrame(rows)
SIG["p_Holm"] = np.nan
for ds in DS_P1:
    k = (SIG.dataset == ds) & SIG.ablation
    SIG.loc[k, "p_Holm"] = holm(SIG.loc[k, "p_DeLong"].values)
SIG["สรุป"] = np.where(SIG.ablation, np.where(SIG.p_Holm < ALPHA, "ต่างจริง", "ไม่ต่างอย่างมีนัยสำคัญ"),
                       np.where(SIG.p_DeLong < ALPHA, "ต่างจริง", "ไม่ต่าง"))
SIG.to_csv(OUT_DIR / "paired_tests.csv", index=False)
show = ["comparison", "AUC_A", "AUC_B", "ΔAUC", "CI_lo", "CI_hi", "p_DeLong", "p_Holm", "MDE80", "paper_ΔCDF", "สรุป"]
for ds in DS_P1:
    print(f"\n=== {ds} (n = {int(SIG[SIG.dataset == ds].n.iloc[0])}) · main branch · Δ = B − A ===")
    print(SIG[SIG.dataset == ds][show].round(4).to_string(index=False))
''')
code(r'''
INK2, GRID = "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
fig, axes = plt.subplots(1, len(DS_P1), figsize=(5.2 * len(DS_P1), 3.8), sharey=True)
LAB_EN = {"+ L_EDU (1 branch)": "+ L_EDU (1 branch)", "+ branch ที่ 2": "+ 2nd branch", "+ L_dec": "+ L_dec",
          "+ UFM": "+ UFM", "+ L_EDU (2 branch)": "+ L_EDU (2 branch)", "FB+CE → CFB ครบ": "FB+CE -> CFB full",
          "ours → checkpoint ผู้เขียน (ตรวจ power)": "ours -> author (power check)"}
for ax, ds in zip(np.atleast_1d(axes), DS_P1):
    d = SIG[SIG.dataset == ds].reset_index(drop=True)
    yy = np.arange(len(d))[::-1]
    for i, r in d.iterrows():
        c = "#7a1f22" if r["สรุป"] == "ต่างจริง" else "#8a8278"
        ax.plot([r.CI_lo, r.CI_hi], [yy[i]] * 2, color=c, lw=2); ax.plot(r["ΔAUC"], yy[i], "o", color=c)
        if not np.isnan(r["paper_ΔCDF"]) and ds == "CDF": ax.plot(r["paper_ΔCDF"], yy[i], "x", color="#2a78d6", ms=8)
    ax.axvline(0, color=INK2, lw=1); ax.set_yticks(yy); ax.set_yticklabels([LAB_EN.get(c, c) for c in d.comparison])
    ax.set_title(f"{ds} (n={int(d.n.iloc[0])})"); ax.set_xlabel("ΔAUC (points) · 95% paired-bootstrap CI"); ax.grid(axis="x", color=GRID)
np.atleast_1d(axes)[0].plot([], [], "x", color="#2a78d6", label="paper Table VI (CDF)"); np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(OUT_DIR / "paired_tests_forest.png", dpi=150); plt.show()

a = SIG[SIG.ablation]
print(f"\nablation ที่ต่างจริงหลัง Holm: {int((a['สรุป'] == 'ต่างจริง').sum())} / {len(a)} การเปรียบเทียบ")
cdf = SIG[(SIG.dataset == "CDF") & SIG.ablation]
if len(cdf):
    print(f"MDE80 บน Celeb-DF เฉลี่ย {cdf.MDE80.mean():.2f} จุด · ผลต่างใน Table VI ของบทความ: "
          f"{cdf['paper_ΔCDF'].abs().min():.2f}–{cdf['paper_ΔCDF'].abs().max():.2f} จุด")
''')

# ------------------------------------------------------------------ part 2
md(r"""
## 2. ความแปรปรวนจาก seed (checkpoint จาก 3d2)

ถ้าไม่มี `ablation_seeds/` ใน Add Input ทุก cell ในส่วนนี้จะไม่โหลดข้อมูลและไม่รันโมเดล

- ประเมิน checkpoint ใหม่ด้วยโค้ดเดียวกับ 3e (main branch, 8 frame, AMP) บนชุดเดียวกัน
- seed 1234 ใช้ score เดิมจาก 3e_V2
""")
code(ec(6) + r'''

MODELS = [(f"ours_{p.parent.name}", p) for p in SEED_CKPTS]
for n, p in MODELS:
    print(f"  {n:<24} {p}  ({p.stat().st_size/1e6:.0f} MB)")
''')
code(ec(9))
code(ec(11))
code(ec(13) + r'''
for k, v in SCORES.items():
    S[k] = v
''')
code(r'''
if SEED_CKPTS:
    rx = re.compile(r"ours_(.+)_s(\d+)$")
    runs = []
    for (nm, ds) in S:
        m = rx.match(nm)
        if m: runs.append((m.group(1), int(m.group(2)), nm, ds))
    for v in sorted({r[0] for r in runs}):
        for ds in DS_P1:
            if (vname(v), ds) in S: runs.append((v, 1234, vname(v), ds))
    R = pd.DataFrame([{"variant": v, "seed": sd, "dataset": ds,
                       "AUC": 100 * roc_auc_score(S[(nm, ds)].label, S[(nm, ds)].p_main)} for v, sd, nm, ds in runs])
    R = R.drop_duplicates(["variant", "seed", "dataset"])
    W = R.pivot_table(index=["variant", "seed"], columns="dataset", values="AUC").round(2)
    print("AUC (%) ต่อ seed · main branch"); print(W.to_string())
    agg = R.groupby(["variant", "dataset"]).AUC.agg(["mean", "std", "min", "max", "count"]).round(2)
    print("\nmean ± SD ข้าม seed"); print(agg.to_string())
    R.to_csv(OUT_DIR / "seed_auc_long.csv", index=False); agg.to_csv(OUT_DIR / "seed_auc_summary.csv")

    vs = sorted(R.variant.unique())
    if {"fb_ce", "cfb_full"} <= set(vs):
        print("\nΔAUC (cfb_full − fb_ce) ต่อ seed  · บทความ (CDF): +1.28")
        D = R.pivot_table(index=["seed", "dataset"], columns="variant", values="AUC")
        D["Δ (full − ce)"] = D["cfb_full"] - D["fb_ce"]
        print(D.round(2).to_string())
        # paired DeLong บน score เฉลี่ยข้าม seed (seed ensemble) — ลดความแปรปรวนจากการฝึก
        ens_rows = []
        for ds in DS_P1:
            seeds = sorted(set(R[(R.dataset == ds) & (R.variant == "fb_ce")].seed) & set(R[(R.dataset == ds) & (R.variant == "cfb_full")].seed))
            if len(seeds) < 2: continue
            nm = lambda v, sd: vname(v) if sd == 1234 else f"ours_{v}_s{sd}"
            names = [nm(v, sd) for v in ["fb_ce", "cfb_full"] for sd in seeds]
            y, T = aligned(ds, names)
            pa = T[[nm("fb_ce", sd) for sd in seeds]].mean(1).values; pb = T[[nm("cfb_full", sd) for sd in seeds]].mean(1).values
            aA, aB, z, p = delong(y, pa, pb)
            ens_rows.append({"dataset": ds, "n_seeds": len(seeds), "AUC fb_ce (ens)": 100 * aA,
                             "AUC cfb_full (ens)": 100 * aB, "ΔAUC": 100 * (aB - aA), "p_DeLong": p})
        if ens_rows:
            ENS = pd.DataFrame(ens_rows); ENS.to_csv(OUT_DIR / "seed_ensemble_test.csv", index=False)
            print("\nseed ensemble (เฉลี่ย p ทุก seed) · paired DeLong"); print(ENS.round(4).to_string(index=False))
else:
    print("ไม่มี checkpoint จาก 3d2 — ข้ามส่วน 2")
''')

md("## 3. สรุป")
code(r'''
print("=" * 72); print("สรุประดับ 3f"); print("=" * 72)
for ds in DS_P1:
    d = SIG[(SIG.dataset == ds)]
    print(f"\n{ds}:")
    for _, r in d.iterrows():
        ph = f"p_Holm={r.p_Holm:.3f}" if r.ablation else f"p={r.p_DeLong:.2g}"
        print(f"  {r.comparison:<38} Δ = {r['ΔAUC']:+6.2f}  [{r.CI_lo:+.2f}, {r.CI_hi:+.2f}]  {ph:<14} → {r['สรุป']}")
print("\nไฟล์ผลลัพธ์:", sorted(p.name for p in OUT_DIR.iterdir()))
print(f"เวลารวม: {(time.time() - T_START) / 60:.1f} นาที")
''')

nb = {"cells": cells, "metadata": e["metadata"], "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("FFD_level3f_significance.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("cells:", len(cells))
