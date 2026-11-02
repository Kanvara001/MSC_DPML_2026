DUP_MD = r"""
## 8. ตรวจภาพซ้ำ: WildDeepfake test vs ข้อมูลที่ใช้ฝึก/ปรับ (perceptual hash)

ตรวจว่ามี frame ของ WDF test ที่ **ซ้ำหรือเกือบซ้ำ** กับ frame ที่โมเดลเคยเห็นหรือไม่
- **FF++ ทั้งหมด** (train / val / test จาก tar ของระดับ 3a) = ข้อมูล fine-tune
- **Celeb-DF test ทั้งหมด** = ข้อมูล few-shot

วิธี: pHash 64 บิต (DCT 32×32 → 8×8) ของภาพหน้าที่ crop แล้ว · ถือว่า "เกือบซ้ำ" เมื่อ Hamming distance ≤ 6 (ตรวจเพิ่มที่ ≤ 10)
ข้อจำกัด: ตรวจได้เฉพาะภาพซ้ำ **ไม่ได้ตรวจ identity ซ้ำ** (คนเดียวกันต่างวิดีโอ) และตรวจกับชุด pre-train ไม่ได้ (ผู้เขียนไม่เปิดเผยรายชื่อภาพ)
"""
DUP_CODE = r'''
def phash(img_bgr):
    g = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = (d > np.median(d[1:])).astype(np.uint8)
    return np.packbits(bits).view(">u8")[0]

def hash_files(paths):
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None: out.append((str(p), phash(im)))
    return out

def hash_ffpp_all():
    """FF++ ทุก split: อ่านจาก tar ของระดับ 3a แบบ stream (ไม่แตกไฟล์) · ถ้าไม่มี tar ใช้ไฟล์ที่แตกแล้ว"""
    tars = [t for t in (INPUT_DIR.rglob("*.tar") if INPUT_DIR.exists() else [])
            if t.stem in ["original"] + FF_METHODS]
    out = []
    if tars:
        for t in tars:
            with tarfile.open(t, "r|*") as tf:
                for m_ in tf:
                    if m_.isfile() and m_.name.endswith(".png") and "/frame/" in m_.name:
                        im = cv2.imdecode(np.frombuffer(tf.extractfile(m_).read(), np.uint8), cv2.IMREAD_COLOR)
                        if im is not None: out.append((m_.name, phash(im)))
    else:
        roots = sorted(INPUT_DIR.rglob("face_v1")) if INPUT_DIR.exists() else []
        if roots: out = hash_files(sorted(roots[0].rglob("frame/*.png")))
    return out

def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int32)

def min_hamming(q, ref, chunk=512):
    """ระยะ Hamming ต่ำสุดของแต่ละ hash ใน q เทียบกับทุก hash ใน ref"""
    q = np.asarray(q, np.uint64); ref = np.asarray(ref, np.uint64)
    best = np.full(len(q), 64, np.int32); arg = np.zeros(len(q), np.int64)
    for i in range(0, len(q), chunk):
        d = popcount64(q[i:i + chunk, None] ^ ref[None, :])
        j = d.argmin(1); best[i:i + chunk] = d[np.arange(len(j)), j]; arg[i:i + chunk] = j
    return best, arg

try:
    t0 = time.time()
    wdf_items = [(u, p) for u, fs in zip(DATA["WDF"].uid, DATA["WDF"].frames) for p in dict.fromkeys(fs)]
    wdf_h = hash_files([p for _, p in wdf_items])
    wdf_uid = {p: u for u, p in wdf_items}
    refs = {"FF++ (ทุก split)": hash_ffpp_all(),
            "Celeb-DF": hash_files(sorted({p for fs in DATA["CDF"].frames for p in fs}))}
    print(f"WDF test: {len(wdf_h):,} frame จาก {DATA['WDF'].uid.nunique()} sequence")
    DUP_ROWS = []
    for name, ref in refs.items():
        if not ref:
            print(f"  {name}: ไม่พบข้อมูลอ้างอิง — ข้าม"); continue
        best, arg = min_hamming([h for _, h in wdf_h], [h for _, h in ref])
        for thr in (6, 10):
            hit = best <= thr
            seqs = {wdf_uid[wdf_h[i][0]] for i in np.where(hit)[0]}
            DUP_ROWS.append({"reference": name, "n_ref_frames": len(ref), "threshold": thr,
                             "wdf_frames_hit": int(hit.sum()), "wdf_sequences_hit": len(seqs),
                             "pct_sequences": round(100 * len(seqs) / DATA["WDF"].uid.nunique(), 2)})
        for i in np.argsort(best)[:5]:
            print(f"  {name}: ใกล้สุด d={best[i]:2d} · {Path(wdf_h[i][0]).parent.name}/{Path(wdf_h[i][0]).name} ↔ {ref[arg[i]][0][-70:]}")
    DUP = pd.DataFrame(DUP_ROWS); DUP.to_csv(OUT_DIR / "wdf_duplicate_check.csv", index=False)
    print(); print(DUP.to_string(index=False))
    # ค่าอ้างอิง: ระยะ Hamming ระหว่าง frame ต่างวิดีโอภายใน WDF เอง (ควรสูงกว่า threshold มาก)
    hs = np.array([h for _, h in wdf_h], np.uint64)
    rng_ = np.random.default_rng(0); a_, b_ = rng_.integers(len(hs), size=(2, 5000))
    base_d = popcount64(hs[a_] ^ hs[b_])
    print(f"\nอ้างอิง: ระยะ Hamming ระหว่าง frame สุ่ม 2 ภาพใน WDF · median {np.median(base_d):.0f}, 1st percentile {np.percentile(base_d, 1):.0f}")
    print(f"ตรวจภาพซ้ำเสร็จใน {(time.time() - t0) / 60:.1f} นาที")
    if DUP.empty or DUP[DUP.threshold == 6].wdf_sequences_hit.sum() == 0:
        print("→ ไม่พบ frame ของ WDF test ที่ซ้ำ/เกือบซ้ำกับข้อมูลที่ใช้ฝึกหรือปรับ (ที่ d ≤ 6)")
    else:
        print("→ พบภาพเกือบซ้ำ — ดูคู่ภาพด้านบน และพิจารณาตัด sequence เหล่านั้นออกแล้วคำนวณ AUC ใหม่")
except Exception as e:
    import traceback; traceback.print_exc()
    print("ตรวจภาพซ้ำไม่สำเร็จ (ไม่กระทบผล few-shot ด้านบน):", e)
'''
