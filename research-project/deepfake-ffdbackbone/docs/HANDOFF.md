# Handoff — ทำซ้ำ FFDBackbone (MSC DPML 2026) · 16 ต.ค. 2026

## บริบท
- ผู้ใช้: toon · ผู้นำเสนอ **นางสาวกัญญ์วรา ถาวโรฤทธิ์ 695020018-8** (ห้ามใส่ VI Lab ในสไลด์) · นำเสนอ **พฤหัส 22 ต.ค. 2026**
- Paper: Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026 · repo `zhenglab/FFDBackbone` @ `0675f03`
- งาน: ทำซ้ำ paper ตาม CRISP-DM + สไลด์ HTML เชิงวิชาการ · อาจารย์ขอให้ **เน้น algorithm ภายใน (ระดับโค้ด) และรันโค้ดแสดง**
- โฟลเดอร์ในเครื่อง: `D:\Master\Master term 1\MSC DPML 2026\paper\` (อ่าน `PROGRESS_LOG.md` ก่อน — สรุปทุกขั้นและตัวเลขทั้งหมด)
- Kaggle user `kanvara` (มีหลาย account ไม่ต้องห่วงโควตา) · notebook: `3a-ff-c23`, `3b-fine-tune-cfb`, `3c-cross-dataset`, `3d-ablation-study-table-vi`, `3e-v2`

## สถานะ
| งาน | สถานะ |
|---|---|
| 3a–3e_V2 (prep, fine-tune, cross-dataset, ablation, UTOM, WildDeepfake) | ✅ เสร็จ · ตัวเลขใน `results_summary/*.csv` |
| สไลด์ 35 หน้า `ffdbackbone-crispdm-deck.html` (artifact `claude.ai/artifact/17mVaC1tuycEVkqj7sBoKr` v10) · ซอร์สใน `deck_src/` (`python3 build.py` → HTML) | ✅ |
| สไลด์ Algorithm หน้า 17–24 (pseudo-code, MIM, ViT, CFB forward, L_dec/u, L_EDU+Eq.9, inference+UTOM, live demo) | ✅ |
| `kaggle/FFD_level4_live_demo.ipynb` (inference demo: RetinaFace→crop, ViT ทีละขั้น, attention, Pearson/UFM, loss Eq.9, UTOM) | ✅ สร้าง + ทดสอบ mock · ⏳ **ยังไม่ได้รันบน Kaggle จริง** (Input: 3b, 3c, 3e_V2, .mp4 ไม่บังคับ · T4×1 · Internet On) |
| `kaggle/FFD_level3d2_ablation_seeds.ipynb` (ฝึก fb_ce, cfb_full ซ้ำ seed 2025, 3407 · ~5 ชม. T4×2 · แบ่ง 2 account ตาม seed ได้) | ✅ สร้างแล้ว · ⏳ ยังไม่รัน |
| `kaggle/FFD_level3f_significance.ipynb` ส่วน 1: paired DeLong + paired bootstrap ΔAUC + Holm + MDE80 บน score 3e_V2 (CPU ~1–2 นาที) | ✅ ทดสอบ mock ผ่าน · ⏳ ยังไม่รันจริง |
| 3f ส่วน 2: ประเมิน checkpoint จาก 3d2 → mean±SD ข้าม seed + seed-ensemble DeLong | ⚠️ **ยังไม่ได้ทดสอบ** (ผู้ใช้หยุดกลางคัน) · จะข้ามอัตโนมัติถ้าไม่มี `ablation_seeds/` |

## ทำต่อ (ตามลำดับ)
1. ให้ผู้ใช้รัน **3f ส่วน 1** บน Kaggle (Add Input = output 3e_V2) → อ่านผล `paired_tests.csv` / forest plot → ใส่สไลด์ Ablation (หน้า 31 ราว ๆ, data-t="Ablation (ours)") ว่า ablation ต่างจริงหรือไม่ + MDE เทียบกับผลต่าง 0.16–1.28 ใน Table VI
2. (ถ้ามีเวลา) รัน 3d2 → 3f ส่วน 2 (Input: 3a, 3c, 3e ที่มี `wdf_crops/`, 3d2 ทุก account, 3e_V2) — ควรทดสอบส่วน 2 ด้วย mock ก่อน (ใช้ `FFD_DATASETS=CDF,WDF`)
3. รัน live demo บน Kaggle 1 รอบ เก็บ output สำรอง (ถ้า gdown ดึง `Final-mainbranch.tar` ไม่ได้ ตั้ง `AUTHOR_CKPT_PATH`)
4. อัปเดต `PROGRESS_LOG.md`, สไลด์ (republish artifact + บันทึกลงโฟลเดอร์), ซ้อมนำเสนอ

## ข้อเท็จจริงสำคัญ (อย่าเข้าใจผิด)
- ผลหลักใช้ **main branch**: เรา FF++ 96.57 / CDF 84.97 / WDF 84.44 · checkpoint ผู้เขียนผ่าน pipeline เรา 99.24 / 90.05 / 89.78 → preprocessing ถูก ผลต่างมาจากการฝึก (กระจุกที่ NeuralTextures 89.37 vs 98.10)
- 96.72 = fused head ของ 3b (ไม่ใช้ในตารางหลักแล้ว)
- Table IX ทำซ้ำได้ (τ_UTOM 85.33 = paper) · λ_auto 0.0236 · UTOM ดีกว่า τ_ot บน target ที่ไม่เคยเห็น แต่ไม่ดีกว่า τ=0.5
- Ablation กลับด้านจาก paper (fb_ce 87.92 สูงสุด, cfb_full 84.97 ต่ำสุดบน CDF) แต่ต่าง ≤ 3 จุด, seed เดียว
- **Paper ไม่ระบุจำนวน seed** และไม่มี ± ในตาราง · config ผู้เขียน `seed: 1234` ค่าเดียว
- Table I (หน้า 13) เป็นตัวเลขจาก paper ไม่ได้รันเอง · ไม่ได้ pre-train เอง (ใช้ `BEiT-1k-Face-55w.tar`)
- WildDeepfake = ทดสอบเพิ่มนอก paper · ทำไม่ได้: DFDC/FFIW/DFDCP/DFD, few-shot
- จากโค้ด: head ใช้ร่วมกัน 3 head · L_dec = Pearson ตรง ๆ (ดันไป −1) · UFM softmax(−10u) แต่ paper เขียน softmax(−u) · L_EDU = log loss + AvU annealing 0.01^(1−t/E), ปิด KL
- Output ต้นฉบับของ 3c/3d/3e_V2 (score CSV, checkpoint ablation) ยังอยู่บน Kaggle เท่านั้น

## ไฟล์ generator
`kaggle/make_3d.py`, `make_3e.py`, `make_4.py` (+ `make_3d2.py`, `make_3f.py` แนบมากับ handoff นี้) — สร้าง notebook จาก notebook ก่อนหน้า (ต้องมี `FFD_level3c/3d/3e*.ipynb` อยู่ข้างกัน)
