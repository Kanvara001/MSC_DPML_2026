# บันทึกการทำงาน — ทำซ้ำ FFDBackbone (MSC DPML 2026)

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026
**Repo:** https://github.com/zhenglab/FFDBackbone (commit `0675f03`)
**ผู้นำเสนอ:** นางสาวกัญญ์วรา ถาวโรฤทธิ์ · 695020018-8 · **กำหนดส่ง/นำเสนอ:** พฤหัสบดี 22 ต.ค. 2026
**อัปเดตล่าสุด:** 16 ต.ค. 2026

---

## 1. สถานะรวม

| ขั้น | สถานะ | Notebook (โฟลเดอร์ `kaggle/`) | Kaggle notebook | ผลลัพธ์ในเครื่อง |
|---|---|---|---|---|
| 1–2 Inference ด้วย checkpoint ผู้เขียน (เบื้องต้น) | ✅ | `FFD_level1_DFF_inference.ipynb`, `FFD_level2_CelebDF_eval.ipynb` | — | — |
| 3a เตรียมข้อมูล FF++ c23 | ✅ | `FFD_level3a_FFpp_prepare.ipynb` | `kanvara/3a-ff-c23` | `FFD_level3a_FFpp_prepare/` |
| 3b Fine-tune CFB 10 epoch | ✅ | `FFD_level3b_CFB_finetune.ipynb` | `kanvara/3b-fine-tune-cfb` | `FFD_level3b_CFB_finetune/` |
| 3c Cross-dataset Celeb-DF | ✅ | `FFD_level3c_eval_crossdataset.ipynb` | `kanvara/3c-cross-dataset` | ตัวเลขอยู่ใน `results_summary/` |
| 3d Ablation (Table VI) 5 variant | ✅ | `FFD_level3d_ablation_train.ipynb` | `kanvara/3d-ablation-study-table-vi` | ตัวเลขอยู่ใน `results_summary/` |
| 3e Eval ทุกโมเดล + UTOM + WildDeepfake | ✅ | `FFD_level3e_eval_utom_wdf.ipynb` | `kanvara/3e-v2` (รอบสุดท้าย) | ตัวเลขอยู่ใน `results_summary/` |
| 4 Live demo (algorithm ภายใน) | ✅ สร้างแล้ว · ⏳ ยังไม่ได้รันบน Kaggle | `FFD_level4_live_demo.ipynb` | — | — |
| สไลด์ CRISP-DM (35 หน้า) | ✅ | — | — | `ffdbackbone-crispdm-deck.html` |
| ซ้อม / ตรวจรอบสุดท้าย | ⏳ | | | |

---

## 2. สิ่งที่ทำในแต่ละขั้น

### Pre-training (Stage 1)
- **ไม่ได้ pre-train เอง** ใช้ weight ของผู้เขียน `BEiT-1k-Face-55w.tar`
- เหตุผล: repo ไม่มีสคริปต์ / tokenizer / config การ pre-train · ใช้ 550k ภาพ บน A100 ×4 · ประมาณการบน T4 ×2 ≈ 450 GPU-hours (300 epoch)
- ผลของ Table I (หน้า 13 ในสไลด์) เป็น**ตัวเลขจากบทความ** ไม่ได้รันเอง

### 3a — Data preparation (FF++ c23)
- Official split 360/70/70 pairs · 5,000 วิดีโอ (real 1,000 + fake 4,000 จาก 4 วิธี)
- 8 frame/วิดีโอ → RetinaFace → align 5 landmark → crop 224 × 224 · IoU tracking แทน FTCN
- ได้ 39,999 / 40,000 frame (ขาด 1 frame: train/Deepfakes/569_921)

### 3b — Fine-tune CFB
- Kaggle T4 ×2 DDP · AMP fp16 · grad checkpointing · Lamb · lr 5e-5 cosine · layer decay 0.75 · batch 32×2 · 10 epoch (~98 นาที)
- Best val AUC 96.45 (fused) · test AUC: fused 96.72 / **main 96.57** (main ใช้ในทุกตารางสุดท้าย)
- Pearson(f_M, f_A) จาก +0.68 → ค่าลบ

### 3c — Cross-dataset Celeb-DF v2 (518 วิดีโอ)
- Main AUC 84.97 · TNR 87.64 · TPR 62.35

### 3d — Ablation (Table VI)
- ฝึก 5 variant เพิ่ม (fb_ce, fb_edu, cfb_base, cfb_dec, cfb_dec_ufm) + cfb_full จาก 3b
- **แนวโน้มกลับด้านจากบทความ** (fb_ce สูงสุด 87.92, cfb_full ต่ำสุด 84.97 บน CDF) แต่ต่างกัน ≤ 3 จุด อยู่ใน 95% CI (±3) · seed เดียว → ยังสรุปไม่ได้
- L_dec ทำงานจริง: Pearson +0.74 (ไม่มี L_dec) → −0.38 / −0.63 / −0.34

### 3e_V2 — ประเมินรวม + UTOM + WildDeepfake
- **checkpoint ผู้เขียนผ่าน pipeline เรา:** FF++ 99.24 · CDF 90.05 · WDF 89.78 → preprocessing ถูกต้อง (ต่างจากบทความ ≤ 0.41)
- **โมเดลเรา:** FF++ 96.57 · CDF 84.97 · WDF 84.44 → ผลต่างมาจากการฝึก กระจุกที่ NeuralTextures (89.37 vs 98.10)
- **Table IX ทำซ้ำได้:** CDF ACC τ_UTOM 85.33 (= บทความ) · λ_auto = 0.0236 (บทความ 0.02)
- **UTOM บน target ที่ไม่เคยเห็น:** ดีกว่า τ_ot (+8.3 ผู้เขียน, +5.3 เรา) แต่ไม่ดีกว่า τ = 0.5
- WildDeepfake = **ทดสอบเพิ่มเติม นอกบทความ** · 806 sequence · realign สำเร็จ 99.4%

### 4 — Live demo + สไลด์ Algorithm (ตามที่อาจารย์ขอให้เน้น algorithm และรันโค้ดแสดง)
- สไลด์ 17–24: pseudo-code + code map, MIM, ViT forward, CFB forward (`models/custom.py` 25–42), L_dec + u, L_EDU + Eq. 9, inference + UTOM, live demo
- จุดที่พบจากโค้ด: head ใช้ร่วมกันทั้ง 3 head · L_dec = ค่า Pearson ตรง ๆ (ดันไป −1) · UFM ใช้ softmax(−10u) แต่บทความเขียน softmax(−u) · L_EDU = log loss + AvU (annealing 0.01^(1−t/E)), ปิด KL
- Notebook demo เป็น **inference** (ไม่ train ใหม่) · แสดง RetinaFace → crop, shape ทุกขั้นของ ViT, attention map, z→e→α→u, Pearson, น้ำหนัก UFM, loss Eq. 9, stop-gradient, UTOM
- ทดสอบในเครื่องด้วย weight สุ่ม + ข้อมูลจำลอง: รันผ่านครบ ~1 นาที

---

## 3. สิ่งที่ทำไม่ได้ / ข้อจำกัด
- ไม่ได้ pre-train เอง (ไม่มีโค้ด + ทรัพยากร)
- ไม่ได้ประเมิน DFDC, FFIW, DFDCP, DFD (ไม่มีสิทธิ์/ข้อมูล) และไม่ได้ทำ few-shot adaptation
- ฝึก seed เดียว และใช้ AMP fp16
- Repo ไม่มีโค้ด UTOM, ablation, few-shot และสคริปต์ FTCN tracking สำหรับ FF++

## 4. สิ่งที่เหลือก่อน 22 ต.ค.
1. รัน `FFD_level4_live_demo.ipynb` บน Kaggle 1 รอบ (T4 ×1, Internet On, Input: 3b, 3c, 3e_V2, .mp4 ไม่บังคับ) แล้วเก็บ output ไว้สำรอง
2. ดาวน์โหลด output ของ 3c / 3d / 3e_V2 จาก Kaggle มาเก็บในเครื่อง (อย่างน้อยไฟล์ CSV)
3. (ไม่บังคับ) ฝึก seed ที่ 2 ของ cfb_full และ fb_ce เพื่อยืนยัน ablation
4. ซ้อมนำเสนอ

## 5. ไฟล์ในโฟลเดอร์นี้
| ไฟล์ / โฟลเดอร์ | เนื้อหา |
|---|---|
| `ffdbackbone-crispdm-deck.html` | สไลด์ (เปิดด้วย browser) |
| `PROGRESS_LOG.md` | ไฟล์นี้ |
| `results_summary/*.csv` | ตัวเลขผลทุกตาราง (3b–3e_V2) |
| `kaggle/*.ipynb` | notebook ทุกขั้น + `make_*.py` ที่ใช้สร้าง notebook |
| `deck_src/` | ซอร์สของสไลด์ (`build.py`, template, รูป) สำหรับแก้ไขภายหลัง |
| `FFD_level3a_FFpp_prepare/`, `FFD_level3b_CFB_finetune/` | output จาก Kaggle (ภาพ crop, checkpoint, history) |
| `Celeb-DF-v2/` | dataset ต้นฉบับ |

---

## 6. อัปเดต 17 ต.ค. 2026 — 3d2 + 3f
- **3d2:** ฝึก FB·CE และ CFB full ซ้ำ seed 2025, 3407 (ไฟล์ใน `results_summary/ระดับ 3d2 — Ablation/`)
- **3f:** paired DeLong + Holm (seed 1234) และ mean ± SD ข้าม 3 seed (ไฟล์ใน `results_summary/ระดับ 3f — ablation/`)
- AUC main branch ราย seed (1234 / 2025 / 3407)
  - CFB full: CDF 84.97 / 87.35 / 87.69 · WDF 84.44 / 85.73 / 85.77 · FF++ 96.57 / 97.42 / 97.70
  - FB·CE: CDF 87.92 / 86.23 / 86.84 · WDF 86.17 / 85.88 / 86.82 · FF++ 97.37 / 97.34 / 97.36
- mean ± SD: CDF FB·CE 87.00 ± 0.85 vs CFB 86.67 ± 1.48 · WDF 86.29 ± 0.48 vs 85.31 ± 0.76 · FF++ 97.36 ± 0.02 vs 97.23 ± 0.59
- Seed ensemble (paired DeLong, CFB − FB·CE): CDF −0.01 (p 0.97) · FF++ 0.00 (p 0.99) · WDF −0.90 (p < 0.001)
- สรุป: seed 1234 เป็นรอบที่ CFB ต่ำสุด · seed เปลี่ยน AUC ได้ 2.7 จุด > ผลต่างใน Table VI (0.16–1.28) · ยังไม่เห็นประโยชน์ของ CFB เหนือ FB·CE
- สไลด์: เพิ่มหน้า "ขอบเขตการทำซ้ำ" และ "Seed variance" (รวม 37 หน้า) · แก้ augmentation/epochs ให้ตรงโค้ด · ลิงก์ใหม่ claude.ai/artifact/6gwqwCAoQg1BYHYpFfGPZR
- เหลือ: รัน live demo บน Kaggle · ซ้อมนำเสนอ

---

## 7. อัปเดต 20 ต.ค. 2569 — จัดโครงสไลด์ใหม่ (Part I บทความ / Part II ของเรา)
**สำรองก่อนแก้ (ตรวจแล้วเปิดอ่านได้):** `deck_src/deck2.tpl.bak-2026-10-20.html` · `deck_src/build.bak-2026-10-20.py` · `ffdbackbone-crispdm-deck.bak-2026-10-20.html` · `speaker_script.bak-2026-10-20.md` · `PROGRESS_LOG.bak-2026-10-20.md`
- (ก่อนหน้านี้ในวันเดียวกันมีการลองจัดเป็น 30 หน้า แล้วคืนกลับตามที่ผู้ใช้ขอ · ไฟล์ทดลองเก็บไว้ที่ `deck_src/*.restructure-20261020*`)

**ไฟล์ที่แก้:** `deck_src/deck2.tpl.html` (สร้างจาก template เดิมด้วยสคริปต์ใหม่ ไม่ได้ใช้ `restructure.py`) · `deck_src/build.py` (แก้ข้อมูลกราฟ, ชื่อไฟล์ output เดิม) · `speaker_script.md` (เขียนใหม่ทั้งไฟล์) · `ffdbackbone-crispdm-deck.html` (build ใหม่)

**โครงสร้าง:** 42 หน้า → หน้าหลัก 40 + ภาคผนวก 12 (+ หน้าคั่น)
- Part I บทความ: A Business 4–5 · B Data Understanding 6–8 · C Data Preparation 9–12 · D Modeling 13–19 · E Evaluation 20–24
- Part II ของเรา: F ข้อมูล/การเตรียม 25–28 · G การฝึก 29–30 · H ผล (Q1–Q6) 31–36 · I Reproducibility/สรุป 37–39
- ย้ายโค้ดยาวและ live demo ไปภาคผนวก A1–A8 · หน้าเดิมที่ถูกแทนด้วยหน้าใหม่ (ไม่อยู่ใน deck แล้ว แต่ยังอยู่ใน backup): Research questions, Datasets, Data composition, Preprocessing, Pre-training (Table I), Analysis, Our results, Few-shot (ours, ผลรอบ AMP), Conclusion

**Build:** `python3 build.py` สำเร็จ · output 53 slides · ไฟล์ที่ build บนเครื่องตรงกับไฟล์ที่ตรวจภาพ (md5 เดียวกัน)
**ตรวจภาพ:** render ทุกหน้า (53) ที่ 1280×720 ด้วย Chromium + ฟอนต์ Sarabun/Fira Sans · ตรวจอัตโนมัติหาข้อความล้น/องค์ประกอบหลุดขอบ · พบ 2 หน้า (23, 31) ล้นเล็กน้อย → แก้แล้ว ตรวจซ้ำผ่าน · ป้าย บทความ/โค้ดผู้เขียน/ของเรา อยู่ทุกหน้าเนื้อหา

**ตัวเลขที่ตรวจกับ PDF และแก้ในสไลด์:**
- UTOM ของบทความ: เดิมเขียน "τ = 0.5 79.42 → 82.29" แต่ 79.42 คือ Avg ACC ของ variant +L_dec ใน Table VI · Table IX (CFB + L_EDU, Avg.) = τ0.5 80.13 · τ_ot 80.63 · τ_UTOM 82.29
- layer decay 0.75 / weight decay 0.05 มีเฉพาะใน DBBF.yaml บทความไม่ระบุ
- L_dec = Eq. 1 (เดิมเขียน Eq. 5) · CFB ไม่ได้อยู่ใน Fig. 4
- Table V มีแถว CDF เป็นเป้าหมาย (CFB 90.46 → 91.33 / 91.31; FF++ 99.25 / 99.10) → ใช้เทียบกับ 3g
- ยืนยันกับ PDF แล้ว: Table I (77.11, 81.92, 82.70, 83.77, 86.56; in-set 97.09–99.16), Table II CFB, Table III, Table VI, Table IX, Table V

**3g รอบ fp32 (ผู้ใช้รันเสร็จ 20 ต.ค.):** notebook ตั้ง `FT_AMP = False` + ตรวจว่า weight เปลี่ยน · 0/24 run ค้าง (รอบ AMP เดิม 6/24 ทั้งหมด k = 10)
- ΔAUC CDF (mean ± SD, 6 run): CFB k10 −0.65 ± 0.70 · k20 −0.08 ± 1.37 · FB·CE k10 −1.01 ± 0.29 · k20 −0.74 ± 2.25 · FF++ Δ −0.01 ถึง −0.13

**ยังยืนยันไม่ได้ / ข้อจำกัด:**
- จำนวนรอบที่รันในตารางของบทความ: ไม่ระบุ
- เหตุผลที่ฝึกเองต่ำกว่าบทความ (10 vs 60 epoch, fp16) ยังเป็นสมมติฐาน ไม่ได้ทดลองแยก
- ผล 3h (ตรวจภาพซ้ำ WildDeepfake) บันทึกไว้ใน `results_summary/ระดับ 3h — dupcheck/` แต่ไม่ได้ใส่ใน deck
- Live demo (level 4) ผู้ใช้รันบน Kaggle สำเร็จแล้ว (20 ต.ค.) · ยังไม่ได้เทียบตัวเลขที่ demo พิมพ์กับสไลด์
