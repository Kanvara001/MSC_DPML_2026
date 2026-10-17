# แผนทำซ้ำ FFDBackbone (TPAMI 2026) บน Kaggle — ระดับ 3

> Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026
> repo: https://github.com/zhenglab/FFDBackbone

ระดับ 1–2 (ที่ทำไปแล้ว) คือ **inference** ด้วย checkpoint ของผู้เขียน
ระดับ 3 คือ **เทรนเอง** ตามกระบวนการใน paper: เตรียมข้อมูล → แบ่ง split → ใช้ backbone ที่ pre-train → fine-tune → ประเมิน

---

## สิ่งที่ paper บอกไว้ (ข้อเท็จจริงที่ดึงมาจาก Sec. IV-A2 + `checkpoints/Ours/DBBF.yaml`)

| ขั้น | รายละเอียดใน paper |
|---|---|
| **Pre-training** | ViT-B/16 ด้วย BEiT v2 MIM บน real-face dataset 550k ภาพ (CelebA + CelebV-Text + FFHQ), 4×A100 จน converge |
| **Data prep** | RetinaFace crop → resize 224×224 → sample 8 frames ระยะเท่ากันต่อวิดีโอ |
| **Split** | official FF++ split `splits/{train,val,test}.json` → 360/70/70 identity pairs = 720/140/140 วิดีโอจริง |
| **Fine-tuning** | FF++ c23, 2×GTX3090, batch 64, Lamb, lr 5e-5 cosine, wd 0.05, layer decay 0.75, **10 epochs** (backbone ที่ pre-train บนหน้าจริง converge ใน 10 epoch) |
| **Loss (Eq. 9)** | `L = L^M_EDU + L^A_EDU + L^F_ce + L_dec` |
| **Inference** | main branch เท่านั้น · เฉลี่ย softmax 8 frame = score วิดีโอ · video-level AUC |
| **เป้าหมาย** | FF++ c23 **99.36** · Celeb-DF **90.46** · DFDC 84.90 · FFIW 90.97 |

### ข้อจำกัดที่เจอ (ต้องระบุในรายงาน)

1. **repo ไม่ได้แนบสคริปต์ SSL pre-training มา** — มีแต่ weight สำเร็จรูป `BEiT-1k-Face-55w.tar`
   → ขั้น pre-train จึงใช้ weight ของผู้เขียน (ถ้าจะ pre-train เองต้องใช้โค้ด BEiT v2 ต้นฉบับ + 4×A100 หลายวัน ซึ่งเกินกำลัง Kaggle)
2. **repo ไม่ได้แนบสคริปต์ face tracking แบบ FTCN** (ตรวจหน้าทุก frame แล้ว track หน้าที่อยู่นานสุด)
   → ใช้ IoU tracking อย่างง่ายบน 8 frame ที่เลือกแทน (ผลเท่ากันสำหรับวิดีโอหน้าเดียวอย่าง FF++)
3. **T4 มี VRAM 16 GB** ส่วน dual ViT-B แบบ fp32 ต้องการ ~22 GB
   → เปิด AMP fp16 + gradient checkpointing (effective batch ยังเป็น 64 เท่า paper)
4. `lr_scheduler.step_update()` ของ repo ไม่มีผลจริง เพราะ timm ตั้ง `t_in_epochs=True`
   → notebook ใช้ cosine รายสเต็ป + warmup 1 epoch ตามค่าใน `DBBF.yaml` (ตรงตามเจตนาของ config)

---

## ไฟล์ที่ได้

| notebook | ทำอะไร | เวลา | GPU |
|---|---|---|---|
| `FFD_level3a_FFpp_prepare.ipynb` | เตรียมข้อมูล FF++ c23: 8 frame/วิดีโอ → RetinaFace → align → crop 224×224 → โครงสร้างโฟลเดอร์ตาม repo + official split | 1.5–3 ชม. | T4 ×1 |
| `FFD_level3b_CFB_finetune.ipynb` | fine-tune CFB (DDBF_BEiT_v2) 10 epochs จาก `BEiT-1k-Face-55w.tar` | 3–6 ชม. | T4 ×2 (DDP) |
| `FFD_level3c_eval_crossdataset.ipynb` | ประเมินบน Celeb-DF v2 + เทียบกับ checkpoint ผู้เขียน + UTOM threshold | 30–60 นาที | T4 ×1 |

---

## ลำดับการรัน

### ขั้น 0 — เตรียม input บน Kaggle (ครั้งเดียว)

| ต้องมี | ที่มา |
|---|---|
| FF++ c23 (วิดีโอ) | Kaggle dataset `xdxd003/ff-c23` (~17.9 GB) หรือ `ahmedelbanby/faceforensicsplusplus-c23-deepfakebench-structure` |
| `BEiT-1k-Face-55w.tar` | Google Drive ของผู้เขียน — notebook 3b ดาวน์โหลดให้ (ถ้าติดโควตา ให้ดาวน์โหลดเองแล้วอัปเป็น private Kaggle Dataset แล้วตั้ง `PRETRAIN_PATH`) |
| `celebdf_v2_test518` | zip ที่สร้างจาก `prepare_celebdf_test.py` อยู่แล้ว — อัปเป็น **private** Kaggle Dataset |

notebook 3a **ตรวจโครงสร้าง FF++ ให้อัตโนมัติ** (แยก real/fake จากชื่อไฟล์ `000.mp4` vs `000_003.mp4`
ไม่ขึ้นกับว่า dataset วางโฟลเดอร์แบบไหน) และข้าม subset `DeepFakeDetection` ซึ่งไม่ใช่ 4 method ของ paper

### ขั้น 1 — `FFD_level3a_FFpp_prepare.ipynb`

- Accelerator = GPU T4 ×2 · Internet = On
- Add Input: FF++ c23
- **Save & Run All (Commit)** → output จะเป็น `.tar` 5 ก้อน (meta / original / 4 method)
- ถ้าชนลิมิต 12 ชม. → แบ่งรันด้วย `DO_METHODS = ["Deepfakes", "Face2Face"]` แล้วอีกรอบ `["FaceSwap", "NeuralTextures"]`

### ขั้น 2 — `FFD_level3b_CFB_finetune.ipynb`

- Accelerator = **GPU T4 ×2** · Internet = On
- Add Input: output ของขั้น 1 (ทุกก้อน)
- **Save & Run All** → ได้ `cfb_run/best.pth`, `history.json`, `final.json`, `test_scores.npy`
- ถ้าไม่ครบ 10 epoch → Add Input เป็น output ของรอบนี้ แล้ว Save Version อีกรอบ จะ resume ต่อเอง
- ถ้า **CUDA OOM** → `CFG['batch_size'] = 16` และ `CFG['grad_accum'] = 2` (effective batch ยังเป็น 64)

### ขั้น 3 — `FFD_level3c_eval_crossdataset.ipynb`

- Add Input: output ของขั้น 2 + `celebdf_v2_test518` (+ output ของระดับ 2 ถ้ามี `cdf_crops` จะข้ามขั้น crop)
- ได้ตารางเทียบ paper / checkpoint ผู้เขียน / โมเดลเรา + ROC + threshold analysis

---

## การตรวจสอบที่ทำไว้แล้ว

รันทดสอบบน CPU ก่อนส่งมอบ ผ่านทั้งหมด:

- `get_FF_list` ที่เขียนใหม่ให้ผลลัพธ์ **ตรงกับของ repo ทุกบรรทัด** (3600 / 700 / 700, unique ครบ)
- official split ไม่มี identity ซ้ำข้าม train/val/test (1000 identity พอดี)
- frame sampling ตรงกับ `get_sampled_idx()` ของ repo สำหรับ n = 3, 8, 9, 17, 100, 557
- โมเดลสร้างได้ 171.5M params, forward [B,1,3,224,224] → (B,1,2) ×3 + pearson
- **gradient-stop ถูกต้อง**: `L_dec` ส่ง gradient ไปที่ aux branch เท่านั้น main ไม่โดน (ตรงกับ Sec. III-C)
- UFM: branch ที่มั่นใจกว่า (u ต่ำกว่า) ได้ weight มากกว่าจริง
- `EvidenceLoss` คืน `loss_cls` + `loss_avu` ตามที่ config `with_kldiv=False, with_avuloss=True` กำหนด
- layer-wise lr decay: 28 กลุ่ม, lr_scale 0.75^13 … 1.0, ครอบคลุมพารามิเตอร์ครบทั้ง 398 ตัว
- cosine + warmup: 1e-6 → 5e-5 ที่สเต็ป warmup → ลงไปใกล้ min_lr 1e-5
- Lamb step เปลี่ยน weight จริง · `evaluate()` ให้ score ระดับวิดีโอได้

---

## สิ่งที่ยังไม่ได้ทำ (ถ้าต้องการต่อ)

- **pre-train เอง** — ต้องใช้โค้ด BEiT v2 ต้นฉบับ (microsoft/unilm) + CelebA/CelebV-Text/FFHQ 550k ภาพ; ประมาณ 4×A100 หลายวัน
- **cross-dataset อื่น** — DFDC (5k วิดีโอ), FFIW (10k วิดีโอ) ใช้ pipeline เดียวกับ 3c เพียงเปลี่ยนตัวอ่านรายชื่อวิดีโอ
- **adaptation protocol** (Table V) — 10-shot / 20-shot fine-tune 2 epochs, lr 2e-5
- **ablation** (Table VI) — ปิด `L_dec` / UFM / `L_EDU` ทีละตัว

---

## อัปเดต 16 ต.ค. 2026 — notebook เพิ่มเติม

| notebook | ทำอะไร | สถานะ |
|---|---|---|
| `FFD_level3d_ablation_train.ipynb` | ฝึก ablation 5 variant (Table VI) · สร้างจาก `make_3d.py` | ✅ รันแล้ว (`kanvara/3d-ablation-study-table-vi`) |
| `FFD_level3e_eval_utom_wdf.ipynb` | ประเมินทุกโมเดลบน FF++ / Celeb-DF / WildDeepfake + UTOM เต็มรูปแบบ · สร้างจาก `make_3e.py` | ✅ รันแล้ว (`kanvara/3e-v2`) |
| `FFD_level4_live_demo.ipynb` | demo สดตอนนำเสนอ: algorithm ภายในทีละขั้น (inference) · สร้างจาก `make_4.py` | ⏳ ยังไม่ได้รันบน Kaggle |

ดูสรุปงานทั้งหมดที่ `../PROGRESS_LOG.md` และตัวเลขผลที่ `../results_summary/`
