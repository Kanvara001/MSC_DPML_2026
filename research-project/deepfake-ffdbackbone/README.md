# ทำซ้ำ FFDBackbone (Face Forgery Detection)

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026
**Code ต้นฉบับ:** https://github.com/zhenglab/FFDBackbone
**แนวทาง:** ทำซ้ำตาม CRISP-DM บน Kaggle (T4) เน้น algorithm ระดับโค้ดและรันแสดง

## โครงสร้าง

| โฟลเดอร์ | เนื้อหา |
|---|---|
| [paper/](paper/) | ไฟล์ PDF ของ paper |
| [slides/](slides/) | สไลด์นำเสนอ 35 หน้า `ffdbackbone-crispdm-deck.html` (ซอร์สและภาพใน `slides/src/`) |
| [notebooks/](notebooks/) | Kaggle notebook ระดับ 1–4 และสคริปต์สร้าง notebook (ดู `README_pipeline.md`) |
| [results/](results/) | ตารางผลลัพธ์ (CSV) |
| [docs/](docs/) | `PROGRESS_LOG.md` (สรุปทุกขั้นและตัวเลข), `HANDOFF.md` (สถานะและงานที่ค้าง) |

## ผลหลัก (main branch, video-level AUC)

| ชุดข้อมูล | ทำซ้ำได้ | Paper |
|---|:--:|:--:|
| FF++ c23 | 96.57 | 99.36 |
| Celeb-DF | 84.97 | 90.46 |

รายละเอียดและข้อจำกัดอยู่ใน [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md)

## ข้อมูลที่ไม่ได้อยู่ใน repo

Dataset และ output ขนาดใหญ่ (Celeb-DF-v2 ~9.5 GB, FF++ crops, checkpoint, zip) ถูกแยกออกด้วย `.gitignore` และเก็บไว้ในเครื่อง / Kaggle เท่านั้น

> หมายเหตุ: `slides/src/build.py` เขียนไฟล์ออกที่ `/mnt/user-data/outputs/` ต้องแก้ตัวแปร `OUT` ก่อนรันในเครื่อง
