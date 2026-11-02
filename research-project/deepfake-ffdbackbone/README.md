# ทำซ้ำ FFDBackbone (Face Forgery Detection)

**Paper:** Guo et al., *Revisiting Face Forgery Detection: From Facial Representation to Forgery Detection*, IEEE TPAMI 2026
**Code ต้นฉบับ:** https://github.com/zhenglab/FFDBackbone
**แนวทาง:** ทำซ้ำตาม CRISP-DM บน Kaggle (T4) เน้น algorithm ระดับโค้ดและรันแสดง
**นำเสนอ:** 22 ต.ค. 2026

## โครงสร้าง

| โฟลเดอร์ | เนื้อหา |
|---|---|
| [paper/](paper/) | ไฟล์ PDF ของ paper |
| [slides/](slides/) | สไลด์นำเสนอ `ffdbackbone-crispdm-deck.html` และฉบับ `.pptx` (ซอร์สและภาพใน `slides/src/`) |
| [notebooks/](notebooks/) | Kaggle notebook ระดับ 1–4 และสคริปต์สร้าง notebook (ดู `README_pipeline.md`) |
| [results/](results/) | ตารางผลลัพธ์ (CSV) แยกตามระดับการทดลอง |
| [docs/](docs/) | `PROGRESS_LOG.md` (สรุปทุกขั้นและตัวเลข), `HANDOFF.md`, `speaker_script.md` (บทพูดประกอบสไลด์) |

## ขั้นการทดลอง

| ระดับ | งาน | Notebook | ผลลัพธ์ |
|:-:|---|---|---|
| 1–2 | Inference ด้วย checkpoint ผู้เขียน | `FFD_level1_*`, `FFD_level2_*` | — |
| 3a | เตรียมข้อมูล FF++ c23 | `FFD_level3a_FFpp_prepare.ipynb` | — |
| 3b | Fine-tune CFB 10 epoch | `FFD_level3b_CFB_finetune.ipynb` | — |
| 3c | Cross-dataset Celeb-DF | `FFD_level3c_eval_crossdataset.ipynb` | [01](results/01_main_results_auc.csv), [02](results/02_auc_95ci.csv), [03](results/03_ffpp_per_method_auc.csv) |
| 3d / 3d2 | Ablation (Table VI) และหลาย seed | `FFD_level3d_*`, `FFD_level3d2_*` | [04](results/04_ablation_table6.csv), [level3d2_seeds/](results/level3d2_seeds/) |
| 3e | Eval ทุกโมเดล + UTOM + WildDeepfake | `FFD_level3e_eval_utom_wdf.ipynb` | [05](results/05_table9_utom_cdf.csv), [06](results/06_utom_unseen_target_avg_acc.csv) |
| 3f | Paired significance test | `FFD_level3f_significance.ipynb` | [level3f_significance/](results/level3f_significance/) |
| 3g | Few-shot adaptation (k = 10, 20) | `FFD_level3g_fewshot.ipynb` | [level3g_fewshot/](results/level3g_fewshot/) |
| 3h | ตรวจภาพซ้ำระหว่าง WildDeepfake กับ FF++/Celeb-DF | `FFD_level3h_wdf_dupcheck.ipynb` | [level3h_dupcheck/](results/level3h_dupcheck/) |
| 3i | Eval บน DF40 (Table IV) | `FFD_level3i_DF40_eval.ipynb` | [level3i_df40/](results/level3i_df40/) |
| 4 | Live demo (algorithm ภายใน) | `FFD_level4_live_demo.ipynb` | — |

## ผลหลัก (main branch, video-level AUC)

| ชุดข้อมูล | ทำซ้ำได้ | Paper |
|---|:--:|:--:|
| FF++ c23 | 96.57 | 99.36 |
| Celeb-DF | 84.97 | 90.46 |

DF40 (ค่าเฉลี่ย 7 วิธี, checkpoint ผู้เขียน, ทุก domain): 84.32 เทียบกับ paper 96.22 — รายละเอียดใน [table4_df40_auc.csv](results/level3i_df40/table4_df40_auc.csv)

รายละเอียดและข้อจำกัดอยู่ใน [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md)

## ข้อมูลที่ไม่ได้อยู่ใน repo

Dataset และ output ขนาดใหญ่ (Celeb-DF-v2 ~9.5 GB, FF++ crops, checkpoint `.pth`, score `.npy`, zip) ถูกแยกออกด้วย `.gitignore` และเก็บไว้ในเครื่อง / Kaggle เท่านั้น

> หมายเหตุ: `slides/src/build.py` เขียนไฟล์ออกตามตัวแปรสภาพแวดล้อม `DECK_OUT` (ค่าเริ่มต้นคือโฟลเดอร์แม่ของ `src/`)
