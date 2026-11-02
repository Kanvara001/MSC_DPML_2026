# MSC DPML 2026

**Data Preparation for Machine Learning and AI** · Master, Term 1
กัญญ์วรา ถาวโรฤทธิ์ · 695020018-8

> 🌐 **หน้ารวมเนื้อหาแบบ interactive:** [index.html](index.html) — ค้นหา/กรองหัวข้อ, ROC/AUC, Gaussian kernel builder, perceptron step-through
> (เปิดผ่าน GitHub Pages: https://kanvara001.github.io/MSC_DPML_2026/)

## เริ่มอ่านตรงไหน

| อยากดู | ไปที่ |
|---|---|
| สรุปบทเรียนทั้งหมด | [course/README.md](course/README.md) |
| Quiz 1–2 (ไฟล์ที่ส่ง + วิธีทำ) | [quizzes/README.md](quizzes/README.md) |
| Research Project: Deepfake Detection | [research-project/README.md](research-project/README.md) |
| เครื่องมือ ROC / AUC | [course/interactive/evaluation-metrics-roc-auc.html](course/interactive/evaluation-metrics-roc-auc.html) |

## สรุปบทเรียน

| # | หัวข้อ | กลุ่ม |
|:-:|---|---|
| 1 | [AI และ Machine Learning เบื้องต้น](course/lectures/lecture_01_intro-ml.md) | พื้นฐาน |
| 2 | [Large Language Model](course/lectures/lecture_02_llm.md) | พื้นฐาน |
| 3 | [Data Preparation S1: A number you can defend](course/lectures/lecture_03_data-prep-session1.md) | Data Preparation |
| 4 | [Data Preprocessing (cleaning, integration, PCA)](course/lectures/lecture_04_data-preprocessing.md) | Data Preparation |
| 5 | [Image Data Class 1: How a Digital Image is Formed](course/lectures/lecture_05_image-formation.md) | Image Data |
| 6 | [Image Data Class 2: Preprocessing Before & During Training](course/lectures/lecture_06_image-preprocessing-training.md) | Image Data |
| 7 | [Classification: Decision Tree, Naive Bayes, Evaluation](course/lectures/lecture_07_classification-basic.md) | Classification |
| 8 | [Neural Network: Perceptron & Delta Rule](course/lectures/lecture_08_neural-network.md) | Classification |

## Quiz

| Quiz | หัวข้อ | วิธีทำ | ไฟล์ที่ส่ง |
|:-:|---|---|---|
| 1 | Gaussian Kernel 5×5 (σ = 1) | [quiz_01.md](quizzes/quiz_01.md) | [PDF](quizzes/quiz_01_gaussian-kernel.pdf) |
| 2 | Perceptron Learning (X1 ∧ X2) ∨ X3 | [quiz_02.md](quizzes/quiz_02.md) | [PDF](quizzes/quiz_02_perceptron.pdf) |

## โครงสร้าง repo

```
MSC_DPML_2026/
├── index.html           หน้ารวมเนื้อหาแบบ interactive
├── course/
│   ├── lectures/        สรุปบทเรียน (Markdown)
│   ├── slides/          สไลด์ / เอกสารประกอบ (PDF)
│   ├── interactive/     หน้า HTML interactive (ROC/AUC)
│   ├── topics/          หัวข้อเสริม: Decision Tree, SMOTE
│   └── materials/       เอกสาร Data Preparation (HTML)
├── quizzes/             Quiz 1–2
└── research-project/    Research Project: AI Detection (Deepfake, FFDBackbone)
```

## สัดส่วนคะแนน

| ส่วนประกอบ | สัดส่วน |
|---|:--:|
| Research Project | 30% |
| Homework | 10% |
| Quiz | 10% |
| Midterm Exam | 20% |
| Final Exam | 20% |
| Github | 10% |

เกณฑ์เกรด: A ≥ 80, ..., F < 50 · รายละเอียดที่ [course/00_grading-and-schedule.md](course/00_grading-and-schedule.md)
