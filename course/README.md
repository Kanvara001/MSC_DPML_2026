# เนื้อหารายวิชา

> 🌐 ดูแบบ interactive (ค้นหา/กรองหัวข้อ, เครื่องมือ ROC/AUC, Gaussian kernel, perceptron): [index.html](../index.html)

| โฟลเดอร์ / ไฟล์ | เนื้อหา |
|---|---|
| [00_grading-and-schedule.md](00_grading-and-schedule.md) | สัดส่วนคะแนน เกณฑ์เกรด และกิจกรรม Research Paper |
| [lectures/](lectures/) | สรุปเนื้อหาแต่ละบท (Markdown) |
| [slides/](slides/) | สไลด์และเอกสารประกอบการสอน (PDF, บางไฟล์มีโน้ตลายมือ) |
| [interactive/](interactive/) | หน้า HTML แบบ interactive: [Evaluation Metrics & ROC/AUC](interactive/evaluation-metrics-roc-auc.html) |
| [topics/](topics/) | หัวข้อเฉพาะที่ศึกษาเพิ่ม: Decision Tree, SMOTE |
| [materials/data-preparation/](materials/data-preparation/) | เอกสารประกอบ *Data Preparation for ML and AI* (HTML, เริ่มที่ `index.html`) |

## สรุปบทเรียน

### 1. พื้นฐาน AI / ML

| บท | หัวข้อ | สรุป | สไลด์ |
|:--:|---|---|---|
| 0 | ตกลงเกณฑ์คะแนนและแนะนำรายวิชา | [00_grading-and-schedule.md](00_grading-and-schedule.md) | — |
| 1 | AI และ Machine Learning เบื้องต้น, ข้อมูลและโมเดล | [lecture_01](lectures/lecture_01_intro-ml.md) | — |
| 2 | Large Language Model: Discriminative vs Generative, 5 ขั้นตอนการพัฒนา LLM | [lecture_02](lectures/lecture_02_llm.md) | — |

### 2. Data Preparation

| บท | หัวข้อ | สรุป | สไลด์ |
|:--:|---|---|---|
| 3 | Session 1: A number you can defend (data quality, leakage, evaluation hygiene) | [lecture_03](lectures/lecture_03_data-prep-session1.md) | [HTML](materials/data-preparation/s1-getting-a-defensible-number.html) |
| 4 | Data Preprocessing: cleaning, integration, reduction, PCA | [lecture_04](lectures/lecture_04_data-preprocessing.md) | [PDF](slides/03_data-preprocessing_annotated.pdf) |
| — | SMOTE และ imbalanced data | [topics/smote/](topics/smote/) | [PDF](topics/smote/smote25_data-section-summary.pdf) |

### 3. Image Data

| บท | หัวข้อ | สรุป | สไลด์ |
|:--:|---|---|---|
| 5 | Class 1: How a Digital Image is Formed | [lecture_05](lectures/lecture_05_image-formation.md) | [PDF](slides/image-data_class1_how-a-digital-image-is-formed.pdf) |
| 6 | Class 2: Preprocessing Before and During Training | [lecture_06](lectures/lecture_06_image-preprocessing-training.md) | [PDF](slides/image-data_class2_preprocessing-before-and-during-training.pdf) |

### 4. Classification & Neural Network

| บท | หัวข้อ | สรุป | สไลด์ |
|:--:|---|---|---|
| 7 | Classification: Decision Tree, Naive Bayes, Linear Classifier, Evaluation | [lecture_07](lectures/lecture_07_classification-basic.md) | [PDF](slides/08_classification-basic_annotated.pdf) |
| — | Decision Tree (หัวข้อเสริม) | [topics/decision-tree/](topics/decision-tree/) | [PDF](topics/decision-tree/Decision_tree_lecture-notes.pdf) |
| — | Evaluation Metrics & ROC/AUC (interactive) | [HTML](interactive/evaluation-metrics-roc-auc.html) | — |
| 8 | Neural Network: Perceptron & Delta Rule | [lecture_08](lectures/lecture_08_neural-network.md) | [PDF](slides/neural-network_perceptron_annotated.pdf) |

> Quiz ที่เกี่ยวข้อง: [Quiz 1](../quizzes/quiz_01.md) (Gaussian kernel → บท 6), [Quiz 2](../quizzes/quiz_02.md) (Perceptron → บท 8)
