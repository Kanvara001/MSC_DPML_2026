# Lecture 7: Classification — Basic Concepts

> สรุปจากสไลด์ *Chapter 8. Classification: Basic Concepts* (Han, Kamber & Pei) พร้อมโน้ตและแบบฝึกหัดในคาบ
> สไลด์ (มีโน้ตลายมือ): [08_classification-basic_annotated.pdf](../slides/08_classification-basic_annotated.pdf)
> หัวข้อเสริม: [topics/decision-tree/](../topics/decision-tree/) · หน้า interactive: [Evaluation Metrics & ROC/AUC](../interactive/evaluation-metrics-roc-auc.html)

## 1. Supervised vs Unsupervised

| | Supervised (classification) | Unsupervised (clustering) |
|---|---|---|
| Label | training data มี class label | ไม่มี label |
| เป้าหมาย | สร้างโมเดลจาก training set เพื่อจำแนกข้อมูลใหม่ | ค้นหากลุ่มที่มีอยู่ในข้อมูล |

**Classification** ทำนาย label แบบไม่ต่อเนื่อง (categorical) ส่วน **numeric prediction** ทำนายค่าต่อเนื่อง
ตัวอย่างงาน: อนุมัติสินเชื่อ, วินิจฉัยเนื้องอก, ตรวจจับการโกง, จัดหมวดหน้าเว็บ

### Model construction → validation → testing → deployment

1. **Construction:** สร้างโมเดลจาก training set (decision tree, rule, สูตรคณิตศาสตร์)
2. **Validation:** ถ้าใช้ชุดข้อมูลเพื่อเลือก/ปรับโมเดล เรียกว่า validation (development) set
3. **Test:** ประเมินความแม่นยำบน test set ที่ **แยกอิสระ** จาก training set
4. **Deployment:** ถ้าความแม่นยำยอมรับได้ จึงนำไปใช้จำแนกข้อมูลใหม่

## 2. Decision Tree Induction

สร้างต้นไม้แบบ **top-down, recursive, divide-and-conquer** — ที่แต่ละ node เลือก attribute ที่แบ่งข้อมูลได้ดีที่สุด

### Entropy และ Information Gain (ID3 / C4.5)

$$Info(D) = -\sum_{i=1}^{m} p_i \log_2 p_i$$

$$Info_A(D) = \sum_{j=1}^{v} \frac{|D_j|}{|D|} \times Info(D_j)$$

$$Gain(A) = Info(D) - Info_A(D)$$

- entropy สูง = ไม่แน่นอนมาก, entropy ต่ำ = ข้อมูลบริสุทธิ์
- เลือก attribute ที่ **Gain สูงสุด** เป็น node

### ตัวอย่าง: buys_computer (14 แถว, yes 9 / no 5)

$Info(D) = I(9,5) = 0.940$

| Attribute | Gain |
|---|:-:|
| **age** | **0.246** |
| student | 0.151 |
| credit_rating | 0.048 |
| income | 0.029 |

→ เลือก `age` เป็น root (age 31…40 ทุกแถวเป็น yes จึงเป็น leaf ทันที)

### แบบฝึกหัดในคาบ: ทำนายไข้หวัด (Flu)

| Temp | Cough | Age | Flu |
|:-:|:-:|:-:|:-:|
| 39.5 | yes | young | yes |
| 38.0 | yes | young | yes |
| 40.0 | yes | old | yes |
| 38.5 | yes | old | yes |
| 39.0 | no | young | no |
| 37.0 | yes | young | no |
| 37.5 | yes | old | no |
| 36.5 | no | old | no |

$Info(D) = I(4,4) = 1$

| Split | Info | Gain |
|---|:-:|:-:|
| Cough | 6/8·I(4,2) + 2/8·I(0,2) = 0.6887 | 0.3113 |
| Age | 4/8·I(2,2) + 4/8·I(2,2) = 1 | 0 |
| **Temp > 37.5** | 5/8·I(4,1) + 3/8·I(0,3) = 0.4512 | **0.5488** |

ต้นไม้ที่ได้

```
            Temp
       > 37.5 / \ ≤ 37.5
          Cough   No
     yes /   \ no
       Yes    No
```

**Test:** Temp = 39.8, Cough = No, Age = Young → **Flu = No**

## 3. Bayes Classification

$$P(C_i \mid X) = \frac{P(X \mid C_i)\,P(C_i)}{P(X)}$$

- **Naïve Bayes:** สมมติว่า attribute เป็นอิสระต่อกันเมื่อรู้ class → $P(X \mid C_i) = \prod_k P(x_k \mid C_i)$
- attribute แบบ categorical: นับสัดส่วน; แบบ continuous: ใช้ Gaussian
- ตัวอย่าง buys_computer: X = (age ≤ 30, income = medium, student = yes, credit = fair)
  - P(X|yes)·P(yes) = 0.044 × 0.643 = 0.028
  - P(X|no)·P(no) = 0.019 × 0.357 = 0.007 → ทำนาย **yes**
- **Zero-probability problem:** ถ้ามีค่าใดเป็น 0 ผลคูณทั้งหมดเป็น 0 → แก้ด้วย **Laplacian correction** (บวก 1 ทุก count)
- จุดแข็ง: ง่าย ผลดีในหลายกรณี / จุดอ่อน: สมมติฐานความอิสระมักไม่จริง

## 4. Linear Classifier

- **Linear regression:** $y = wx + b$ ทำนายค่าต่อเนื่อง
- **Linear classifier:** ใช้ $f(x) = w^\top x + b$ แล้วตัดสินจากเครื่องหมาย (เส้นตรง/hyperplane แบ่ง 2 class)
- มีหลายเส้นที่แบ่งได้ → คำถามคือเส้นไหนดีกว่า (นำไปสู่ SVM)
- **Logistic regression:** ใช้ sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$ แปลงผลเชิงเส้นเป็นความน่าจะเป็น
- **Generative vs Discriminative:** generative จำลอง $p(X, Y)$ (Naïve Bayes) / discriminative จำลอง $p(Y \mid X)$ โดยตรง (Logistic regression)

## 5. Model Evaluation

### Confusion Matrix

| Actual \ Predicted | Positive | Negative |
|---|:-:|:-:|
| **Positive** | TP | FN |
| **Negative** | FP | TN |

| Metric | สูตร | ความหมาย |
|---|---|---|
| Accuracy | (TP + TN) / All | ทายถูกทั้งหมดกี่ % |
| Error rate | (FP + FN) / All | 1 − accuracy |
| Sensitivity / Recall | TP / (TP + FN) | จาก positive จริง จับได้กี่ % |
| Specificity | TN / (TN + FP) | จาก negative จริง ปฏิเสธถูกกี่ % |
| Precision | TP / (TP + FP) | จากที่ทายว่า positive ถูกกี่ % |
| F1 | 2·P·R / (P + R) | ค่าเฉลี่ยฮาร์มอนิกของ precision และ recall |

โน้ตในคาบ
- **ข้อมูลไม่สมดุล → accuracy หลอกได้** ต้องดู recall / precision / specificity ประกอบ
- การปรับ **threshold** (เช่น จาก 0.5) เป็นการแลก recall กับ precision
- ROC curve แสดงทุก threshold พร้อมกัน, AUC = พื้นที่ใต้ ROC curve

> ลองเล่นได้ที่หน้า interactive [Evaluation Metrics & ROC/AUC](../interactive/evaluation-metrics-roc-auc.html): เลื่อน threshold, prevalence และ separation แล้วดูว่า metric ไหนเปลี่ยน metric ไหนไม่เปลี่ยน
