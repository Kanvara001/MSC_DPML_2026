# Lecture 4: Data Preprocessing

> สรุปจากสไลด์ *Chapter 3: Data Preprocessing* (Han, Kamber & Pei — *Data Mining: Concepts and Techniques*) พร้อมโน้ตที่จดในคาบ
> สไลด์ (มีโน้ตลายมือ): [03_data-preprocessing_annotated.pdf](../slides/03_data-preprocessing_annotated.pdf)

## ภาพรวม

| หัวข้อ | คำถามที่ตอบ |
|---|---|
| Data Quality | ข้อมูล "ดี" วัดจากอะไร |
| Data Cleaning | จัดการ missing, noise, inconsistency อย่างไร |
| Data Integration | รวมหลายแหล่งข้อมูลโดยไม่ซ้ำ/ไม่ขัดกันอย่างไร |
| Data Reduction & Transformation | ลดขนาดข้อมูล/มิติ และปรับข้อมูลไม่สมดุลอย่างไร |
| Dimensionality Reduction (PCA) | ฉายข้อมูลลงมิติที่น้อยลงโดยเก็บข้อมูลไว้มากที่สุด |

---

## 1. ทำไมต้อง preprocess: Data Quality

คุณภาพข้อมูลมองได้หลายมิติ

| มิติ | ความหมาย | ตัวอย่างปัญหา |
|---|---|---|
| Accuracy | ถูกต้องหรือไม่ | ค่าที่บันทึกผิด |
| Completeness | ครบหรือไม่ | ไม่ได้บันทึก / ไม่มีข้อมูล |
| Consistency | สอดคล้องกันหรือไม่ | แก้บางที่แต่บางที่ไม่แก้ |
| Timeliness | ทันสมัยหรือไม่ | ข้อมูลการเสียชีวิตไม่ถูกอัปเดต |
| Believability | เชื่อถือได้แค่ไหน | แหล่งข้อมูลไม่น่าเชื่อถือ |
| Interpretability | เข้าใจง่ายแค่ไหน | รหัสที่ไม่มีคำอธิบาย |

## 2. Data Cleaning

ข้อมูลจริง "สกปรก" เพราะเครื่องมือเสีย, ความผิดพลาดของคนหรือคอมพิวเตอร์, การส่งข้อมูลผิดพลาด

| ประเภท | ตัวอย่าง |
|---|---|
| **Incomplete** (ขาดหาย) | `Occupation = ""` |
| **Noisy** (มี noise / error / outlier) | `Salary = "-10"` |
| **Inconsistent** (ขัดแย้งกัน) | `Age = "42"` แต่ `Birthday = "03/07/2010"`; rating เคยเป็น 1,2,3 แล้วเปลี่ยนเป็น A,B,C |
| **Intentional** (disguised missing) | ทุกคนเกิดวันที่ 1 ม.ค. |

### การจัดการ Missing Data

สาเหตุ: อุปกรณ์เสีย, ถูกลบเพราะขัดกับข้อมูลอื่น, ไม่ได้กรอกเพราะเข้าใจผิด, ตอนบันทึกคิดว่าไม่สำคัญ

| วิธี | หมายเหตุ |
|---|---|
| ตัดแถวทิ้ง (ignore the tuple) | ใช้เมื่อ class label หาย ไม่ดีถ้าสัดส่วน missing แต่ละ attribute ต่างกันมาก |
| เติมด้วยมือ | น่าเบื่อและทำไม่ได้กับข้อมูลใหญ่ |
| เติมด้วยค่าคงที่ เช่น `"unknown"` | ระวังโมเดลมองเป็น class ใหม่ |
| เติมด้วย mean ของ attribute | ง่าย |
| เติมด้วย mean ของ attribute **ภายใน class เดียวกัน** | ฉลาดกว่า |
| เติมด้วยค่าที่น่าจะเป็นที่สุด (Bayesian, decision tree) | เป็นการ "ทำนาย" ค่าที่หาย |

### การจัดการ Noisy Data

- **Binning:** เรียงข้อมูลแล้วแบ่ง bin (equal-frequency) จากนั้น smooth ด้วย bin mean / median / boundary
- **Regression:** fit ฟังก์ชันให้ข้อมูลแล้วใช้ค่าบนเส้นแทน
- **Clustering:** หาและตัด outlier ที่ไม่อยู่ใน cluster ใด
- **Semi-supervised:** คอมพิวเตอร์ตรวจหาค่าน่าสงสัย แล้วให้คนตรวจซ้ำ

### Data Cleaning as a Process

1. **Discrepancy detection:** ใช้ metadata (domain, range, dependency, distribution), ตรวจ field overloading, ตรวจ uniqueness / consecutive / null rule
2. **Data scrubbing:** ใช้ domain knowledge ง่าย ๆ (รหัสไปรษณีย์, spell-check)
3. **Data auditing:** วิเคราะห์หากฎความสัมพันธ์ (correlation, clustering) เพื่อหาค่าที่ผิดกฎ
4. **Migration & integration:** ใช้เครื่องมือ ETL (Extraction / Transformation / Loading)
5. ทำซ้ำแบบ iterative และ interactive

## 3. Data Integration

รวมข้อมูลจากหลายแหล่งให้เป็นชุดเดียวที่สอดคล้องกัน

- **Schema integration:** `A.cust-id` ≡ `B.cust-#`
- **Entity identification:** Bill Clinton = William Clinton
- **Value conflict:** entity เดียวกันแต่ค่าต่างกัน เพราะหน่วยหรือ scale ต่างกัน (metric vs British units)

### Redundancy

- attribute เดียวกันอาจชื่อต่างกันในแต่ละ database
- attribute ที่คำนวณได้จาก attribute อื่น (เช่น annual revenue)
- ตรวจได้ด้วย **correlation analysis** และ **covariance analysis**
- ข้อมูลซ้ำทำให้โมเดลให้น้ำหนักกับข้อมูลชุดนั้นมากเกินจริง

## 4. Data Reduction

แนวทางลดขนาดข้อมูล

- **Feature selection:** เลือก subset ของ feature เดิม
- **Feature extraction:** แปลงข้อมูลไปยัง space ที่มีมิติน้อยลง (เช่น PCA)
- **Feature creation / feature engineering:** สร้าง feature ใหม่จากความรู้เรื่อง distribution และ correlation

### Imbalanced data และ data augmentation (โน้ตในคาบ)

เมื่อข้อมูลไม่สมดุล เช่น class 0 : class 1 = 95 : 5 โมเดลมักเอนไปทาง majority class

| วิธี | แนวคิด |
|---|---|
| **Under-sampling** | ลดตัวอย่างฝั่ง majority |
| **Over-sampling** | เพิ่มตัวอย่างฝั่ง minority (เช่น duplicate) |
| **SMOTE** | สร้างตัวอย่าง minority ใหม่ด้วยการ interpolate เชิงเส้นระหว่างจุดข้างเคียง |

ดูเพิ่มที่ [topics/smote/](../topics/smote/)

## 5. Principal Component Analysis (PCA)

- ใช้ **orthogonal transformation** แปลงตัวแปรที่อาจมี correlation กัน ให้เป็นตัวแปรที่ไม่มี correlation เชิงเส้น เรียกว่า **principal components**
- ตัวอย่าง: ลูกบอลเคลื่อนที่เป็นเส้นตรง ข้อมูลจากกล้อง 3 ตัวจึงซ้ำซ้อนกันมาก

### ขั้นตอน

1. Normalize ข้อมูลให้แต่ละ attribute อยู่ในช่วงเดียวกัน
2. คำนวณ covariance matrix แล้วหา **eigenvectors** ซึ่งจะเป็นแกนของ space ใหม่
3. เรียง components ตาม variance (ความสำคัญ) จากมากไปน้อย
4. ตัด components ที่ variance ต่ำทิ้ง เหลือ k แกน (k ≤ n)

ข้อจำกัด: ใช้กับข้อมูลตัวเลขเท่านั้น

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2).fit(X_scaled)
print(pca.explained_variance_ratio_)
```

> เชื่อมกับ [Lecture 3](lecture_03_data-prep-session1.md): scaler และ PCA ต้อง `fit` บน training split เท่านั้น ไม่เช่นนั้นจะเกิด leakage
