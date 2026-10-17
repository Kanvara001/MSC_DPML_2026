# Lecture 3: Data Preparation — Session 1: A number you can defend

> สรุปเนื้อหาสำคัญจาก *Data Preparation for ML and AI (condensed edition)* Session 1 รวมข้อสังเกตจาก speaker notes
> ต้นฉบับ: [s1-getting-a-defensible-number.html](../materials/data-preparation/s1-getting-a-defensible-number.html) (ภาพรวมทั้ง 5 session ดูที่ [index-condensed.html](../materials/data-preparation/index-condensed.html))
> ผู้สอน: Thanapong Intharah, Khon Kaen University

## แนวคิดหลัก

**ตัวเลขผลลัพธ์ (เช่น accuracy, AUC) ขึ้นอยู่กับวิธีแบ่งข้อมูล คุณภาพ label และวินัยในการประเมินผล ไม่น้อยไปกว่าตัวโมเดล** Session นี้เป็นฐานของอีก 4 session ที่เหลือ

หัวข้อ 3 ส่วน (ตามสไลด์ §0.1–§0.3)

| ส่วน | หัวข้อ |
|:-:|---|
| 0.1 | Framing: การเตรียมข้อมูลอยู่ตรงไหนของงาน |
| 0.2 | Data quality, profiling และ validation |
| 0.3 | Splitting, leakage และ evaluation hygiene (**ส่วนสำคัญที่สุด**) |

เป้าหมายการเรียนรู้ 4 ข้อ: **Diagnose** (หา leak ใน pipeline ของคนอื่น), **Measure** (วัดคุณภาพข้อมูลเป็นตัวเลข), **Decide** (ตัดสินใจก่อนแตะข้อมูล), **Defend** (เขียนให้ผู้ตรวจสอบเช็กได้)

---

## 0.1 Framing: การเตรียมข้อมูลอยู่ตรงไหน

### Data-centric turn
- **Model-centric**: เลือกสถาปัตยกรรม ปรับ hyper-parameter ฝึกซ้ำ รายงานตัวเลข โดยปล่อยให้ dataset คงที่
- **Data-centric**: ตรวจ/แก้ label กรอง ลบข้อมูลซ้ำ เลือก/augment/rebalance แล้ว ablate การเปลี่ยนแปลงของข้อมูล โดย **ตรึงโมเดลไว้**
- ข้อเสนอ: เมื่อโมเดลคงที่ การปรับปรุงข้อมูลอย่างเป็นระบบคือ lever ที่ให้ผลสูงกว่าและยังถูกศึกษาน้อย (ไม่ได้หมายความว่าสถาปัตยกรรมไม่สำคัญ)
- *Notes:* ถามตัวเองว่าโปรเจกต์ล่าสุดใช้เวลากับ loop ฝั่งโมเดลกี่ชั่วโมง ฝั่งข้อมูลกี่ชั่วโมง

### งานด้านข้อมูล 3 อย่างที่มักถูกรวมเป็นอย่างเดียว (Zha et al.)

| งาน | ตัวอย่าง | Metric |
|---|---|---|
| Training-data development | collection, labelling, cleaning, dedup, augmentation, selection | ประสิทธิภาพ downstream task ที่ compute คงที่ |
| Inference-data development | eval set, slice/subgroup set, adversarial set, decontamination | eval set ตอบคำถามที่เราถามจริงไหม |
| Data maintenance | monitoring, drift detection, versioning, consent/deletion | time-to-detect, time-to-repair |

ข้อผิดพลาดคลาสสิก: ปรับปรุง training data แล้วประเมินบน inference set ที่สร้างจาก pool เดียวกันซึ่งปนเปื้อน แล้วรายงาน "ผลที่ดีขึ้น" ที่จริงคือ **ข้อมูลซ้อนทับกัน**

### ข้อควรระวังและแนวปฏิบัติ
- **"งาน data prep กิน 80% ของเวลา"** เป็นความเชื่อจากแบบสำรวจในอุตสาหกรรมที่กลุ่มตัวอย่างเล็กและเลือกเอง ใช้เปิดบทสนทนาได้ แต่ไม่ควรใช้อ้างอิง ให้บันทึกชั่วโมงงานของตัวเองตามเฟส CRISP-DM และแยกบันทึก **rework** (เวลาที่ต้องทำ preparation ซ้ำหลังพบปัญหาภายหลัง)
- **CRISP-DM** ยังเป็นภาษากลางที่ดี แต่ไม่ได้ครอบคลุม label ในฐานะวัตถุที่ต้องศึกษา การ curate ข้อมูลระดับ corpus และการที่ preparation เป็นตัวแปรทดลองหลัก
- **Task-first design:** ตัดสิน 4 อย่างก่อนแตะข้อมูล

| ต้องตัดสินก่อน | ส่งผลต่อ |
|---|---|
| Downstream task (classification, ranking, segmentation, generation) | อะไรคือ outlier กับ rare event |
| Unit of analysis (patient, visit, image, camera-hour, document, chunk) | **รูปแบบการ split และ grain ของการ join** |
| Evaluation metric (และการตัดสินใจที่ metric แทน) | ต้องสนใจ calibration หรือแค่ ranking |
| Deployment constraint (latency, device, ใครใช้ผลลัพธ์) | งบประมาณ preprocessing ตอน inference |

- **DC-Check:** checklist แบบ data-centric ที่ตอบก่อนสร้างโมเดล บังคับให้ถามเรื่อง provenance, subgroup coverage, drift plan ตั้งแต่ต้น
- **Automated data preparation ตามหลัง AutoML** เพราะ search space ไม่ปิด, objective ไม่ใช่ตัวเลขเดียว (fidelity/fairness/privacy/cost), ลำดับขั้นสลับกันไม่ได้ (impute→scale ≠ scale→impute) และ validation เปราะบาง (ส่วนที่ค้นหาคือส่วนที่ทำให้ leak)

---

## 0.2 Data quality, profiling และ validation

> *"A dimension you cannot compute is an adjective, not a quality dimension."*

### 6 มิติคุณภาพข้อมูล (Wang & Strong, 1996) พร้อมวิธีวัด

| มิติ | คำถาม | วัดเป็น |
|---|---|---|
| Completeness | มีอะไรขาดหายที่ควรมีไหม | จำนวน null ต่อคอลัมน์ ต่อกลุ่มย่อย |
| Validity | ค่าทุกตัวตรงตาม type และช่วงที่กำหนดไหม | อัตราผ่าน schema/constraint |
| Accuracy | ค่าตรงกับความจริงไหม | ตรวจเทียบ reference standard |
| Consistency | ฟิลด์ที่เกี่ยวข้องกันสอดคล้องกันไหม | จำนวนการละเมิดกฎข้ามฟิลด์ |
| Uniqueness | แต่ละเอนทิตีในโลกจริงปรากฏครั้งเดียวไหม | อัตรา duplicate / linkage |
| Timeliness | ข้อมูลใหม่พอจะนำไปใช้ไหม | การกระจายอายุข้อมูล ณ เวลาทำนาย |

*Notes (แบบฝึกหัด 5 นาที):* เลือก dataset หนึ่งชุด เขียน test ให้ครบ 6 ข้อ ส่วนใหญ่จะติดที่ **Accuracy** เพราะต้องมี reference standard ที่ยังไม่ได้นิยาม

### Profiling: ขั้นแรกที่ห้ามข้าม
- **ต่อคอลัมน์:** type, cardinality, null rate, min/max, รูปร่าง distribution, ค่าที่พบบ่อย, จำนวนค่าที่เป็นไปไม่ได้
- **ต่อคู่คอลัมน์:** correlation และแผนที่ missingness co-occurrence (คอลัมน์ที่หายพร้อมกันบอกกระบวนการเก็บข้อมูล)
- **ต่อแถว:** อัตรา duplicate / near-duplicate
- **ต่อกลุ่ม:** แยกตาม site, scanner, camera, เวลา (ตรงนี้เห็น site effect)
- รายงาน profiling ไม่ใช่ผลงาน สิ่งที่ส่งมอบคือ **assertion ที่เขียนหลังอ่านรายงาน**

### Schema contract และ unit test สำหรับข้อมูล
- กำหนด contract: type, ช่วงค่า, หมวดที่อนุญาต, referential integrity, ขอบเขตจำนวนแถว, null rate ที่ยอมรับ
- **Assert สองรอบ:** บน raw source และหลัง transform (transform ที่สร้าง NaN เงียบๆ จะผ่านรอบแรกแต่ตกรอบสอง)
- **Fail loudly:** warning ใน log ไม่ใช่ gate ต้องหยุด pipeline
- ปฏิบัติต่อข้อมูลเหมือนโค้ด (ML Test Score, Breck et al. 2017) และ version contract ไปพร้อมข้อมูล
- ตัวอย่างจริงใน notes: หน่วยเปลี่ยนจาก mg/dL เป็น mmol/L ผ่านการตรวจ type ทุกอย่าง แต่ทำให้ sensitivity ของโมเดลลดลงครึ่งหนึ่งนาน 4 เดือน

### Drift 3 แบบ ที่ dashboard มองเหมือนกันแต่แก้ต่างกัน

| แบบ | อะไรเปลี่ยน | ตัวอย่าง | ตอบสนอง |
|---|---|---|---|
| Covariate shift | P(x) เปลี่ยน, P(y\|x) คงเดิม | เครื่อง ultrasound ใหม่ที่โรงพยาบาลอำเภอ | reweight / recalibrate |
| Label (prior) shift | P(y) เปลี่ยน, P(x\|y) คงเดิม | ย้ายการคัดกรองจากคลินิกส่งต่อไปยังหมู่บ้าน (prevalence ลด) | ขยับ threshold |
| Concept drift | P(y\|x) เปลี่ยน | แนวทางวินิจฉัยฉบับใหม่ | ต้อง retrain / relabel |

### สถิติตรวจ drift และข้อจำกัด

| สถิติ | เทียบอะไร | ข้อควรระวัง |
|---|---|---|
| PSI | marginal distribution แบบแบ่ง bin | ค่าขึ้นกับการเลือก bin; เกณฑ์ 0.1/0.25 เป็นธรรมเนียม ไม่ใช่ทฤษฎี |
| Kolmogorov–Smirnov | CDF ทีละมิติ | univariate; ที่ n > 10⁵ จับความต่างเล็กจนไม่สำคัญ |
| Chi-square | ความถี่ของ category | ไวต่อ category หายากและขนาดตัวอย่าง |
| MMD | distribution หลายมิติ | ต้องเลือก kernel/bandwidth; แพง; ระบุคอลัมน์ยาก |
| Classifier two-sample test | โมเดลแยกข้อมูลเก่า/ใหม่ได้ไหม | ตีความง่ายสุด (feature importance บอกว่าอะไรเปลี่ยน) แต่ต้อง split ที่ซื่อตรงเอง |

**ข้อควรระวังที่สำคัญที่สุด:** ทั้งหมดเป็น hypothesis test ถ้ามีแถวมากพอทุกอย่างจะ "มีนัยสำคัญ" ให้ดู **effect size และผลต่อ downstream metric** ไม่ใช่ p-value

### Audit corpus: อย่าเชื่อ README
- ตรวจ corpus ด้วยการค้นหาและสุ่มอ่าน เช่น *What's In My Big Data?* (Elazar et al., ICLR 2024)
- ปัญหาที่เคยพบ: benchmark contamination, ข้อความที่สร้างโดยเครื่องและซ้ำกันจำนวนมาก, PII ใน data ที่ควรเป็นสาธารณะ, blocklist filtering ที่ตัดข้อความที่ถูกต้องออกทั้ง register (Dodge et al., 2021 กับ C4)
- Audit ขั้นต่ำ: **สุ่ม 200 เอกสารมาอ่านจริง** แล้ววัด language mix, source mix, duplication rate, date distribution, overlap กับ benchmark ที่จะรายงาน

---

## 0.3 Splitting, leakage และ evaluation hygiene

### Grouping unit คือการตัดสินใจเชิงโมเดล ไม่ใช่รายละเอียดเทคนิค

ข้อมูลเดียวกัน แบ่งได้ 3 แบบ (ตัวอย่าง: ผู้ป่วย 8 คน คนละ 3 ภาพ)

| แบบ split | ปัญหา / ความเหมาะสม |
|---|---|
| Random (ตามภาพ) | ผู้ป่วยคนเดียวกันอยู่ทั้ง train และ test โมเดลจำ **ตัวผู้ป่วย** แทนที่จะเรียนรู้โรค |
| Grouped (ตามผู้ป่วย) | ภาพทุกภาพของผู้ป่วยอยู่ฝั่งเดียวกัน เป็น **ขั้นต่ำ** สำหรับ claim ทางคลินิก |
| Temporal (ตามวันที่ได้ภาพ) | ฝึกด้วยอดีต ทดสอบด้วยอนาคต ตรงกับการใช้งานจริงที่สุด |

**สัญญาณเตือน:** ถ้า accuracy ตกแรงเมื่อเปลี่ยนจาก random เป็น grouped แปลว่า random split วัด "ตัวตนของผู้ป่วย" ให้ **รายงานทั้งสองตัวเลข**

ตัวอย่าง grouping unit ในงานจริง: ผู้ป่วย (ultrasound), specimen ของชุดทดสอบ (OV-RDT), กล้องและ session (CCTV) *Notes:* ให้แต่ละกลุ่มเขียน grouping unit ของโปรเจกต์ตนบนกระดาน ความเห็นไม่ตรงกันในกลุ่มคือส่วนที่น่าสนใจ

### Cross-validation เลือกแบบไหน

| แบบ | ใช้เมื่อ |
|---|---|
| k-fold | แถว exchangeable และมีมาก (ข้อมูลสุขภาพแทบไม่เข้าเงื่อนไข) |
| Stratified k-fold | class imbalance (ยังผิดถ้าข้อมูลเป็นกลุ่ม) |
| **Group k-fold / leave-one-group-out** | **ค่าเริ่มต้นที่ถูกต้อง** สำหรับ patient, site, scanner, camera, session |
| Nested CV | วงในปรับ hyper-parameter วงนอกประเมินผล (ไม่งั้นคะแนนรวมการค้นหาโมเดลไว้ด้วย; Cawley & Talbot, 2010) |
| Blocked / walk-forward | ข้อมูลมี time index |
| Repeated CV | ลด variance แต่ **ไม่ลด bias จากการออกแบบที่ผิด** (ทำ split ที่ leak ซ้ำ = ได้ตัวเลขผิดที่แม่นยำ) |

กฎง่ายๆ: เลือก split ที่ทำให้ train **ต่างจาก test มากที่สุด** ในบรรดาแบบที่ยังซื่อตรง เพราะ optimism มาจากความคล้ายที่เราไม่ได้ตั้งใจ

### Leakage 8 ประเภท (Kapoor & Narayanan)

รีวิว 22 ชิ้น พบ leakage ใน 17 สาขา กระทบ paper รวม 294 ฉบับ

| # | ประเภท | ที่พบในทางปฏิบัติ |
|---|---|---|
| L1.1 | ไม่มี test set | ตัวเลขที่รายงานคือผลบน training |
| L1.2 | Pre-processing บนข้อมูลทั้งหมด | scaler/imputer/encoder/PCA fit ก่อน split |
| L1.3 | Feature selection บนข้อมูลทั้งหมด | คัดฟีเจอร์ด้วย correlation กับ y ทุกแถว แล้วค่อย CV |
| L1.4 | Duplicate ข้าม split | แถว/ภาพ/เฟรม/เอกสารซ้ำหรือเกือบซ้ำอยู่สองฝั่ง |
| L2 | Illegitimate features | ฟีเจอร์ที่เป็น proxy ของ target หรือบันทึกหลังรู้ผลแล้ว |
| L3.1 | Temporal leakage | ฝึกด้วยข้อมูลหลังเวลาทำนาย |
| L3.2 | Non-independence | ผู้ป่วย/site/อุปกรณ์/ผู้เขียน/session เดียวกันอยู่ทั้งสองฝั่ง |
| L3.3 | Sampling bias ใน test | test ไม่ได้สุ่มจากประชากรที่ claim พูดถึง |

ผลที่น่าจดจำ: เมื่อแก้ leakage ในงานทำนายสงครามกลางเมือง โมเดล ML ที่ซับซ้อนไม่ได้ดีกว่า logistic regression รุ่นเก่าอย่างมีนัยสำคัญ

### Fit transform ใน fold เสมอ
- ความผิดพลาดที่พบบ่อยที่สุด และป้องกันได้ง่ายที่สุดด้วยโค้ด
- **ป้องกันด้วยสถาปัตยกรรม:** ห่อ stateful transform ทุกตัวใน `Pipeline` / `ColumnTransformer` แล้วส่ง pipeline (ไม่ใช่ array ที่ fit ไว้แล้ว) ให้ cross-validator
- ตัวอย่างพิสูจน์ (Varma & Simon, 2006): เลือก feature จากข้อมูลทั้งหมดแล้ว CV ได้ error ใกล้ศูนย์บน **noise ล้วน**

### Optimism gap

Leakage ไม่ได้ทำให้เอนเอียงเล็กน้อย แต่คือความต่างระหว่างผลที่ตีพิมพ์ได้กับผลที่จริง ใหญ่สุดเมื่อข้อมูลน้อยและ grouping แข็งแรง (เช่นงานคลินิก) แนวปฏิบัติ: **รายงานสองตัวเลข** เช่น "random split 0.94, patient-grouped split 0.79" (ตัวเลขนี้เป็นตัวอย่างประกอบ ไม่ใช่ผลวิจัยจริง)

### Duplicate และ grouping ในข้อมูลคลินิก/CCTV
- **Near-duplicate** คือตัวเพิ่ม accuracy เงียบๆ เช่น เฟรมห่างกัน 1 วินาที, สไลด์ที่สแกนสองครั้ง, รายงานที่ copy-forward
- วิธีตรวจ: exact hashing → perceptual hashing / MinHash-LSH → embedding nearest-neighbour (ทำ **ก่อน split**)
- Grouping unit: ภาพทางคลินิก = ผู้ป่วยเป็นอย่างน้อย (site/scanner ถ้าอ้าง generalisation), วิดีโอ/CCTV = กล้องและ session (**ไม่ใช่เฟรม**), ข้อความ = เอกสารหรือผู้เขียน, RAG = เอกสารต้นทาง (ไม่ใช่ chunk)
- **ทดสอบ:** ถ้าสองแถวจะยังอยู่ใน dataset เมื่อสุ่มผู้ป่วย/กล้อง/เอกสารน้อยลงหนึ่ง แปลว่าอยู่กลุ่มเดียวกัน

### Temporal validity

ถ้ามี time index การประเมินที่ซื่อตรงคือ train อดีต test อนาคต และเว้นช่วง **purge/embargo** เพราะฟีเจอร์และ label มีความกว้างในเวลา เช่น label "เสียชีวิตภายใน 30 วัน" ทำให้ 30 วันสุดท้ายของ training window ใช้ไม่ได้ เพราะยังไม่รู้ label

### Contamination, test-set reuse และทางแก้
- **Benchmark contamination:** eval set อยู่ใน training corpus (ที่ระดับ web-scale คือค่าเริ่มต้น) ตรวจด้วย n-gram overlap/embedding แล้ว decontaminate ก่อนฝึก
- **Adaptive overfitting:** งานจำนวนมากจูนบน test set เดียวกัน Recht et al. (2019) สร้าง test set ใหม่ของ CIFAR-10/ImageNet แล้ว accuracy ลดลงทั่วกัน แต่ลำดับของโมเดลส่วนใหญ่ยังคงเดิม (claim เชิงเปรียบเทียบทนกว่า claim เชิงสัมบูรณ์)
- **Model info sheet:** แบบฟอร์ม 1 หน้า 8 แถว ให้ผู้เขียนระบุว่าแต่ละ leakage type สะอาดเพราะอะไร ควรเป็นภาคผนวกบังคับในทุกโปรเจกต์

---

## Paper clinic (50 นาที): Find the leakage

นำ paper applied-ML จากสาขาของตัวเองมา 1 เรื่อง
1. 10 นาที ทำคนเดียว: กรอก model info sheet ว่า paper พูดอะไรในแต่ละ leakage type หรือไม่ได้พูดถึง
2. 15 นาที จับคู่: สลับ paper คนหนึ่งปกป้อง อีกคนโต้แย้ง แล้วสลับข้าง
3. 20 นาที ทั้งห้อง: 3 คน คนละ 3 นาที ระบุประเภท อ้างประโยคจริง และบอกว่าผลหลักยังรอดไหม
4. 5 นาที โหวต: มีกี่เรื่องที่รายงาน grouping unit ชัดเจน

กติกา: เฉพาะเจาะจงและยุติธรรม "split ไม่ชัดเจน" คือ finding แต่ "paper นี้แย่" ไม่ใช่ ผู้เขียนส่วนใหญ่แค่พลาดตามจุดบอดของสาขา

**Take home:** กรอก sheet เดียวกันกับโปรเจกต์ของตัวเองก่อน Session 2 และนำแถวที่กรอกไม่ได้มาด้วย

---

## สรุป 5 ประโยค

1. **Curation คือตัวแปรทดลอง** เมื่อโมเดลคงที่ การปรับปรุงข้อมูลอย่างเป็นระบบคือ lever ที่ให้ผลสูงและยังถูกสำรวจน้อย
2. **มิติคุณภาพที่คำนวณไม่ได้คือคำคุณศัพท์** เขียน assertion และให้ pipeline หยุดเมื่อไม่ผ่าน
3. **Grouping unit คือการตัดสินใจเชิงโมเดล** ประกาศก่อนทดลองครั้งแรก (patient, site, camera, session, document, author)
4. **Fit ทุก stateful transform ใน fold** บังคับด้วยสถาปัตยกรรม ไม่ใช่ด้วยความจำ
5. **รายงานสองตัวเลข** random split vs grouped split แสดงว่าคุณรู้ว่าต้องดู

> ตัวเลขใน abstract เป็นคุณสมบัติของ split design, คุณภาพ label และวินัย fold ไม่น้อยไปกว่าคุณสมบัติของโมเดล

## Session ถัดไป

| Session | หัวข้อ |
|:-:|---|
| 2 | Tabular data: missingness, encoding, imbalance vs calibration, EHR cohorts |
| 3 | Text และ LLM data: Thai segmentation, tokenisation, corpus filtering, dedup |
| 4 | Image และ video data: DICOM, normalisation, augmentation, CCTV |
| 5 | Labels, reduction และ governance: label noise, coreset, synthetic data, privacy, fairness |

อ่านก่อน Session 2: Kapoor & Narayanan, *Patterns* 2023
