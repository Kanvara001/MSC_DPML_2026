# Lecture 6: Image Data — Class 2: Preprocessing Before and During Training

> สรุปจากสไลด์ *Data Preparation for ML and AI · Image Data Module · Class 2* (105 หน้า)
> สไลด์: [image-data_class2_preprocessing-before-and-during-training.pdf](../slides/image-data_class2_preprocessing-before-and-during-training.pdf)
> แบบฝึกหัดที่เกี่ยวข้อง: [Quiz 1 — Gaussian Kernel](../../quizzes/quiz_01.md)

## Run sheet (3 ชั่วโมง)

| ส่วน | หัวข้อ |
|:-:|---|
| D1 | Filtering: point ops, convolution, Gaussian vs box, sharpening |
| D2 | Classical ops บนภาพการแพทย์: histogram CT, median filter, MRI denoising, SNR/CNR |
| D3 | Contrast, color, crop: histogram equalization, CLAHE, cropping/padding |
| E1 | Preprocessing for the model: zero-center, normalize, per-channel stats |
| E2 | **The rule:** fit สถิติบน training split เท่านั้น + MRI intensity standardization |
| E3 | Transfer learning: pretrained backbone และสถิติที่มันคาดหวัง |
| F1–F2 | Augmentation ทั่วไป และ augmentation ภาพการแพทย์ |
| F3 | Pipelines in code: torchvision v2, Albumentations, Keras, MONAI |

---

## D1. Image Filtering

**Linear shift-invariant filtering:** แทนแต่ละ pixel ด้วย linear combination ของเพื่อนบ้าน (= convolution กับ kernel)

| Filter | ลักษณะ |
|---|---|
| Box filter | เฉลี่ยเท่ากันทุกช่อง เกิด artifact เป็นเหลี่ยม |
| **Gaussian filter** | ถ่วงน้ำหนักตามระยะห่างจากจุดกลาง → blur นุ่มนวล ใช้ก่อน downsample |
| Sharpening | original + α(original − blurred) ใช้อย่างระมัดระวัง |
| Median filter | ไม่เชิงเส้น เหมาะกับ salt-and-pepper noise รักษาขอบ |

### Gaussian kernel

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

ตัวอย่าง 5×5, σ = 1 (ทำใน [Quiz 1](../../quizzes/quiz_01.md))

```
          | 1  4  7  4 1 |
  1/273 × | 4 16 26 16 4 |
          | 7 26 41 26 7 |
          | 4 16 26 16 4 |
          | 1  4  7  4 1 |
```

## D2–D3. ภาพการแพทย์ และ Contrast / Color / Crop

- วัดคุณภาพภาพด้วย **SNR** (signal-to-noise) และ **CNR** (contrast-to-noise)
- **Histogram equalization** กระจาย histogram ทั้งภาพ แต่บางบริเวณอาจซีด → **CLAHE** (adaptive, จำกัด contrast) ดีกว่า
- การตัดสินใจเรื่องสีอยู่ใน pipeline ไม่ใช่ในโมเดล
- **Crop** สิ่งที่สำคัญ แล้ว **pad** ส่วนที่เหลือ

```python
from skimage import exposure
eq    = exposure.equalize_hist(img)
clahe = exposure.equalize_adapthist(img, clip_limit=0.03)
```

## E1. Preprocessing for the Model

- **Zero-center** และ **normalize** ทำให้ loss surface สมมาตร optimizer ลู่เข้าเร็วขึ้น
- ใช้สถิติ **per-channel** (mean/std ของแต่ละ R, G, B)

## E2. The Rule — Fit preprocessing on the training split only

```
train  ──fit──▶  mean, std, min/max, PCA, landmarks  (frozen)
                         │ apply
              ┌──────────┼──────────┐
            train    validation    test
```

> "Any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data." — CS231n

- fit บนข้อมูลทั้งหมด = test set รั่วเข้า training (**leakage**)
- ภาพการแพทย์: ทั้ง split และสถิติต้องเคารพขอบเขต **patient** (ผู้ป่วยคนเดียวห้ามอยู่หลาย split)
- **MRI intensity standardization:** ค่าความเข้ม MRI ไม่มีหน่วยมาตรฐาน ต้อง normalize โดยอิง anatomy (เช่น histogram landmark ที่ fit บน train)

## E3. Transfer Learning — Pretrained Statistics

```python
# สถิติของ ImageNet ที่ backbone ส่วนใหญ่ใช้
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
tfm = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=mean, std=std),
])
```

- คือ training-set statistics ของ ImageNet ที่ถูกแช่แข็งเป็น "สัญญา" ของ checkpoint
- ViT / CLIP บางตัวใช้ค่าต่างกัน ต้องอ่าน model card
- ภาพ grayscale ผ่าน RGB backbone: ทำซ้ำเป็น 3 channel แล้วใช้สถิติของ backbone
- สถิติผิดจะ **ไม่ crash** แต่ทำให้ input เลื่อนออกจาก distribution ที่โมเดลคาดหวังแบบเงียบ ๆ

## F. Augmentation

- **Augmentation = preprocessing ที่สุ่มใหม่ทุก epoch** (flip, rotate, crop, color jitter, blur, …)
- **MixUp / CutMix** ทำระดับ batch หลัง DataLoader
- **Medical augmentation:** ต้องสมจริงทางฟิสิกส์ (เช่น จำลอง artifact ของเครื่อง scanner, bias field, motion)
- **ตอน evaluate ต้องใช้ภาพที่ไม่ถูก augment** (หรือเฉลี่ยผลแบบ TTA)

### Pipelines in code

| Library | จุดเด่น |
|---|---|
| torchvision v2 | มาตรฐาน PyTorch, transform ภาพ + box + mask พร้อมกัน |
| Albumentations | เร็ว เหมาะกับงานที่ label เป็น pixel (segmentation) |
| Keras preprocessing layers | preprocessing อยู่ในโมเดลตอน deploy |
| MONAI / TorchIO | ภาพการแพทย์ 3D: DICOM, spacing, windowing, artifact |

## ห้าประโยคที่ต้องจำ

1. Classical filter ไม่ได้ล้าสมัย มันคือภาษาของ CNN layer แรก และของทุก artifact ที่ต้อง debug
2. สถิติ preprocessing ทุกตัว fit บน training split แล้ว freeze; ข้อมูลการแพทย์ต้อง "ไม่ข้ามขอบเขตผู้ป่วย" ด้วย
3. Pretrained backbone คือสัญญา: resize, ลำดับ channel, ค่า normalize ต้องตรง ไม่เช่นนั้นประโยชน์หายไป
4. Augmentation คือความสุ่มตอน train ที่ต้องปิด (หรือเฉลี่ย) ตอน test; การประเมินผลเห็นภาพที่ไม่ถูกแก้ไขเสมอ
5. Pipeline ที่ไม่มีใครรันซ้ำได้ คือผลลัพธ์ที่ไม่มีใครเชื่อได้: seed, parameter, ค่าคงที่ ต้องอยู่ใน code ที่มี version
