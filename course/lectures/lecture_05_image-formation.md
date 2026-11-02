# Lecture 5: Image Data — Class 1: How a Digital Image is Formed

> สรุปจากสไลด์ *Data Preparation for ML and AI · Image Data Module · Class 1* (130 หน้า)
> สไลด์: [image-data_class1_how-a-digital-image-is-formed.pdf](../slides/image-data_class1_how-a-digital-image-is-formed.pdf)
> แหล่งที่มาของสไลด์: CMU 16-385 Computer Vision, Bagci Lab BME 495 (Medical Imaging), MIT OCW 6.003 และส่วนที่ผู้สอนเขียนเอง

## แนวคิดหลัก

**ค่า pixel ที่เราป้อนให้โมเดลไม่ใช่ "ข้อมูลดิบ"** มันผ่านเลนส์ เซนเซอร์ การ sample/quantize และ pipeline ในกล้องมาแล้วหลายขั้น การเข้าใจว่าภาพเกิดขึ้นอย่างไรช่วยให้ตัดสินใจ preprocessing ได้ถูก

## Run sheet (3 ชั่วโมง)

| ส่วน | หัวข้อ | ประเด็น |
|:-:|---|---|
| A1 | Pinhole & lens cameras | ทำไมเซนเซอร์เปล่า ๆ ไม่เห็นภาพ |
| A2 | Imaging sensor | photon → electron → ตัวเลข, CCD vs CMOS |
| A3 | Medical image formation | X-ray, Ultrasound, CT, MRI |
| A4 | Sampling & quantization | bit depth และสิ่งที่ 8 bit ทิ้งไป |
| A5 | Aliasing & downsampling | Moiré, Nyquist, การ resize แบบผิด |
| B | Representation & color | ภาพคือ array, DICOM, color space, in-camera pipeline |
| C | First preprocessing | Resize และ Normalize |

---

## A1. กล้อง: Pinhole และ Lens

- **Bare sensor:** ทุกจุดในฉากส่งแสงไปทุก pixel จึงไม่เกิดภาพ
- **Pinhole camera (camera obscura):** รูเล็ก ๆ ทำให้แต่ละจุดในฉากไปตกที่ pixel เดียว → เกิดภาพกลับหัว
  - รูใหญ่ → ภาพเบลอ, รูเล็ก → ภาพมืด (แสงน้อย) และเกิด diffraction
- **Lens camera:** รวมแสงได้มากขึ้นโดยยังโฟกัสได้ แต่มีระยะชัด (depth of field)
- **Focal length:** pinhole = ระยะรู–เซนเซอร์, lens = ระยะที่รังสีขนานมาตัดกัน

## A2. Imaging Sensor

- **Photoelectric effect:** photon กระทบ silicon แล้วปล่อย electron
- **Quantum efficiency (QE):** สัดส่วน photon ที่แปลงเป็น electron ได้
- **Full well capacity:** จำนวน electron สูงสุดต่อ pixel (เกินแล้ว saturate)
- **CCD vs CMOS:** CMOS สมัยใหม่อ่านค่าได้เร็ว ใช้พลังงานต่ำ เป็นมาตรฐานปัจจุบัน
- **Vignetting:** pixel ที่ไกลจากกึ่งกลางได้รับแสงน้อยกว่า

## A3. Medical Image Formation

| Modality | หลักการ | ลักษณะ |
|---|---|---|
| X-ray | การดูดกลืนรังสีของเนื้อเยื่อ | ภาพ 2D แบบ projection |
| Ultrasound | การสะท้อนคลื่นเสียง | real-time, มี speckle noise |
| CT | X-ray หลายมุม + reconstruction | ภาพ 3D, หน่วย Hounsfield (HU) |
| MRI | สนามแม่เหล็ก + คลื่นวิทยุ | หลาย sequence (T1, T2, …) ค่าความเข้มไม่มีหน่วยมาตรฐาน |

ฟิสิกส์ต่างกันมาก แต่สุดท้ายได้ **array ของตัวเลข** เหมือนกัน

## A4. Sampling และ Quantization

- **Sampling:** แบ่งภาพต่อเนื่องเป็น grid ของ pixel
- **Quantization:** แปลงความเข้มต่อเนื่องเป็นระดับไม่ต่อเนื่อง (8 bit = 256 ระดับ)
- ลด bit depth มาก ๆ จะเกิด **false contouring** (แถบสี)

## A5. Aliasing

- การย่อภาพด้วยการลบแถว/คอลัมน์ (naive downsampling) ทำให้เกิด **aliasing / Moiré**
- **Nyquist–Shannon:** ต้อง sample ด้วยความถี่อย่างน้อย 2 เท่าของความถี่สูงสุดในสัญญาณ
- **Anti-aliasing:** blur ด้วย **Gaussian filter** ก่อนย่อ → ได้ **Gaussian image pyramid**

## B. Representation และ Color

- ภาพสี = array 3 channel (R, G, B), ภาพ grayscale = 1 channel, ภาพการแพทย์ = voxel 3D
- **Point processing:** ปรับค่าแต่ละ pixel แยกกัน (invert, brightness, contrast, gamma)
- **Color space:** RGB, HSV, LAB, YCbCr — แปลงด้วยโค้ดบรรทัดเดียว แต่ bug ก็บรรทัดเดียว (เช่น OpenCV อ่านเป็น **BGR**)
- **Color sensing:** สีไม่ใช่สมบัติทางฟิสิกส์ของแสง แต่เกิดจากการรับรู้ ใช้ **Color Filter Array (Bayer)** เลียนแบบ cone cell

### In-camera image processing pipeline

```
RAW → white balance → demosaicing → denoising → color transform → tone reproduction (gamma) → compression (JPEG)
```

| ขั้น | ความหมาย |
|---|---|
| White balancing | ชดเชยสีของแสง (เช่น Grey world assumption) |
| Demosaicing | เติมค่า RGB ครบทุก pixel จาก Bayer mosaic (เช่น bilinear interpolation) |
| Noise | shot noise, read noise, dark current |
| Tone reproduction | gamma correction เพราะการรับรู้ความสว่างของคนไม่เป็นเชิงเส้น |

> **Take-home:** ค่า pixel ในรูปถ่าย กับค่าที่เซนเซอร์วัดได้ เป็นคนละอย่างกัน ภาพ JPEG ผ่าน white balance, demosaic, denoise, gamma มาแล้ว — "raw data" ของเรา raw จริงหรือ?

## C. First Preprocessing: Resize และ Normalize

### Resize = Resampling

```python
# OpenCV: เลือก interpolation ให้ชัด
small = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)   # ย่อ
big   = cv2.resize(img, (896, 896), interpolation=cv2.INTER_CUBIC)  # ขยาย

# torchvision
t = v2.Resize((224, 224), antialias=True)
```

- **ย่อ:** ใช้ area averaging หรือ anti-aliasing ห้ามใช้ nearest neighbor
- **ขยาย:** bilinear / bicubic (ไม่ได้สร้างรายละเอียดจริง แค่ interpolate)
- **Nearest neighbor** ใช้กับ **label mask** ใน segmentation เท่านั้น (เฉลี่ยแล้วจะเกิด class ปลอม)
- ใช้ resize path เดียวกันทั้ง training และ deployment

### ทุกการ resize คือการตัดสินใจ

- **Squash หรือ letterbox:** บีบเป็น 224×224 ทำให้รูปร่างผิด, padding รักษารูปทรงแต่เปลือง pixel
- **Target size คืองบประมาณ:** compute เพิ่มแบบกำลังสอง
- **ภาพการแพทย์วัดเป็นมิลลิเมตร:** resample ให้ voxel spacing เท่ากันก่อน แล้วบันทึกไว้
- **จดตัวเลขทั้งหมดลง methods:** interpolation, output size, padding color, spacing

### Bit depth และ Normalization

- CT เก็บค่า HU เกิน 255 มาก การบีบเป็น 8 bit จะซ่อนเนื้อเยื่อ ต้องใช้ **windowing** (center, width)
- หารด้วย 255 เปลี่ยนหน่วย ไม่เปลี่ยนรูปร่าง distribution; **z-score** ย้ายจุดศูนย์กลางให้ optimizer ทำงานดี
- คำถามที่ทิ้งไว้: mean/std ของใคร? ของภาพนี้, ของ dataset, หรือของ ImageNet? → เป็นคำถามเรื่อง **leakage** (ต่อใน [Lecture 6](lecture_06_image-preprocessing-training.md))

## การบ้าน (Take-home)

นำภาพ 3 ภาพ ย่อ 4 เท่าแบบ naive และแบบ anti-aliasing, ลดเหลือ 4-bit, แปลงเป็น grayscale และ HSV แล้วอธิบาย artifact ที่แย่ที่สุดใน 1 ประโยค
