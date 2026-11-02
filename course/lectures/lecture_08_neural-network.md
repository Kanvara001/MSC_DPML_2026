# Lecture 8: Neural Network — Perceptron และ Delta Rule

> สรุปจากเอกสาร *โครงข่ายประสาทเทียม (Artificial Neural Network)* บทที่ 6.7 (บุญเสริม กิจศิริกุล) พร้อมโน้ตในคาบ
> เอกสาร (มีโน้ตลายมือ): [neural-network_perceptron_annotated.pdf](../slides/neural-network_perceptron_annotated.pdf)
> แบบฝึกหัดที่เกี่ยวข้อง: [Quiz 2 — Perceptron Learning](../../quizzes/quiz_02.md)

## 1. จากเซลล์ประสาทสู่ Perceptron

เซลล์ประสาทชีวภาพรับสัญญาณผ่าน dendrite รวมที่ตัวเซลล์ แล้วส่งออกทาง axon — perceptron จำลองแนวคิดนี้

```
x0 = 1 ──w0──┐
x1 ─────w1───┤
x2 ─────w2───┼──▶ Σ wi·xi ──▶ activation ──▶ o
 ⋮            │
xn ─────wn───┘
```

- **Net sum (linear combination):** $g(\vec{x}) = \sum_{i=0}^{n} w_i x_i = \vec{w}\cdot\vec{x}$ โดย $x_0 = 1$ คือ **bias**
- **Activation function**
  - bipolar: output 1 / −1
  - binary: output 1 / 0

$$o(\vec{x}) = \begin{cases} 1 & \text{if } g(\vec{x}) > 0 \\ -1 & \text{otherwise} \end{cases}$$

- เส้น $g(\vec{x}) = 0$ คือ **decision surface (hyperplane)** ใน 2 มิติคือเส้นตรง $x_2 = -\frac{w_1}{w_2}x_1 - \frac{w_0}{w_2}$

## 2. Perceptron Learning Rule

```
Algorithm: Perceptron-Learning-Rule
1. Initialize weights wi
2. UNTIL termination condition DO
     FOR EACH training example DO
       - คำนวณ output o
       - ถ้า o ≠ t ปรับ weight:  wi ← wi + Δwi,  Δwi = α (t − o) xi
```

- $t$ = target, $o$ = output, $\alpha$ = learning rate (เช่น 0.5, 0.01, 0.005)
- ถ้าทายถูก $(t-o) = 0$ → weight ไม่เปลี่ยน
- ทายต่ำเกินไป → เพิ่ม weight ของ input ที่เป็นบวก, ทายสูงเกินไป → ลด
- **1 epoch** = ใช้ training example ครบทุกตัว 1 รอบ หยุดเมื่อไม่มี error ทั้ง epoch

## 3. AND vs XOR

ตัวอย่าง AND ($\alpha = 0.5$, weight เริ่มต้น 0.1 ทั้งหมด) เรียนรู้จนได้ $w_0 = -1.4,\ w_1 = 1.1,\ w_2 = 0.6$ และไม่มี error

| x1 | x2 | AND | XOR |
|:-:|:-:|:-:|:-:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 0 |

- **AND** เป็น **linearly separable** → perceptron เรียนรู้ได้
- **XOR** เป็น **linearly non-separable** → weight วนซ้ำไม่มีวันลู่เข้า perceptron ชั้นเดียวแก้ไม่ได้ ต้องใช้ **multilayer network + back-propagation**

## 4. Delta Rule (Gradient Descent)

เมื่อข้อมูลแบ่งด้วยเส้นตรงไม่ได้ perceptron rule จะไม่ลู่เข้า **Delta rule** แก้โดยใช้ unit เชิงเส้น $o(\vec{x}) = \vec{w}\cdot\vec{x}$ (ไม่ผ่าน threshold) แล้วลด error

$$E(\vec{w}) = \frac{1}{2}\sum_{d \in D}(t_d - o_d)^2$$

ปรับ weight ในทิศตรงข้ามกับ gradient

$$\Delta w_i = -\alpha \frac{\partial E}{\partial w_i} = \alpha \sum_{d \in D}(t_d - o_d)\,x_{id}$$

- error surface เป็นรูปชาม (parabolic) จึงมีจุดต่ำสุดเดียว
- เป็นพื้นฐานของ **back-propagation** ใน neural network หลายชั้น

### โน้ตในคาบ

- คำที่เกี่ยวข้องที่ต่อยอดได้: loss, epoch, batch / mini-batch, learning rate, dropout, batch normalization, overfitting
- ค่า weight เริ่มต้นและ learning rate มีผลต่อจำนวน epoch ที่ใช้ลู่เข้า

## 5. ตัวอย่างโค้ด

```python
import numpy as np

def train_perceptron(X, t, alpha=0.5, w=None, max_epoch=100, thr=0.0):
    X = np.hstack([np.ones((len(X), 1)), X])        # bias x0 = 1
    w = np.full(X.shape[1], 0.1) if w is None else w
    for epoch in range(max_epoch):
        errors = 0
        for xi, ti in zip(X, t):
            o = int(xi @ w > thr)
            w = w + alpha * (ti - o) * xi
            errors += ti != o
        if errors == 0:
            return w, epoch + 1
    return w, None                                   # ไม่ลู่เข้า (เช่น XOR)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print(train_perceptron(X, [0, 0, 0, 1]))  # AND → ลู่เข้า
print(train_perceptron(X, [0, 1, 1, 0]))  # XOR → None
```
