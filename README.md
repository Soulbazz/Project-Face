# Face to BMI, Body Fat & NCDs Risk Analysis

โปรเจกต์นี้เป็นการพัฒนาระบบ AI สำหรับคัดกรองและประเมินสุขภาพแบบหลายขั้นตอน (Multi-Stage Health Screening Pipeline) ประกอบด้วย 3 ส่วนหลัก:
1. **ประเมินค่าดัชนีมวลกาย (BMI) จากรูปภาพใบหน้า:** สกัดคุณลักษณะทางกายภาพจากภาพเซลฟี่หน้าตรงด้วยโมเดล Deep Learning (Vision Transformer)
2. **ประเมินเปอร์เซ็นต์ไขมันในร่างกาย (Total Body Fat %):** ทำนายด้วย Machine Learning (XGBoost) พร้อมระบบ **Lifestyle Calibration** ที่ปรับแก้ค่าตามพฤติกรรมการใช้ชีวิตจริงของผู้ใช้
3. **คัดกรองความเสี่ยงโรคเรื้อรัง (NCDs):** ประเมินความน่าจะเป็นของโรคเบาหวาน (Type 2 Diabetes) และโรคความดันโลหิตสูง (Hypertension) อ้างอิงเกณฑ์ประชากรจากฐานข้อมูลสุขภาพแห่งชาติสหรัฐฯ (NHANES)

---

## 🎯 ข้อมูลนำเข้าและผลลัพธ์ที่คาดหวัง (Expected Inputs & Outputs)

### ข้อมูลนำเข้า (System Inputs)

| ข้อมูล (Input) | รูปแบบจริงในระบบ | ความจำเป็น | คำอธิบาย |
|---|---|---|---|
| ภาพถ่ายใบหน้า (Face Image) | อัปโหลดไฟล์ `.jpg` `.jpeg` `.png` `.webp` หรือถ่ายผ่านเว็บแคมในหน้าเว็บได้เลย | **จำเป็น** | ตรวจสอบผ่าน Face Guard ทันทีที่มีภาพ ก่อนจะกดปุ่มวิเคราะห์ได้ |
| อายุ (Age) | ตัวเลขเต็ม หน่วยปี เลือกในหน้าเว็บได้ช่วง 12–99 ปี (ค่าเริ่มต้น 30) | **จำเป็น** | โค้ด Backend ยอมรับกว้างกว่านั้นคือ 10–100 ปี แต่ตัวเลือกบนหน้าเว็บจำกัดไว้แคบกว่า |
| เพศ (Gender) | เลือก "ชาย" หรือ "หญิง" จากปุ่มตัวเลือก (radio) | **จำเป็น** | ระบบแปลงเป็นค่าภายใน male/female เพื่อใช้แยกเกณฑ์ Body Fat % และหมวด BMI |
| รูปแบบการใช้ชีวิต (Lifestyle) | เลือก 1 ใน 3: 🏋️ ออกกำลังกายเป็นประจำ (Active) / 🚶 กิจกรรมปานกลาง (Normal — ค่าเริ่มต้น) / 💺 แทบไม่ออกกำลังกาย (Sedentary) | **จำเป็น** | ใช้ปรับ Body Fat % ผ่าน Lifestyle Calibration (Active -10% / Sedentary +5% / Normal ไม่ปรับ) โดยมีเพดานล่างไม่ต่ำกว่า 5% เสมอ (ไขมันจำเป็นขั้นต่ำของร่างกาย) |
| เส้นรอบเอว (Waist Circumference) | เปิดผ่าน toggle "ฉันมีสายวัด..." แล้วปรับ slider ได้ 50–160 ซม. (ค่าเริ่มต้น 80 ซม.) | *ทางเลือกเสริม* | หากเปิดใช้ ระบบจะสลับไปใช้โมเดลชุด `with_waist` ทั้ง Body Fat และ NCDs (Dual-Route) |
| Face Guard (เปิด/ปิดการตรวจสอบใบหน้า) | toggle เปิด/ปิดได้ในเมนู "ตัวเลือกขั้นสูง" (ค่าเริ่มต้น: เปิด) | *ทางเลือกเสริม* | ถ้าปิด ภาพจะถูกส่งเข้า Stage 1 โดยไม่ผ่านการตรวจสอบใบหน้าก่อน |

### ผลลัพธ์ที่ได้จากการวิเคราะห์ (Expected Outputs)

| ขั้นตอน | ผลลัพธ์ (Output) | รูปแบบการแสดงผล |
|---|---|---|
| Stage 1 (ViT) | ค่าดัชนีมวลกาย (BMI) | ตัวเลขทศนิยม พร้อมจัดหมวด 4 ระดับด้วยเกณฑ์ BMI แบบเอเชีย (ไม่ใช่เกณฑ์ WHO 25/30): น้ำหนักต่ำกว่าเกณฑ์ (<18.5) / สมส่วน (18.5–23) / ท้วม (23–25) / อ้วน (≥25) |
| Stage 1.5 (XGBoost + Lifestyle Calibration) | เปอร์เซ็นต์ไขมันในร่างกาย (Total Body Fat %) | แสดงทั้งค่าดิบก่อนปรับและค่าหลังปรับตามไลฟ์สไตล์ พร้อมจัดหมวดตามเกณฑ์ที่ต่างกันระหว่างชาย/หญิง: กล้ามเนื้อชัด / หุ่นฟิต / ทั่วไป / อ้วน |
| Stage 2 (Classifiers) | ความเสี่ยงโรคเบาหวาน (Diabetes Risk) | ค่าความน่าจะเป็นแปลงเป็นเปอร์เซ็นต์ 0–100% (ทศนิยม 1 ตำแหน่ง) จัดระดับ 4 ขั้น: ต่ำ (<40%) / เฝ้าระวัง (40–49%) / สูง (50–74%) / สูงมาก (≥75%) |
| Stage 2 (Classifiers) | ความเสี่ยงโรคความดันโลหิตสูง (Hypertension Risk) | รูปแบบเดียวกับความเสี่ยงโรคเบาหวานด้านบน |
| Export Engine | เอกสารรายงานผลสุขภาพ (PDF) | ค่าเริ่มต้นเป็นภาษาไทย (ใช้ฟอนต์ Sarabun) แต่ถ้าเครื่องไม่พบไฟล์ฟอนต์ ระบบจะสลับไปออกรายงานเป็นภาษาอังกฤษให้อัตโนมัติ |
| Stage 1 (ViT + MC Dropout) | ค่าดัชนีมวลกาย (BMI) & Epistemic Uncertainty | ตัวเลขทศนิยม พร้อมค่าความไม่แน่นอน ($\pm\text{SD}$) จัดหมวด 4 ระดับเกณฑ์เอเชีย: น้ำหนักต่ำกว่าเกณฑ์ (<18.5) / สมส่วน (18.5–23) / ท้วม (23–25) / อ้วน (≥25) โดยมีระบบ Rejection หาก $\sigma > 1.8$ |
| Stage 1.5 (XGBoost Regressor) | เปอร์เซ็นต์ไขมันในร่างกาย (Total Body Fat %) | ทำนายมวลไขมันรวมโดยตรงจากโมเดล Dual-Route ที่ผสานฟีเจอร์พฤติกรรม (`LIFESTYLE_LEVEL`) จัดหมวดตามเพศสรีรวิทยา: กล้ามเนื้อชัด / หุ่นฟิต / ทั่วไป / อ้วน |
| Stage 2 (Monotonic Calibrated XGBoost) | คัดกรองความเสี่ยงโรคเบาหวาน (Type 2 Diabetes Risk) | ความน่าจะเป็นหลังสอบเทียบพลาตต์ (Platt Calibrated Posterior Probability $P(Y=1|X)$) จัดกลุ่มตาม **4-Tier Clinical Triage**: ความเสี่ยงต่ำ (<4.5%) / เฝ้าระวัง (4.5–6.1%) / คัดกรองเป็นบวก (≥6.1% F2 Cutoff) / เสี่ยงสูงเร่งด่วน (≥12.0%) |
| Stage 2 (Monotonic Calibrated XGBoost) | คัดกรองความเสี่ยงโรคความดันโลหิตสูง (Hypertension Risk) | ความน่าจะเป็นหลังสอบเทียบ จัดกลุ่มตาม **4-Tier Clinical Triage**: ความเสี่ยงต่ำ (<20.0%) / เฝ้าระวัง (20.0–33.0%) / คัดกรองเป็นบวก (≥33.0% Youden Cutoff / ≥15.8% F2 Cutoff) / เสี่ยงสูงเร่งด่วน (≥55.0%) |
| Export Engine | เอกสารรายงานผลสุขภาพ (PDF / DOCX) | รายงานผลสรุปทางการแพทย์พร้อมคำแนะนำการตรวจยืนยันทางคลินิก (Confirmatory Phlebotomy / Sphygmomanometry) รองรับทั้งภาษาไทยและอังกฤษ |

> ตารางนี้ตรวจสอบและยืนยันตรงกับซอร์สโค้ดจริงแล้ว (`app.py`, `predict_pipeline.py`, `face_guard.py`)

---

## 🔄 สถาปัตยกรรมและกระบวนการทำงาน (Pipeline Architecture)

ระบบทำงานประสานกันผ่าน Pipeline 3 ลำดับขั้นแบบ **Adaptive Dual-Route** (ปรับเส้นทางตามการมีหรือไม่มีข้อมูลรอบเอวอัตโนมัติ):
ระบบทำงานประสานกันผ่าน Pipeline ลำดับขั้นแบบ **Cascading Multimodal Architecture** พร้อม **Two-Tier Safety Guard** และ **Adaptive Dual-Route**:

```text
               [ รูปถ่ายใบหน้า (Face Image) ]
                             │
                             ▼
                [ Face Guard Validation ]
              (OpenCV / MediaPipe ตรวจจับหน้า)
              [ Safety Layer 1: Face Guard ]
       (ตรวจจับใบหน้า, มุมเอียง |Roll| ≤ 15°, ความสว่าง)
                             │
                             ▼
         [ Stage 1: Vision Transformer (ViT-H/14) ]
         [ Stage 1: Vision Transformer (ViT-B/16) ]
      (Monte Carlo Dropout n=25-30 → Point Estimate + SD)
                             │
                             ▼
                    [ Predicted BMI ]
         [ Safety Layer 2: Epistemic Rejection ]
              (หาก SD > 1.8 → ปฏิเสธการประมวลผล)
                             │
                             ▼
                   [ Predicted BMI (ŷ₁) ]
                             │
      ┌──────────────────────┴──────────────────────┐
      │  Demographics: อายุ (Age), เพศ (Gender)     │
      │  Optional: รอบเอว (Waist Circumference)     │
      │  Lifestyle: รูปแบบการใช้ชีวิต               │
      │  Lifestyle: ระดับกิจกรรม (Sedentary/Norm/Act)│
      └──────────────────────┬──────────────────────┘
                             │
                             ▼
        [ Stage 1.5: Total Body Fat % Estimation ]
             (XGBoost Regressor + Dual Route)
                             │
                             ▼
               [ Lifestyle Calibration Engine ]
        (Active: -10% | Sedentary: +5% | Normal: 0%)
                 [ Predicted Body Fat (ŷ₂) ]
                             │
      ┌──────────────────────┴──────────────────────┐
      │  Biological Sex Error Shock Absorber (α=0.39)│
      │  Monotonicity Constraints (Non-decreasing)   │
      └──────────────────────┬──────────────────────┘
                             │
                             ▼
             [ Stage 2: NCDs Risk Screening ]
      (XGBoost Classifiers: เบาหวาน & ความดันโลหิตสูง)
      (Monotonic XGBoost + Platt Calibration CalibratedClassifierCV)
                             │
                             ▼
           [ 4-Tier Clinical Triage & F2 Cutoffs ]
     (Diabetes τ* = 0.061 | Hypertension τ* = 0.330 / 0.158)
                             │
                             ▼
         [ Interactive Dashboard & PDF Health Report ]
```

---

## 🔬 นวัตกรรมอัลกอริทึมและการปรับปรุงใน Stage 2 (Algorithmic Innovations)

### 1. Monotonicity Constraints & Biological Sex Error Shock Absorber ($\alpha = 0.39$)
* **Monotonicity Constraints ($+1$):** ในการทำนายโรค NCDs หากปล่อยให้ Decision Trees เรียนรู้อย่างอิสระ อาจเกิดสภาวะผิดปกติทางสรีรวิทยา (Physiological Paradox) เช่น ผู้ที่มีอายุหรือมวลไขมันสูงขึ้น แต่ความเสี่ยงที่ทำนายได้กลับลดลง เราจึงกำหนดข้อจำกัด `monotone_constraints` เป็น $+1$ สำหรับตัวแปร `AGE`, `BMI`, `WAIST_CM`, และ `TOTAL_BODY_FAT_PCT` เพื่อการันตีว่าฟังก์ชันความเสี่ยงจะเป็นแบบไม่ลดลง (Monotonically Non-Decreasing) เสมอ
* **Biological Sex Shock Absorber Effect:** ตามธรรมชาติ เพศหญิงจะมีเปอร์เซ็นต์ไขมันในร่างกาย (Total Body Fat %) สูงกว่าเพศชายประมาณ $10\%$ เพื่อทำหน้าที่ปกป้องระบบสืบพันธุ์ แต่มีอัตราความเสี่ยงพื้นฐานต่อโรคระบบหัวใจและหลอดเลือดต่ำกว่าเพศชาย โมเดล XGBoost ได้เรียนรู้ค่าน้ำหนักเพศ $w_{\text{GENDER}} = -0.39$ ซึ่งทำหน้าที่เป็น "โช้คอัพชีววิทยา" คอยตัดทอนและดูดซับความคลาดเคลื่อน (Error Propagation) ที่ส่งต่อมาจาก Stage 1.5 ทำให้ความเสี่ยงปลายทางของโรค NCDs ไม่ถูกดันให้สูงเกินจริง

### 2. The 0-Recall Paradox & การแก้ปัญหาด้วย Platt Scaling + $F_2$-Score Threshold Optimization
* **0-Recall Paradox:** ในงานคัดกรองโรคเรื้อรังที่มีความชุกต่ำ (เช่น เบาหวานในประชากรทั่วไปมีประมาณ $\sim 10\%$) หากใช้เกณฑ์ตัดสินมาตรฐานทั่วไปที่ $\tau = 0.50$ โมเดลจะทำนายทุกคนเป็นผลลบทั้งหมด ส่งผลให้ **Sensitivity (Recall) กลายเป็น 0.0%** (พลาดคนไข้ที่เป็นโรคเบาหวานจริงไปทั้งหมด 55 ราย)
* **Platt Calibration (`CalibratedClassifierCV`):** เราประยุกต์ใช้การสอบเทียบ Sigmoid Fitting บน 5-Fold Cross-Validation เพื่อแปลงคะแนน Logit ดิบของโมเดลให้เป็นความน่าจะเป็นเชิงสถิติจริง (True Posterior Probabilities $P(Y=1|X)$) ส่งผลให้ความแม่นยำในการทำนายความน่าจะเป็น (Brier Score) อยู่ในเกณฑ์ดีเยี่ยม ($0.0881$)
* **$F_2$-Score Screening Threshold ($\tau^* = 0.061$):** ในบริบทงานคัดกรองสุขภาพเบื้องต้น การปล่อยผู้ป่วยหลุดรอด (False Negative) ก่อให้เกิดอันตรายถึงชีวิตและค่ารักษาพยาบาลระยะยาวสูงกว่าการส่งคนปกติไปเจาะเลือดตรวจยืนยัน (False Positive) อย่างมหาศาล ระบบจึงใช้เกณฑ์ $F_\beta$ ($\beta=2.0$ ให้น้ำหนัก Recall สูงกว่า Precision $2\times$) ในการค้นหา Threshold บน Validation Set (Train 70% / Val 15% / Test 15%) ปราศจากปัญหาข้อมูลรั่วไหล (Zero Test Leakage) ทำให้ค่า Threshold ปรับลดลงมาอยู่ที่ $\tau^* = 0.061$ **ช่วยกอบกู้ผู้ป่วยเบาหวานกลับมาได้ 49 ราย (Sensitivity เพิ่มขึ้นจาก 0.0% เป็น 89.1%)**

---

## 🩺 เกณฑ์การคัดกรองและแบ่งระดับความเสี่ยงทางคลินิก (4-Tier Clinical Triage Strata)

ความน่าจะเป็นที่ผ่านการสอบเทียบ (Calibrated Probabilities) จะถูกแปลงเป็นระดับการดูแลรักษาทางคลินิก 4 ระดับ (บันทึกและโหลดแบบไดนามิกจาก `weights/thresholds.json`):

| ระดับความเสี่ยง (Triage Strata) | รหัสสี & สัญลักษณ์ | เกณฑ์ความน่าจะเป็น เบาหวาน (T2DM) | เกณฑ์ความน่าจะเป็น ความดันโลหิตสูง (HTN) | คำแนะนำและการปฏิบัติทางคลินิก (Clinical Action) |
|---|:---:|:---:|:---:|---|
| **Low Risk (ความเสี่ยงต่ำ)** | 🟢 เขียว | $p < 4.5\%$ | $p < 20.0\%$ | ความเสี่ยงอยู่ในเกณฑ์มาตรฐานประชากรทั่วไป แนะนำตรวจสุขภาพประจำปีสม่ำเสมอและรักษาพฤติกรรมสุขภาพที่ดี |
| **Watchful (เฝ้าระวัง)** | 🟡 เหลือง | $4.5\% \le p < 6.1\%$ | $20.0\% \le p < 33.0\%$ | พบแนวโน้มความเสี่ยงระยะเริ่มต้น แนะนำปรับเปลี่ยนพฤติกรรม ลดหวาน-มัน-เค็ม เพิ่มการออกกำลังกาย และคัดกรองซ้ำใน 6–12 เดือน |
| **Screen Positive (คัดกรองเป็นบวก)** | 🟠 ส้ม | $p \ge 6.1\%$ ($F_2$ Cutoff) | $p \ge 33.0\%$ (Youden) / $\ge 15.8\%$ ($F_2$) | ผลคัดกรองเบื้องต้นเป็นบวก แนะนำเข้ารับการตรวจยืนยันด้วยผลเลือด (FPG หรือ HbA1c) หรือตรวจวัดความดันซ้ำด้วยเครื่องมาตรฐานทางการแพทย์ |
| **Urgent Risk (ความเสี่ยงสูงมาก)** | 🔴 แดง | $p \ge 12.0\%$ | $p \ge 55.0\%$ | ระดับความเสี่ยงสูงมากอย่างมีนัยสำคัญทางคลินิก แนะนำพบแพทย์ที่สถานพยาบาลโดยด่วนเพื่อประเมินภาวะแทรกซ้อนและรับการรักษาทันที |

---

## ⚙️ การติดตั้งและวิธีใช้งาน (Installation & Usage)

ระบบพัฒนาบน Python 3.10 รองรับการประมวลผลแบบเร่งความเร็วผ่าน GPU (NVIDIA CUDA 11.8 หรือเทียบเท่า) และ CPU อัตโนมัติ

### 1. การติดตั้งสภาพแวดล้อม (Environment Setup)

Clone โครงการและเข้าสู่โฟลเดอร์:

```bash
git clone https://github.com/Soulbazz/Project-Face.git
cd Project-Face
```

สร้างและเปิดใช้งาน Conda Environment:

```bash
conda env create -f environment.yml
conda activate face2bmi
```

(หรือติดตั้งผ่าน pip: `pip install -r requirements.txt`)

สำหรับผู้ที่มี Environment อยู่แล้ว และต้องการอัปเดต Library ให้ตรงกัน:
```bash
conda env update -f environment.yml --prune
```

### 2. การจัดเตรียมไฟล์น้ำหนักโมเดล (Model Weights)

ตรวจสอบให้แน่ใจว่ามีไฟล์โมเดลครบถ้วนในโฟลเดอร์ `weights/`:

- `vit_head_split_v2.pt` (โมเดล ViT สำหรับทำนาย BMI)
ลิงค์โหลด: https://drive.google.com/file/d/1CKdD7CUrFYImvhA-I1r0-SUvxbZOaZsU/view?usp=sharing
- `xgboost_bodyfat_with_waist.pkl` & `xgboost_bodyfat_no_waist.pkl`
- `xgb_classifier_diabetes.pkl` & `xgb_classifier_diabetes_no_waist.pkl`
- `xgb_classifier_hypertension.pkl` & `xgb_classifier_hypertension_no_waist.pkl`
- `vit_bmi_model.pt` (โมเดล ViT สำหรับทำนาย BMI จากภาพใบหน้า)
- `xgboost_bodyfat_with_waist.pkl` & `xgboost_bodyfat_no_waist.pkl` (โมเดลทำนาย Body Fat %)
- `xgb_classifier_diabetes.pkl` & `xgb_classifier_diabetes_no_waist.pkl` (โมเดล Calibrated Monotonic XGBoost คัดกรองเบาหวาน)
- `xgb_classifier_hypertension.pkl` & `xgb_classifier_hypertension_no_waist.pkl` (โมเดล Calibrated Monotonic XGBoost คัดกรองความดันสูง)
- `thresholds.json` (ไฟล์บันทึกค่า Cutoff, เมทริกซ์การทดสอบ, และเกณฑ์ Triage ทางคลินิก)

### 3. การรันแอปพลิเคชัน (Launch UI)
---

วิธีที่ 1: ดับเบิลคลิกไฟล์ Launcher (แนะนำสำหรับ Windows)
## 🚀 ลำดับการรันโปรเจกต์และการทดสอบ (Execution & Reproducibility Workflow)

ดับเบิลคลิกที่ไฟล์ run_app.bat ที่อยู่โฟลเดอร์หลักของโปรเจกต์ ระบบจะเปิดใช้งาน Environment และรันเว็บแอปพลิเคชันให้อัตโนมัติ
เพื่อให้การทำงานเป็นไปอย่างถูกต้องและสามารถตรวจสอบซ้ำได้ (Reproducible) โปรดรันสคริปต์ตามลำดับดังนี้:

วิธีที่ 2: รันผ่าน Command Line
เริ่มต้นรันหน้าต่างเว็บแอปพลิเคชันด้วย Streamlit:

```bash
conda activate face2bmi
cd scripts
streamlit run app.py
```
# 1. เทรนโมเดล Stage 2 Calibrated Monotonic XGBoost พร้อมค้นหาเกณฑ์ F2 Cutoff และส่งออก thresholds.json
python scripts/stage2_train_optimized.py

ขั้นตอนการใช้งานสำหรับผู้ใช้:
# 2. คอมไพล์รายงานวิชาการ IEEE Conference Paper เป็นไฟล์ Word (.docx), PDF (.pdf), และ Markdown (.md)
python scripts/generate_academic_paper_docx.py

1. อัปโหลดภาพถ่ายใบหน้าตรง หรือถ่ายผ่านเว็บแคมได้เลย (ระบบมี Face Guard ตรวจสอบภาพก่อนเสมอ — บล็อกทันทีถ้าไม่พบใบหน้า / พบหลายใบหน้า / ใบหน้าเล็กเกินไป / ภาพเบลอหนักมาก ส่วนกรณีภาพมืด สว่างเกิน หรือคอนทราสต์ต่ำ ระบบจะแค่เตือนแต่ไม่บล็อก สามารถปิดการตรวจสอบนี้ได้ในเมนู "ตัวเลือกขั้นสูง")
2. ระบุอายุและเพศ
3. เลือกรูปแบบไลฟ์สไตล์ (ออกกำลังกายเป็นประจำ, กิจกรรมปานกลาง, หรือแทบไม่ออกกำลังกาย) เพื่อให้ระบบปรับความแม่นยำของเปอร์เซ็นต์ไขมัน
4. (ทางเลือกเสริม) เปิดใส่ค่ารอบเอว (50–160 ซม.)
5. กดวิเคราะห์ผลสุขภาพและสามารถคลิกดาวน์โหลดรายงานผลเป็นเอกสาร PDF ภาษาไทยได้ทันที
# 3. รันประมวลผล Pipeline คัดกรองสุขภาพเบื้องต้นผ่าน CLI พร้อมรัน Sanity Check
python scripts/predict_pipeline.py

# 4. เปิดใช้งานหน้าเว็บแอปพลิเคชันแบบอินเตอร์แอคทีฟ (Streamlit Dashboard)
streamlit run scripts/app.py
```

---

## 📂 โครงสร้างโปรเจกต์และหน้าที่ของแต่ละไฟล์ (Repository & Scripts)

```
Project Face/
├── assets/fonts/              # ฟอนต์ Sarabun สำหรับสร้างรายงานภาษาไทย
├── data/                      # โฟลเดอร์ชุดข้อมูล
│   ├── raw/                   # ข้อมูลดิบจาก NHANES (.xpt)
│   ├── processed/             # ข้อมูลตารางที่คลีนแล้ว (.csv)
│   ├── Images/                # รูปภาพในหน้าที่รับเทรน ViT (.bmp)
│   ├── test_images/           # รูปภาพสำหรับทดสอบระบบ
│   ├── split_fixed.csv        # ข้อมูลแบ่งชุด Train/Val/Test (v1)
│   └── split_fixed_v2.csv     # ข้อมูลแบ่งชุด Train/Val/Test สำหรับโมเดล Production (v2)
├── assets/fonts/                    # ฟอนต์ Sarabun สำหรับสร้างรายงานภาษาไทย
├── data/                            # โฟลเดอร์ชุดข้อมูล
│   ├── raw/                         # ข้อมูลดิบจาก NHANES (.xpt)
│   ├── processed/                   # ข้อมูลตารางที่คลีนแล้ว (.csv)
│   ├── test_images/                 # รูปภาพสำหรับทดสอบระบบ
│   └── split_fixed_v2.csv           # ข้อมูลแบ่งชุด Train/Val/Test
├── reports/                         # รายงานผลงานวิจัยฉบับสมบูรณ์
│   ├── Face2Health_IEEE_Conference_Paper.docx # เอกสารเปเปอร์รูปแบบ IEEE 2-Column Word
│   ├── Face2Health_Academic_Paper_IEEE.pdf    # เอกสารเปเปอร์ฉบับ PDF
│   └── Face2Health_IEEE_FullText.md           # เนื้อหาเปเปอร์ฉบับเต็มภาษาอังกฤษ
├── scripts/
│   ├── app.py                 # Streamlit Web Application (หน้าบ้าน UI)
│   ├── predict_pipeline.py    # ท่อประมวลผลหลักเชื่อมต่อโมเดลทุก Stage (Backend Engine)
│   ├── face_guard.py          # ตัวตรวจจับใบหน้า ป้องกันภาพที่ไม่ใช่ใบหน้าคน
│   ├── pdf_export.py          # ระบบเรนเดอร์และสร้างรายงานผลสุขภาพเป็น PDF
│   ├── models.py              # โครงสร้างสถาปัตยกรรม Vision Transformer (ViT-H/14)
│   ├── loader.py              # Image Transforms, DataLoader และตัวจัดการ Dataset
│   ├── train_vit_bmi.py       # สคริปต์เทรนโมเดล Stage 1 (Face to BMI) Test
│   ├── eval_vit.py            # สคริปต์ประเมินผลความแม่นยำโมเดล ViT บนชุด Test
│   ├── train_xgboost.py       # สคริปต์เทรนโมเดล Stage 1.5 ทำนาย Body Fat %
│   ├── Stage2.py              # สคริปต์เทรนโมเดล Stage 2 ประเมินโรค (แบบใช้รอบเอว)
│   ├── stage2_nowaist.py      # สคริปต์เทรนโมเดล Stage 2 ประเมินโรค (แบบไม่ใช้รอบเอว)
│   ├── prepare_dataset.py     # สคริปต์คลีนและตัดข้อมูลสูญหายของชุดข้อมูล NHANES
│   ├── convert_xpt_to_csv.py  # เครื่องมือแปลงไฟล์ดิบสถาบัน SAS (.xpt) เป็น .csv
│   └── build_dataset.py       # รวมตารางสรุปข้อมูลเข้าเป็น Dataframe ก้อนเดียว
├── weights/                   # โฟลเดอร์จัดเก็บโมเดลเทรนแล้ว (.pt, .pkl)
├── archive/                   # โฟลเดอร์เก็บโมเดลทดลองย้อนหลังและ Benchmark เดิม
├── environment.yml            # รายการ Dependencies สำหรับ Conda Environment
└── requirements.txt           # รายการ Dependencies สำหรับ Pip
│   ├── app.py                       # Streamlit Web Application (หน้าบ้าน UI)
│   ├── predict_pipeline.py          # ท่อประมวลผลหลักเชื่อมต่อโมเดลทุก Stage (Production Inference Engine)
│   ├── stage2_train_optimized.py    # สคริปต์เทรน Calibrated Monotonic XGBoost พร้อม F2 Threshold Tuning
│   ├── generate_academic_paper_docx.py # ตัวสร้างเปเปอร์งานวิจัย IEEE .docx / .pdf อัตโนมัติ
│   ├── face_guard.py                # ระบบ Two-Tier Face Guard ตรวจสอบคุณภาพภาพถ่ายและ Aleatoric noise
│   ├── facial_concepts.py           # สกัดตัวชี้วัดสัดส่วนสรีรวิทยาใบหน้า (LFWR, CJWR, PAR, FWHR)
│   ├── pdf_export.py                # ระบบเรนเดอร์และสร้างรายงานผลสุขภาพผู้ป่วยเป็น PDF
│   ├── models.py                    # โครงสร้างสถาปัตยกรรม Vision Transformer (ViT)
│   ├── loader.py                    # Image Transforms, DataLoader และตัวจัดการ Dataset
│   ├── train_vit_bmi.py             # สคริปต์เทรนโมเดล Stage 1 (Face to BMI)
│   ├── train_xgboost.py             # สคริปต์เทรนโมเดล Stage 1.5 ทำนาย Body Fat %
│   ├── prepare_dataset.py           # สคริปต์คลีนและตัดข้อมูลสูญหายของชุดข้อมูล NHANES
│   ├── convert_xpt_to_csv.py        # เครื่องมือแปลงไฟล์ดิบ SAS (.xpt) เป็น .csv
│   └── build_dataset.py             # รวมตารางสรุปข้อมูลเข้าเป็น Dataframe ก้อนเดียว
├── weights/                         # โฟลเดอร์จัดเก็บโมเดลเทรนแล้ว (.pt, .pkl) และ thresholds.json
├── environment.yml                  # รายการ Dependencies สำหรับ Conda Environment
└── requirements.txt                 # รายการ Dependencies สำหรับ Pip
```

---

## 💾 ข้อมูลที่ใช้ในการพัฒนา (Datasets)

1. **NHANES Dataset (Tabular Data):**
   - ข้อมูลผลตรวจทางสรีรวิทยา การตรวจวัด DEXA Scan (มวลไขมัน) และแบบสอบถามประวัติโรคเรื้อรังจาก CDC สหรัฐฯ นำมาประมวลผลเป็น `nhanes_cleaned_merged_final.csv`

2. **Face Images Dataset (Image Data):**
   - ข้อมูลภาพถ่ายใบหน้าพร้อมฉลากค่า BMI สำหรับเทรนโมเดล Vision Transformer
   - ดาวน์โหลดรูปภาพชุดข้อมูลต้นฉบับได้ที่: `liujie-zheng/face-to-bmi-vit`
   - การติดตั้ง: แตกไฟล์รูปภาพทั้งหมด (.bmp) นำไปวางไว้ในโฟลเดอร์ `data/Images/`

---

## 🧪 การเทรนโมเดลใหม่ตั้งแต่ต้น (Model Retraining)

เข้าไปโหลด data เพิ่มเติมจาก  
https://github.com/liujie-zheng/face-to-bmi-vit/tree/main/data/Images  
https://github.com/liujie-zheng/face-to-bmi-vit/blob/main/data/data.csv

และทำตามขั้นตอน:

```bash
cd scripts

# 1. จัดเตรียมและเชื่อมโยงชุดข้อมูล NHANES
python convert_xpt_to_csv.py
python prepare_dataset.py
python build_dataset.py

# 2. เทรนโมเดล Stage 1: Face to BMI (Vision Transformer)
#    --augmented เป็น flag (store_true) ไม่ต้องใส่ True/False ต่อท้าย
python train_vit_bmi.py --augmented

# 3. เทรนโมเดล Stage 1.5: Body Fat Regressor (XGBoost)
python train_xgboost.py

# 4. เทรนโมเดล Stage 2: NCDs Classifiers (Diabetes & Hypertension)
python Stage2.py
python stage2_nowaist.py
```
---

## ⚠️ ข้อจำกัดความรับผิดชอบทางการแพทย์ (Medical Disclaimer)

ระบบนี้พัฒนาขึ้นเพื่อการวิจัย การศึกษา และเป็นเครื่องมือคัดกรองความเสี่ยงเบื้องต้นเท่านั้น ผลลัพธ์ที่ได้ไม่ใช้การวินิจฉัยโรคทางการแพทย์ ไม่สามารถใช้แทนการตรวจเลือด การวัดมวลกล้ามเนื้อจริง หรือการปรึกษาแพทย์ผู้เชี่ยวชาญได้
ระบบนี้พัฒนาขึ้นเพื่อการวิจัย การศึกษา และเป็นเครื่องมือคัดกรองความเสี่ยงเบื้องต้นเท่านั้น (Opportunistic Screening Tool) ผลลัพธ์ที่ได้ไม่ใช่การวินิจฉัยโรคทางการแพทย์ ไม่สามารถใช้แทนการเจาะตรวจเลือด (Laboratory Phlebotomy) หรือการวัดความดันโลหิตด้วยเครื่องมาตรฐานทางการแพทย์ (Clinical Sphygmomanometry) ได้

---

## 🎉 กิตติกรรมประกาศและเครดิต (Acknowledgements)

โครงสร้างพื้นฐานในการทำนายค่า BMI จากใบหน้า ถูกนำมาและดัดแปลงจากงานวิจัย Open Source ของคุณ Liujie Zheng ภายใต้ลิขสิทธิ์ MIT License:

- **Original Repository:** `liujie-zheng/face-to-bmi-vit`
- **Original License Notice:** Copyright (c) 2023 Liujie Zheng

คณะผู้จัดทำขอขอบคุณสำหรับแนวทางสถาปัตยกรรมและชุดข้อมูลใบหน้านี้ ซึ่งนำมาพัฒนาเชื่อมโยงเข้ากับฐานข้อมูล NHANES จนเกิดเป็นระบบประเมินความเสี่ยงสุขภาพแบบครบวงจร
คณะผู้จัดทำขอขอบคุณสำหรับแนวทางสถาปัตยกรรมและชุดข้อมูลใบหน้านี้ ซึ่งนำมาพัฒนาเชื่อมโยงเข้ากับฐานข้อมูล CDC NHANES จนเกิดเป็นระบบประเมินความเสี่ยงสุขภาพแบบครบวงจร