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
| เส้นรอบเอว (Waist Circumference) | เปิดผ่าน toggle "ฉันมีสายวัด..." แล้วปรับ slider ได้ 50–160 ซม. (ค่าเริ่มต้น 80 ซม.) | *ทางเลือกเสริม* | หากเปิดใช้ ระบบจะสลับไปใช้โมเดลชุด `with_waist` ทั้ง Body Fat และ NCDs (Dual-Route) เพื่อความแม่นยำที่สูงขึ้น |
| Face Guard (เปิด/ปิดการตรวจสอบใบหน้า) | toggle เปิด/ปิดได้ในเมนู "ตัวเลือกขั้นสูง" (ค่าเริ่มต้น: เปิด) | *ทางเลือกเสริม* | ถ้าปิด ภาพจะถูกส่งเข้า Stage 1 โดยไม่ผ่านการตรวจสอบใบหน้าก่อน |

### ผลลัพธ์ที่ได้จากการวิเคราะห์ (Expected Outputs)

| ขั้นตอน | ผลลัพธ์ (Output) | รูปแบบการแสดงผล |
|---|---|---|
| Stage 1 (ViT) | ค่าดัชนีมวลกาย (BMI) | ตัวเลขทศนิยม พร้อมจัดหมวด 4 ระดับด้วยเกณฑ์ BMI แบบเอเชีย (ไม่ใช่เกณฑ์ WHO 25/30): น้ำหนักต่ำกว่าเกณฑ์ (<18.5) / สมส่วน (18.5–23) / ท้วม (23–25) / อ้วน (≥25) |
| Stage 1.5 (XGBoost + Lifestyle Calibration) | เปอร์เซ็นต์ไขมันในร่างกาย (Total Body Fat %) | แสดงทั้งค่าดิบก่อนปรับและค่าหลังปรับตามไลฟ์สไตล์ พร้อมจัดหมวดตามเกณฑ์ที่ต่างกันระหว่างชาย/หญิง: กล้ามเนื้อชัด / หุ่นฟิต / ทั่วไป / อ้วน |
| Stage 2 (Classifiers) | ความเสี่ยงโรคเบาหวาน (Diabetes Risk) | ค่าความน่าจะเป็นแปลงเป็นเปอร์เซ็นต์ 0–100% (ทศนิยม 1 ตำแหน่ง) จัดระดับ 4 ขั้น: ต่ำ (<40%) / เฝ้าระวัง (40–49%) / สูง (50–74%) / สูงมาก (≥75%) |
| Stage 2 (Classifiers) | ความเสี่ยงโรคความดันโลหิตสูง (Hypertension Risk) | รูปแบบเดียวกับความเสี่ยงโรคเบาหวานด้านบน |
| Export Engine | เอกสารรายงานผลสุขภาพ (PDF) | ค่าเริ่มต้นเป็นภาษาไทย (ใช้ฟอนต์ Sarabun) แต่ถ้าเครื่องไม่พบไฟล์ฟอนต์ ระบบจะสลับไปออกรายงานเป็นภาษาอังกฤษให้อัตโนมัติ |

> ตารางนี้ตรวจสอบและยืนยันตรงกับซอร์สโค้ดจริงแล้ว (`app.py`, `predict_pipeline.py`, `face_guard.py`)

---

## 🔄 สถาปัตยกรรมและกระบวนการทำงาน (Pipeline Architecture)

ระบบทำงานประสานกันผ่าน Pipeline 3 ลำดับขั้นแบบ **Adaptive Dual-Route** (ปรับเส้นทางตามการมีหรือไม่มีข้อมูลรอบเอวอัตโนมัติ):

```text
               [ รูปถ่ายใบหน้า (Face Image) ]
                             │
                             ▼
                [ Face Guard Validation ]
              (OpenCV / MediaPipe ตรวจจับหน้า)
                             │
                             ▼
         [ Stage 1: Vision Transformer (ViT-H/14) ]
                             │
                             ▼
                    [ Predicted BMI ]
                             │
      ┌──────────────────────┴──────────────────────┐
      │  Demographics: อายุ (Age), เพศ (Gender)     │
      │  Optional: รอบเอว (Waist Circumference)     │
      │  Lifestyle: รูปแบบการใช้ชีวิต               │
      └──────────────────────┬──────────────────────┘
                             │
                             ▼
        [ Stage 1.5: Total Body Fat % Estimation ]
             (XGBoost Regressor + Dual Route)
                             │
                             ▼
               [ Lifestyle Calibration Engine ]
        (Active: -10% | Sedentary: +5% | Normal: 0%)
                             │
                             ▼
             [ Stage 2: NCDs Risk Screening ]
      (XGBoost Classifiers: เบาหวาน & ความดันโลหิตสูง)
                             │
                             ▼
         [ Interactive Dashboard & PDF Health Report ]
```

---

## ⚙️ การติดตั้งและวิธีใช้งาน (Installation & Usage)

ระบบพัฒนาบน Python 3.10 รองรับการประมวลผลแบบเร่งความเร็วผ่าน GPU (NVIDIA CUDA 11.8 หรือเทียบเท่า)

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

### 2. การจัดเตรียมไฟล์น้ำหนักโมเดล (Model Weights)

ตรวจสอบให้แน่ใจว่ามีไฟล์โมเดลครบถ้วนในโฟลเดอร์ `weights/`:

- `vit_head_split_v2.pt` (โมเดล ViT สำหรับทำนาย BMI)
ลิงค์โหลด: https://drive.google.com/file/d/1CKdD7CUrFYImvhA-I1r0-SUvxbZOaZsU/view?usp=sharing
- `xgboost_bodyfat_with_waist.pkl` & `xgboost_bodyfat_no_waist.pkl`
- `xgb_classifier_diabetes.pkl` & `xgb_classifier_diabetes_no_waist.pkl`
- `xgb_classifier_hypertension.pkl` & `xgb_classifier_hypertension_no_waist.pkl`

### 3. การรันแอปพลิเคชัน (Launch UI)

เริ่มต้นรันหน้าต่างเว็บแอปพลิเคชันด้วย Streamlit:

```bash
cd scripts
streamlit run app.py
```

ขั้นตอนการใช้งานสำหรับผู้ใช้:

1. อัปโหลดภาพถ่ายใบหน้าตรง หรือถ่ายผ่านเว็บแคมได้เลย (ระบบมี Face Guard ตรวจสอบภาพก่อนเสมอ — บล็อกทันทีถ้าไม่พบใบหน้า / พบหลายใบหน้า / ใบหน้าเล็กเกินไป / ภาพเบลอหนักมาก ส่วนกรณีภาพมืด สว่างเกิน หรือคอนทราสต์ต่ำ ระบบจะแค่เตือนแต่ไม่บล็อก สามารถปิดการตรวจสอบนี้ได้ในเมนู "ตัวเลือกขั้นสูง")
2. ระบุอายุและเพศ
3. เลือกรูปแบบไลฟ์สไตล์ (ออกกำลังกายเป็นประจำ, กิจกรรมปานกลาง, หรือแทบไม่ออกกำลังกาย) เพื่อให้ระบบปรับความแม่นยำของเปอร์เซ็นต์ไขมัน
4. (ทางเลือกเสริม) เปิดใส่ค่ารอบเอว (50–160 ซม.)
5. กดวิเคราะห์ผลสุขภาพและสามารถคลิกดาวน์โหลดรายงานผลเป็นเอกสาร PDF ภาษาไทยได้ทันที

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

หากต้องการเทรนโมเดลใหม่ทั้งหมดตามขั้นตอน:

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

---

## 🎉 กิตติกรรมประกาศและเครดิต (Acknowledgements)

โครงสร้างพื้นฐานในการทำนายค่า BMI จากใบหน้า ถูกนำมาและดัดแปลงจากงานวิจัย Open Source ของคุณ Liujie Zheng ภายใต้ลิขสิทธิ์ MIT License:

- **Original Repository:** `liujie-zheng/face-to-bmi-vit`
- **Original License Notice:** Copyright (c) 2023 Liujie Zheng

คณะผู้จัดทำขอขอบคุณสำหรับแนวทางสถาปัตยกรรมและชุดข้อมูลใบหน้านี้ ซึ่งนำมาพัฒนาเชื่อมโยงเข้ากับฐานข้อมูล NHANES จนเกิดเป็นระบบประเมินความเสี่ยงสุขภาพแบบครบวงจร