# Face to BMI, Body Fat & NCDs Risk Analysis

โปรเจกต์นี้เป็นการพัฒนาระบบ AI สำหรับวิเคราะห์และทำนายข้อมูลสุขภาพ แบ่งออกเป็น 3 ส่วนหลัก ได้แก่:

1. **ทำนายค่าดัชนีมวลกาย (BMI) จากรูปภาพใบหน้า** โดยใช้โมเดล Deep Learning (Vision Transformer)
2. **ทำนายเปอร์เซ็นต์ไขมันในร่างกาย (Body Fat %)** โดยใช้โมเดล Machine Learning (XGBoost) จากข้อมูลสัดส่วนร่างกาย พร้อมระบบ Lifestyle Calibration
3. **คัดกรองความเสี่ยงโรคเรื้อรัง (NCDs)** ได้แก่ โรคเบาหวาน และโรคความดันโลหิตสูง โดยใช้ชุดข้อมูล NHANES

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

- **assets/**: รูปภาพสำหรับเอกสารประกอบ (แผนภาพ, ภาพตัวอย่าง) และไฟล์ฟอนต์สำหรับ UI
- **data/**: ข้อมูลสำหรับเทรนโมเดล
  - `raw/`: ข้อมูลต้นฉบับ
  - `processed/`: ข้อมูลแบบตาราง (CSV) ที่ผ่านการทำความสะอาดแล้ว
  - `Images/`: โฟลเดอร์สำหรับใส่รูปภาพใบหน้าเพื่อเทรนโมเดล ViT
  - `test_images/`: รูปภาพตัวอย่างสำหรับทดสอบแอปพลิเคชัน
- **scripts/**: สคริปต์หลักสำหรับการประมวลผลและเทรนโมเดล
  - `app.py`: โค้ดหลักสำหรับรันหน้าเว็บแอปพลิเคชัน (UI)
  - `predict_pipeline.py`: ท่อประมวลผลหลักเชื่อมต่อ AI ทั้ง 3 Stage
  - `models.py` / `loader.py`: โครงสร้างโมเดล ViT และ Data Dataloader
  - `train_vit_bmi.py`: เทรนโมเดล Deep Learning ทาย BMI จากใบหน้า
  - `train_xgboost.py`: เทรนโมเดลทายเปอร์เซ็นต์ไขมัน (Stage 1.5)
  - `Stage2.py` / `stage2_nowaist.py`: เทรนโมเดลคัดกรองโรคเบาหวานและความดันโลหิตสูง
- **weights/**: โฟลเดอร์เก็บไฟล์น้ำหนักโมเดล (.pt, .pkl) ทั้งหมด
- **environment.yml**: ไฟล์ตั้งค่า Conda Environment (สำหรับ Windows/CUDA)

## 💾 ข้อมูลที่ใช้ (Datasets)

โปรเจกต์นี้ใช้ข้อมูล 2 ส่วน:

1. **NHANES Dataset (Tabular Data):** ข้อมูลสำหรับเทรน Body Fat % และความเสี่ยงโรค NCDs (`nhanes_cleaned_merged_final.csv`)
2. **Face Images Dataset (Image Data):** ข้อมูลรูปภาพใบหน้าคนสำหรับเทรนโมเดล BMI
   - 📥 **ดาวน์โหลดรูปภาพชุดข้อมูลได้จากโปรเจกต์ต้นฉบับ:** https://github.com/liujie-zheng/face-to-bmi-vit/tree/main/data/Images
   - **วิธีติดตั้ง:** เมื่อดาวน์โหลดข้อมูลภาพเสร็จแล้ว ให้นำรูปภาพทั้งหมด (เช่นไฟล์ `.bmp`) มาวางไว้ในพาธ `data/Images/` ของโปรเจกต์นี้

## 👏 กิตติกรรมประกาศและเครดิต (Acknowledgements)

โปรเจกต์นี้เป็นการต่อยอดและพัฒนาระบบเพิ่มเติม โดยโครงสร้างหลักในส่วนของการทำนายค่า BMI จากใบหน้า ถูกนำมาและดัดแปลงจากโปรเจกต์ Open Source ของคุณ **Liujie Zheng** ภายใต้ลิขสิทธิ์ MIT License

- **Original Repository:** [liujie-zheng/face-to-bmi-vit](https://github.com/liujie-zheng/face-to-bmi-vit)
- **Original License Notice:** Copyright (c) 2023 Liujie Zheng

ทางผู้จัดทำขอขอบคุณสำหรับโมเดลพื้นฐานและชุดข้อมูล ซึ่งทำให้สามารถนำมาพัฒนาต่อยอดร่วมกับข้อมูล NHANES เพื่อประเมินเปอร์เซ็นต์ไขมันในร่างกาย (Body Fat %) ในโครงงานนี้ได้

## ⚙️ การติดตั้งและวิธีใช้งาน (Installation & Usage)

โปรเจกต์นี้รันบน Python 3.10 และรองรับการประมวลผลผ่าน GPU (NVIDIA CUDA 11.8) แนะนำให้ใช้ Conda ในการจัดการ Environment

### Step 1: การติดตั้ง Environment

Clone repository นี้ลงมาที่เครื่องและเข้าสู่โฟลเดอร์โปรเจกต์:

```bash
git clone https://github.com/Soulbazz/Project-Face.git
cd Project-Face
```

สร้างและเปิดใช้งาน Conda Environment:

```bash
conda env create -f environment.yml
conda activate face2bmi
```

### Step 2: การเปิดใช้งานแอปพลิเคชัน (UI)

หากมีไฟล์โมเดลที่เทรนไว้แล้วครบถ้วนอยู่ในโฟลเดอร์ `weights/` สามารถเปิดแอปพลิเคชันเพื่อทดสอบระบบแบบ End-to-End ได้ทันที:

```bash
cd scripts
streamlit run app.py
```

### Step 3: การเทรนโมเดลใหม่ทั้งหมด (Optional)

หากต้องการเทรนโมเดลใหม่ทั้งหมด ให้ตรวจสอบว่ามีรูปภาพวางอยู่ในโฟลเดอร์ `data/Images/` เรียบร้อยแล้ว จากนั้นรันคำสั่งตามลำดับดังนี้:

```bash
cd scripts

# 1. เทรนโมเดลทำนาย BMI จากใบหน้า (Vision Transformer)
python train_vit_bmi.py --augmented True

# 2. เทรนโมเดลทำนาย Body Fat (Machine Learning)
python train_xgboost.py

# 3. เทรนโมเดลคัดกรองความเสี่ยงโรค NCDs (แยก 2 เส้นทาง)
python Stage2.py
python stage2_nowaist.py
```
