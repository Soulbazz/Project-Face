# Face to BMI & Body Fat Analysis

โปรเจกต์นี้เป็นการพัฒนาระบบ AI สำหรับวิเคราะห์และทำนายข้อมูลสุขภาพ แบ่งออกเป็น 2 ส่วนหลัก ได้แก่:
1. **ทำนายค่าดัชนีมวลกาย (BMI) จากรูปภาพใบหน้า** โดยใช้โมเดล Deep Learning (Vision Transformer)
2. **ทำนายเปอร์เซ็นต์ไขมันในร่างกาย (Body Fat %)** โดยใช้โมเดล Machine Learning (XGBoost และ Linear Regression) จากข้อมูลสัดส่วนร่างกาย

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

- **assets/**: รูปภาพสำหรับเอกสารประกอบ (เช่น แผนภาพ, ภาพตัวอย่าง)
- **data/**: ข้อมูลแบบตาราง (CSV) สำหรับเทรนโมเดล Body Fat
- **scripts/**: สคริปต์หลักสำหรับการประมวลผลและเทรนโมเดล
  - `benchmark_models.py`: เปรียบเทียบประสิทธิภาพโมเดล
  - `cleandata.py`: ทำความสะอาดข้อมูล
  - `demo.py`: โค้ดสำหรับทดสอบรันโมเดลทำนาย BMI จากใบหน้า
  - `loader.py`: จัดการ Data Dataloader และ Augmentation
  - `models.py`: โครงสร้างโมเดล ViT
  - `run.py`: สคริปต์รัน Pipeline การเทรน Deep Learning
  - `train_bodyfat.py`: เทรนโมเดล Linear Regression
  - `train_xgboost.py`: เทรนโมเดล XGBoost
- **weights/**: โฟลเดอร์เก็บไฟล์น้ำหนักโมเดล (.pt, .pkl)
- **environment.yml**: ไฟล์ตั้งค่า Conda Environment (สำหรับ Windows/CUDA)

## 💾 ข้อมูลที่ใช้ (Datasets)

โปรเจกต์นี้ใช้ข้อมูล 2 ส่วน ซึ่งผู้ที่นำโค้ดไปรันต้องดาวน์โหลดข้อมูลรูปภาพเพิ่มเติมเพื่อใช้ในการเทรนโมเดล Vision Transformer:

1. **NHANES Dataset (Tabular Data):** 
   - ข้อมูลตารางสำหรับเทรน Body Fat % (`nhanes_cleaned_merged_final.csv`) ถูกจัดเตรียมไว้แล้วในโฟลเดอร์ `data/`
2. **Face Images Dataset (Image Data):**
   - ข้อมูลรูปภาพใบหน้าคนสำหรับเทรนโมเดล BMI มีขนาดใหญ่และไม่ได้รวมอยู่ใน Repository นี้
   - 📥 **ดาวน์โหลดรูปภาพชุดข้อมูลได้ที่:** [ใส่ลิงก์ Google Drive หรือลิงก์ดาวน์โหลดไฟล์ ZIP ของคุณที่นี่]
   - **วิธีติดตั้ง:** เมื่อดาวน์โหลดและแตกไฟล์ ZIP แล้ว ให้นำโฟลเดอร์ `Images` ไปวางไว้ในพาธ `data/Images/`

## ⚙️ การติดตั้ง (Installation)

โปรเจกต์นี้รันบน Python 3.10 และรองรับการประมวลผลผ่าน GPU (NVIDIA CUDA 11.8) แนะนำให้ใช้ Conda ในการจัดการ Environment

1. Clone repository นี้ลงมาที่เครื่อง:
> git clone https://github.com/Usernameของคุณ/Project-Face.git
> cd Project-Face

2. สร้างและเปิดใช้งาน Conda Environment:
> conda env create -f environment.yml
> conda activate face2bmi

## 🚀 วิธีใช้งาน (Usage)

### 1. ทดสอบโมเดลทำนาย BMI จากใบหน้า (Demo)
คุณสามารถทดสอบโมเดลกับภาพตัวอย่างได้ทันที หากยังไม่มีไฟล์น้ำหนักโมเดล (`aug_epoch_7.pt`) สคริปต์จะทำการดาวน์โหลดให้อัตโนมัติ:
> cd scripts
> python demo.py

### 2. การเทรนโมเดลทำนาย Body Fat
เทรนและสร้างไฟล์โมเดล `.pkl` ทั้งแบบมีรอบเอวและไม่มีรอบเอว:
> python scripts/train_xgboost.py
> python scripts/train_bodyfat.py

### 3. การเทรนโมเดล Vision Transformer (ต้องมี Dataset รูปภาพ)
หากต้องการเทรนโมเดลประเมินใบหน้าใหม่ทั้งหมดด้วยตัวเอง (ใช้ GPU):
> python scripts/run.py --augmented True

## 🧠 สถาปัตยกรรมโมเดล (Models Used)
- **Vision Transformer (ViT_H_14):** ใช้โครงสร้างแบบ Pre-trained ร่วมกับการทำ Fine-tuning ที่ชั้น Head (Linear Layers + GELU + Dropout)
- **XGBoost Regressor:** ใช้พารามิเตอร์ `n_estimators=150`, `max_depth=4`
- **Data Augmentation:** รองรับเทคนิค Random Rotation, Horizontal Flip, Color Jitter และ Random Distortion