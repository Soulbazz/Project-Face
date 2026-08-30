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

โปรเจกต์นี้ใช้ข้อมูล 2 ส่วน:
1. **NHANES Dataset (Tabular Data):** ข้อมูลสำหรับเทรน Body Fat % (`nhanes_cleaned_merged_final.csv`) รวมอยู่ใน Repository นี้แล้ว
2. **Face Images Dataset (Image Data):** ข้อมูลรูปภาพใบหน้าคนสำหรับเทรนโมเดล BMI 
   - 📥 **ดาวน์โหลดรูปภาพชุดข้อมูลได้จากโปรเจกต์ต้นฉบับ:** https://github.com/liujie-zheng/face-to-bmi-vit/tree/main/data/Images
   - **วิธีติดตั้ง:** เมื่อดาวน์โหลดข้อมูลภาพเสร็จแล้ว ให้นำรูปภาพทั้งหมด (เช่นไฟล์ .bmp) มาวางไว้ในพาธ `data/Images/` ของโปรเจกต์นี้

## 👏 กิตติกรรมประกาศและเครดิต (Acknowledgements)

โปรเจกต์นี้เป็นการต่อยอดและพัฒนาระบบเพิ่มเติม โดยโครงสร้างหลักในส่วนของการทำนายค่า BMI จากใบหน้า ถูกนำมาและดัดแปลงจากโปรเจกต์ Open Source ของคุณ **Liujie Zheng** ภายใต้ลิขสิทธิ์ MIT License
- **Original Repository:** [liujie-zheng/face-to-bmi-vit](https://github.com/liujie-zheng/face-to-bmi-vit)
- **Original License Notice:** Copyright (c) 2023 Liujie Zheng

ทางผู้จัดทำขอขอบคุณสำหรับโมเดลพื้นฐานและชุดข้อมูล ซึ่งทำให้สามารถนำมาพัฒนาต่อยอดร่วมกับข้อมูล NHANES เพื่อประเมินเปอร์เซ็นต์ไขมันในร่างกาย (Body Fat %) ในโครงงานนี้ได้

## ⚙️ การติดตั้ง (Installation)

โปรเจกต์นี้รันบน Python 3.10 และรองรับการประมวลผลผ่าน GPU (NVIDIA CUDA 11.8) แนะนำให้ใช้ Conda ในการจัดการ Environment

1. Clone repository นี้ลงมาที่เครื่อง:
> git clone https://github.com/Usernameของคุณ/Project-Face.git
> cd Project-Face

2. สร้างและเปิดใช้งาน Conda Environment:
> conda env create -f environment.yml
> conda activate face2bmi

🚀 วิธีใช้งาน (Usage)
Step 1: การติดตั้ง Environment
โปรเจกต์นี้รันบน Python 3.10 และรองรับการประมวลผลผ่าน GPU (NVIDIA CUDA 11.8)

ทำการ Clone repository และเข้าสู่โฟลเดอร์โปรเจกต์:

git clone https://github.com/Soulbazz/Project-Face.git
cd Project-Face

สร้างและเปิดใช้งาน Conda Environment:

conda env create -f environment.yml
conda activate face2bmi

Step 2: การเตรียมข้อมูลรูปภาพ (สำหรับเทรนโมเดลใบหน้า)
📥 ดาวน์โหลดชุดข้อมูลภาพใบหน้าได้จากโปรเจกต์ต้นฉบับ: คลิกเพื่อดาวน์โหลดรูปภาพ Dataset

เมื่อแตกไฟล์เรียบร้อย ให้นำไฟล์ภาพทั้งหมด (นามสกุล .bmp) ไปวางไว้ในโฟลเดอร์ data/Images/

Step 3: การเทรนโมเดลทำนาย BMI จากใบหน้า (Vision Transformer)
(หมายเหตุ: หากมีไฟล์โมเดล aug_epoch_7.pt อยู่ในโฟลเดอร์ weights/ แล้ว สามารถข้ามขั้นตอนนี้ได้ทันที)
รันคำสั่งเพื่อเทรนโมเดล Deep Learning (แนะนำให้รันด้วย GPU):

cd scripts
python train_vit_bmi.py --augmented True

Step 4: การเทรนโมเดลทำนาย Body Fat (Machine Learning)
รันสคริปต์เพื่อสร้างไฟล์น้ำหนักโมเดล .pkl เข้าไปเก็บไว้ในโฟลเดอร์ weights/:

python train_linear_bodyfat.py
python train_xgboost.py
python train_randomforest.py

Step 5: ทดสอบการใช้งานโมเดล (Inference / Demo)
คุณสามารถเลือกทดสอบระบบผ่าน 4 รูปแบบตามความต้องการ:

1. ทดสอบโมเดลวิเคราะห์ใบหน้าอย่างเดียว (CLI):

python demo.py

2. ทดสอบทำนาย Body Fat ผ่านหน้าต่างโปรแกรม (GUI):

python demo_bodyfat_gui.py

3. ทดสอบทำนาย Body Fat ด้วยโมเดล Random Forest (CLI):

python demo_bodyfat_rf.py

4. ทดสอบเปรียบเทียบกับสูตรทางการแพทย์ Deurenberg (CLI):

python demo_bodyfat_deurenberg.py