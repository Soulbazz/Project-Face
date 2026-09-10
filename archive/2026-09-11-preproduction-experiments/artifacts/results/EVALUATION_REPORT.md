# Evaluation Report — Face-to-BMI Health Risk System

_Generated: 2026-09-06 10:40_

> ผลทั้งหมดคำนวณบน **held-out test set** ที่โมเดลไม่เคยเห็นระหว่างการฝึก

> Root: `C:\Users\Ninenine\Downloads\Project Face`

---

## Stage 1 — Face-to-BMI (Vision Transformer)

> ⚠️ ข้าม: ไม่พบ `data\test_faces.csv`
> รัน `make_test_faces.py` ก่อนเพื่อสร้างไฟล์นี้

---

## Stage 1.5 — Body Fat % (XGBoost vs Baseline)

> ชุดทดสอบ: 1230 แถว

### Baseline: Deurenberg Formula (1991)

| Metric | Value |
|---|---|
| N (test) | 1230 |
| MAE | 6.397 |
| RMSE | 8.312 |
| R² | 0.168 |
| Bias (mean error) | -4.115 |
| คลาดเคลื่อน ≤ 2 | 20.2% |
| คลาดเคลื่อน ≤ 3 | 31.1% |

> `custom_bodyfat_model_with_waist.pkl` → features: `['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']`

### Body Fat — XGBoost (+waist)

| Metric | Value |
|---|---|
| N (test) | 1230 |
| MAE | 3.648 |
| RMSE | 4.694 |
| R² | 0.735 |
| Bias (mean error) | -0.055 |
| คลาดเคลื่อน ≤ 2 | 34.7% |
| คลาดเคลื่อน ≤ 3 | 51.3% |

> `custom_bodyfat_model_no_waist.pkl` → features: `['BMI', 'AGE', 'GENDER_NUM']`

### Body Fat — XGBoost (no waist)

| Metric | Value |
|---|---|
| N (test) | 1230 |
| MAE | 3.739 |
| RMSE | 4.796 |
| R² | 0.723 |
| Bias (mean error) | -0.069 |
| คลาดเคลื่อน ≤ 2 | 33.6% |
| คลาดเคลื่อน ≤ 3 | 50.7% |

### 📊 ตารางเปรียบเทียบ Body Fat

| Model              |    n |   mae |   rmse |    r2 |   bias |
|:-------------------|-----:|------:|-------:|------:|-------:|
| Deurenberg (1991)  | 1230 | 6.397 |  8.312 | 0.168 | -4.115 |
| XGBoost (+waist)   | 1230 | 3.648 |  4.694 | 0.735 | -0.055 |
| XGBoost (no waist) | 1230 | 3.739 |  4.796 | 0.723 | -0.069 |

> **ข้อสรุป:** โมเดลที่แม่นที่สุดคือ **XGBoost (+waist)**
> Machine Learning ลด error ได้ **43.0%** เทียบกับสูตรสำเร็จปี 1991

---

## Stage 2 — NCD Risk Classifiers

> คอลัมน์ในไฟล์ test: `['SEQN', 'GENDER', 'AGE', 'RACE', 'WEIGHT_KG', 'HEIGHT_CM', 'BMI', 'WAIST_CM', 'ANDROID_FAT_PCT', 'GYNOID_FAT_PCT', 'TOTAL_BODY_FAT_PCT', 'GENDER_NUM']`
> ตรวจพบ: diabetes = `None` | hypertension = `None`

> ⚠️ **ไม่พบคอลัมน์โรคในไฟล์นี้**
> แปลว่า `Stage2.py` เทรนจากไฟล์ CSV คนละตัวกับ `train_xgboost.py`
> วิธีแก้: เปิด `Stage2.py` ดูว่าอ่านไฟล์ไหน แล้วให้สคริปต์นั้น
> เซฟ test set ของตัวเองออกมา แล้วชี้ `ncd_test_csv` มาที่ไฟล์นั้น

---

## Sensitivity Analysis — Error Propagation

> **จุดประสงค์:** Stage 1 มี MAE ระดับหนึ่ง — คำถามคือ error นั้นถูก
> *ขยาย (amplify)* หรือถูก *กลืน (absorb)* เมื่อไหลไปถึงความเสี่ยงโรคที่ Stage 2

> ⚠️ ข้าม: ไม่มีข้อมูล Stage 2

---

## ข้อจำกัดของงาน (Limitations)
- ข้อมูลฝึกมาจาก NHANES (ประชากรสหรัฐฯ) อาจเกิด **domain shift** เมื่อใช้กับคนไทย
- ทำนายจากภาพใบหน้าเพียงอย่างเดียว ไม่ครอบคลุมพันธุกรรมและพฤติกรรม
- ผลลัพธ์เป็นเครื่องมือ **คัดกรองเบื้องต้น** ไม่ใช่การวินิจฉัยทางการแพทย์