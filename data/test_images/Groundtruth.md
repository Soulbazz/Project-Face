# 📁 Test Images & Ground Truth Reference

เอกสารรวบรวมข้อมูลอ้างอิง (Ground Truth) ของรูปภาพทั้ง 14 ไฟล์ในโฟลเดอร์ `data/test_images/` สำหรับใช้ทดสอบความแม่นยำและสาธิตการทำงานของระบบคัดกรองความเสี่ยงโรค **Face2Health**

---

## 🎯 1. Out-of-Sample Verified Benchmark Set (ชุดทดสอบจริงจังนอกชุดฝึกสอน — 10 ภาพ)

ชุดภาพบุคคลที่มีข้อมูลสถิติทางกายภาพอ้างอิงตามช่วงเวลาที่ถ่ายภาพ (ส่วนสูง, น้ำหนัก, ค่า BMI จริง, และเปอร์เซ็นต์ไขมันร่างกายประเมินตามสรีระ/ผลตรวจร่างกายทางการแพทย์) และ **ไม่เคยอยู่ในชุดข้อมูลเทรนของโมเดล** ใช้สำหรับประเมินความแม่นยำของ Stage 1 (ViT BMI), Stage 1.5 (DEXA Body Fat), และ Stage 2 (Disease Risk Triage)

| File Name | Subject / Identity | Year | Age | Sex | PA Level | Height / Weight | True BMI (kg/m²) | Est. / True Body Fat (%) | Expected Triage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bench_adriana_lima.png` | Adriana Lima (Model/Boxing) | ~2012 | 30 | Female (0) | Vigorous (2) | 178 cm / 58.0 kg | **18.3** | ~15.0% | 🟢 Low Risk |
| `bench_cristiano_ronaldo_01.png` | Cristiano Ronaldo (Photo 1) | 2018 | 33 | Male (1) | Vigorous (2) | 187 cm / 84.0 kg | **24.0** | **7.0%** *(Juventus Medical)* | 🟢 Low Risk |
| `bench_cristiano_ronaldo_02.png` | Cristiano Ronaldo (Photo 2) | 2018 | 33 | Male (1) | Vigorous (2) | 187 cm / 84.0 kg | **24.0** | **7.0%** *(Juventus Medical)* | 🟢 Low Risk |
| `bench_leonardo_dicaprio.png` | Leonardo DiCaprio (Oscars) | 2016 | 41 | Male (1) | Moderate (1) | 183 cm / 88.0 kg | **26.3** *(Overweight)* | ~23.0% | 🟡 Watchful / 🟠 Screen Pos. |
| `bench_lionel_messi.png` | Lionel Messi (Athletic) | 2022 | 35 | Male (1) | Vigorous (2) | 170 cm / 72.0 kg | **24.9** | ~9.5% | 🟢 Low / 🟡 Watchful |
| `bench_ronda_rousey.png` | Ronda Rousey (UFC 190) | 2015 | 28 | Female (0) | Vigorous (2) | 170 cm / 61.2 kg | **21.2** | ~13.0% | 🟢 Low Risk |
| `bench_selena_gomez.png` | Selena Gomez (Golden Globes) | 2023 | 30 | Female (0) | Moderate (1) | 165 cm / 59.5 kg | **21.9** | ~25.0% | 🟢 Low Risk |
| `bench_tom_holland_01.png` | Tom Holland (Smiling / Tilted) | 2017 | 21 | Male (1) | Vigorous (2) | 173 cm / 64.0 kg | **21.4** | ~11.5% | 🟢 Low Risk |
| `bench_tom_holland_02.png` | Tom Holland (Frontal Neutral) | 2017 | 21 | Male (1) | Vigorous (2) | 173 cm / 64.0 kg | **21.4** | ~11.5% | 🟢 Low Risk |
| `bench_zac_efron.jpg` | Zac Efron (Baywatch Lean) | 2016 | 28 | Male (1) | Vigorous (2) | 173 cm / 68.0 kg | **22.7** | **~5.5%** | 🟢 Low Risk |

*หมายเหตุการตั้งค่าพารามิเตอร์ตอนทดสอบ:*
- **Sex (เพศชีววิทยา):** `1` = ชาย (Male), `0` = หญิง (Female)
- **PA Level (ระดับกิจกรรมทางกาย):** `0` = Sedentary (นั่งทำงานเป็นหลัก), `1` = Moderate (ขยับร่างกาย/ออกกำลังกายปานกลาง), `2` = Vigorous/Active (ออกกำลังกายหนักสม่ำเสมอ)
- **การทดสอบความเสถียร (Consistency Check):** ภาพของ `Cristiano Ronaldo` (01 vs 02) และ `Tom Holland` (01 vs 02) มีไว้เพื่อทดสอบว่าเมื่อเปลี่ยนภาพถ่ายหรือสีหน้าเล็กน้อยของบุคคลเดียวกัน โมเดลยังคงให้ค่า BMI และความเสี่ยงที่ใกล้เคียงเดิมหรือไม่

---

## 🧪 2. In-Distribution Dataset Sanity-Check Set (ภาพอ้างอิงจากชุดข้อมูลเทรน Stage 1 — 2 ภาพ)

ภาพในหมวดหมู่นี้เป็นตัวอย่างที่ดึงมาจากชุดข้อมูลที่ใช้พัฒนาโมเดล Stage 1 (BMI Estimation) มีค่า Ground Truth BMI จากชุดข้อมูลต้นฉบับ เก็บแยกไว้เพื่อทำ **Pipeline Sanity Check** (ตรวจสอบว่าการโหลดโมเดลและการส่งค่าผ่านท่อประมวลผลทำงานได้ตรงตามต้นฉบับ) โดยแยกออกจากชุดทดสอบหมวดที่ 1 เพื่อป้องกันปัญหา Data Leakage

| File Name | Source Dataset | True BMI (kg/m²) |
| :--- | :--- | :---: |
| `dataset_ref_male.png` | Stage 1 Training Set | **30.41** *(30.4065)* |
| `dataset_ref_female.png` | Stage 1 Training Set | **38.27** *(38.2737)* |

---

## 🎮 3. Wild Playground & Live Demo Set (ชุดภาพสำหรับทดลองเล่นและสาธิต UI — 2 ภาพ)

ภาพในหมวดหมู่นี้เป็นภาพบุคคลสาธารณะของไทยที่ **ไม่มีข้อมูล Ground Truth ทางการแพทย์ (ไม่ทราบค่า BMI หรือผลสแกน DEXA ที่แท้จริง)** จัดเตรียมไว้สำหรับกดสาธิตการทำงานบนหน้าเว็บแอปพลิเคชัน (Live UI Demo) เพื่อเปรียบเทียบผลลัพธ์ตามลักษณะโครงหน้าและระดับกิจกรรมทางกาย

| File Name | Subject / Identity | Recommended Demo Inputs | Expected Demo Behavior |
| :--- | :--- | :--- | :--- |
| `demo_anutin.png` | อนุทิน ชาญวีรกูล | Age: `58`, Sex: `Male (1)`, PA: `Sedentary (0)` หรือ `Moderate (1)` | สาธิตเคสผู้ใหญ่วัยใกล้ 60 ปีที่มีไขมันสะสมบริเวณแก้มและใต้คาง (Buccal/Submental fat) ระบบควรจัดเข้าสู่กลุ่มเฝ้าระวังหรือส่งตรวจยืนยัน (🟡 Watchful / 🟠 Screen Positive) |
| `demo_chadchart.png` | ชัชชาติ สิทธิพันธุ์ | Age: `58`, Sex: `Male (1)`, PA: `Vigorous (2)` | สาธิตเคสผู้ใหญ่วัยเดียวกันกับเคสแรก แต่มีโครงหน้ากระชับและออกกำลังกายระดับ Vigorous (`PA = 2`) เพื่อแสดงให้เห็นว่าโมเดลประเมินเปอร์เซ็นต์ไขมันและความเสี่ยงโรคต่ำกว่าอย่างชัดเจน |