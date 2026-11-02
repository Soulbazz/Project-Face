# แผนปฏิบัติการแก้ไขข้อท้วงติงของ Reviewer และปรับปรุงความสอดคล้องของเอกสารวิจัย Face2Health

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ปรับปรุงความถูกต้องทางวิชาการ ความสอดคล้องระหว่างข้อมูลดิบ โค้ด และข้อความในเอกสารวิจัย Face2Health ทั้ง 2 ภาษา (English: `Paper.pdf` และ Thai: `paper_thai.pdf`) ให้ตรงกับหลักฐานเชิงประจักษ์ในโค้ดและชุดข้อมูลจริง 100% พร้อมแก้ไขข้อท้วงติงของ Reviewer ครบทุกข้อ

**Architecture:** 
1. ตรวจสอบและระบุที่มาของข้อมูล (Data Provenance) จาก Cycle C (NHANES 2003–2004) และชุดข้อมูลภาพ VisualBMI อย่างโปร่งใส
2. ปรับปรุงโค้ดสร้างภาพประกอบ (`scripts/regenerate_figures_fix.py`) ให้ใช้ผลทำนายจากโมเดลที่บันทึกไว้จริงใน `weights/` แทนการสุ่มตัวเลขเชิงสถิติ (Synthetic Generation)
3. ปรับแก้เนื้อหา สถิติ รหัสตัวแปร และข้อความจำกัดขอบเขตใน `scripts/generate_academic_paper_docx.py` และ `scripts/generate_academic_paper_thai.py` ให้สอดคล้องกันทุกจุด
4. คอมไพล์และตรวจสอบ PDF ทั้งสองฉบับแบบหน้าต่อหน้า (Visual Inspection)

**Tech Stack:** Python 3.14, PyMuPDF (fitz), XGBoost, Scikit-learn, Pandas, Matplotlib, ReportLab/HTML-CSS rendering via MuPDF Story.

**Spec:** ข้อท้วงติงและข้อเสนอแนะจาก Reviewer (5 ข้อหลัก + 4 จุดย่อย) ควบคู่กับผลการตรวจสอบไฟล์ข้อมูลจริงใน `data/` และน้ำหนักโมเดลใน `weights/`.

---

## Global Constraints

- คงไว้เฉพาะไฟล์เปเปอร์ต้นฉบับหลัก 2 ไฟล์: `reports/Paper.pdf` (English) และ `reports/paper_thai.pdf` (Thai) ไม่สร้างไฟล์ซ้ำซ้อน
- ความยาวของเปเปอร์ทั้งสองฉบับต้องคงไว้ที่ 9 หน้าเท่าเดิม ไม่ให้เนื้อหาล้นหน้าหรือทำให้ฟิกเกอร์/ตารางตกไปหน้าถัดไป
- สัดส่วนของตารางและรูปภาพใน MuPDF ต้องคงความคมชัดและไม่มีการตกขอบหรือตัวอักษรตัดบรรทัดกลางคำ
- ข้อมูลทางสถิติและตัวแปรต้องตรงกับโค้ดจริง: ใช้ `PAD200`, `PAD320` จาก `PAQ_C.XPT` และระบุรอบสำรวจเป็น NHANES 2003–2004 (Cycle C) อย่างสม่ำเสมอทั่วทั้งสองฉบับ

---

## Review Focus

1. **ความถูกต้องของรอบข้อมูล (NHANES Data Year Consistency):** ต้องไม่มีการอ้างว่าผู้เข้าร่วมกลุ่ม DEXA ปี 2003–2004 มีผลแล็บ HbA1c จากรอบ 2017–2018 ในคนเดียวกัน
2. **ความถูกต้องของตัวแปร Physical Activity:** ต้องใช้ `PAD200` และ `PAD320` เท่านั้น ห้ามมีรหัสปี 2007+ (`PAQ605`, `PAD680`) หลงเหลือในเอกสาร
3. **ความซื่อตรงของกราฟ Figure 2 (Scatter Plot):** ต้องพล็อตจากโมเดลจริง `xgboost_bodyfat_no_waist.pkl` และ `xgboost_bodyfat_with_waist.pkl` บน `data/test_tabular.csv`
4. **ความสอดคล้องระหว่างฉบับไทยและอังกฤษ:** ฉบับไทยต้องไม่เคลมเกินจริง โดยต้องระบุชัดเจนว่าที่ $\sigma = 5.60\text{ kg/m}^2$ ความสามารถในการจำแนกเบาหวานไม่มีนัยสำคัญทางสถิติเหนือกว่า Demographics ($p = 0.068$) และเป็นเพียงเครื่องมือคัดกรองสำรอง (Fallback)
5. **ความถูกต้องของคำศัพท์ทางคลินิก:** ใช้ "โรคเบาหวาน" (Diabetes Mellitus) แทน "โรคเบาหวานประเภทที่ 2" เนื่องจากคำถาม DIQ010 ไม่ได้แยกประเภท และใช้คำว่า "Screening/Triage" แทน "Diagnostic" ในบทคัดย่อ

---

## Proposed Changes

### Task 1: อัปเดตโค้ดสร้างภาพประกอบให้ใช้ผลจากโมเดลจริง (`regenerate_figures_fix.py`)

**Files:**
- Modify: `scripts/regenerate_figures_fix.py`
- Test Data: `data/test_tabular.csv`
- Models: `weights/xgboost_bodyfat_no_waist.pkl`, `weights/xgboost_bodyfat_with_waist.pkl`

- [x] **Step 1: ปรับแก้ฟังก์ชัน `generate_fig1_scatter()` ใน `scripts/regenerate_figures_fix.py`**
  - ยกเลิกการสร้างข้อมูลสุ่ม (`0.77 * y_actual_sub + ... + noise`)
  - โหลดโมเดลจริงจาก `weights/xgboost_bodyfat_no_waist.pkl` และ `weights/xgboost_bodyfat_with_waist.pkl`
  - ทำนายค่าจริงบนชุดข้อมูลทดสอบ `data/test_tabular.csv`
  - คำนวณค่า MAE, RMSE, R² จริง และพล็อต Scatter Plot ลงบน `fig1_stage15_scatter.png`

- [x] **Step 2: ตรวจสอบและรัน `scripts/regenerate_figures_fix.py`**
  - Run: `python scripts/regenerate_figures_fix.py`
  - ตรวจสอบว่ารูปภาพทั้ง 5 รูปถูกบันทึกลง `reports/figures/` และ `scripts/results/` อย่างสมบูรณ์และไม่มี error

---

### Task 2: แก้ไขเปเปอร์ภาษาอังกฤษ (`generate_academic_paper_docx.py`)

**Files:**
- Modify: `scripts/generate_academic_paper_docx.py`
- Output: `reports/Paper.pdf`

- [x] **Step 1: แก้ไข Section Abstract**
  - เปลี่ยนคำว่า "statistically significant diagnostic gains" เป็น "statistically significant pre-screening triage gains"
  - เพิ่มตัวเลขประสิทธิภาพหลักภายใต้สัญญาณรบกวนจำลอง (Sensitivity 80.9%–85.1% ที่ $\sigma_{\text{sim}} = 4.40\text{--}5.60\text{ kg/m}^2$ เทียบกับ 90.91% / 90.61% ที่ nominal reference $\sigma_{\text{sim}} = 2.14\text{ kg/m}^2$)
  - ระบุอย่างชัดเจนว่าที่ $\sigma_{\text{sim}} = 5.60\text{ kg/m}^2$ การตรวจคัดกรองเบาหวานลดทอนลงจนไม่มีนัยสำคัญทางสถิติเหนือข้อมูลพื้นฐาน ($p = 0.068$)

- [x] **Step 2: แก้ไข Section II.C (Dataset Cohort & Participant Flow) และ Table I**
  - ปรับข้อมูลที่มาเป็น CDC NHANES 2003–2004 (Cycle C) อย่างสม่ำเสมอ
  - แก้ไขตัวเลขประชากรเริ่มต้นใน Table I จาก 39,156 (ซึ่งเป็นตัวเลขรวมหลายรอบ) ให้ระบุให้ชัดเจนว่าเป็น Cycle C cohort (10,122 participants) หรือระบุเกณฑ์การคัดกรองจาก Cycle C ตามจริง
  - ปรับรหัสกิจกรรมทางกายเป็น `PAD200` (vigorous activity) และ `PAD320` (moderate activity) จาก `PAQ_C.XPT` ตัดรหัส `PAQ605, 620, 650, 665, PAD680` ออก
  - แก้ไขหมายเหตุท้าย Table I และเอกสารอ้างอิง Ref [10] ให้ระบุเฉพาะรอบ 2003–2004

- [x] **Step 3: แก้ไข Section III.A.1 และ Table II Note เกี่ยวกับที่มาของภาพ**
  - ระบุอย่างชัดเจนและโปร่งใสว่าชุดข้อมูลภาพใบหน้า 3,962 ภาพมาจาก VisualBMI benchmark (Kocabey et al., 2017)
  - นิยามคำว่า "controlled frontal subset (n = 412)" ว่าเป็นภาพที่ผ่านเกณฑ์มุมหน้าตรงเข้มงวด ($\le \pm 15^\circ$) จาก VisualBMI เพื่อไม่ให้เกิดความสับสนกับ CFD

- [x] **Step 4: ตรวจสอบ Table III และ Table IV**
  - ตรวจสอบคอลัมน์ AUC และช่วงความเชื่อมั่น 95% CI ไม่ให้มีการตัดบรรทัดกลางช่วง

---

### Task 3: แก้ไขเปเปอร์ภาษาไทย (`generate_academic_paper_thai.py`)

**Files:**
- Modify: `scripts/generate_academic_paper_thai.py`
- Output: `reports/paper_thai.pdf`

- [x] **Step 1: แก้ไขคำศัพท์ทางคลินิก (จุดเล็กข้อ 1 & 3)**
  - แก้ "โรคเบาหวานประเภทที่ 2" ในบทคัดย่อ, วัตถุประสงค์ (1.2.2), ผลการทดลอง (4.1), และหัวตารางที่ 3 เป็น "โรคเบาหวาน" (Diabetes Mellitus) ให้ตรงกับคำถาม DIQ010
  - แก้ "แบบจำลองหลักแบบไร้การสัมผัส" ในข้อ 4.4 เป็น "แบบจำลองหลักแบบไม่ใช้รอบเอว (Contactless / No Waist)"

- [x] **Step 2: ปรับแก้เนื้อหาบทคัดย่อ สรุปผล (5.1) และข้อจำกัด เพื่อไม่ให้กล่าวอ้างเกินจริง**
  - ระบุในบทคัดย่อและสรุปผลว่า ประสิทธิภาพ 90.91% / 90.61% เป็นผลที่ระดับสัญญาณรบกวนอ้างอิงต่ำสุด ($\sigma_{\text{sim}} = 2.14\text{ kg/m}^2$)
  - เพิ่มผลประเมินในสภาวะการใช้งานจริง ($\sigma_{\text{sim}} = 4.40\text{--}5.60\text{ kg/m}^2$) ซึ่งได้ Sensitivity อยู่ที่ 81%–86% และที่ $\sigma_{\text{sim}} = 5.60\text{ kg/m}^2$ ประสิทธิภาพการคัดกรองเบาหวานไม่มีนัยสำคัญทางสถิติเหนือกว่าข้อมูลพื้นฐาน ($p = 0.068$)
  - เพิ่มการเปรียบเทียบกับแบบจำลองที่ใช้ค่า BMI จากการรายงานตนเอง (Self-Report BMI) และระบุชัดเจนว่าระบบทำหน้าที่เป็น "เครื่องมือคัดกรองสำรองเบื้องต้น (Zero-Hardware Fallback) ที่มีประโยชน์จำกัดภายใต้เงื่อนไข"

- [x] **Step 3: แก้ไข Section 3.1.1 (ข้อมูล NHANES) และหมายเหตุตารางที่ 1**
  - ปรับแก้ให้ตรงกับฉบับภาษาอังกฤษ: ระบุรอบ 2003–2004 (Cycle C), ปรับรหัสกิจกรรมทางกายเป็น `PAD200` และ `PAD320`

- [x] **Step 4: แก้ไข Section 4.3 เกี่ยวกับ Brier Score (จุดเล็กข้อ 2)**
  - ปรับแก้คำอธิบาย Brier Score ให้มีบริบทครบถ้วนว่าค่า 0.142 คือแบบจำลองที่ยังไม่ได้ใส่เงื่อนไขความชันแบบโมโนโทนิกและยังไม่ผ่านการปรับเทียบ และเปรียบเทียบกับค่า 0.081 ของโมเดลหลักที่ปรับเทียบแล้วอย่างเป็นธรรม

---

### Task 4: คอมไพล์และตรวจสอบความถูกต้องของเอกสารวิจัยทั้งสองฉบับ

**Files:**
- Target PDFs: `reports/Paper.pdf`, `reports/paper_thai.pdf`

- [x] **Step 1: รันคอมไพล์เปเปอร์ทั้ง 2 ภาษา**
  - Run: `python scripts/generate_academic_paper_docx.py`
  - Run: `python scripts/generate_academic_paper_thai.py`
  - ยืนยันว่าคอมไพล์ผ่าน 100% โดยไม่มีข้อผิดพลาด

- [x] **Step 2: ตรวจสอบความยาวหน้าและการจัดวาง (Layout Verification)**
  - ตรวจสอบว่าทั้งสองฉบับมีความยาว 9 หน้าพอดี
  - ตรวจสอบว่าชื่อเรื่องภาษาไทยยังคงตัดบรรทัด 2 บรรทัดถูกต้อง และเครื่องหมายผู้แต่ง `*1`, `*2` เป็นตัวยกสมบูรณ์
  - ตรวจสอบรูปภาพและตารางทั้งหมดว่าไม่มีการตกขอบหรือตัดบรรทัดผิดปกติ

---

## Verification Plan

### Automated Tests
1. รันสคริปต์ตรวจสอบข้อมูลและค่า Thresholds:
   `python scripts/checking.py`
   ผลที่คาดหวัง: ตรวจพบค่า N และ Metrics ตรงกับ Table III และ Table IV ในเปเปอร์
2. รันสคริปต์คอมไพล์รูปภาพและเปเปอร์:
   `python scripts/regenerate_figures_fix.py`
   `python scripts/generate_academic_paper_docx.py`
   `python scripts/generate_academic_paper_thai.py`
   ผลที่คาดหวัง: สร้างไฟล์ PDF ทั้ง 2 ฉบับสำเร็จโดยขนาดหน้าเท่ากับ 9 หน้า

### Manual Verification
- ตรวจสอบเอกสาร PDF ทั้งสองฉบับ (`Paper.pdf` และ `paper_thai.pdf`) ในส่วน Abstract, Table I, Table II, Table III, Table IV, Table VII, และ Conclusion ว่าได้รับการแก้ไขตรงตามข้อท้วงติงของ Reviewer ครบถ้วนทุกข้อ

