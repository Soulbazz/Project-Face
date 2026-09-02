import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

os.makedirs('../weights', exist_ok=True)

# 1. โหลดข้อมูล Features หลักที่คุณมีอยู่แล้ว
print("กำลังโหลดข้อมูลสัดส่วนร่างกาย...")
df_main = pd.read_csv('../data/processed/nhanes_cleaned_merged_final.csv')
df_main['GENDER_NUM'] = df_main['GENDER'].map({'Male': 1, 'Female': 0})

# [ตัวเลือกเสริม] หากต้องการใช้ RACE ให้ลบเครื่องหมาย # ด้านล่างออก
# df_main = pd.get_dummies(df_main, columns=['RACE'], drop_first=True)

# 2. โหลดไฟล์เฉลย (Labels) จากโฟลเดอร์ raw
print("กำลังดึงข้อมูลเฉลยผู้ป่วยเบาหวานและความดัน...")
# เบาหวาน (DIQ_I.XPT) - DIQ010: 1=เป็น, 2=ไม่เป็น, 3=เสี่ยง (Borderline)
df_diab = pd.read_sas('data/raw/DIQ_C.XPT')
df_diab['DIABETES'] = df_diab['DIQ010'].apply(lambda x: 1 if x == 1 else 0) 
df_diab = df_diab[['SEQN', 'DIABETES']]

# ความดันโลหิตสูง (BPQ_I.XPT) - BPQ020: 1=เป็น, 2=ไม่เป็น
df_hyper = pd.read_sas('data/raw/BPQ_C.XPT')
df_hyper['HYPERTENSION'] = df_hyper['BPQ020'].apply(lambda x: 1 if x == 1 else 0)
df_hyper = df_hyper[['SEQN', 'HYPERTENSION']]

# 3. นำข้อมูลมาเชื่อมกัน (Merge) ด้วยรหัส SEQN
df_merged = pd.merge(df_main, df_diab, on='SEQN', how='inner')
df_merged = pd.merge(df_merged, df_hyper, on='SEQN', how='inner')

# กำหนดตัวแปรต้น (Input Features) 
# ถ้าเปิดใช้ RACE อย่าลืมเพิ่มชื่อคอลัมน์ RACE_... เข้าไปในลิสต์นี้ด้วย
features = ['AGE', 'GENDER_NUM', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT']
df_merged = df_merged.dropna(subset=features + ['DIABETES', 'HYPERTENSION'])

X = df_merged[features]

# ==========================================
# โมเดลที่ 1: คัดกรองเบาหวาน (Diabetes)
# ==========================================
y_diab = df_merged['DIABETES']
# ใช้ stratify=y_diab เพื่อรักษาอัตราส่วนคนป่วยและคนปกติใน Train/Test ให้เท่ากัน
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_diab, test_size=0.2, random_state=42, stratify=y_diab)

# คำนวณสัดส่วนคนปกติ/คนป่วย เพื่อให้ XGBoost ไม่ละเลยคลาสคนป่วย (Class Imbalance)
scale_weight_d = len(y_train_d[y_train_d == 0]) / len(y_train_d[y_train_d == 1])

xgb_diab = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, 
                         scale_pos_weight=scale_weight_d, random_state=42)
xgb_diab.fit(X_train_d, y_train_d)
y_pred_d = xgb_diab.predict(X_test_d)

print("\n📊 === ผลประเมินโมเดลคัดกรองเบาหวาน (Diabetes) ===")
print(classification_report(y_test_d, y_pred_d, target_names=['Healthy', 'Diabetic']))
joblib.dump(xgb_diab, '../weights/xgb_classifier_diabetes.pkl')

# ==========================================
# โมเดลที่ 2: คัดกรองความดันโลหิตสูง (Hypertension)
# ==========================================
y_hyp = df_merged['HYPERTENSION']
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_hyp, test_size=0.2, random_state=42, stratify=y_hyp)

scale_weight_h = len(y_train_h[y_train_h == 0]) / len(y_train_h[y_train_h == 1])

xgb_hyp = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, 
                        scale_pos_weight=scale_weight_h, random_state=42)
xgb_hyp.fit(X_train_h, y_train_h)
y_pred_h = xgb_hyp.predict(X_test_h)

print("\n📊 === ผลประเมินโมเดลคัดกรองความดันโลหิตสูง (Hypertension) ===")
print(classification_report(y_test_h, y_pred_h, target_names=['Healthy', 'Hypertensive']))
joblib.dump(xgb_hyp, '../weights/xgb_classifier_hypertension.pkl')

print("\n✅ บันทึกไฟล์โมเดล Stage 2 ลงในโฟลเดอร์ weights/ เรียบร้อย!")