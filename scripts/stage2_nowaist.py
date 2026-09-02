import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

os.makedirs('../weights', exist_ok=True)

print("กำลังสร้างโมเดล NCDs แบบไม่ใช้รอบเอว (Optional Waist)...")
df_main = pd.read_csv('../data/processed/nhanes_cleaned_merged_final.csv')
df_main['GENDER_NUM'] = df_main['GENDER'].map({'Male': 1, 'Female': 0})

df_diab = pd.read_sas('data/raw/DIQ_C.XPT')
df_diab['DIABETES'] = df_diab['DIQ010'].apply(lambda x: 1 if x == 1 else 0) 

df_hyper = pd.read_sas('data/raw/BPQ_C.XPT')
df_hyper['HYPERTENSION'] = df_hyper['BPQ020'].apply(lambda x: 1 if x == 1 else 0)

df_merged = pd.merge(df_main, df_diab[['SEQN', 'DIABETES']], on='SEQN', how='inner')
df_merged = pd.merge(df_merged, df_hyper[['SEQN', 'HYPERTENSION']], on='SEQN', how='inner')

# 💡 ตัด WAIST_CM ออกจาก Features
features_no_waist = ['AGE', 'GENDER_NUM', 'BMI', 'TOTAL_BODY_FAT_PCT']
df_merged = df_merged.dropna(subset=features_no_waist + ['DIABETES', 'HYPERTENSION'])

X = df_merged[features_no_waist]

# 1. เบาหวาน
y_diab = df_merged['DIABETES']
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y_diab, test_size=0.2, random_state=42, stratify=y_diab)
scale_weight_d = len(y_train_d[y_train_d == 0]) / len(y_train_d[y_train_d == 1])
xgb_diab_nw = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, scale_pos_weight=scale_weight_d, random_state=42)
xgb_diab_nw.fit(X_train_d, y_train_d)
joblib.dump(xgb_diab_nw, '../weights/xgb_classifier_diabetes_no_waist.pkl')

# 2. ความดัน
y_hyp = df_merged['HYPERTENSION']
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X, y_hyp, test_size=0.2, random_state=42, stratify=y_hyp)
scale_weight_h = len(y_train_h[y_train_h == 0]) / len(y_train_h[y_train_h == 1])
xgb_hyp_nw = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, scale_pos_weight=scale_weight_h, random_state=42)
xgb_hyp_nw.fit(X_train_h, y_train_h)
joblib.dump(xgb_hyp_nw, '../weights/xgb_classifier_hypertension_no_waist.pkl')

print("✅ สร้างไฟล์ NCDs แบบไม่ใช้รอบเอวเสร็จสมบูรณ์!")