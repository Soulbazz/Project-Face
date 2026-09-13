import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report

# 1. กำหนด Path ต่างๆ
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, '..', 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MAIN_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'nhanes_cleaned_merged_final.csv')
DIAB_RAW_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'DIQ_C.XPT')
HYP_RAW_PATH = os.path.join(BASE_DIR, '..', 'data', 'raw', 'BPQ_C.XPT')

print(f"Loading main dataset: {MAIN_DATA_PATH}")
df_main = pd.read_csv(MAIN_DATA_PATH)

# กรองเฉพาะผู้ใหญ่อายุ 20 ปีขึ้นไป
df_main = df_main[df_main['AGE'] >= 20].copy()

# แปลงเพศเป็นตัวเลข (ชาย: 1, หญิง: 0)
if 'GENDER_NUM' not in df_main.columns and 'GENDER' in df_main.columns:
    df_main['GENDER_NUM'] = df_main['GENDER'].map({'Male': 1, 'Female': 0})

# 2. โหลดข้อมูลโรคเบาหวาน (DIQ_C.XPT)
print(f"Loading Diabetes raw data: {DIAB_RAW_PATH}")
df_diab_raw = pd.read_sas(DIAB_RAW_PATH)[['SEQN', 'DIQ010']]
# คัดกรองเฉพาะรหัส 1 (เป็น) และ 2 (ไม่เป็น)
df_diab_raw = df_diab_raw[df_diab_raw['DIQ010'].isin([1, 2])].copy()
df_diab_raw['TARGET_DIABETES'] = (df_diab_raw['DIQ010'] == 1).astype(int)

df_diab = pd.merge(df_main, df_diab_raw[['SEQN', 'TARGET_DIABETES']], on='SEQN', how='inner')

# 3. โหลดข้อมูลโรคความดันโลหิตสูง (BPQ_C.XPT)
print(f"Loading Hypertension raw data: {HYP_RAW_PATH}")
df_hyp_raw = pd.read_sas(HYP_RAW_PATH)[['SEQN', 'BPQ020']]
# คัดกรองเฉพาะรหัส 1 (เป็น) และ 2 (ไม่เป็น)
df_hyp_raw = df_hyp_raw[df_hyp_raw['BPQ020'].isin([1, 2])].copy()
df_hyp_raw['TARGET_HYPERTENSION'] = (df_hyp_raw['BPQ020'] == 1).astype(int)

df_hyp = pd.merge(df_main, df_hyp_raw[['SEQN', 'TARGET_HYPERTENSION']], on='SEQN', how='inner')

# 4. ฟังก์ชัน Train พร้อม Probability Calibration
def train_and_calibrate(data, target_col, feature_cols, model_filename):
    print(f"\n" + "="*50)
    print(f"Training & Calibrating: {model_filename}")
    print(f"Features: {feature_cols}")
    
    clean_data = data.dropna(subset=feature_cols + [target_col])
    X = clean_data[feature_cols]
    y = clean_data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # คำนวณ scale_pos_weight รับมือ Imbalanced Data
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    # ดัดเส้นกราฟความน่าจะเป็นด้วย Platt Scaling (Sigmoid)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method='sigmoid',
        cv=5
    )
    
    calibrated_model.fit(X_train, y_train)
    
    # ตรวจสอบค่าชี้วัด
    preds_prob = calibrated_model.predict_proba(X_test)[:, 1]
    preds = calibrated_model.predict(X_test)
    
    auc = roc_auc_score(y_test, preds_prob)
    brier = brier_score_loss(y_test, preds_prob)
    print(f"ROC-AUC: {auc:.4f} | Brier Score: {brier:.4f}")
    print(classification_report(y_test, preds, digits=4))
    
    save_path = os.path.join(WEIGHTS_DIR, model_filename)
    joblib.dump(calibrated_model, save_path)
    print(f"Saved -> {save_path}")

# 5. สั่งเทรนทั้ง 4 โมเดล
features_with_waist = ['AGE', 'GENDER_NUM', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT']
features_no_waist = ['AGE', 'GENDER_NUM', 'BMI', 'TOTAL_BODY_FAT_PCT']

# Diabetes
train_and_calibrate(df_diab, 'TARGET_DIABETES', features_with_waist, 'xgb_classifier_diabetes.pkl')
train_and_calibrate(df_diab, 'TARGET_DIABETES', features_no_waist, 'xgb_classifier_diabetes_no_waist.pkl')

# Hypertension
train_and_calibrate(df_hyp, 'TARGET_HYPERTENSION', features_with_waist, 'xgb_classifier_hypertension.pkl')
train_and_calibrate(df_hyp, 'TARGET_HYPERTENSION', features_no_waist, 'xgb_classifier_hypertension_no_waist.pkl')

print("\n" + "="*50)
print("All Stage 2 models trained, calibrated, and saved successfully!")