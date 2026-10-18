import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

print("==================================================")
print(" 1. CHECKING WEIGHTS/THRESHOLDS.JSON")
print("==================================================")
thresholds_path = os.path.join(WEIGHTS_DIR, "thresholds.json")
if os.path.exists(thresholds_path):
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresh_data = json.load(f)
    print(json.dumps(thresh_data, indent=2, ensure_ascii=False))

print("\n==================================================")
print(" 2. EVALUATING CLASSIFICATION TRIAGE METRICS")
print("==================================================")

candidate_files = [
    os.path.join(PROJECT_ROOT, "data", "processed", "nhanes_stage2.parquet"),
    os.path.join(PROJECT_ROOT, "data", "test_tabular.csv"),
    os.path.join(PROJECT_ROOT, "data", "split_fixed_v2.csv"),
    os.path.join(PROJECT_ROOT, "data", "split_fixed.csv"),
]

target_df = None
found_file_path = None

for p in candidate_files:
    if os.path.exists(p):
        target_df = pd.read_parquet(p) if p.endswith('.parquet') else pd.read_csv(p)
        found_file_path = p
        break

if target_df is not None:
    print(f"[+] โหลดข้อมูลสำเร็จจาก: {found_file_path}")

    # Map ชื่อคอลัมน์รหัส NHANES ให้ตรงกับที่โมเดลใช้
    col_mapping = {
        'RIDAGEYR': 'AGE',
        'RIAGENDR': 'GENDER_NUM',
        'BMXBMI': 'BMI',
        'BMXWAIST': 'WAIST_CM',
        'DXDTOPF': 'TOTAL_BODY_FAT_PCT',
        'GENDER': 'GENDER_NUM',
        'SEX': 'GENDER_NUM',
        'sex': 'GENDER_NUM'
    }
    target_df = target_df.rename(columns=col_mapping)

    tier_names = ["Low Risk", "Watchful", "Screen Positive", "Urgent Risk"]

    def find_col(df, candidates):
        cols_upper = {c.upper(): c for c in df.columns}
        for cand in candidates:
            if cand.upper() in cols_upper:
                return cols_upper[cand.upper()]
        return None

    dia_target_col = find_col(target_df, ['DIQ010', 'dm', 'diabetes_target', 'diabetes'])
    htn_target_col = find_col(target_df, ['BPQ020', 'htn', 'hypertension_target', 'hypertension'])

    def run_eval(model_filename, target_col, name, thresholds):
        model_path = os.path.join(WEIGHTS_DIR, model_filename)
        if not os.path.exists(model_path):
            print(f"[!] ไม่พบโมเดล: {model_filename}")
            return
            
        if not target_col:
            print(f"[!] ไม่พบคอลัมน์ Target สำหรับ {name}")
            return

        model = joblib.load(model_path)
        sub_df = target_df.dropna(subset=[target_col]).copy()

        raw_feature_cols = getattr(model, 'feature_names_in_', None)
        if raw_feature_cols is None and hasattr(model, 'get_booster'):
            raw_feature_cols = model.get_booster().feature_names

        feature_cols = [str(c) for c in raw_feature_cols] if raw_feature_cols is not None else []

        missing = [c for c in feature_cols if c not in sub_df.columns]
        if missing:
            print(f"[!] ฟีเจอร์ไม่ครบสำหรับ {name}: ขาด {missing}")
            return

        X = sub_df[feature_cols]
        y = (sub_df[target_col] == 1).astype(int)

        # หากเป็น Dataset ใหญ่ ให้ Split 15% Test Set ให้ตรงกับรายงาน
        if len(X) > 1000:
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
        else:
            X_test, y_test = X, y

        preds = model.predict_proba(X_test)[:, 1]

        print(f"\n--- {name.upper()} TRIAGE METRICS (TEST SET N={len(y_test)}, Total Positive={y_test.sum()}) ---")
        for i in range(len(tier_names)):
            low, high = thresholds[i], thresholds[i+1]
            mask = (preds >= low) if i == len(tier_names)-1 else ((preds >= low) & (preds < high))
            total = mask.sum()
            positives = y_test[mask].sum()
            prev = (positives / total * 100) if total > 0 else 0
            print(f"{tier_names[i]}: Total={total}, Positive Cases={positives} ({prev:.2f}%)")

    run_eval("xgb_classifier_diabetes.pkl", dia_target_col, "Diabetes", [0.00, 0.060, 0.100, 0.200, 1.00])
    run_eval("xgb_classifier_hypertension.pkl", htn_target_col, "Hypertension", [0.00, 0.155, 0.250, 0.500, 1.00])

else:
    print("[!] ไม่พบไฟล์ข้อมูลในโฟลเดอร์ data/")