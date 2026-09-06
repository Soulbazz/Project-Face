import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# ---------- PATH กลาง แก้ที่เดียวจบ ----------
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project Face/
DATA_DIR  = os.path.join(ROOT, 'data')
PROC_DIR  = os.path.join(DATA_DIR, 'processed')
WEIGHTS   = os.path.join(ROOT, 'weights')
RESULTS   = os.path.join(ROOT, 'scripts', 'results')

for d in (WEIGHTS, RESULTS):
    os.makedirs(d, exist_ok=True)

print("กำลังโหลดข้อมูลและฝึกสอน XGBoost...")
df = pd.read_csv(os.path.join(PROC_DIR, 'nhanes_cleaned_merged_final.csv'))
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})

# ---------- ลำดับ feature: ล็อกไว้ตรงนี้ ห้ามสลับ ----------
FEATS_FULL  = ['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']
FEATS_BASIC = ['BMI', 'AGE', 'GENDER_NUM']
TARGET      = 'TOTAL_BODY_FAT_PCT'

df_model = df.dropna(subset=FEATS_FULL + [TARGET]).reset_index(drop=True)
print(f"ข้อมูลที่ใช้ได้: {len(df_model)} / {len(df)} แถว")

train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)
y_train, y_test = train_df[TARGET], test_df[TARGET]
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ---------- เซฟลง data/ ตัวหลัก (ไม่ใช่ scripts/data/) ----------
test_df.to_csv(os.path.join(DATA_DIR, 'test_tabular.csv'), index=False)
train_df.to_csv(os.path.join(DATA_DIR, 'train_tabular.csv'), index=False)
print(f"✅ เซฟ test/train_tabular.csv ลง {DATA_DIR}")

# ---------- โมเดล 1: มีรอบเอว ----------
xgb_full = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
xgb_full.fit(train_df[FEATS_FULL], y_train)
mae_full = mean_absolute_error(y_test, xgb_full.predict(test_df[FEATS_FULL]))
print(f"\n=== XGBoost (มีรอบเอว) ===\nMAE: {mae_full:.3f} %")
joblib.dump(xgb_full, os.path.join(WEIGHTS, 'xgboost_bodyfat_with_waist.pkl'))

# ---------- โมเดล 2: ไม่มีรอบเอว ----------
xgb_basic = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
xgb_basic.fit(train_df[FEATS_BASIC], y_train)
mae_basic = mean_absolute_error(y_test, xgb_basic.predict(test_df[FEATS_BASIC]))
print(f"\n=== XGBoost (ไม่มีรอบเอว) ===\nMAE: {mae_basic:.3f} %")
joblib.dump(xgb_basic, os.path.join(WEIGHTS, 'xgboost_bodyfat_no_waist.pkl'))

# ---------- สรุป ----------
gain = mae_basic - mae_full
print("\n" + "="*52)
print(f"MAE มีรอบเอว    : {mae_full:.3f} %")
print(f"MAE ไม่มีรอบเอว  : {mae_basic:.3f} %")
print(f"รอบเอวช่วยลด error {gain:.3f} % ({gain/mae_basic*100:.1f}% relative)")
print("="*52)

pd.DataFrame([
    {"model": "with_waist", "mae": round(mae_full, 3),  "n_test": len(test_df)},
    {"model": "no_waist",   "mae": round(mae_basic, 3), "n_test": len(test_df)},
]).to_csv(os.path.join(RESULTS, 'xgboost_train_summary.csv'), index=False)

print("\n✅ เสร็จเรียบร้อย")