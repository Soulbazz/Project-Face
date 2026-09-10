import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# สร้างโฟลเดอร์ weights อัตโนมัติหากยังไม่มี
os.makedirs('../weights', exist_ok=True)

# 1. โหลดข้อมูลจากโฟลเดอร์ data (../data/)
print("กำลังโหลดข้อมูลและฝึกสอน Random Forest...")
df = pd.read_csv('../data/processed/nhanes_cleaned_merged_final.csv')
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})

y = df['TOTAL_BODY_FAT_PCT']

# hyperparameter เบื้องต้น (ยังไม่ได้ tune) 
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,  # ใช้ทุก core ช่วยให้เทรนเร็วขึ้น
)

# ==========================================
# โมเดลที่ 1: Random Forest แบบ "มีรอบเอว" (Full)
# ==========================================
X_full = df[['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']]
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

rf_full = RandomForestRegressor(**RF_PARAMS)
rf_full.fit(X_train_full, y_train)

y_pred_full = rf_full.predict(X_test_full)
r2_full = r2_score(y_test, y_pred_full)
mae_full = mean_absolute_error(y_test, y_pred_full)

print("\n📊 === Random Forest (แบบมีรอบเอว) ===")
print(f"R2 : {r2_full:.4f}")
print(f"MAE: ±{mae_full:.2f} %")

# เกร็ดความรู้เสริม: ดูว่าฟีเจอร์ไหนมีน้ำหนักในการทำนายมากสุด
importances_full = pd.Series(
    rf_full.feature_importances_, index=X_full.columns
).sort_values(ascending=False)
print("\n🔍 Feature Importance (Full Model):")
print(importances_full.to_string())

# เซฟลงโฟลเดอร์ weights
joblib.dump(rf_full, '../weights/randomforest_bodyfat_with_waist.pkl')

# ==========================================
# โมเดลที่ 2: Random Forest แบบ "ไม่มีรอบเอว" (Basic)
# ==========================================
X_basic = df[['BMI', 'AGE', 'GENDER_NUM']]
X_train_basic, X_test_basic, _, _ = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)

rf_basic = RandomForestRegressor(**RF_PARAMS)
rf_basic.fit(X_train_basic, y_train)

y_pred_basic = rf_basic.predict(X_test_basic)
r2_basic = r2_score(y_test, y_pred_basic)
mae_basic = mean_absolute_error(y_test, y_pred_basic)

print("\n📊 === Random Forest (แบบไม่มีรอบเอว) ===")
print(f"R2 : {r2_basic:.4f}")
print(f"MAE: ±{mae_basic:.2f} %")

importances_basic = pd.Series(
    rf_basic.feature_importances_, index=X_basic.columns
).sort_values(ascending=False)
print("\n🔍 Feature Importance (Basic Model):")
print(importances_basic.to_string())

# เซฟลงโฟลเดอร์ weights
joblib.dump(rf_basic, '../weights/randomforest_bodyfat_no_waist.pkl')

print("\n✅ เซฟไฟล์ Random Forest (.pkl) ทั้ง 2 ตัวลงใน weights/ เรียบร้อย!")