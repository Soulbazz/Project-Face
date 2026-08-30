import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. โหลดข้อมูลจากโฟลเดอร์ data (../data/)
print("กำลังโหลดข้อมูลและฝึกสอน XGBoost...")
df = pd.read_csv('../data/nhanes_cleaned_merged_final.csv')
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})

y = df['TOTAL_BODY_FAT_PCT']

# ==========================================
# โมเดลที่ 1: XGBoost แบบ "มีรอบเอว" (Full)
# ==========================================
X_full = df[['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']]
X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)

xgb_full = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
xgb_full.fit(X_train_full, y_train)

y_pred_full = xgb_full.predict(X_test_full)
print("\n📊 === XGBoost (แบบมีรอบเอว) ===")
print(f"MAE: ±{mean_absolute_error(y_test, y_pred_full):.2f} %")

# เซฟโมเดล
joblib.dump(xgb_full, 'xgboost_bodyfat_with_waist.pkl')


# ==========================================
# โมเดลที่ 2: XGBoost แบบ "ไม่มีรอบเอว" (Basic)
# ==========================================
X_basic = df[['BMI', 'AGE', 'GENDER_NUM']]
X_train_basic, X_test_basic, _, _ = train_test_split(X_basic, y, test_size=0.2, random_state=42)

xgb_basic = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
xgb_basic.fit(X_train_basic, y_train)

y_pred_basic = xgb_basic.predict(X_test_basic)
print("\n📊 === XGBoost (แบบไม่มีรอบเอว) ===")
print(f"MAE: ±{mean_absolute_error(y_test, y_pred_basic):.2f} %")

# เซฟโมเดล
joblib.dump(xgb_basic, 'xgboost_bodyfat_no_waist.pkl')

print("\n✅ เซฟไฟล์ XGBoost (.pkl) ทั้ง 2 ตัวเรียบร้อย พร้อมใช้งานกับ GUI แล้ว!")