import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

os.makedirs('../weights', exist_ok=True)

# 1. โหลดข้อมูล
df = pd.read_csv('../data/processed/nhanes_cleaned_merged_final.csv')
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})
y = df['TOTAL_BODY_FAT_PCT']

# 2. เทรนโมเดลที่ 1: แบบ "มี" รอบเอว (Full Model)
X_full = df[['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']]
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)
model_full = LinearRegression().fit(X_train_full, y_train)
joblib.dump(model_full, '../weights/custom_bodyfat_model_with_waist.pkl')

# 3. เทรนโมเดลที่ 2: แบบ "ไม่มี" รอบเอว (Basic Model)
X_basic = df[['BMI', 'AGE', 'GENDER_NUM']]
X_train_basic, X_test_basic, _, _ = train_test_split(
    X_basic, y, test_size=0.2, random_state=42
)
model_basic = LinearRegression().fit(X_train_basic, y_train)
joblib.dump(model_basic, '../weights/custom_bodyfat_model_no_waist.pkl')

print("✅ เทรนและเซฟ Linear Regression (.pkl) ทั้ง 2 ตัวลงใน weights/ เรียบร้อย!")