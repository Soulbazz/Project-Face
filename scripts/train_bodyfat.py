import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 1. โหลดข้อมูล
df = pd.read_csv('nhanes_cleaned_merged_final.csv')
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})

# 2. เทรนโมเดลที่ 1: แบบ "มี" รอบเอว (Full Model)
X_full = df[['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']]
y = df['TOTAL_BODY_FAT_PCT']
model_full = LinearRegression().fit(X_full, y)
joblib.dump(model_full, 'custom_bodyfat_model_with_waist.pkl')

# 3. เทรนโมเดลที่ 2: แบบ "ไม่มี" รอบเอว (Basic Model ใช้แค่ 3 ตัวแปร)
X_basic = df[['BMI', 'AGE', 'GENDER_NUM']]
model_basic = LinearRegression().fit(X_basic, y)
joblib.dump(model_basic, 'custom_bodyfat_model_no_waist.pkl')

print("✅ เทรนและเซฟทั้ง 2 โมเดลเสร็จเรียบร้อย! พร้อมนำไปประกอบร่าง")