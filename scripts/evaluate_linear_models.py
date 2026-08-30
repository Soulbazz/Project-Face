import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# 1. โหลดข้อมูลฐานข้อมูล (เปรียบเสมือนสมุดข้อสอบพร้อมเฉลย)
df = pd.read_csv('nhanes_cleaned_merged_final.csv')
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})

# 2. โหลดเฉลย (Ground Truth จากเครื่อง DXA Scan)
y_true = df['TOTAL_BODY_FAT_PCT']

# ==========================================
# ประเมินโมเดลที่ 1: แบบ "ไม่มี" รอบเอว (Basic Model)
# ==========================================
X_basic = df[['BMI', 'AGE', 'GENDER_NUM']]
model_basic = joblib.load('custom_bodyfat_model_no_waist.pkl')
y_pred_basic = model_basic.predict(X_basic)

r2_basic = r2_score(y_true, y_pred_basic)
mae_basic = mean_absolute_error(y_true, y_pred_basic)

# ==========================================
# ประเมินโมเดลที่ 2: แบบ "มี" รอบเอว (Full Model)
# ==========================================
X_full = df[['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']]
model_full = joblib.load('custom_bodyfat_model_with_waist.pkl')
y_pred_full = model_full.predict(X_full)

r2_full = r2_score(y_true, y_pred_full)
mae_full = mean_absolute_error(y_true, y_pred_full)

# ==========================================
# พิมพ์สรุปผลลัพธ์เพื่อนำไปใส่รายงาน
# ==========================================
print("\n📊 === สรุปความแม่นยำของโมเดล Body Fat (เปรียบเทียบกับ DXA Scan) ===")
print("-" * 60)
print("1. โมเดลพื้นฐาน (ใช้แค่ BMI, อายุ, เพศ):")
print(f"   • R-squared (ความเชื่อมั่น): {r2_basic:.4f}")
print(f"   • MAE (ความคลาดเคลื่อนเฉลี่ย): ±{mae_basic:.2f} %")
print("-" * 60)
print("2. โมเดลจัดเต็ม (เพิ่ม 'รอบเอว' เข้ามาช่วย):")
print(f"   • R-squared (ความเชื่อมั่น): {r2_full:.4f}")
print(f"   • MAE (ความคลาดเคลื่อนเฉลี่ย): ±{mae_full:.2f} %")
print("-" * 60)
print("💡 สรุป: การเพิ่ม 'รอบเอว' ช่วยลดความคลาดเคลื่อนได้เท่าไหร่ ลองดูที่ค่า MAE ได้เลยครับ!")