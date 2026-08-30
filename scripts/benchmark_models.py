import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 1. โหลดและเตรียมข้อมูล
df = pd.read_csv('nhanes_cleaned_merged_final.csv')
df['GENDER_NUM'] = df['GENDER'].map({'Male': 1, 'Female': 0})

X = df[['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM']]
y = df['TOTAL_BODY_FAT_PCT']

# 2. แบ่งข้อมูล Train 80% / Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. สร้างโมเดลทั้ง 3 ตัว
models = {
    "Linear Regression (Baseline)": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
}

# 4. ฟังก์ชันคำนวณ Custom Accuracy (ยอมรับ Error ได้ไม่เกิน margin %)
def calculate_custom_accuracy(y_true, y_pred, margin=3.0):
    diff = np.abs(y_true - y_pred)
    correct = np.sum(diff <= margin)
    return (correct / len(y_true)) * 100

results = []

print("กำลังฝึกสอนและทดสอบโมเดลทั้งหมด กรุณารอสักครู่...\n")
for name, model in models.items():
    # เทรนโมเดล
    model.fit(X_train, y_train)
    
    # ทดสอบกับ Test Set
    y_pred = model.predict(X_test)
    
    # คำนวณตัวชี้วัดต่างๆ
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc_3pct = calculate_custom_accuracy(y_test.values, y_pred, margin=3.0)
    acc_5pct = calculate_custom_accuracy(y_test.values, y_pred, margin=5.0)
    
    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Acc (±3%)": acc_3pct,
        "Acc (±5%)": acc_5pct
    })
    
    # เซฟโมเดลเก็บไว้
    filename = f"model_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(model, filename)

# 5. แสดงผลตารางเปรียบเทียบ
results_df = pd.DataFrame(results)
print("=" * 80)
print(f"{'Model Name':<30} | {'R2':<6} | {'MAE':<6} | {'RMSE':<6} | {'Acc(±3%)':<9} | {'Acc(±5%)':<9}")
print("=" * 80)
for _, row in results_df.iterrows():
    print(f"{row['Model']:<30} | {row['R2']:.4f} | {row['MAE']:.2f}% | {row['RMSE']:.2f}% | {row['Acc (±3%)']:.1f}%    | {row['Acc (±5%)']:.1f}%")
print("=" * 80)