import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os

# 1. โหลดข้อมูล (เปลี่ยนชื่อไฟล์ให้ตรงกับที่คุณมีอยู่)
# สมมติว่าไฟล์นี้มีคอลัมน์: ['y_pred', 'y_true', 'resid']
df = pd.read_csv('stage1_residuals.csv')

# 2. แยก Features และ Target
X = df[['y_pred']] # ค่าที่ ViT ทายได้
y = df['resid']    # ค่าความผิดพลาดจริง

# 3. เทรนโมเดล XGBoost
model_resid = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
model_resid.fit(X, y)

# 4. วัดผลหลังแก้
resid_pred = model_resid.predict(X)
y_corrected = X['y_pred'] + resid_pred

mae_raw = mean_absolute_error(df['y_pred'], df['y_true'])
mae_corrected = mean_absolute_error(y_corrected, df['y_true'])

print(f"MAE ก่อนแก้: {mae_raw:.4f}")
print(f"MAE หลังแก้: {mae_corrected:.4f}")

# 5. บันทึกโมเดล (แยกไปเก็บในโฟลเดอร์ models ตามที่เราคุยกัน)
if not os.path.exists('models'):
    os.makedirs('models')
model_resid.save_model('../models/residual_corrector.json')
print("Saved model to ../models/residual_corrector.json")