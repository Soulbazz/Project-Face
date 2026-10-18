import os
import joblib
import pandas as pd

# 1. กำหนด Path อ้างอิงโฟลเดอร์ weights ของโปรเจกต์
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# 2. โหลดโมเดล XGBoost Body Fat (ที่มีการใช้รอบเอว)
model_path = os.path.join(WEIGHTS_DIR, "xgboost_bodyfat_with_waist.pkl")

if not os.path.exists(model_path):
    print(f"[!] ไม่พบไฟล์โมเดลที่: {model_path}")
else:
    model = joblib.load(model_path)
    
    # 3. ดึงชื่อฟีเจอร์และค่า Feature Importance
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        feature_names = model.get_booster().feature_names
        
    importances = model.feature_importances_
    
    # 4. แปลงเป็น Percentage (%) และจัดเรียงจากมากไปน้อย
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance (%)': importances * 100
    }).sort_values(by='Importance (%)', ascending=False)
    
    print("=== AUXILIARY MODEL (WITH WAIST) FEATURE IMPORTANCE ===")
    print(df_imp.to_string(index=False))