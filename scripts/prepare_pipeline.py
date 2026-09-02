import torch
import joblib
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from loader import vit_transforms
from models import get_model

def load_all_models(device):
    print("กำลังโหลดระบบ AI ทั้งหมด...")
    # 1. โหลด ViT (Face to BMI) - ตัว 10 ชม.
    vit_model = get_model().float().to(device)
    vit_model.load_state_dict(torch.load('../weights/aug_epoch_7.pt', map_location=device))
    vit_model.eval()

    # 2. โหลด XGBoost Stage 1 (BMI to Body Fat)
    xgb_bodyfat = joblib.load('../weights/xgboost_bodyfat_with_waist.pkl')

    # 3. โหลด XGBoost Stage 2 (NCDs Classifiers)
    xgb_diab = joblib.load('../weights/xgb_classifier_diabetes.pkl')
    xgb_hyp = joblib.load('../weights/xgb_classifier_hypertension.pkl')
    
    return vit_model, xgb_bodyfat, xgb_diab, xgb_hyp

def analyze_body_shape(bmi, body_fat, gender):
    """ฟังก์ชันช่วยประเมินเกณฑ์รูปร่างจากตัวเลขที่ AI ทายได้"""
    if bmi < 18.5: bmi_cat = "น้ำหนักต่ำกว่าเกณฑ์"
    elif 18.5 <= bmi < 23.0: bmi_cat = "สมส่วน (Normal)"
    elif 23.0 <= bmi < 25.0: bmi_cat = "ท้วม (Overweight)"
    else: bmi_cat = "อ้วน (Obese)"
        
    if gender.lower() == 'male' or gender == 1:
        if body_fat < 14: bf_cat = "กล้ามเนื้อชัด (Lean/Athlete)"
        elif 14 <= body_fat < 18: bf_cat = "หุ่นฟิต (Fitness)"
        elif 18 <= body_fat < 25: bf_cat = "ทั่วไป (Acceptable)"
        else: bf_cat = "อ้วน (Obese)"
    else:
        if body_fat < 21: bf_cat = "กล้ามเนื้อชัด (Lean/Athlete)"
        elif 21 <= body_fat < 25: bf_cat = "หุ่นฟิต (Fitness)"
        elif 25 <= body_fat < 32: bf_cat = "ทั่วไป (Acceptable)"
        else: bf_cat = "อ้วน (Obese)"

    return bmi_cat, bf_cat

# 💡 เปลี่ยน Keyword ให้เข้าใจง่าย: normal, active, sedentary
def predict_health_risk(image_path, age, gender, waist_cm, lifestyle='normal'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vit_model, xgb_bodyfat, xgb_diab, xgb_hyp = load_all_models(device)
    
    gender_num = 1 if gender.lower() == 'male' else 0
    
    # จับคู่ Keyword เป็นข้อความภาษาไทยเพื่อแสดงผล
    lifestyle_map = {
        'active': 'ออกกำลังกายเป็นประจำ / มวลกล้ามเนื้อมาก (Active)',
        'sedentary': 'ไม่ออกกำลังกาย / อาจมีไขมันซ่อนรูป (Sedentary)',
        'normal': 'ทั่วไป / กิจกรรมปานกลาง (Normal)'
    }
    lifestyle_display = lifestyle_map.get(lifestyle.lower(), lifestyle_map['normal'])
    
    # --- STEP 1: Computer Vision (Face -> BMI) ---
    img = Image.open(image_path)
    if img.mode != 'RGB': img = img.convert('RGB')
        
    img_tensor = ToTensor()(img)
    img_tensor = vit_transforms(img_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_bmi = vit_model(img_tensor).item() 
        
    # --- STEP 1.5: Body Fat Prediction (Calibrate) ---
    df_stage1 = pd.DataFrame([[pred_bmi, age, gender_num, waist_cm]], 
                             columns=['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM'])
    raw_bodyfat = xgb_bodyfat.predict(df_stage1)[0]
    
    # 💡 ปรับเทียบข้อจำกัดตาม Lifestyle ใหม่
    if lifestyle.lower() == 'active':
        pred_bodyfat = max(5.0, raw_bodyfat - 10.0)
    elif lifestyle.lower() == 'sedentary':
        pred_bodyfat = raw_bodyfat + 5.0
    else:
        pred_bodyfat = raw_bodyfat
    
    bmi_cat, bf_cat = analyze_body_shape(pred_bmi, pred_bodyfat, gender)
    
    # --- STEP 2: NCDs Risk Prediction ---
    df_stage2 = pd.DataFrame([[age, gender_num, pred_bmi, waist_cm, pred_bodyfat]], 
                             columns=['AGE', 'GENDER_NUM', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT'])
    
    diab_risk_pct = xgb_diab.predict_proba(df_stage2)[0][1] * 100
    hyp_risk_pct = xgb_hyp.predict_proba(df_stage2)[0][1] * 100
    
    # --- แสดงผลสรุป ---
    print("\n" + "="*55)
    print("🏥 สรุปผลการวิเคราะห์สุขภาพ AI Pipeline 🏥")
    print("="*55)
    print(f"ผู้รับการประเมิน: {gender.capitalize()}, อายุ {age} ปี, รอบเอว {waist_cm} ซม.")
    print(f"รูปแบบการใช้ชีวิต: {lifestyle_display}")
    print("-" * 55)
    
    print(f"📷 [Stage 1] ประเมินจากใบหน้า (ViT Model)")
    print(f"   > ค่า BMI ที่ทำนายได้   : {pred_bmi:.2f} [{bmi_cat}]")
    print(f"📊 [Stage 1.5] ประเมินไขมัน (XGBoost)")
    print(f"   > เปอร์เซ็นต์ไขมันรวม    : {pred_bodyfat:.2f} % [{bf_cat}]")
    
    print("-" * 55)
    print(f"🔍 [ปรับเทียบข้อจำกัดตามไลฟ์สไตล์ (Boundary Adjustment)]")
    if lifestyle.lower() == 'active':
        print(f"   💪 บริบท: ออกกำลังกายสม่ำเสมอ/มวลกล้ามเนื้อมาก")
        print(f"      - AI ตั้งต้นประเมินไขมันจากน้ำหนักตัวได้: {raw_bodyfat:.2f}%")
        print(f"      - Calibrate หักลบมวลกล้ามเนื้อออก เหลือไขมันจริง: {pred_bodyfat:.2f}%")
    elif lifestyle.lower() == 'sedentary':
        print(f"   ⚠️ บริบท: ไม่ออกกำลังกาย/ไขมันสะสมซ่อนรูป")
        print(f"      - AI ตั้งต้นประเมินไขมันจากน้ำหนักตัวได้: {raw_bodyfat:.2f}%")
        print(f"      - Calibrate บวกไขมันสะสมช่วงท้องเพิ่ม เป็น: {pred_bodyfat:.2f}%")
    else:
        print(f"   🚶‍♂️ บริบท: คนทั่วไป (ค่าที่ได้สอดคล้องกับสถิติประชากรปกติ)")
        
    print("-" * 55)
    print(f"🩺 [Stage 2] คัดกรองความเสี่ยงโรค NCDs")
    diab_status = "🔴 เสี่ยงสูง" if diab_risk_pct > 50 else "🟢 ปกติ"
    hyp_status = "🔴 เสี่ยงสูง" if hyp_risk_pct > 50 else "🟢 ปกติ"
    print(f"   > โรคเบาหวาน           : {diab_risk_pct:.1f}% ({diab_status})")
    print(f"   > โรคความดันโลหิตสูง     : {hyp_risk_pct:.1f}% ({hyp_status})")
    print("="*55 + "\n")

if __name__ == "__main__":
    test_image = "../data/test_images/testpic11.png"
    # ทดสอบรันด้วย CR7 โดยป้อนไลฟ์สไตล์เป็น 'active'
    predict_health_risk(test_image, age=32, gender='Male', waist_cm=80.0, lifestyle='active')