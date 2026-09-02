import torch
import joblib
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from loader import vit_transforms
from models import get_model

def load_all_models(device):
    print("กำลังโหลดระบบ AI ทั้งหมด...")
    vit_model = get_model().float().to(device)
    vit_model.load_state_dict(torch.load('../weights/aug_epoch_7.pt', map_location=device))
    vit_model.eval()

    # โหลดเตรียมไว้ทั้ง 2 ชุด (แบบมีเอว และ ไม่มีเอว)
    models = {
        'vit': vit_model,
        'bf_waist': joblib.load('../weights/xgboost_bodyfat_with_waist.pkl'),
        'bf_nowaist': joblib.load('../weights/xgboost_bodyfat_no_waist.pkl'),
        'diab_waist': joblib.load('../weights/xgb_classifier_diabetes.pkl'),
        'diab_nowaist': joblib.load('../weights/xgb_classifier_diabetes_no_waist.pkl'),
        'hyp_waist': joblib.load('../weights/xgb_classifier_hypertension.pkl'),
        'hyp_nowaist': joblib.load('../weights/xgb_classifier_hypertension_no_waist.pkl')
    }
    return models

def analyze_body_shape(bmi, body_fat, gender):
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

# 💡 กำหนดให้ waist_cm=None เป็นค่าเริ่มต้น (Optional)
def predict_health_risk(image_path, age, gender, waist_cm=None, lifestyle='normal'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = load_all_models(device)
    gender_num = 1 if gender.lower() == 'male' else 0
    
    lifestyle_map = {
        'active': 'ออกกำลังกายเป็นประจำ / มวลกล้ามเนื้อมาก (Active)',
        'sedentary': 'ไม่ออกกำลังกาย / อาจมีไขมันซ่อนรูป (Sedentary)',
        'normal': 'ทั่วไป / กิจกรรมปานกลาง (Normal)'
    }
    lifestyle_display = lifestyle_map.get(lifestyle.lower(), lifestyle_map['normal'])
    
    # 💡 เช็กว่า User ใส่รอบเอวมาหรือไม่?
    has_waist = waist_cm is not None and waist_cm > 0
    mode_text = "ประเมินแบบเต็มรูปแบบ (ใช้รอบเอว)" if has_waist else "ประเมินแบบพื้นฐาน (ไม่ใช้รอบเอว)"
    
    # --- STEP 1: Face -> BMI ---
    img = Image.open(image_path)
    if img.mode != 'RGB': img = img.convert('RGB')
    img_tensor = ToTensor()(img)
    img_tensor = vit_transforms(img_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_bmi = models['vit'](img_tensor).item() 
        
    # --- STEP 1.5: Body Fat & NCDs (แยก Route ตามการใส่รอบเอว) ---
    if has_waist:
        # Route A: แบบมีรอบเอว
        df_bf = pd.DataFrame([[pred_bmi, age, gender_num, waist_cm]], columns=['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM'])
        raw_bodyfat = models['bf_waist'].predict(df_bf)[0]
        
        # Calibrate Body Fat
        if lifestyle.lower() == 'active': pred_bodyfat = max(5.0, raw_bodyfat - 10.0)
        elif lifestyle.lower() == 'sedentary': pred_bodyfat = raw_bodyfat + 5.0
        else: pred_bodyfat = raw_bodyfat
        
        df_ncd = pd.DataFrame([[age, gender_num, pred_bmi, waist_cm, pred_bodyfat]], columns=['AGE', 'GENDER_NUM', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT'])
        diab_risk_pct = models['diab_waist'].predict_proba(df_ncd)[0][1] * 100
        hyp_risk_pct = models['hyp_waist'].predict_proba(df_ncd)[0][1] * 100
        
    else:
        # Route B: แบบไม่มีรอบเอว
        df_bf = pd.DataFrame([[pred_bmi, age, gender_num]], columns=['BMI', 'AGE', 'GENDER_NUM'])
        raw_bodyfat = models['bf_nowaist'].predict(df_bf)[0]
        
        # Calibrate Body Fat
        if lifestyle.lower() == 'active': pred_bodyfat = max(5.0, raw_bodyfat - 10.0)
        elif lifestyle.lower() == 'sedentary': pred_bodyfat = raw_bodyfat + 5.0
        else: pred_bodyfat = raw_bodyfat
        
        df_ncd = pd.DataFrame([[age, gender_num, pred_bmi, pred_bodyfat]], columns=['AGE', 'GENDER_NUM', 'BMI', 'TOTAL_BODY_FAT_PCT'])
        diab_risk_pct = models['diab_nowaist'].predict_proba(df_ncd)[0][1] * 100
        hyp_risk_pct = models['hyp_nowaist'].predict_proba(df_ncd)[0][1] * 100

    bmi_cat, bf_cat = analyze_body_shape(pred_bmi, pred_bodyfat, gender)
    
    # --- แสดงผลสรุป ---
    print("\n" + "="*60)
    print("🏥 สรุปผลการวิเคราะห์สุขภาพ AI Pipeline 🏥")
    print("="*60)
    waist_display = f"{waist_cm} ซม." if has_waist else "ไม่ได้ระบุ"
    print(f"ผู้รับการประเมิน: {gender.capitalize()}, อายุ {age} ปี, รอบเอว: {waist_display}")
    print(f"รูปแบบการใช้ชีวิต: {lifestyle_display}")
    print(f"โหมดการประเมิน: {mode_text}")
    print("-" * 60)
    
    print(f"📷 [Stage 1] ประเมินจากใบหน้า (ViT Model)")
    print(f"   > ค่า BMI ที่ทำนายได้   : {pred_bmi:.2f} [{bmi_cat}]")
    print(f"📊 [Stage 1.5] ประเมินไขมัน (XGBoost)")
    print(f"   > เปอร์เซ็นต์ไขมันรวม    : {pred_bodyfat:.2f} % [{bf_cat}]")
    
    print("-" * 60)
    print(f"🩺 [Stage 2] คัดกรองความเสี่ยงโรค NCDs")
    diab_status = "🔴 เสี่ยงสูง" if diab_risk_pct > 50 else "🟢 ปกติ"
    hyp_status = "🔴 เสี่ยงสูง" if hyp_risk_pct > 50 else "🟢 ปกติ"
    print(f"   > โรคเบาหวาน           : {diab_risk_pct:.1f}% ({diab_status})")
    print(f"   > โรคความดันโลหิตสูง     : {hyp_risk_pct:.1f}% ({hyp_status})")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_image = "../data/test_images/testpic11.png"
    
    # เทสต์แบบใส่รอบเอว
    print("\n>>> ทดสอบแบบที่ 1: ใส่รอบเอว <<<")
    predict_health_risk(test_image, age=32, gender='Male', waist_cm=80.0, lifestyle='active')
    
    # เทสต์แบบ ไม่ใส่รอบเอว (Optional)
    print("\n>>> ทดสอบแบบที่ 2: ไม่ใส่รอบเอว (ทิ้งว่าง) <<<")
    predict_health_risk(test_image, age=32, gender='Male', waist_cm=None, lifestyle='active')