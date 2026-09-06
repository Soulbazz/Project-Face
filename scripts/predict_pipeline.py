"""
predict_pipeline.py  (v2 - Web-Ready)
──────────────────────────────────────────────────────────────
สิ่งที่แก้จากเวอร์ชันเดิม:
  [1] load_all_models() แยกออกมาให้เรียกครั้งเดียวได้ (cache ได้)
  [2] predict_health_risk() คืน dict แทน print() → ใช้กับ UI ได้
  [3] Path ใช้ pathlib อ้างอิงจากตำแหน่งไฟล์ → ไม่พังเมื่อเปลี่ยน cwd
  [4] รับภาพได้ทั้ง path / bytes / PIL.Image / file-like object
  [5] แก้ EXIF rotation อัตโนมัติ (รูปจากมือถือ)
  [6] เพิ่ม Face Detection Guard (optional)
  [7] validate input (อายุ, รอบเอว, lifestyle)
  [8] CLI เดิมยังใช้ได้ผ่าน print_report()
──────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Optional, Union, Any

import joblib
import pandas as pd
import torch
from PIL import Image, ImageOps

from torchvision.transforms import ToTensor
from loader import vit_transforms
from models import get_model

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════
# [FIX 3] PATH CONFIG — อ้างอิงจากไฟล์นี้ ไม่ใช่ current working directory
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = (BASE_DIR.parent / "weights").resolve()

WEIGHT_FILES = {
    "vit":          "aug_epoch_7_backup2.pt",
    "bf_waist":     "xgboost_bodyfat_with_waist.pkl",
    "bf_nowaist":   "xgboost_bodyfat_no_waist.pkl",
    "diab_waist":   "xgb_classifier_diabetes.pkl",
    "diab_nowaist": "xgb_classifier_diabetes_no_waist.pkl",
    "hyp_waist":    "xgb_classifier_hypertension.pkl",
    "hyp_nowaist":  "xgb_classifier_hypertension_no_waist.pkl",
}

# ══════════════════════════════════════════════════════════
# CONFIG: Lifestyle Calibration (แยกเป็น config → แก้ง่าย ไม่ต้องไล่หา if-else)
# ══════════════════════════════════════════════════════════
LIFESTYLE_META = {
    "active": {
        "icon": "🏋️",
        "title": "ออกกำลังกายเป็นประจำ",
        "subtitle": "มวลกล้ามเนื้อมาก / เล่นเวทสม่ำเสมอ",
        "display": "ออกกำลังกายเป็นประจำ / มวลกล้ามเนื้อมาก (Active)",
        "delta": -10.0,
        "reason": ("ผู้ที่มีมวลกล้ามเนื้อสูงมักมีค่า BMI สูงกว่าปกติทั้งที่ไขมันจริงต่ำ "
                   "ระบบจึงปรับลดเปอร์เซ็นต์ไขมันลงเพื่อชดเชยน้ำหนักส่วนที่เป็นกล้ามเนื้อ"),
    },
    "normal": {
        "icon": "🚶",
        "title": "กิจกรรมปานกลาง",
        "subtitle": "ใช้ชีวิตทั่วไป ออกกำลังกายบ้าง",
        "display": "ทั่วไป / กิจกรรมปานกลาง (Normal)",
        "delta": 0.0,
        "reason": "ใช้ค่าที่โมเดลทำนายโดยตรง ไม่มีการปรับชดเชย",
    },
    "sedentary": {
        "icon": "💺",
        "title": "แทบไม่ออกกำลังกาย",
        "subtitle": "นั่งทำงานทั้งวัน / อาจมีไขมันแทรก (Skinny Fat)",
        "display": "ไม่ออกกำลังกาย / อาจมีไขมันซ่อนรูป (Sedentary)",
        "delta": +5.0,
        "reason": ("กลุ่มนี้มักมีไขมันแทรกในช่องท้องสูงกว่าที่ BMI บ่งชี้ (ภาวะ Skinny Fat) "
                   "ระบบจึงปรับเพิ่มเปอร์เซ็นต์ไขมันเพื่อไม่ให้ประเมินความเสี่ยงต่ำเกินจริง"),
    },
}

BODYFAT_FLOOR = 5.0    # Essential fat — เพดานล่างเชิงสรีรวิทยา
RISK_THRESHOLD = 50.0  # เกณฑ์ "เสี่ยงสูง" ตาม pipeline เดิม


# ══════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_weights() -> list[str]:
    """คืน list ชื่อไฟล์ weight ที่หายไป (ใช้เช็คตอนเปิดแอป)"""
    return [f for f in WEIGHT_FILES.values() if not (WEIGHTS_DIR / f).exists()]


# ══════════════════════════════════════════════════════════
# [FIX 1] LOAD MODELS — เรียกครั้งเดียว แล้วส่ง dict ไปใช้ซ้ำ
# ══════════════════════════════════════════════════════════
def load_all_models(device: Optional[str] = None, verbose: bool = False) -> dict:
    device = device or get_device()
    if verbose:
        print(f"กำลังโหลดระบบ AI ทั้งหมด... (device={device})")

    missing = check_weights()
    if missing:
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดลใน {WEIGHTS_DIR}: {', '.join(missing)}"
        )

    vit_model = get_model().float().to(device)
    vit_model.load_state_dict(
        torch.load(WEIGHTS_DIR / WEIGHT_FILES["vit"], map_location=device)
    )
    vit_model.eval()

    models: dict[str, Any] = {"vit": vit_model, "_device": device}
    for key in ("bf_waist", "bf_nowaist", "diab_waist",
                "diab_nowaist", "hyp_waist", "hyp_nowaist"):
        models[key] = joblib.load(WEIGHTS_DIR / WEIGHT_FILES[key])

    if verbose:
        print("โหลดโมเดลครบทั้งหมดแล้ว ✅")
    return models


# ══════════════════════════════════════════════════════════
# [FIX 4+5] IMAGE LOADER — รับได้ทุกรูปแบบ + แก้ EXIF
# ══════════════════════════════════════════════════════════
def to_pil(image: Union[str, Path, bytes, bytearray, Image.Image, Any]) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif hasattr(image, "read"):          # file-like (UploadedFile / BytesIO)
        try:
            image.seek(0)
        except Exception:
            pass
        img = Image.open(image)
    else:
        raise TypeError(f"ไม่รองรับ input ชนิด {type(image).__name__}")

    img = ImageOps.exif_transpose(img)    # [FIX 5] แก้รูปหมุนจากมือถือ
    return img.convert("RGB")


# ══════════════════════════════════════════════════════════
# CLASSIFICATION HELPERS
# ══════════════════════════════════════════════════════════
def analyze_body_shape(bmi: float, body_fat: float, gender) -> dict:
    if bmi < 18.5:
        bmi_cat, bmi_key = "น้ำหนักต่ำกว่าเกณฑ์", "under"
    elif bmi < 23.0:
        bmi_cat, bmi_key = "สมส่วน (Normal)", "normal"
    elif bmi < 25.0:
        bmi_cat, bmi_key = "ท้วม (Overweight)", "over"
    else:
        bmi_cat, bmi_key = "อ้วน (Obese)", "obese"

    is_male = (str(gender).lower() == "male") or (gender == 1)
    cuts = [14, 18, 25] if is_male else [21, 25, 32]

    if body_fat < cuts[0]:
        bf_cat, bf_key = "กล้ามเนื้อชัด (Lean/Athlete)", "lean"
    elif body_fat < cuts[1]:
        bf_cat, bf_key = "หุ่นฟิต (Fitness)", "fit"
    elif body_fat < cuts[2]:
        bf_cat, bf_key = "ทั่วไป (Acceptable)", "normal"
    else:
        bf_cat, bf_key = "อ้วน (Obese)", "obese"

    return {"bmi_cat": bmi_cat, "bmi_key": bmi_key,
            "bf_cat": bf_cat, "bf_key": bf_key, "bf_cuts": cuts}


def get_risk_level(pct: float) -> dict:
    if pct < 40:
        return {"label": "ความเสี่ยงต่ำ", "key": "low",
                "color": "#16a34a", "bg": "#dcfce7", "icon": "🟢",
                "rgb": (22, 163, 74)}
    if pct < RISK_THRESHOLD: # (RISK_THRESHOLD คือ 50)
        return {"label": "เฝ้าระวัง", "key": "watch",
                "color": "#ca8a04", "bg": "#fef9c3", "icon": "🟡",
                "rgb": (202, 138, 4)}
    if pct < 75:
        return {"label": "ความเสี่ยงสูง", "key": "high",
                "color": "#ea580c", "bg": "#ffedd5", "icon": "🟠",
                "rgb": (234, 88, 12)}
    return {"label": "ความเสี่ยงสูงมาก", "key": "critical",
            "color": "#dc2626", "bg": "#fee2e2", "icon": "🔴",
            "rgb": (220, 38, 38)}


# ══════════════════════════════════════════════════════════
# [FIX 2] MAIN PREDICTION — คืน dict แทน print
# ══════════════════════════════════════════════════════════
def predict_health_risk(
    image: Union[str, Path, bytes, Image.Image, Any],
    age: int,
    gender: str,
    waist_cm: Optional[float] = None,
    lifestyle: str = "normal",
    models: Optional[dict] = None,
    face_guard: bool = True,
) -> dict:
    # ---------- [FIX 7] Validate ----------
    age = int(age)
    if not (10 <= age <= 100):
        raise ValueError("อายุต้องอยู่ระหว่าง 10–100 ปี")

    lifestyle = str(lifestyle).lower()
    if lifestyle not in LIFESTYLE_META:
        lifestyle = "normal"
    ls_meta = LIFESTYLE_META[lifestyle]

    has_waist = waist_cm is not None and float(waist_cm) > 0
    if has_waist:
        waist_cm = float(waist_cm)
        if not (40.0 <= waist_cm <= 200.0):
            raise ValueError("รอบเอวต้องอยู่ระหว่าง 40–200 ซม.")

    gender_num = 1 if str(gender).lower() == "male" else 0

    models = models or load_all_models()
    device = models.get("_device", get_device())

    # ---------- [FIX 4] Load image ----------
    img = to_pil(image)

    # ---------- [FIX 6] FACE GUARD ----------
    face_report = None
    if face_guard:
        from face_guard import check_face
        face_report = check_face(img)
        if not face_report.ok:
            raise FaceGuardError(face_report)

    # ---------- STAGE 1: Face → BMI (ViT) ----------
    img_tensor = vit_transforms(ToTensor()(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_bmi = float(models["vit"](img_tensor).item())

    # ---------- STAGE 1.5: Body Fat (Dual Route) ----------
    if has_waist:
        df_bf = pd.DataFrame([[pred_bmi, age, gender_num, waist_cm]],
                             columns=["BMI", "AGE", "GENDER_NUM", "WAIST_CM"])
        raw_bodyfat = float(models["bf_waist"].predict(df_bf)[0])
    else:
        df_bf = pd.DataFrame([[pred_bmi, age, gender_num]],
                             columns=["BMI", "AGE", "GENDER_NUM"])
        raw_bodyfat = float(models["bf_nowaist"].predict(df_bf)[0])

    # ---------- Lifestyle Calibration ----------
    pred_bodyfat = max(BODYFAT_FLOOR, raw_bodyfat + ls_meta["delta"])
    actual_delta = pred_bodyfat - raw_bodyfat
    was_clamped = ls_meta["delta"] < 0 and abs(pred_bodyfat - BODYFAT_FLOOR) < 1e-6

    # ---------- STAGE 2: NCDs ----------
    if has_waist:
        df_ncd = pd.DataFrame(
            [[age, gender_num, pred_bmi, waist_cm, pred_bodyfat]],
            columns=["AGE", "GENDER_NUM", "BMI", "WAIST_CM", "TOTAL_BODY_FAT_PCT"])
        diab = float(models["diab_waist"].predict_proba(df_ncd)[0][1] * 100)
        hyp = float(models["hyp_waist"].predict_proba(df_ncd)[0][1] * 100)
    else:
        df_ncd = pd.DataFrame(
            [[age, gender_num, pred_bmi, pred_bodyfat]],
            columns=["AGE", "GENDER_NUM", "BMI", "TOTAL_BODY_FAT_PCT"])
        diab = float(models["diab_nowaist"].predict_proba(df_ncd)[0][1] * 100)
        hyp = float(models["hyp_nowaist"].predict_proba(df_ncd)[0][1] * 100)

    shape = analyze_body_shape(pred_bmi, pred_bodyfat, gender)

    return {
        "age": age, "gender": gender, "gender_num": gender_num,
        "waist_cm": waist_cm if has_waist else None,
        "has_waist": has_waist,
        "mode_text": ("ประเมินแบบเต็มรูปแบบ (ใช้รอบเอว)" if has_waist
                      else "ประเมินแบบพื้นฐาน (ไม่ใช้รอบเอว)"),
        "lifestyle": lifestyle, "lifestyle_meta": ls_meta,
        "bmi": pred_bmi, "bmi_cat": shape["bmi_cat"], "bmi_key": shape["bmi_key"],
        "raw_bodyfat": raw_bodyfat, "bodyfat": pred_bodyfat,
        "calibration_delta": actual_delta, "was_clamped": was_clamped,
        "bf_cat": shape["bf_cat"], "bf_key": shape["bf_key"],
        "bf_cuts": shape["bf_cuts"],
        "diabetes_pct": diab, "diabetes_level": get_risk_level(diab),
        "hypertension_pct": hyp, "hypertension_level": get_risk_level(hyp),
        "image": img,
        "face_report": face_report,
        "device": device,
    }


class FaceGuardError(Exception):
    """ยกขึ้นเมื่อภาพไม่ผ่านการตรวจใบหน้า"""
    def __init__(self, report):
        self.report = report
        super().__init__(report.message)


# ══════════════════════════════════════════════════════════
# [FIX 8] CLI REPORT — คงพฤติกรรมเดิมของคุณไว้ทุกประการ
# ══════════════════════════════════════════════════════════
def print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("🏥 สรุปผลการวิเคราะห์สุขภาพ AI Pipeline 🏥")
    print("=" * 60)
    waist_display = f"{r['waist_cm']} ซม." if r["has_waist"] else "ไม่ได้ระบุ"
    print(f"ผู้รับการประเมิน: {str(r['gender']).capitalize()}, "
          f"อายุ {r['age']} ปี, รอบเอว: {waist_display}")
    print(f"รูปแบบการใช้ชีวิต: {r['lifestyle_meta']['display']}")
    print(f"โหมดการประเมิน: {r['mode_text']}")
    print("-" * 60)
    print("📷 [Stage 1] ประเมินจากใบหน้า (ViT Model)")
    print(f"   > ค่า BMI ที่ทำนายได้   : {r['bmi']:.2f} [{r['bmi_cat']}]")
    print("📊 [Stage 1.5] ประเมินไขมัน (XGBoost)")
    print(f"   > ค่าดิบก่อนปรับ        : {r['raw_bodyfat']:.2f} %")
    print(f"   > เปอร์เซ็นต์ไขมันรวม    : {r['bodyfat']:.2f} % [{r['bf_cat']}]")
    print("-" * 60)
    print("🩺 [Stage 2] คัดกรองความเสี่ยงโรค NCDs")
    d, h = r["diabetes_level"], r["hypertension_level"]
    print(f"   > โรคเบาหวาน           : {r['diabetes_pct']:.1f}% ({d['icon']} {d['label']})")
    print(f"   > โรคความดันโลหิตสูง     : {r['hypertension_pct']:.1f}% ({h['icon']} {h['label']})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_image = BASE_DIR.parent / "data" / "test_images" / "testpic11.png"
    m = load_all_models(verbose=True)   # ✅ โหลดครั้งเดียว ใช้ซ้ำได้

    print("\n>>> ทดสอบแบบที่ 1: ใส่รอบเอว <<<")
    print_report(predict_health_risk(test_image, age=32, gender="Male",
                                     waist_cm=80.0, lifestyle="active", models=m))

    print("\n>>> ทดสอบแบบที่ 2: ไม่ใส่รอบเอว <<<")
    print_report(predict_health_risk(test_image, age=32, gender="Male",
                                     waist_cm=None, lifestyle="active", models=m))

# import torch
# import joblib
# import pandas as pd
# from PIL import Image
# from torchvision.transforms import ToTensor
# from loader import vit_transforms
# from models import get_model
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# def load_all_models(device):
#     print("กำลังโหลดระบบ AI ทั้งหมด...")
#     vit_model = get_model().float().to(device)
#     vit_model.load_state_dict(torch.load('../weights/aug_epoch_7.pt', map_location=device))
#     vit_model.eval()

#     # โหลดเตรียมไว้ทั้ง 2 ชุด (แบบมีเอว และ ไม่มีเอว)
#     models = {
#         'vit': vit_model,
#         'bf_waist': joblib.load('../weights/xgboost_bodyfat_with_waist.pkl'),
#         'bf_nowaist': joblib.load('../weights/xgboost_bodyfat_no_waist.pkl'),
#         'diab_waist': joblib.load('../weights/xgb_classifier_diabetes.pkl'),
#         'diab_nowaist': joblib.load('../weights/xgb_classifier_diabetes_no_waist.pkl'),
#         'hyp_waist': joblib.load('../weights/xgb_classifier_hypertension.pkl'),
#         'hyp_nowaist': joblib.load('../weights/xgb_classifier_hypertension_no_waist.pkl')
#     }
#     return models

# def analyze_body_shape(bmi, body_fat, gender):
#     if bmi < 18.5: bmi_cat = "น้ำหนักต่ำกว่าเกณฑ์"
#     elif 18.5 <= bmi < 23.0: bmi_cat = "สมส่วน (Normal)"
#     elif 23.0 <= bmi < 25.0: bmi_cat = "ท้วม (Overweight)"
#     else: bmi_cat = "อ้วน (Obese)"
        
#     if gender.lower() == 'male' or gender == 1:
#         if body_fat < 14: bf_cat = "กล้ามเนื้อชัด (Lean/Athlete)"
#         elif 14 <= body_fat < 18: bf_cat = "หุ่นฟิต (Fitness)"
#         elif 18 <= body_fat < 25: bf_cat = "ทั่วไป (Acceptable)"
#         else: bf_cat = "อ้วน (Obese)"
#     else:
#         if body_fat < 21: bf_cat = "กล้ามเนื้อชัด (Lean/Athlete)"
#         elif 21 <= body_fat < 25: bf_cat = "หุ่นฟิต (Fitness)"
#         elif 25 <= body_fat < 32: bf_cat = "ทั่วไป (Acceptable)"
#         else: bf_cat = "อ้วน (Obese)"
#     return bmi_cat, bf_cat

# # 💡 กำหนดให้ waist_cm=None เป็นค่าเริ่มต้น (Optional)
# def predict_health_risk(image_path, age, gender, waist_cm=None, lifestyle='normal'):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     models = load_all_models(device)
#     gender_num = 1 if gender.lower() == 'male' else 0
    
#     lifestyle_map = {
#         'active': 'ออกกำลังกายเป็นประจำ / มวลกล้ามเนื้อมาก (Active)',
#         'sedentary': 'ไม่ออกกำลังกาย / อาจมีไขมันซ่อนรูป (Sedentary)',
#         'normal': 'ทั่วไป / กิจกรรมปานกลาง (Normal)'
#     }
#     lifestyle_display = lifestyle_map.get(lifestyle.lower(), lifestyle_map['normal'])
    
#     # 💡 เช็กว่า User ใส่รอบเอวมาหรือไม่?
#     has_waist = waist_cm is not None and waist_cm > 0
#     mode_text = "ประเมินแบบเต็มรูปแบบ (ใช้รอบเอว)" if has_waist else "ประเมินแบบพื้นฐาน (ไม่ใช้รอบเอว)"
    
#     # --- STEP 1: Face -> BMI ---
#     img = Image.open(image_path)
#     if img.mode != 'RGB': img = img.convert('RGB')
#     img_tensor = ToTensor()(img)
#     img_tensor = vit_transforms(img_tensor).unsqueeze(0).to(device)
    
#     with torch.no_grad():
#         pred_bmi = models['vit'](img_tensor).item() 
        
#     # --- STEP 1.5: Body Fat & NCDs (แยก Route ตามการใส่รอบเอว) ---
#     if has_waist:
#         # Route A: แบบมีรอบเอว
#         df_bf = pd.DataFrame([[pred_bmi, age, gender_num, waist_cm]], columns=['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM'])
#         raw_bodyfat = models['bf_waist'].predict(df_bf)[0]
        
#         # Calibrate Body Fat
#         if lifestyle.lower() == 'active': pred_bodyfat = max(5.0, raw_bodyfat - 10.0)
#         elif lifestyle.lower() == 'sedentary': pred_bodyfat = raw_bodyfat + 5.0
#         else: pred_bodyfat = raw_bodyfat
        
#         df_ncd = pd.DataFrame([[age, gender_num, pred_bmi, waist_cm, pred_bodyfat]], columns=['AGE', 'GENDER_NUM', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT'])
#         diab_risk_pct = models['diab_waist'].predict_proba(df_ncd)[0][1] * 100
#         hyp_risk_pct = models['hyp_waist'].predict_proba(df_ncd)[0][1] * 100
        
#     else:
#         # Route B: แบบไม่มีรอบเอว
#         df_bf = pd.DataFrame([[pred_bmi, age, gender_num]], columns=['BMI', 'AGE', 'GENDER_NUM'])
#         raw_bodyfat = models['bf_nowaist'].predict(df_bf)[0]
        
#         # Calibrate Body Fat
#         if lifestyle.lower() == 'active': pred_bodyfat = max(5.0, raw_bodyfat - 10.0)
#         elif lifestyle.lower() == 'sedentary': pred_bodyfat = raw_bodyfat + 5.0
#         else: pred_bodyfat = raw_bodyfat
        
#         df_ncd = pd.DataFrame([[age, gender_num, pred_bmi, pred_bodyfat]], columns=['AGE', 'GENDER_NUM', 'BMI', 'TOTAL_BODY_FAT_PCT'])
#         diab_risk_pct = models['diab_nowaist'].predict_proba(df_ncd)[0][1] * 100
#         hyp_risk_pct = models['hyp_nowaist'].predict_proba(df_ncd)[0][1] * 100

#     bmi_cat, bf_cat = analyze_body_shape(pred_bmi, pred_bodyfat, gender)
    
#     # --- แสดงผลสรุป ---
#     print("\n" + "="*60)
#     print("🏥 สรุปผลการวิเคราะห์สุขภาพ AI Pipeline 🏥")
#     print("="*60)
#     waist_display = f"{waist_cm} ซม." if has_waist else "ไม่ได้ระบุ"
#     print(f"ผู้รับการประเมิน: {gender.capitalize()}, อายุ {age} ปี, รอบเอว: {waist_display}")
#     print(f"รูปแบบการใช้ชีวิต: {lifestyle_display}")
#     print(f"โหมดการประเมิน: {mode_text}")
#     print("-" * 60)
    
#     print(f"📷 [Stage 1] ประเมินจากใบหน้า (ViT Model)")
#     print(f"   > ค่า BMI ที่ทำนายได้   : {pred_bmi:.2f} [{bmi_cat}]")
#     print(f"📊 [Stage 1.5] ประเมินไขมัน (XGBoost)")
#     print(f"   > เปอร์เซ็นต์ไขมันรวม    : {pred_bodyfat:.2f} % [{bf_cat}]")
    
#     print("-" * 60)
#     print(f"🩺 [Stage 2] คัดกรองความเสี่ยงโรค NCDs")
#     diab_status = "🔴 เสี่ยงสูง" if diab_risk_pct > 50 else "🟢 ปกติ"
#     hyp_status = "🔴 เสี่ยงสูง" if hyp_risk_pct > 50 else "🟢 ปกติ"
#     print(f"   > โรคเบาหวาน           : {diab_risk_pct:.1f}% ({diab_status})")
#     print(f"   > โรคความดันโลหิตสูง     : {hyp_risk_pct:.1f}% ({hyp_status})")
#     print("="*60 + "\n")

# if __name__ == "__main__":
#     test_image = "../data/test_images/testpic11.png"
    
#     # เทสต์แบบใส่รอบเอว
#     print("\n>>> ทดสอบแบบที่ 1: ใส่รอบเอว <<<")
#     predict_health_risk(test_image, age=32, gender='Male', waist_cm=80.0, lifestyle='active')
    
#     # เทสต์แบบ ไม่ใส่รอบเอว (Optional)
#     print("\n>>> ทดสอบแบบที่ 2: ไม่ใส่รอบเอว (ทิ้งว่าง) <<<")
#     predict_health_risk(test_image, age=32, gender='Male', waist_cm=None, lifestyle='active')