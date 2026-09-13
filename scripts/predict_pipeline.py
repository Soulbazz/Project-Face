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

import cv2
import numpy as np

# นำเข้า facial_concepts แบบ local import เพื่อไม่ให้พังเวลาอยู่ใน scripts/
try:
    from facial_concepts import extract_facial_morphometry
except ImportError:
    from scripts.facial_concepts import extract_facial_morphometry

# ══════════════════════════════════════════════════════════
# PATH CONFIG
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = (BASE_DIR.parent / "weights").resolve()

WEIGHT_FILES = {
    "vit":          "vit_bmi_model.pt",
    "bf_waist":     "xgboost_bodyfat_with_waist.pkl",
    "bf_nowaist":   "xgboost_bodyfat_no_waist.pkl",
    "diab_waist":   "xgb_classifier_diabetes.pkl",
    "diab_nowaist": "xgb_classifier_diabetes_no_waist.pkl",
    "hyp_waist":    "xgb_classifier_hypertension.pkl",
    "hyp_nowaist":  "xgb_classifier_hypertension_no_waist.pkl",
}

# ══════════════════════════════════════════════════════════
# CONFIG: Lifestyle Calibration
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

BODYFAT_FLOOR = 5.0    
RISK_THRESHOLD = 50.0  

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
    return [f for f in WEIGHT_FILES.values() if not (WEIGHTS_DIR / f).exists()]

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

def to_pil(image: Union[str, Path, bytes, bytearray, Image.Image, Any]) -> Image.Image:
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, (str, Path)):
        img = Image.open(image)
    elif hasattr(image, "read"):
        try:
            image.seek(0)
        except Exception:
            pass
        img = Image.open(image)
    else:
        raise TypeError(f"ไม่รองรับ input ชนิด {type(image).__name__}")

    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

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
    if pct < RISK_THRESHOLD:
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

def enable_mc_dropout(model):
    """เปิด Dropout layer ค้างไว้ขณะทำนายเพื่อประเมิน Epistemic Uncertainty"""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# ══════════════════════════════════════════════════════════
# MAIN PREDICTION PIPELINE
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

    # แปลงภาพเป็น PIL Image
    img = to_pil(image)
    
    # แปลง PIL Image เป็นภาพ OpenCV BGR สำหรับตรวจสอบ Landmark
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ---------- [SAFETY LAYER 1] Aleatoric Proxy Check & Morphometry ----------
    morph_data, aleatoric_err = extract_facial_morphometry(img_bgr)
    if aleatoric_err:
        raise ValueError(f"[Aleatoric Rejection] ภาพไม่ได้มาตรฐาน: {aleatoric_err}")

    # ---------- FACE GUARD ----------
    face_report = None
    if face_guard:
        from face_guard import check_face
        face_report = check_face(img)
        if not face_report.ok:
            raise FaceGuardError(face_report)

    # ---------- STAGE 1: Face → BMI (ViT with Monte Carlo Dropout) ----------
    img_tensor = vit_transforms(ToTensor()(img)).unsqueeze(0).to(device)
    
    vit_model = models["vit"]
    vit_model.eval()
    enable_mc_dropout(vit_model)
    
    mc_preds = []
    with torch.no_grad():
        for _ in range(25):
            val = vit_model(img_tensor).item()
            mc_preds.append(val)
            
    pred_bmi = float(np.mean(mc_preds))
    std_bmi = float(np.std(mc_preds))
    ci_lower = float(np.percentile(mc_preds, 2.5))
    ci_upper = float(np.percentile(mc_preds, 97.5))

    # ---------- [SAFETY LAYER 2] Epistemic Uncertainty Rejection ----------
    if std_bmi > 1.8:
        raise ValueError(
            f"[Epistemic Rejection] ความไม่แน่นอนของโครงสร้างใบหน้าสูงเกินเกณฑ์ (±SD: {std_bmi:.2f} > 1.8) "
            "โมเดลไม่คุ้นเคยกับลักษณะโครงหน้านี้ แนะนำให้ปรึกษาแพทย์เพื่อตรวจวัดโดยตรง"
        )

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
        "bmi_std": std_bmi, "ci_range": [ci_lower, ci_upper],
        "morphometry": morph_data,
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
    def __init__(self, report):
        self.report = report
        super().__init__(report.message)

def print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("🏥 สรุปผลการวิเคราะห์สุขภาพ AI Pipeline 🏥")
    print("=" * 60)
    waist_display = f"{r['waist_cm']} ซม." if r["has_waist"] else "ไม่ได้ระบุ"
    print(f"ผู้รับการประเมิน: {str(r['gender']).capitalize()}, อายุ {r['age']} ปี, รอบเอว: {waist_display}")
    print(f"รูปแบบการใช้ชีวิต: {r['lifestyle_meta']['display']}")
    print(f"โหมดการประเมิน: {r['mode_text']}")
    print("-" * 60)
    print("📷 [Stage 1] ประเมินจากใบหน้า (ViT Model + Morphometry)")
    print(f"   > ค่า BMI ที่ทำนายได้   : {r['bmi']:.2f} (±{r['bmi_std']:.2f}) [{r['bmi_cat']}]")
    m = r['morphometry']
    print(f"   > Morphometry: LFWR={m['LFWR']:.2f}, CJWR={m['CJWR']:.2f}, PAR={m['PAR']:.2f}")
    print("📊 [Stage 1.5] ประเมินไขมัน (XGBoost)")
    print(f"   > ค่าดิบก่อนปรับ        : {r['raw_bodyfat']:.2f} %")
    print(f"   > เปอร์เซ็นต์ไขมันรวม    : {r['bodyfat']:.2f} % [{r['bf_cat']}]")
    print("-" * 60)
    print("🩺 [Stage 2] คัดกรองความเสี่ยงโรค NCDs")
    d, h = r["diabetes_level"], r["hypertension_level"]
    print(f"   > โรคเบาหวาน           : {r['diabetes_pct']:.1f}% ({d['icon']} {d['label']})")
    print(f"   > โรคความดันโลหิตสูง     : {r['hypertension_pct']:.1f}% ({h['icon']} {h['label']})")
    print("=" * 60 + "\n")