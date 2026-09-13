"""
predict_pipeline.py  (v3 - Web-Ready + MC Dropout)
──────────────────────────────────────────────────────────────
สิ่งที่แก้จากเวอร์ชันเดิม (v2):
  [9] เพิ่ม Monte Carlo Dropout (MC Dropout) สำหรับประเมิน "ความไม่แน่นอน"
      (uncertainty) ของค่า BMI ที่โมเดลทำนาย พร้อม propagate ต่อไปยัง
      body fat / diabetes / hypertension เพื่อให้ได้ confidence interval
      ของผลลัพธ์ปลายทางทั้งหมด ไม่ใช่แค่ BMI
──────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Optional, Union, Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from torchvision.transforms import ToTensor
from loader import vit_transforms
from models import get_model

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════
# PATH CONFIG — อ้างอิงจากไฟล์นี้ ไม่ใช่ current working directory
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

BODYFAT_FLOOR = 5.0    # Essential fat — เพดานล่างเชิงสรีรวิทยา
RISK_THRESHOLD = 50.0  # เกณฑ์ "เสี่ยงสูง" ตาม pipeline เดิม

# ค่า z-score สำหรับช่วงความเชื่อมั่นที่ใช้บ่อย (สมมติ distribution ~ normal)
_Z_TABLE = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


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
# LOAD MODELS — เรียกครั้งเดียว แล้วส่ง dict ไปใช้ซ้ำ
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
# [NEW] MONTE CARLO DROPOUT
# ══════════════════════════════════════════════════════════
def enable_mc_dropout(model: nn.Module) -> None:
    """
    เปิดเฉพาะเลเยอร์ nn.Dropout ให้ทำงานแบบ train() (สุ่ม mask ทุกครั้งที่ forward)
    โดยส่วนอื่นของโมเดล (ViT backbone ที่ freeze ไว้, LayerNorm ฯลฯ) ยังอยู่ใน eval()

    สำคัญ: ต้องเรียก model.eval() ก่อนเสมอ แล้วค่อยเรียกฟังก์ชันนี้ทับ
    ไม่ใช้ model.train() ตรงๆ เพราะจะไปกระทบ behavior ของเลเยอร์อื่นด้วย
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def _z_score(ci: float) -> float:
    return _Z_TABLE.get(round(float(ci), 2), 1.960)


def _summarize(arr: np.ndarray, ci: float = 0.95) -> dict:
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    z = _z_score(ci)
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = 100 - lo_pct
    return {
        "mean": mean,
        "std": std,
        "ci": ci,
        # ช่วงความเชื่อมั่นแบบ parametric (normal approx จาก mean ± z*std)
        "ci_normal": (mean - z * std, mean + z * std),
        # ช่วงความเชื่อมั่นแบบ empirical percentile (ไม่ต้องสมมติ distribution)
        "ci_percentile": (float(np.percentile(arr, lo_pct)),
                           float(np.percentile(arr, hi_pct))),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_samples": int(len(arr)),
    }


def mc_dropout_bmi(
    img_tensor: torch.Tensor,
    vit_model: nn.Module,
    n_samples: int = 30,
) -> np.ndarray:
    """
    รัน forward pass ซ้ำ n_samples ครั้งโดยเปิด dropout ค้างไว้ (MC Dropout)
    คืนค่าเป็น array ของ BMI ที่ทำนายได้ในแต่ละรอบ (ยังไม่ aggregate)

    หมายเหตุ: ต้องเรียกด้วย img_tensor เดิม (deterministic input) ความแตกต่าง
    ระหว่างรอบมาจาก dropout mask ที่สุ่มใหม่ทุกครั้งเท่านั้น
    """
    enable_mc_dropout(vit_model)
    samples = np.empty(n_samples, dtype=np.float64)
    try:
        with torch.no_grad():
            for i in range(n_samples):
                samples[i] = float(vit_model(img_tensor).item())
    finally:
        vit_model.eval()  # คืนโมเดลให้อยู่ใน eval mode ปกติเสมอ แม้เกิด error
    return samples


# ══════════════════════════════════════════════════════════
# IMAGE LOADER — รับได้ทุกรูปแบบ + แก้ EXIF
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

    img = ImageOps.exif_transpose(img)    # แก้รูปหมุนจากมือถือ
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


# ══════════════════════════════════════════════════════════
# MAIN PREDICTION — คืน dict; รองรับ MC Dropout uncertainty (opt-in)
# ══════════════════════════════════════════════════════════
def predict_health_risk(
    image: Union[str, Path, bytes, Image.Image, Any],
    age: int,
    gender: str,
    waist_cm: Optional[float] = None,
    lifestyle: str = "normal",
    models: Optional[dict] = None,
    face_guard: bool = True,
    mc_dropout: bool = False,
    mc_samples: int = 30,
    mc_ci: float = 0.95,
) -> dict:
    # ---------- Validate ----------
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

    if mc_dropout and mc_samples < 2:
        raise ValueError("mc_samples ต้องมีอย่างน้อย 2 รอบ")

    gender_num = 1 if str(gender).lower() == "male" else 0

    models = models or load_all_models()
    device = models.get("_device", get_device())

    # ---------- Load image ----------
    img = to_pil(image)

    # ---------- FACE GUARD ----------
    face_report = None
    if face_guard:
        from face_guard import check_face
        face_report = check_face(img)
        if not face_report.ok:
            raise FaceGuardError(face_report)

    # ---------- STAGE 1: Face → BMI (ViT), deterministic point estimate ----------
    vit_model = models["vit"]
    vit_model.eval()  # การันตี dropout ปิดสำหรับค่าประเมินหลัก (ไม่เปลี่ยนพฤติกรรมเดิม)
    img_tensor = vit_transforms(ToTensor()(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_bmi = float(vit_model(img_tensor).item())

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

    result = {
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
        "uncertainty": None,
    }

    # ---------- [NEW] STAGE 3: Monte Carlo Dropout uncertainty (opt-in) ----------
    if mc_dropout:
        bmi_samples = mc_dropout_bmi(img_tensor, vit_model, n_samples=mc_samples)

        n = len(bmi_samples)
        ages_arr = np.full(n, age, dtype=np.float64)
        genders_arr = np.full(n, gender_num, dtype=np.float64)

        if has_waist:
            waists_arr = np.full(n, waist_cm, dtype=np.float64)
            df_bf_mc = pd.DataFrame({"BMI": bmi_samples, "AGE": ages_arr,
                                      "GENDER_NUM": genders_arr, "WAIST_CM": waists_arr})
            raw_bf_samples = models["bf_waist"].predict(df_bf_mc)
        else:
            df_bf_mc = pd.DataFrame({"BMI": bmi_samples, "AGE": ages_arr,
                                      "GENDER_NUM": genders_arr})
            raw_bf_samples = models["bf_nowaist"].predict(df_bf_mc)

        bf_samples = np.maximum(BODYFAT_FLOOR, raw_bf_samples + ls_meta["delta"])

        if has_waist:
            df_ncd_mc = pd.DataFrame({
                "AGE": ages_arr, "GENDER_NUM": genders_arr, "BMI": bmi_samples,
                "WAIST_CM": waists_arr, "TOTAL_BODY_FAT_PCT": bf_samples,
            })
            diab_samples = models["diab_waist"].predict_proba(df_ncd_mc)[:, 1] * 100
            hyp_samples = models["hyp_waist"].predict_proba(df_ncd_mc)[:, 1] * 100
        else:
            df_ncd_mc = pd.DataFrame({
                "AGE": ages_arr, "GENDER_NUM": genders_arr, "BMI": bmi_samples,
                "TOTAL_BODY_FAT_PCT": bf_samples,
            })
            diab_samples = models["diab_nowaist"].predict_proba(df_ncd_mc)[:, 1] * 100
            hyp_samples = models["hyp_nowaist"].predict_proba(df_ncd_mc)[:, 1] * 100

        result["uncertainty"] = {
            "n_samples": mc_samples,
            "ci": mc_ci,
            "bmi": _summarize(bmi_samples, mc_ci),
            "bodyfat": _summarize(bf_samples, mc_ci),
            "diabetes_pct": _summarize(diab_samples, mc_ci),
            "hypertension_pct": _summarize(hyp_samples, mc_ci),
        }

    return result


class FaceGuardError(Exception):
    """ยกขึ้นเมื่อภาพไม่ผ่านการตรวจใบหน้า"""
    def __init__(self, report):
        self.report = report
        super().__init__(report.message)


# ══════════════════════════════════════════════════════════
# CLI REPORT — คงพฤติกรรมเดิมไว้ทุกประการ + แสดง uncertainty ถ้ามี
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

    u = r.get("uncertainty")
    if u:
        print("-" * 60)
        print(f"🎲 [Stage 3] Monte Carlo Dropout (n={u['n_samples']}, CI={u['ci']*100:.0f}%)")
        b = u["bmi"]
        print(f"   > BMI                  : {b['mean']:.2f} ± {b['std']:.2f} "
              f"(95%CI norm: {b['ci_normal'][0]:.2f}–{b['ci_normal'][1]:.2f})")
        bf = u["bodyfat"]
        print(f"   > Body Fat             : {bf['mean']:.2f}% ± {bf['std']:.2f}%")
        dd = u["diabetes_pct"]
        print(f"   > เบาหวาน (unc.)        : {dd['mean']:.1f}% ± {dd['std']:.1f}%")
        hh = u["hypertension_pct"]
        print(f"   > ความดัน (unc.)        : {hh['mean']:.1f}% ± {hh['std']:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_image = BASE_DIR.parent / "data" / "test_images" / "testpic11.png"
    m = load_all_models(verbose=True)   # ✅ โหลดครั้งเดียว ใช้ซ้ำได้

    print("\n>>> ทดสอบแบบที่ 1: ใส่รอบเอว (ไม่มี MC Dropout) <<<")
    print_report(predict_health_risk(test_image, age=32, gender="Male",
                                     waist_cm=80.0, lifestyle="active", models=m))

    print("\n>>> ทดสอบแบบที่ 2: ไม่ใส่รอบเอว + MC Dropout (n=50) <<<")
    print_report(predict_health_risk(test_image, age=32, gender="Male",
                                     waist_cm=None, lifestyle="active", models=m,
                                     mc_dropout=True, mc_samples=50))
