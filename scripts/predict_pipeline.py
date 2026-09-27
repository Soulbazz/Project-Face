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
import sys
import json
import copy
import warnings
from pathlib import Path
from typing import Optional, Union, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps

# ══════════════════════════════════════════════════════════
# PATH CONFIG & SYS.PATH SETUP
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.resolve()
WEIGHTS_DIR = (PROJECT_ROOT / "weights").resolve()

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchvision.transforms import ToTensor
from loader import vit_transforms
from models import get_model

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import cv2

# นำเข้า facial_concepts แบบ local import เพื่อไม่ให้พังเวลาอยู่ใน scripts/
try:
    from facial_concepts import extract_facial_morphometry
except ImportError:
    from scripts.facial_concepts import extract_facial_morphometry

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
LIFESTYLE_LEVEL_MAP = {

    "sedentary": 0,

    "normal": 1,

    "active": 2,

}


LIFESTYLE_DISPLAY = {

    0: "ไม่ออกกำลังกาย (Sedentary)",

    1: "กิจกรรมปานกลาง (Normal)",

    2: "ออกกำลังกายหนัก (Active)",

}


RISK_THRESHOLD = 50.0  # เกณฑ์ "เสี่ยงสูง" ตาม pipeline เดิม

# ค่า z-score สำหรับช่วงความเชื่อมั่นที่ใช้บ่อย (สมมติ distribution ~ normal)
_Z_TABLE = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


# ══════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════
def normalize_lifestyle_level(
    lifestyle: Union[str, int],
) -> tuple[int, str]:
    """
    รองรับค่าจาก UI เดิม:
      sedentary -> 0
      normal    -> 1
      active    -> 2

    และรองรับค่าตัวเลข 0, 1, 2 โดยตรง
    """
    if isinstance(lifestyle, str):
        value = lifestyle.strip().lower()

        if value in LIFESTYLE_LEVEL_MAP:
            level = LIFESTYLE_LEVEL_MAP[value]
            return level, value

        try:
            lifestyle = int(value)
        except ValueError as exc:
            raise ValueError(
                "lifestyle ต้องเป็น sedentary, normal, active "
                "หรือค่า LIFESTYLE_LEVEL 0, 1, 2"
            ) from exc

    try:
        level = int(lifestyle)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LIFESTYLE_LEVEL ต้องเป็นจำนวนเต็ม 0, 1 หรือ 2"
        ) from exc

    if level not in (0, 1, 2):
        raise ValueError(
            "LIFESTYLE_LEVEL ต้องเป็น 0 (Sedentary), "
            "1 (Normal) หรือ 2 (Active)"
        )

    lifestyle_key = {
        0: "sedentary",
        1: "normal",
        2: "active",
    }[level]

    return level, lifestyle_key

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def check_weights() -> list[str]:
    return [f for f in WEIGHT_FILES.values() if not (WEIGHTS_DIR / f).exists()]

# ══════════════════════════════════════════════════════════
# ตรวจสอบ Feature ก่อนทำนาย เพื่อป้องกันโหลด Weight รุ่นเก่าโดยไม่รู้ตัว
# ══════════════════════════════════════════════════════════

def validate_bodyfat_models(models: dict) -> None:
    expected = {
        "bf_waist": [
            "BMI",
            "AGE",
            "GENDER_NUM",
            "WAIST_CM",
            "LIFESTYLE_LEVEL",
        ],
        "bf_nowaist": [
            "BMI",
            "AGE",
            "GENDER_NUM",
            "LIFESTYLE_LEVEL",
        ],
    }

    for model_key, expected_features in expected.items():
        model = models[model_key]

        actual_features = getattr(
            model,
            "feature_names_in_",
            None,
        )

        if actual_features is None:
            continue

        actual_features = list(actual_features)

        if actual_features != expected_features:
            raise ValueError(
                f"โมเดล {model_key} ใช้ Features ไม่ตรงกับ Pipeline\n"
                f"โมเดล: {actual_features}\n"
                f"Pipeline: {expected_features}\n"
                "กรุณารัน scripts/train_xgboost.py ใหม่"
            )


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

    validate_bodyfat_models(models)

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
    elif hasattr(image, "read"):
        try:
            image.seek(0)
        except Exception:
            pass
        img = Image.open(image)
    else:
        raise TypeError(f"ไม่รองรับ input ชนิด {type(image).__name__}")

    img = ImageOps.exif_transpose(img)    # แก้รูปหมุนจากมือถือ
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

# ══════════════════════════════════════════════════════════
# DYNAMIC THRESHOLD & CLINICAL TRIAGE STRATA CONFIGURATION
# ══════════════════════════════════════════════════════════
DEFAULT_CLINICAL_TIERS = {
    "diabetes": {
        "low_max": 4.5,          # < 4.5% : Green (Low Risk)
        "watch_max": 6.1,        # 4.5% - 6.1% : Yellow (Watchful)
        "f2_cutoff": 6.1,        # >= 6.1% : Orange (Screen Positive / High Risk)
        "urgent_min": 12.0,      # >= 12.0% : Red (Urgent Risk)
        "youden_cutoff": 7.41,
    },
    "hypertension": {
        "low_max": 20.0,         # < 20.0% : Green (Low Risk)
        "watch_max": 33.0,       # 20.0% - 33.0% : Yellow (Watchful)
        "f2_cutoff": 15.8,       # >= 15.8% (initial F2) / >= 33.0% (Screen Positive)
        "urgent_min": 55.0,      # >= 55.0% : Red (Urgent Risk)
        "youden_cutoff": 33.1,
    }
}

def load_thresholds_config(weights_dir: Optional[Path] = None) -> dict:
    """
    โหลดค่า Cutoff และ Tiers จาก weights/thresholds.json อย่างปลอดภัย
    หากไฟล์ขาดหายหรือไม่สมบูรณ์ จะ fallback กลับสู่เกณฑ์ทางคลินิกที่ผ่านการทดสอบ (Validated Clinical Tiers)
    """
    weights_path = Path(weights_dir) if weights_dir else WEIGHTS_DIR
    thresholds_file = weights_path / "thresholds.json"
    tiers = copy.deepcopy(DEFAULT_CLINICAL_TIERS)
    tiers["raw_thresholds"] = {}

    if not thresholds_file.exists():
        return tiers

    try:
        with open(thresholds_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tiers["raw_thresholds"] = raw

        # ปรับปรุง threshold แบบไดนามิกถ้ามีข้อมูลใน thresholds.json
        if "xgb_classifier_diabetes.pkl" in raw:
            d_cfg = raw["xgb_classifier_diabetes.pkl"]
            if "f2_threshold" in d_cfg and d_cfg["f2_threshold"] is not None:
                tiers["diabetes"]["f2_cutoff"] = round(float(d_cfg["f2_threshold"]) * 100, 2)
            if "youden_threshold" in d_cfg and d_cfg["youden_threshold"] is not None:
                tiers["diabetes"]["youden_cutoff"] = round(float(d_cfg["youden_threshold"]) * 100, 2)

        if "xgb_classifier_hypertension.pkl" in raw:
            h_cfg = raw["xgb_classifier_hypertension.pkl"]
            if "f2_threshold" in h_cfg and h_cfg["f2_threshold"] is not None:
                tiers["hypertension"]["f2_cutoff"] = round(float(h_cfg["f2_threshold"]) * 100, 2)
            if "youden_threshold" in h_cfg and h_cfg["youden_threshold"] is not None:
                tiers["hypertension"]["youden_cutoff"] = round(float(h_cfg["youden_threshold"]) * 100, 2)

    except Exception as e:
        warnings.warn(f"ไม่สามารถอ่าน weights/thresholds.json ({e}) - ใช้ค่าเกณฑ์ทางคลินิกเริ่มต้น")

    return tiers

# โหลด Config สำหรับ Module
GLOBAL_THRESHOLDS_CONFIG = load_thresholds_config()

def get_risk_level(
    pct: float,
    disease: str = "diabetes",
    config: Optional[dict] = None
) -> dict:
    """
    จัดกลุ่มระดับความเสี่ยงทางคลินิก (Clinical Triage Strata) 4 ระดับ
    รองรับทั้งค่าเปอร์เซ็นต์ (0.0 - 100.0) และค่าทศนิยมความน่าจะเป็น (0.0 - 1.0)
    
    คืนค่า Dictionary:
    {
        "label": str,       # เช่น "Screen Positive / ความเสี่ยงสูง"
        "level": str,       # Triage Tier สากล ("Low Risk", "Watchful", "Screen Positive / High Risk", "Urgent Risk")
        "key": str,         # Styling key ("low", "watch", "high", "critical")
        "color": str,       # สี HEX
        "bg": str,          # สีพื้นหลัง HEX
        "icon": str,        # Emoji ("🟢", "🟡", "🟠", "🔴")
        "rgb": tuple,       # (r, g, b)
        "action": str,      # ข้อเสนอแนะทางคลินิกและการปฏิบัติตน
        "cutoff_used": float # เกณฑ์ Cutoff ที่ใช้ในการแบ่งกลุ่ม (% points)
    }
    """
    p = float(pct)
    # จัดการกรณีผู้ใช้ส่งค่าเป็นความน่าจะเป็นทศนิยม (0.0 < p <= 1.0)
    if 0.0 < p <= 1.0:
        p = p * 100.0

    cfg = config or GLOBAL_THRESHOLDS_CONFIG
    disease_str = str(disease).strip().lower()
    is_hyp = any(k in disease_str for k in ("hyp", "htn", "ความดัน"))

    if not is_hyp:
        # ─── DIABETES (T2DM) ───
        # เกณฑ์มาตรฐาน: Low < 4.5%, Watchful 4.5-6.1%, Screen Positive >= 6.1%, Urgent >= 12.0%
        d_cfg = cfg.get("diabetes", DEFAULT_CLINICAL_TIERS["diabetes"])
        low_max = float(d_cfg.get("low_max", 4.5))
        watch_max = float(d_cfg.get("watch_max", 6.1))
        f2_cutoff = float(d_cfg.get("f2_cutoff", 6.1))
        urgent_min = float(d_cfg.get("urgent_min", 12.0))
        cutoff_used = f2_cutoff

        if p < low_max:
            return {
                "label": "ความเสี่ยงต่ำ",
                "level": "Low Risk",
                "key": "low",
                "color": "#16a34a",
                "bg": "#dcfce7",
                "icon": "🟢",
                "rgb": (22, 163, 74),
                "action": "ความเสี่ยงอยู่ในเกณฑ์มาตรฐานประชากรทั่วไป แนะนำตรวจสุขภาพประจำปีและรักษาพฤติกรรมสุขภาพที่ดี",
                "cutoff_used": cutoff_used,
            }
        elif p < watch_max:
            return {
                "label": "เฝ้าระวัง",
                "level": "Watchful",
                "key": "watch",
                "color": "#ca8a04",
                "bg": "#fef9c3",
                "icon": "🟡",
                "rgb": (202, 138, 4),
                "action": "พบสัญญาณความเสี่ยงในระดับเฝ้าระวัง แนะนำปรับพฤติกรรมโภชนาการ ลดอาหารหวาน-แป้งแปรรูป และคัดกรองซ้ำใน 6-12 เดือน",
                "cutoff_used": cutoff_used,
            }
        elif p < urgent_min:
            return {
                "label": "Screen Positive / ความเสี่ยงสูง",
                "level": "Screen Positive / High Risk",
                "key": "high",
                "color": "#ea580c",
                "bg": "#ffedd5",
                "icon": "🟠",
                "rgb": (234, 88, 12),
                "action": f"ผลคัดกรองเบื้องต้นเป็นบวก (ความน่าจะเป็น {p:.1f}% เกินเกณฑ์ F2 screening cutoff {cutoff_used:.1f}%) แนะนำเข้ารับการตรวจยืนยันด้วยผลเลือดทางคลินิก (FPG หรือ HbA1c)",
                "cutoff_used": cutoff_used,
            }
        else:
            return {
                "label": "ความเสี่ยงสูงมาก (Urgent Risk)",
                "level": "Urgent Risk",
                "key": "critical",
                "color": "#dc2626",
                "bg": "#fee2e2",
                "icon": "🔴",
                "rgb": (220, 38, 38),
                "action": "ระดับความน่าจะเป็นสูงมากอย่างมีนัยสำคัญทางคลินิก แนะนำพบแพทย์เพื่อรับการประเมินและตรวจทางห้องปฏิบัติการทันที",
                "cutoff_used": cutoff_used,
            }
    else:
        # ─── ESSENTIAL HYPERTENSION ───
        # เกณฑ์มาตรฐาน: Low < 20.0%, Watchful 20.0-33.0%, Screen Positive >= 33.0% (Youden) หรือ F2 >= 15.8%, Urgent >= 55.0%
        h_cfg = cfg.get("hypertension", DEFAULT_CLINICAL_TIERS["hypertension"])
        low_max = float(h_cfg.get("low_max", 20.0))
        watch_max = float(h_cfg.get("watch_max", 33.0))
        f2_cutoff = float(h_cfg.get("f2_cutoff", 15.8))
        urgent_min = float(h_cfg.get("urgent_min", 55.0))
        cutoff_used = watch_max  # 33.0% Youden cutoff for Screen Positive tier

        if p < low_max:
            return {
                "label": "ความเสี่ยงต่ำ",
                "level": "Low Risk",
                "key": "low",
                "color": "#16a34a",
                "bg": "#dcfce7",
                "icon": "🟢",
                "rgb": (22, 163, 74),
                "action": "ระดับความดันโลหิตคาดว่าอยู่ในเกณฑ์ปกติ แนะนำตรวจวัดความดันประจำปีและควบคุมอาหารลดโซเดียม",
                "cutoff_used": cutoff_used,
            }
        elif p < watch_max:
            return {
                "label": "เฝ้าระวัง",
                "level": "Watchful",
                "key": "watch",
                "color": "#ca8a04",
                "bg": "#fef9c3",
                "icon": "🟡",
                "rgb": (202, 138, 4),
                "action": "พบแนวโน้มความเสี่ยงระยะก่อนความดันสูง (Pre-hypertension) แนะนำลดอาหารเค็ม ควบคุมน้ำหนักตัว และหมั่นวัดความดันโลหิตเป็นระยะ",
                "cutoff_used": cutoff_used,
            }
        elif p < urgent_min:
            return {
                "label": "Screen Positive / ความเสี่ยงสูง",
                "level": "Screen Positive / High Risk",
                "key": "high",
                "color": "#ea580c",
                "bg": "#ffedd5",
                "icon": "🟠",
                "rgb": (234, 88, 12),
                "action": f"ผลคัดกรองเบื้องต้นเป็นบวก (ความน่าจะเป็น {p:.1f}% เกินเกณฑ์ตัดสิน {cutoff_used:.1f}%) แนะนำวัดความดันโลหิตซ้ำด้วยเครื่องวัดมาตรฐานทางการแพทย์เพื่อยืนยันผล",
                "cutoff_used": cutoff_used,
            }
        else:
            return {
                "label": "ความเสี่ยงสูงมาก (Urgent Risk)",
                "level": "Urgent Risk",
                "key": "critical",
                "color": "#dc2626",
                "bg": "#fee2e2",
                "icon": "🔴",
                "rgb": (220, 38, 38),
                "action": "ระดับความเสี่ยงสูงมากเข้าข่ายความดันโลหิตสูงอย่างมีนัยสำคัญ แนะนำพบแพทย์ที่สถานพยาบาลโดยด่วนเพื่อประเมินภาวะแทรกซ้อน",
                "cutoff_used": cutoff_used,
            }

# ══════════════════════════════════════════════════════════
# MAIN PREDICTION — คืน dict; รองรับ MC Dropout uncertainty (opt-in)
# ══════════════════════════════════════════════════════════
def predict_health_risk(
    image: Union[str, Path, bytes, Image.Image, Any],
    age: int,
    gender: str,
    waist_cm: Optional[float] = None,
    lifestyle: Union[str, int] = 1,
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

    lifestyle_level, lifestyle_key = normalize_lifestyle_level(
        lifestyle
    )
    lifestyle_display = LIFESTYLE_DISPLAY[lifestyle_level]

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
    
    # แปลง PIL Image เป็นภาพ OpenCV BGR สำหรับตรวจสอบ Landmark
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ---------- [SAFETY LAYER 1] Aleatoric Proxy Check & Morphometry ----------
    morph_data, aleatoric_err = extract_facial_morphometry(img_bgr)
    if face_guard and aleatoric_err:
        raise ValueError(f"[Aleatoric Rejection] ภาพไม่ได้มาตรฐาน: {aleatoric_err}")

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
        df_bf = pd.DataFrame(
            [[
                pred_bmi,
                age,
                gender_num,
                waist_cm,
                lifestyle_level,
            ]],
            columns=[
                "BMI",
                "AGE",
                "GENDER_NUM",
                "WAIST_CM",
                "LIFESTYLE_LEVEL",
            ],
        )

        pred_bodyfat = float(
            models["bf_waist"].predict(df_bf)[0]
        )
    else:
        df_bf = pd.DataFrame(
            [[
                pred_bmi,
                age,
                gender_num,
                lifestyle_level,
            ]],
            columns=[
                "BMI",
                "AGE",
                "GENDER_NUM",
                "LIFESTYLE_LEVEL",
            ],
        )

        pred_bodyfat = float(
            models["bf_nowaist"].predict(df_bf)[0]
        )

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
        "lifestyle": lifestyle_key,
        "lifestyle_level": lifestyle_level,
        "lifestyle_display": lifestyle_display,
        "bmi": pred_bmi, "bmi_cat": shape["bmi_cat"], "bmi_key": shape["bmi_key"],
        "bmi_std": std_bmi, "ci_range": [ci_lower, ci_upper],
        "morphometry": morph_data,
        "bodyfat": pred_bodyfat,
        "bf_cat": shape["bf_cat"], "bf_key": shape["bf_key"],
        "bf_cuts": shape["bf_cuts"],
        "diabetes_pct": diab, "diabetes_level": get_risk_level(diab, disease="diabetes"),
        "hypertension_pct": hyp, "hypertension_level": get_risk_level(hyp, disease="hypertension"),
        "image": img,
        "face_report": face_report,
        "device": device,
        "uncertainty": None,
    }

    # ---------- [NEW] STAGE 3: Monte Carlo Dropout uncertainty (opt-in) ----------
    if mc_dropout:
        bmi_samples = mc_dropout_bmi(img_tensor, vit_model, n_samples=mc_samples)

        n = len(bmi_samples)

        ages_arr = np.full(
            n,
            age,
            dtype=np.float64,
        )

        genders_arr = np.full(
            n,
            gender_num,
            dtype=np.float64,
        )

        lifestyle_arr = np.full(
            n,
            lifestyle_level,
            dtype=np.int8,
        )

        if has_waist:
            waists_arr = np.full(
                n,
                waist_cm,
                dtype=np.float64,
            )

            df_bf_mc = pd.DataFrame({
                "BMI": bmi_samples,
                "AGE": ages_arr,
                "GENDER_NUM": genders_arr,
                "WAIST_CM": waists_arr,
                "LIFESTYLE_LEVEL": lifestyle_arr,
            })

            bf_samples = models["bf_waist"].predict(
                df_bf_mc
            )
        else:
            df_bf_mc = pd.DataFrame({
                "BMI": bmi_samples,
                "AGE": ages_arr,
                "GENDER_NUM": genders_arr,
                "LIFESTYLE_LEVEL": lifestyle_arr,
            })

            bf_samples = models["bf_nowaist"].predict(
                df_bf_mc
            )

        bf_samples = np.asarray(
            bf_samples,
            dtype=np.float64,
        )

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
    print(f"ผู้รับการประเมิน: {str(r['gender']).capitalize()}, อายุ {r['age']} ปี, รอบเอว: {waist_display}")
    print(
        f"รูปแบบการใช้ชีวิต: {r['lifestyle_display']} "
        f"(LIFESTYLE_LEVEL={r['lifestyle_level']})"
    )
    print(f"โหมดการประเมิน: {r['mode_text']}")
    print("-" * 60)
    print("📷 [Stage 1] ประเมินจากใบหน้า (ViT Model + Morphometry)")
    print(f"   > ค่า BMI ที่ทำนายได้   : {r['bmi']:.2f} (±{r['bmi_std']:.2f}) [{r['bmi_cat']}]")
    m = r['morphometry']
    print(f"   > Morphometry: LFWR={m['LFWR']:.2f}, CJWR={m['CJWR']:.2f}, PAR={m['PAR']:.2f}")
    print("📊 [Stage 1.5] ประเมินไขมัน (XGBoost + Lifestyle Feature)")
    print(
        f"   > เปอร์เซ็นต์ไขมันรวม    : "
        f"{r['bodyfat']:.2f} % [{r['bf_cat']}]"
    )
    print("-" * 60)
    print("🩺 [Stage 2] คัดกรองความเสี่ยงโรค NCDs (Calibrated Screening Strata)")
    d, h = r["diabetes_level"], r["hypertension_level"]
    print(f"   > โรคเบาหวาน           : {r['diabetes_pct']:.1f}% ({d['icon']} {d['label']} [{d['level']}])")
    print(f"     เกณฑ์ตัดสิน (Cutoff)   : {d['cutoff_used']:.1f}%")
    print(f"     คำแนะนำทางคลินิก     : {d['action']}")
    print(f"   > โรคความดันโลหิตสูง     : {r['hypertension_pct']:.1f}% ({h['icon']} {h['label']} [{h['level']}])")
    print(f"     เกณฑ์ตัดสิน (Cutoff)   : {h['cutoff_used']:.1f}%")
    print(f"     คำแนะนำทางคลินิก     : {h['action']}")

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
    test_image = PROJECT_ROOT / "data" / "test_images" / "testpic11.png"
    m = load_all_models(verbose=True)   # ✅ โหลดครั้งเดียว ใช้ซ้ำได้

    print("\n>>> [Sanity Test] ตรวจสอบเกณฑ์ Calibrated Clinical Tiers <<<")
    sanity_cases = [
        (3.0, "diabetes", "low"),
        (5.2, "diabetes", "watch"),
        (8.0, "diabetes", "high"),
        (15.0, "diabetes", "critical"),
        (12.0, "hypertension", "low"),
        (25.0, "hypertension", "watch"),
        (40.0, "hypertension", "high"),
        (60.0, "hypertension", "critical"),
    ]
    for test_p, test_dis, expected_k in sanity_cases:
        tier_out = get_risk_level(test_p, disease=test_dis)
        status_sym = "✅" if tier_out["key"] == expected_k else "❌"
        print(f"   {status_sym} {test_dis.capitalize()} {test_p:4.1f}% -> {tier_out['icon']} {tier_out['label']} "
              f"[{tier_out['level']}] (key={tier_out['key']}, cutoff={tier_out['cutoff_used']:.1f}%)")

    print("\n>>> ทดสอบแบบที่ 1: ใส่รอบเอว (ไม่มี MC Dropout) <<<")
    print_report(predict_health_risk(test_image, age=32, gender="Male",
                                     waist_cm=80.0, lifestyle="active", models=m))

    print("\n>>> ทดสอบแบบที่ 2: ไม่ใส่รอบเอว + MC Dropout (n=30) <<<")
    print_report(predict_health_risk(test_image, age=32, gender="Male",
                                     waist_cm=None, lifestyle="active", models=m,
                                     mc_dropout=True, mc_samples=30))
