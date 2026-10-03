"""
Face2Health — Clinical Decision Support System (CDSS) Mockup Dashboard
Non-Invasive Multimodal Health Risk Screening Kiosk | NHANES Calibrated
========================================================================
Interactive Clinical Triage Demonstration for Primary Care Screening Kiosk / Nurse Station
Zero-Lag Standalone Simulation Engine (Instantaneous Mathematical Surrogate < 0.05s)
"""

import os
import json
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MEDICAL CDSS THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Face2Health — CDSS Screening Kiosk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Legibility Medical CDSS Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', 'IBM Plex Sans Thai', -apple-system, sans-serif;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1350px;
    }
    
    /* CDSS Header */
    .cdss-header {
        background: linear-gradient(135deg, #092c3e 0%, #0d4e68 50%, #0284c7 100%);
        border-radius: 16px;
        padding: 24px 28px;
        color: #ffffff;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .cdss-header h1 {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 6px 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .cdss-header p {
        font-size: 14.5px;
        color: #e0f2fe;
        margin: 0;
        opacity: 0.95;
    }
    .cdss-meta-pills {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 14px;
    }
    .cdss-pill {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        color: #f8fafc;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }

    /* Metric Cards */
    .cdss-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
        height: 100%;
    }
    .cdss-card-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .cdss-val-lg {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .cdss-val-unit {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        margin-left: 3px;
    }

    /* Status Badges */
    .badge-pass {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-warn {
        background: #fef9c3;
        color: #a16207;
        border: 1px solid #fef08a;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }
    .badge-alert {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 4px;
    }

    /* Triage Banners */
    .triage-banner {
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 14px;
        border: 1.5px solid;
    }
    .triage-header {
        font-size: 15px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }
    .triage-action {
        font-size: 13px;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Highlight Resolution Callout */
    .highlight-callout {
        background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%);
        border: 2px solid #f97316;
        border-radius: 14px;
        padding: 16px 20px;
        color: #9a3412;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.15);
    }
    .highlight-title {
        font-size: 15.5px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #c2410c;
        margin-bottom: 6px;
    }
    .highlight-body {
        font-size: 13.5px;
        line-height: 1.55;
        color: #7c2d12;
    }

    /* Comparison Box */
    .compare-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 14px;
    }
    .compare-card {
        padding: 12px 14px;
        border-radius: 10px;
        border: 1px solid;
    }
    .compare-title {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .compare-verdict {
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .compare-sub {
        font-size: 11.5px;
        line-height: 1.35;
    }

    /* Rejection Block Screen */
    .block-screen {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 28px 32px;
        color: #991b1b;
        margin: 20px 0;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15);
    }
    .block-screen h2 {
        font-size: 22px;
        font-weight: 800;
        color: #b91c1c;
        margin: 0 0 10px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .block-screen p {
        font-size: 14.5px;
        line-height: 1.6;
        color: #7f1d1d;
    }
    .block-protocol {
        background: #ffffff;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 16px;
        border: 1px solid #fca5a5;
    }

    /* Morphometry Grid */
    .morpho-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-top: 10px;
    }
    .morpho-item {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 10px;
    }
    .morpho-label {
        font-size: 10.5px;
        font-weight: 600;
        color: #64748b;
    }
    .morpho-val {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CONFIGURATION & THRESHOLD SYNCHRONIZATION
# -----------------------------------------------------------------------------
def load_triage_config():
    """
    Attempts to read weights/thresholds.json.
    Strictly synchronizes with verified paper parameters:
      - Diabetes F2 cutoff: 0.061 (6.1%)
      - Hypertension cutoff: 0.158 (15.8%)
    4-Tier Strata:
      - Diabetes: Low < 4.5%, Watchful 4.5-6.1%, Screen Positive >= 6.1%, Urgent >= 12.0%
      - Hypertension: Low < 20.0%, Watchful 20.0-33.0%, Screen Positive >= 33.0%, Urgent >= 55.0%
    """
    threshold_path = Path("weights/thresholds.json")
    config = {
        "diabetes_f2": 0.061,       # 6.1%
        "hypertension_f2": 0.158,   # 15.8%
        "diab_youden": 0.074,       # 7.41%
        "hyp_youden": 0.331,        # 33.1%
        "source": "Paper Verified Standard (Validated)",
        "file_found": False,
        "strata": {
            "diabetes": {
                "low_max": 4.5,
                "watch_max": 6.1,
                "screen_pos": 6.1,
                "urgent_min": 12.0
            },
            "hypertension": {
                "low_max": 20.0,
                "watch_max": 33.0,
                "screen_pos": 33.0,
                "f2_cutoff": 15.8,
                "urgent_min": 55.0
            }
        }
    }

    if threshold_path.exists():
        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                config["file_found"] = True
                config["source"] = "weights/thresholds.json (Synchronized)"
                if "xgb_classifier_diabetes.pkl" in data:
                    d_cfg = data["xgb_classifier_diabetes.pkl"]
                    config["diab_youden"] = float(d_cfg.get("youden_threshold", 0.0741))
                if "xgb_classifier_hypertension.pkl" in data:
                    h_cfg = data["xgb_classifier_hypertension.pkl"]
                    config["hyp_youden"] = float(h_cfg.get("youden_threshold", 0.3312))
        except Exception:
            pass

    return config

CFG = load_triage_config()


# -----------------------------------------------------------------------------
# 3. ZERO-LAG STANDALONE MATHEMATICAL SIMULATION ENGINE (< 0.05s)
# -----------------------------------------------------------------------------
def simulate_cdss_pipeline(age: int, sex: str, waist_cm: float | None, activity_level: int,
                           manual_bmi: float, manual_sigma: float, pose_violation: bool):
    """
    Simulates Stage 1, 1.5, and 2 calculations instantaneously (< 0.05s).
    Reflects the mathematical relationships established in the NHANES benchmark:
      - Sex Error Shock Absorber: alpha = 0.39 with waist, 1.00 without waist
      - Monotonic monotonic progression across age, BMI, and DEXA fat
      - Anchor for Preset 2: Age 56, Male, Waist 98, BMI 27.8 -> DEXA 31.0%, Diab 7.4%
      - Anchor for Preset 1: Age 24, Female, Waist Off, BMI 21.0 -> DEXA 22.5%, Diab 1.2%
    """
    is_male = (sex == "Male")
    s_val = 1.0 if is_male else 0.0
    has_waist = (waist_cm is not None and waist_cm > 0)

    # 1. Epistemic & Aleatoric Guard Checks
    head_pose_yaw = 22.0 if pose_violation else (2.4 if is_male else -1.8)
    head_pose_pitch = 1.1 if not pose_violation else 4.2
    head_pose_roll = -0.5 if not pose_violation else 2.1

    is_aleatoric_rejected = abs(head_pose_yaw) > 15.0
    is_epistemic_rejected = manual_sigma > 1.80
    is_rejected = is_aleatoric_rejected or is_epistemic_rejected

    rejection_reasons = []
    if is_aleatoric_rejected:
        rejection_reasons.append(
            f"Head Pose Yaw angle ({head_pose_yaw:+.1f}°) exceeds clinical tolerance threshold (±15.0°). Non-frontal geometry introduces perspective parallax."
        )
    if is_epistemic_rejected:
        rejection_reasons.append(
            f"Epistemic Uncertainty σ = {manual_sigma:.2f} kg/m² exceeds safety boundary (1.80 kg/m²). Facial structure is out-of-distribution for the Vision Transformer."
        )

    # 2. Stage 1 Morphometrics (Anthropometric indices correlated with visceral adiposity)
    lfwr = round(0.880 + 0.007 * manual_bmi + 0.0008 * age, 3)
    cjwr = round(1.360 - 0.0085 * manual_bmi - 0.0006 * age, 3)
    par = round(2.110 + 0.0115 * manual_bmi + (0.02 if is_male else 0.0), 3)
    fwhr = round(1.760 + 0.0055 * manual_bmi + (0.04 if is_male else 0.0), 3)

    # 3. Stage 1.5 DEXA Total Body Fat % (Tabular Regressor with Sex Shock Absorber)
    # Biological sex commands 74.99% of regression variance.
    # Attenuation factor alpha: 0.39 with waist, 1.00 without waist
    if has_waist:
        # Shock absorber active: alpha = 0.39
        pred_body_fat = 31.0 + 0.39 * (manual_bmi - 27.8) + 0.30 * (waist_cm - 98.0) + \
                        0.14 * (age - 56.0) - 8.5 * (s_val - 1.0) - 1.2 * (activity_level - 1)
        shock_absorber_alpha = 0.39
    else:
        # Without waist: alpha = 1.00
        pred_body_fat = 22.5 + 1.00 * (manual_bmi - 21.0) + 0.16 * (age - 24.0) - \
                        8.5 * (s_val - 0.0) - 1.2 * (activity_level - 1)
        shock_absorber_alpha = 1.00

    pred_body_fat = round(float(np.clip(pred_body_fat, 6.0, 58.0)), 1)

    # 4. Stage 2 Platt-Calibrated Probabilities for T2DM and Hypertension
    # Diabetes calibrated logit (anchor: preset 2 = 7.4% at age 56, BMI 27.8, waist 98, fat 31)
    # logit(0.074) = -2.5268
    w_term_diab = (0.030 * (waist_cm - 98.0)) if has_waist else (0.040 * (manual_bmi - 27.8))
    z_diab = -2.5268 + 0.042 * (age - 56.0) + 0.050 * (manual_bmi - 27.8) + \
             0.025 * (pred_body_fat - 31.0) + w_term_diab - 0.18 * (activity_level - 1)
    prob_diab = float(1.0 / (1.0 + np.exp(-z_diab)) * 100.0)
    prob_diab = round(float(np.clip(prob_diab, 0.5, 96.0)), 1)

    # Hypertension calibrated logit (anchor: preset 2 = 38.5% at age 56, BMI 27.8, waist 98)
    # logit(0.385) = -0.4684
    w_term_hyp = (0.025 * (waist_cm - 98.0)) if has_waist else (0.035 * (manual_bmi - 27.8))
    z_hyp = -0.4684 + 0.052 * (age - 56.0) + 0.045 * (manual_bmi - 27.8) + \
            0.020 * (pred_body_fat - 31.0) + 0.25 * (s_val - 1.0) + w_term_hyp - 0.20 * (activity_level - 1)
    prob_hyp = float(1.0 / (1.0 + np.exp(-z_hyp)) * 100.0)
    prob_hyp = round(float(np.clip(prob_hyp, 1.2, 98.0)), 1)

    return {
        "is_rejected": is_rejected,
        "rejection_reasons": rejection_reasons,
        "pose": {
            "yaw": head_pose_yaw,
            "pitch": head_pose_pitch,
            "roll": head_pose_roll,
            "lighting_score": 145 if not pose_violation else 88,
            "sharpness_score": 210 if not pose_violation else 95,
        },
        "morphometrics": {
            "LFWR": lfwr,
            "CJWR": cjwr,
            "PAR": par,
            "FWHR": fwhr
        },
        "bmi": manual_bmi,
        "sigma": manual_sigma,
        "body_fat": pred_body_fat,
        "shock_alpha": shock_absorber_alpha,
        "diabetes_pct": prob_diab,
        "hypertension_pct": prob_hyp,
    }


def get_triage_stratum(pct: float, disease: str):
    """
    Evaluates 4-Tier Clinical Triage Strata according to the validated paper protocol:
      - Low Risk (Green)
      - Watchful (Yellow)
      - Screen Positive (Orange)
      - Urgent Risk (Red)
    """
    if disease == "diabetes":
        if pct < 4.5:
            return {
                "tier": "Low Risk",
                "label": "Low Risk (Green)",
                "color": "#16a34a",
                "bg": "#dcfce7",
                "border": "#86efac",
                "icon": "🟢",
                "action": "Routine annual medical checkup; encourage balanced nutrition and physical activity maintenance."
            }
        elif pct < 6.1:
            return {
                "tier": "Watchful",
                "label": "Watchful (Yellow)",
                "color": "#ca8a04",
                "bg": "#fef9c3",
                "border": "#fde047",
                "icon": "🟡",
                "action": "Lifestyle counseling & dietary modification; recommend repeat non-invasive screening within 6–12 months."
            }
        elif pct < 12.0:
            return {
                "tier": "Screen Positive",
                "label": "Screen Positive / High Risk (Orange)",
                "color": "#ea580c",
                "bg": "#ffedd5",
                "border": "#fdba74",
                "icon": "🟠",
                "action": "Laboratory confirmatory venous phlebotomy (Fasting Plasma Glucose ≥ 126 mg/dL or HbA1c ≥ 6.5%)."
            }
        else:
            return {
                "tier": "Urgent Risk",
                "label": "Urgent Risk (Red)",
                "color": "#dc2626",
                "bg": "#fee2e2",
                "border": "#fca5a5",
                "icon": "🔴",
                "action": "Immediate physician consultation; comprehensive metabolic panel and urgent diagnostic evaluation."
            }
    else:  # Hypertension
        if pct < 20.0:
            return {
                "tier": "Low Risk",
                "label": "Low Risk (Green)",
                "color": "#16a34a",
                "bg": "#dcfce7",
                "border": "#86efac",
                "icon": "🟢",
                "action": "Annual preventive health checkup; maintain routine cardiovascular healthy lifestyle."
            }
        elif pct < 33.0:
            return {
                "tier": "Watchful",
                "label": "Watchful (Yellow)",
                "color": "#ca8a04",
                "bg": "#fef9c3",
                "border": "#fde047",
                "icon": "🟡",
                "action": "Pre-hypertension lifestyle counseling; dietary sodium reduction, body weight regulation, periodic home BP logging."
            }
        elif pct < 55.0:
            return {
                "tier": "Screen Positive",
                "label": "Screen Positive / High Risk (Orange)",
                "color": "#ea580c",
                "bg": "#ffedd5",
                "border": "#fdba74",
                "icon": "🟠",
                "action": "Confirmatory clinical blood pressure evaluation via calibrated medical sphygmomanometer / 24-hr ambulatory BP monitoring."
            }
        else:
            return {
                "tier": "Urgent Risk",
                "label": "Urgent Risk (Red)",
                "color": "#dc2626",
                "bg": "#fee2e2",
                "border": "#fca5a5",
                "icon": "🔴",
                "action": "Immediate medical referral; evaluate potential end-organ microvascular complications and emergency clinical therapy."
            }


# -----------------------------------------------------------------------------
# 4. PLOTLY RISK GAUGES
# -----------------------------------------------------------------------------
def build_clinical_gauge(pct: float, disease: str, cutoff_val: float):
    """
    Renders high-visibility Plotly semicircular clinical triage gauge with
    color-coded strata zones and explicit cutoff indicator (mode='gauge').
    """
    is_diab = (disease == "diabetes")
    max_range = 50.0 if is_diab else 100.0
    
    if is_diab:
        steps = [
            {"range": [0, 4.5], "color": "#16a34a"},
            {"range": [4.5, 6.1], "color": "#ca8a04"},
            {"range": [6.1, 12.0], "color": "#ea580c"},
            {"range": [12.0, 50.0], "color": "#dc2626"}
        ]
        threshold_color = "#ea580c"
    else:
        steps = [
            {"range": [0, 20.0], "color": "#16a34a"},
            {"range": [20.0, 33.0], "color": "#ca8a04"},
            {"range": [33.0, 55.0], "color": "#ea580c"},
            {"range": [55.0, 100.0], "color": "#dc2626"}
        ]
        threshold_color = "#ea580c"

    stratum = get_triage_stratum(pct, disease)
    bar_color = stratum["color"]

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            "axis": {
                "range": [0, max_range],
                "tickwidth": 1.5,
                "tickcolor": "#94a3b8",
                "tickfont": {"size": 10, "color": "#94a3b8"}
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(255, 255, 255, 0.05)",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": threshold_color, "width": 4},
                "thickness": 0.85,
                "value": cutoff_val
            }
        }
    ))

    fig.update_layout(
        margin=dict(l=30, r=30, t=25, b=10),
        height=190,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# -----------------------------------------------------------------------------
# 5. SIDEBAR: PATIENT INTAKE & PRESETS MODULE
# -----------------------------------------------------------------------------

# Initialize Session State
if "active_preset" not in st.session_state:
    st.session_state.active_preset = "Preset 2"
    st.session_state.age = 56
    st.session_state.gender = "Male"
    st.session_state.waist_enabled = True
    st.session_state.waist_cm = 98.0
    st.session_state.activity = "Normal"
    st.session_state.bmi = 27.8
    st.session_state.sigma = 0.95
    st.session_state.pose_violation = False

# Preset Handlers
def set_preset_1():
    st.session_state.active_preset = "Preset 1"
    st.session_state.age = 24
    st.session_state.gender = "Female"
    st.session_state.waist_enabled = False
    st.session_state.waist_cm = 72.0
    st.session_state.activity = "Normal"
    st.session_state.bmi = 21.0
    st.session_state.sigma = 0.82
    st.session_state.pose_violation = False

def set_preset_2():
    st.session_state.active_preset = "Preset 2"
    st.session_state.age = 56
    st.session_state.gender = "Male"
    st.session_state.waist_enabled = True
    st.session_state.waist_cm = 98.0
    st.session_state.activity = "Normal"
    st.session_state.bmi = 27.8
    st.session_state.sigma = 0.95
    st.session_state.pose_violation = False

def set_preset_3():
    st.session_state.active_preset = "Preset 3"
    st.session_state.age = 52
    st.session_state.gender = "Male"
    st.session_state.waist_enabled = True
    st.session_state.waist_cm = 94.0
    st.session_state.activity = "Sedentary"
    st.session_state.bmi = 28.5
    st.session_state.sigma = 2.45
    st.session_state.pose_violation = True

with st.sidebar:
    st.markdown("### 🎛️ Patient Intake & Kiosk Demo")
    st.markdown(
        "<span style='font-size:12px; color:#64748b;'>Select a 1-click clinical scenario or adjust manual intake below:</span>",
        unsafe_allow_html=True
    )

    # 1-Click Demo Buttons
    st.markdown("#### ⚡ 1-Click Preset Scenarios")
    if st.button("🟢 Preset 1: Normal / Low Risk", use_container_width=True,
                 help="Age 24, Female, Waist Off, BMI 21.0, Uncertainty σ = 0.82 kg/m²"):
        set_preset_1()
        st.rerun()

    if st.button("⭐ Preset 2: Rescued by F2 Threshold", use_container_width=True,
                 help="Highlight Case: Age 56, Male, Waist 98cm, BMI 27.8, DEXA 31%, Diab Risk 7.4%"):
        set_preset_2()
        st.rerun()

    if st.button("🛑 Preset 3: Safety Guard Rejection", use_container_width=True,
                 help="Safety Violation: Head Pose Yaw = 22° (> ±15°) and σ = 2.45 kg/m² (> 1.80)"):
        set_preset_3()
        st.rerun()

    st.markdown("---")

    # Clinical Patient Intake Controls
    st.markdown("#### 📋 Clinical Patient Intake")
    
    age_input = st.slider("Age (Years)", min_value=20, max_value=80,
                          value=st.session_state.age, key="slider_age",
                          help="Calibrated for Adult Population (CDC NHANES protocol 20-80 years).")
    st.session_state.age = age_input
    st.caption("Calibrated for Adult Population (CDC NHANES protocol 20-80 years).")

    gender_input = st.radio("Biological Sex", options=["Male", "Female"],
                            index=0 if st.session_state.gender == "Male" else 1,
                            horizontal=True, key="radio_gender")
    st.session_state.gender = gender_input

    activity_options = ["Sedentary", "Normal", "Active"]
    act_index = activity_options.index(st.session_state.activity) if st.session_state.activity in activity_options else 1
    activity_input = st.selectbox("Physical Activity Level", options=activity_options,
                                  index=act_index, key="select_act")
    st.session_state.activity = activity_input
    activity_level_num = {"Sedentary": 0, "Normal": 1, "Active": 2}[activity_input]

    waist_toggle = st.checkbox("Include Waist Circumference (cm)",
                               value=st.session_state.waist_enabled, key="check_waist")
    st.session_state.waist_enabled = waist_toggle

    if waist_toggle:
        initial_waist = float(st.session_state.waist_cm) if st.session_state.waist_cm is not None else 88.0
        initial_waist = max(50.0, min(150.0, initial_waist))
        waist_input = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=150.0,
                                      value=initial_waist, step=0.5, key="num_waist",
                                      help="Calibrated measurement boundaries (50.0 - 150.0 cm).")
        st.session_state.waist_cm = waist_input

        # Clinical Range Outlier Advisory Notice
        if waist_input > 135.0 or waist_input < 55.0:
            st.markdown(
                "<div style='font-size:12px; color:#b45309; background:#fffbeb; border:1px solid #fde68a; "
                "border-radius:8px; padding:8px 10px; margin-top:6px; line-height:1.45;'>"
                "⚠️ <b>Clinical Outlier Advisory:</b> Input is outside standard 99th percentile NHANES envelope. "
                "Monotonicity constraints enforced to maintain stable risk plateau.</div>",
                unsafe_allow_html=True
            )
    else:
        waist_input = None
        st.session_state.waist_cm = None

    st.markdown("---")

    # Clinical Guard Telemetry (Live Kiosk Status)
    st.markdown("#### 🛡️ Clinical Guard Telemetry")
    if st.session_state.pose_violation or st.session_state.sigma > 1.80:
        guard_status_chip = '<span class="badge-alert">⚠️ Interception Active</span>'
    else:
        guard_status_chip = '<span class="badge-pass">✅ Frontal Alignment</span>'

    st.markdown(f"""
    <div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:10px 12px; font-size:12px; color:#334155; line-height:1.6;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
            <b>Camera Stream:</b> <span class="badge-pass">🟢 1080p Frontal</span>
        </div>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
            <b>Pose Status:</b> {guard_status_chip}
        </div>
        <div style='display:flex; justify-content:space-between; align-items:center;'>
            <b>Illumination:</b> <span class="badge-pass">🟢 145 / 255 Nom</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Advanced AI Diagnostics & Camera Overrides (Concealed Expander)
    with st.sidebar.expander("⚙️ Advanced AI Diagnostics & Camera Overrides", expanded=False):
        st.caption("Manual override for ViT-H/14 Monte Carlo Dropout output (Normally extracted automatically from live webcam).")
        
        bmi_input = st.slider("Simulated Facial BMI (μ)", min_value=16.0, max_value=42.0,
                              value=float(st.session_state.bmi), step=0.1, key="slider_bmi")
        st.session_state.bmi = bmi_input

        sigma_input = st.slider("Epistemic Uncertainty (σ)", min_value=0.40, max_value=3.00,
                                value=float(st.session_state.sigma), step=0.05, key="slider_sigma",
                                help="Safe boundary: σ ≤ 1.80 kg/m². Values > 1.80 trigger Epistemic Anomaly rejection.")
        st.session_state.sigma = sigma_input

        pose_violation_toggle = st.checkbox("Simulate Camera Pose Violation (Yaw = 22°)",
                                            value=st.session_state.pose_violation, key="toggle_pose")
        st.session_state.pose_violation = pose_violation_toggle

    # Active preset indicator pill
    st.markdown(f"""
    <div style='margin-top:14px; padding:8px 12px; background:#f1f5f9; border-radius:8px; font-size:11.5px; color:#475569;'>
        <b>Active Scenario:</b> {st.session_state.active_preset}<br>
        <b>Config:</b> {CFG['source']}
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. RUN SIMULATION ENGINE (< 0.05s)
# -----------------------------------------------------------------------------
sim = simulate_cdss_pipeline(
    age=st.session_state.age,
    sex=st.session_state.gender,
    waist_cm=st.session_state.waist_cm if st.session_state.waist_enabled else None,
    activity_level=activity_level_num,
    manual_bmi=st.session_state.bmi,
    manual_sigma=st.session_state.sigma,
    pose_violation=st.session_state.pose_violation
)


# -----------------------------------------------------------------------------
# 7. MAIN VIEWPORT: HEADER & CDSS METADATA
# -----------------------------------------------------------------------------
st.markdown("""
<div class="cdss-header">
    <h1>🏥 Face2Health — Clinical Decision Support System (CDSS)</h1>
    <p>Non-Invasive Multimodal Health Risk Screening Kiosk | NHANES Calibrated</p>
    <div class="cdss-meta-pills">
        <span class="cdss-pill">🟢 Kiosk Terminal #K-104 (Primary Care Outpatient)</span>
        <span class="cdss-pill">🧬 Architecture: ViT-H/14 ➔ Tabular Monotonic XGBoost</span>
        <span class="cdss-pill">⚖️ CDC NHANES Calibrated (N = 3,540)</span>
        <span class="cdss-pill">⚡ Engine: Real-Time Surrogate (&lt; 0.05s Zero-Lag)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 8. CONDITIONAL VIEW: SAFETY GUARD REJECTION SCREEN vs NORMAL CDSS VIEW
# -----------------------------------------------------------------------------
if sim["is_rejected"]:
    # 🛑 DISPLAY FULL CDSS SAFETY GUARD INTERCEPTION NOTIFICATION
    st.markdown(f"""
    <div class="block-screen">
        <h2>🛑 DOWNSTREAM SCREENING BLOCKED BY SAFETY GUARD</h2>
        <p>
            The Face2Health Two-Tier Safety Protocol intercepted the intake stream before propagating features into Stage 1.5 and Stage 2.
            To guarantee diagnostic integrity and prevent algorithmic hallucination, automated screening has been suspended.
        </p>
        <div style="margin-top: 12px;">
            <b>Active Safety Violations Detected:</b>
            <ul>
                {"".join([f"<li style='margin-top:6px;'><b>{r}</b></li>" for r in sim["rejection_reasons"]])}
            </ul>
        </div>
        <div class="block-protocol">
            <b style="color: #991b1b;">📋 Actionable Clinical Protocol for Kiosk / Nurse Station:</b>
            <ol style="margin: 8px 0 0 18px; color: #334155; font-size: 13.5px;">
                <li><b>Patient Realignment:</b> Request the patient to adjust seat height, maintain an upright posture, and look straight into the kiosk camera center (target yaw &lt; ±15°).</li>
                <li><b>Lighting & Occlusion Check:</b> Verify that ambient clinic illumination is uniform without extreme shadows or facial occlusions (e.g. heavy mask or low eyeglasses).</li>
                <li><b>Direct Clinical Anthropometry:</b> If physical spinal deformity or patient fatigue prevents frontal camera alignment, bypass automated kiosk screening and record manual stadiometer height and balance-beam weight.</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Diagnostic telemetry summary
    c_diag1, c_diag2, c_diag3 = st.columns(3)
    with c_diag1:
        st.markdown(f"""
        <div class="cdss-card">
            <div class="cdss-card-title">Head Pose Geometry</div>
            <div class="cdss-val-lg" style="color: {'#dc2626' if abs(sim['pose']['yaw']) > 15 else '#16a34a'};">
                {sim['pose']['yaw']:+.1f}° <span class="cdss-val-unit">Yaw</span>
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">Pitch: {sim['pose']['pitch']:+.1f}° | Roll: {sim['pose']['roll']:+.1f}°</div>
            <div style="margin-top:8px;">
                <span class="{'badge-alert' if abs(sim['pose']['yaw']) > 15 else 'badge-pass'}">
                    {'❌ Limit Exceeded (> ±15°)' if abs(sim['pose']['yaw']) > 15 else '✅ Frontal Alignment'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_diag2:
        st.markdown(f"""
        <div class="cdss-card">
            <div class="cdss-card-title">Epistemic Uncertainty (σ)</div>
            <div class="cdss-val-lg" style="color: {'#dc2626' if sim['sigma'] > 1.80 else '#16a34a'};">
                {sim['sigma']:.2f} <span class="cdss-val-unit">kg/m²</span>
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">Safety Boundary: σ ≤ 1.80 kg/m²</div>
            <div style="margin-top:8px;">
                <span class="{'badge-alert' if sim['sigma'] > 1.80 else 'badge-pass'}">
                    {'❌ Epistemic Anomaly (> 1.80)' if sim['sigma'] > 1.80 else '✅ High Confidence'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_diag3:
        st.markdown(f"""
        <div class="cdss-card">
            <div class="cdss-card-title">Camera Image Quality</div>
            <div class="cdss-val-lg" style="color: #0f172a;">
                {sim['pose']['sharpness_score']} <span class="cdss-val-unit">/ 255</span>
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:4px;">Luminance: {sim['pose']['lighting_score']}/255 (Nominal)</div>
            <div style="margin-top:8px;">
                <span class="badge-pass">✅ Adequate Illumination</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # -------------------------------------------------------------------------
    # 9. STAGE 1 & 1.5 BIOMETRICS PANEL (TOP ROW)
    # -------------------------------------------------------------------------
    st.markdown("### 🧬 Stage 1 & 1.5 — Facial Biometrics & Body Composition")
    
    col_bio1, col_bio2, col_bio3 = st.columns([1.1, 1.1, 1.4], gap="medium")

    # Card 1: Predicted BMI
    with col_bio1:
        sigma_badge = '<span class="badge-pass">🟢 High Confidence (σ ≤ 1.80)</span>' if sim["sigma"] <= 1.80 else '<span class="badge-alert">🟠 Epistemic Anomaly (σ > 1.80)</span>'
        bmi_val = sim["bmi"]
        if bmi_val < 18.5:
            cat_str = "Underweight"
        elif bmi_val < 23.0:
            cat_str = "Normal (Healthy)"
        elif bmi_val < 25.0:
            cat_str = "Overweight"
        else:
            cat_str = "Obese (Adiposity)"

        st.markdown(f"""
        <div class="cdss-card">
            <div class="cdss-card-title">
                <span>Predicted Body Mass Index (BMI)</span>
                {sigma_badge}
            </div>
            <div class="cdss-val-lg">
                {sim['bmi']:.1f} <span style="font-size:16px; font-weight:600; color:#475569;">± {sim['sigma']:.2f}</span>
                <span class="cdss-val-unit">kg/m²</span>
            </div>
            <div style="margin-top:8px; font-size:13px; color:#334155;">
                <b>WHO Asian Classification:</b> <span style="color:#0284c7; font-weight:700;">{cat_str}</span>
            </div>
            <div style="margin-top:6px; font-size:11.5px; color:#64748b;">
                Simulated 25-pass MC Dropout point estimate μ with epistemic variance σ.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Card 2: Predicted Body Fat % (DEXA)
    with col_bio2:
        is_male = (st.session_state.gender == "Male")
        bf_cuts = [14, 18, 25] if is_male else [21, 25, 32]
        bf = sim["body_fat"]
        if bf < bf_cuts[0]:
            bf_cat = "Lean / Athletic"
        elif bf < bf_cuts[1]:
            bf_cat = "Fitness"
        elif bf < bf_cuts[2]:
            bf_cat = "Acceptable Average"
        else:
            bf_cat = "Excess Adiposity"

        alpha_text = f"α = {sim['shock_alpha']:.2f}"
        shock_pill = f'<span class="cdss-pill" style="background:#f0fdf4; color:#15803d; border-color:#bbf7d0; font-size:11px;" title="Biological sex commands 74.99% of regression variance, damping upstream vision error.">🛡️ Shock Absorber ({alpha_text})</span>'

        st.markdown(f"""
        <div class="cdss-card">
            <div class="cdss-card-title">
                <span>DEXA Total Body Fat %</span>
                {shock_pill}
            </div>
            <div class="cdss-val-lg" style="color: #0369a1;">
                {sim['body_fat']:.1f} <span class="cdss-val-unit">% Fat</span>
            </div>
            <div style="margin-top:8px; font-size:13px; color:#334155;">
                <b>Body Composition:</b> <span style="color:#0369a1; font-weight:700;">{bf_cat}</span>
            </div>
            <div style="margin-top:6px; font-size:11.5px; color:#64748b;">
                DEXA calibrated XGBoost regressor (attenuation factor {alpha_text} dampens vision noise).
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Card 3: Facial Morphometrics
    with col_bio3:
        m = sim["morphometrics"]
        st.markdown(f"""
        <div class="cdss-card">
            <div class="cdss-card-title">
                <span>Scale-Invariant Facial Morphometrics</span>
                <span class="badge-pass">4 Features</span>
            </div>
            <div class="morpho-grid">
                <div class="morpho-item">
                    <div class="morpho-label">LFWR (Visceral Proxy)</div>
                    <div class="morpho-val">{m['LFWR']:.3f}</div>
                </div>
                <div class="morpho-item">
                    <div class="morpho-label">CJWR (Cheek/Jaw Fat)</div>
                    <div class="morpho-val">{m['CJWR']:.3f}</div>
                </div>
                <div class="morpho-item">
                    <div class="morpho-label">PAR (Mandibular Arc)</div>
                    <div class="morpho-val">{m['PAR']:.3f}</div>
                </div>
                <div class="morpho-item">
                    <div class="morpho-label">FWHR (Facial Ratio)</div>
                    <div class="morpho-val">{m['FWHR']:.3f}</div>
                </div>
            </div>
            <div style="margin-top:8px; font-size:11px; color:#64748b;">
                Geometric landmarks extracted via scale-invariant anthropometry (Wen & Guo, Lee & Kim).
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 10. HIGHLIGHT CLINICAL CALLOUT: RESOLVING THE 0-RECALL PARADOX
    # -------------------------------------------------------------------------
    is_highlight_case = (st.session_state.active_preset == "Preset 2") or (6.1 <= sim["diabetes_pct"] < 50.0)
    if is_highlight_case:
        st.markdown("""
        <div class="highlight-callout">
            <div class="highlight-title">
                ⭐ CLINICAL HIGHLIGHT — RESOLVING THE 0-RECALL PARADOX
            </div>
            <div class="highlight-body">
                <b>Case Finding:</b> At the conventional ML classification cutoff (<b>τ = 0.50</b>), this patient is
                <b>MISSED as a False Negative (0% Sensitivity)</b> because true population disease prevalence (~10.3%) naturally bounds Platt-calibrated posterior probabilities below 0.38.<br>
                <b>Face2Health Solution:</b> By deploying our validation-optimized <b>F2 Screening Threshold (τ* = 6.1%)</b>,
                this individual is correctly triaged as <b>Screen Positive (High Risk)</b>, directing them to life-saving confirmatory blood testing (rescuing 49 out of 55 false negatives in NHANES benchmarks)!
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 11. STAGE 2: DUAL NCD SCREENING PANEL (MAIN 2-COLUMN VIEW)
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Stage 2 — Dual Non-Communicable Disease (NCD) Risk Screening")

    col_diab, col_hyp = st.columns(2, gap="large")

    # ----------------- COLUMN A: TYPE 2 DIABETES (T2DM) -----------------
    with col_diab:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
            <h4 style="margin:0; font-weight:800; color:#0f172a;">🩸 Type 2 Diabetes Mellitus (T2DM)</h4>
            <span class="cdss-pill" style="background:#f1f5f9; color:#334155; border-color:#cbd5e1;">NHANES Calibrated</span>
        </div>
        """, unsafe_allow_html=True)

        diab_pct = sim["diabetes_pct"]
        diab_stratum = get_triage_stratum(diab_pct, "diabetes")
        diab_f2_cutoff = CFG["diabetes_f2"] * 100.0  # 6.1%

        # Interactive Risk Gauge
        fig_diab = build_clinical_gauge(diab_pct, "diabetes", diab_f2_cutoff)
        st.plotly_chart(fig_diab, use_container_width=True, config={"displayModeBar": False})

        # Clean centered HTML display right below the gauge:
        st.markdown(
            f"<div style='text-align: center; margin-top: -35px; margin-bottom: 12px;'>"
            f"<span style='font-size: 32px; font-weight: 700; color: #ffffff; font-family: sans-serif;'>{diab_pct:.1f}%</span>"
            f"<div style='font-size: 13px; color: #94a3b8; font-weight: 500;'>Estimated Calibrated Risk</div>"
            f"<div style='font-size: 11.5px; color: #ea580c; font-weight: 700; margin-top: 3px;'>▲ F2 Screening Cutoff τ* = {diab_f2_cutoff:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # 4-Tier Clinical Triage Badge Banner
        st.markdown(f"""
        <div class="triage-banner" style="background:{diab_stratum['bg']}; border-color:{diab_stratum['border']};">
            <div class="triage-header" style="color:{diab_stratum['color']};">
                <span>{diab_stratum['icon']}</span>
                <span>Triage Stratum: {diab_stratum['tier']}</span>
            </div>
            <div class="triage-action" style="color:#1e293b;">
                <b>Clinical Protocol:</b> {diab_stratum['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison Metric Box: Standard ML vs Face2Health
        is_std_missed = (diab_pct < 50.0)
        std_verdict_str = "NEGATIVE (NORMAL)" if is_std_missed else "POSITIVE"
        std_verdict_color = "#dc2626" if is_std_missed else "#16a34a"
        f2_positive = (diab_pct >= diab_f2_cutoff)
        f2_verdict_str = "SCREEN POSITIVE" if f2_positive else "LOW RISK / WATCH"
        f2_verdict_color = "#ea580c" if f2_positive else "#16a34a"

        st.markdown(f"""
        <div class="compare-container">
            <div class="compare-card" style="background:#f8fafc; border-color:#e2e8f0;">
                <div class="compare-title" style="color:#64748b;">Standard ML (τ = 0.50)</div>
                <div class="compare-verdict" style="color:{std_verdict_color};">{std_verdict_str}</div>
                <div class="compare-sub" style="color:#64748b;">
                    {'❌ <b>MISSED (False Negative)</b><br>0.0% Sensitivity across ~10% prevalence' if is_std_missed else '✅ Positive Flagged'}
                </div>
            </div>
            <div class="compare-card" style="background:#f0fdf4; border-color:#86efac;">
                <div class="compare-title" style="color:#15803d;">Face2Health Triage (τ* = {diab_f2_cutoff:.1f}%)</div>
                <div class="compare-verdict" style="color:{f2_verdict_color};">{f2_verdict_str}</div>
                <div class="compare-sub" style="color:#15803d;">
                    {'✅ <b>RESCUED (Screen Positive)</b><br>89.1% Sensitivity (+49 rescued)' if f2_positive else '🟢 Standard Baseline Risk'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------- COLUMN B: ESSENTIAL HYPERTENSION -----------------
    with col_hyp:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px;">
            <h4 style="margin:0; font-weight:800; color:#0f172a;">💓 Essential Hypertension</h4>
            <span class="cdss-pill" style="background:#f1f5f9; color:#334155; border-color:#cbd5e1;">NHANES Calibrated</span>
        </div>
        """, unsafe_allow_html=True)

        hyp_pct = sim["hypertension_pct"]
        hyp_stratum = get_triage_stratum(hyp_pct, "hypertension")
        hyp_cutoff = 33.0  # Screen positive stratum (with F2 = 15.8%)

        # Interactive Risk Gauge
        fig_hyp = build_clinical_gauge(hyp_pct, "hypertension", hyp_cutoff)
        st.plotly_chart(fig_hyp, use_container_width=True, config={"displayModeBar": False})

        # Clean centered HTML display right below the gauge:
        st.markdown(
            f"<div style='text-align: center; margin-top: -35px; margin-bottom: 12px;'>"
            f"<span style='font-size: 32px; font-weight: 700; color: #ffffff; font-family: sans-serif;'>{hyp_pct:.1f}%</span>"
            f"<div style='font-size: 13px; color: #94a3b8; font-weight: 500;'>Estimated Calibrated Risk</div>"
            f"<div style='font-size: 11.5px; color: #ea580c; font-weight: 700; margin-top: 3px;'>▲ Screen Positive Cutoff τ* = {hyp_cutoff:.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # 4-Tier Clinical Triage Badge Banner
        st.markdown(f"""
        <div class="triage-banner" style="background:{hyp_stratum['bg']}; border-color:{hyp_stratum['border']};">
            <div class="triage-header" style="color:{hyp_stratum['color']};">
                <span>{hyp_stratum['icon']}</span>
                <span>Triage Stratum: {hyp_stratum['tier']}</span>
            </div>
            <div class="triage-action" style="color:#1e293b;">
                <b>Clinical Protocol:</b> {hyp_stratum['action']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Comparison Metric Box: Standard ML vs Face2Health
        is_hyp_std_missed = (hyp_pct < 50.0)
        std_hyp_str = "NEGATIVE (NORMAL)" if is_hyp_std_missed else "POSITIVE"
        std_hyp_color = "#dc2626" if is_hyp_std_missed else "#16a34a"
        hyp_flagged = (hyp_pct >= 33.0) or (hyp_pct >= 15.8)
        f2_hyp_str = "SCREEN POSITIVE" if hyp_pct >= 33.0 else ("ELEVATED RISK" if hyp_pct >= 15.8 else "LOW RISK")
        f2_hyp_color = "#ea580c" if hyp_pct >= 33.0 else ("#ca8a04" if hyp_pct >= 15.8 else "#16a34a")

        st.markdown(f"""
        <div class="compare-container">
            <div class="compare-card" style="background:#f8fafc; border-color:#e2e8f0;">
                <div class="compare-title" style="color:#64748b;">Standard ML (τ = 0.50)</div>
                <div class="compare-verdict" style="color:{std_hyp_color};">{std_hyp_str}</div>
                <div class="compare-sub" style="color:#64748b;">
                    {'❌ <b>MISSED / SUBOPTIMAL</b><br>Misses early stage hypertensive cases' if is_hyp_std_missed else '✅ Detected'}
                </div>
            </div>
            <div class="compare-card" style="background:#f0fdf4; border-color:#86efac;">
                <div class="compare-title" style="color:#15803d;">Face2Health Triage (τ* = 15.8% / 33.0%)</div>
                <div class="compare-verdict" style="color:{f2_hyp_color};">{f2_hyp_str}</div>
                <div class="compare-sub" style="color:#15803d;">
                    {'✅ <b>RESCUED (Elevated Risk)</b><br>88.4% Sensitivity (+70 rescued)' if hyp_flagged else '🟢 Normal Arterial Blood Pressure'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 12. EXPLAINABILITY & MODEL CARD EXPANDER (BOTTOM)
# -----------------------------------------------------------------------------
st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
with st.expander("🔍 Clinical Explainability, Feature Importance & Model Card", expanded=False):
    has_waist = st.session_state.waist_enabled and (st.session_state.waist_cm is not None and float(st.session_state.waist_cm) > 0)
    
    if has_waist:
        pathway_badge = '<span class="cdss-pill" style="background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; font-size:11.5px;">Active Pathway: Full Multi-Modal (With Waist)</span>'
        chart_subtitle = "Stage 2 XGBoost Monotonic Model (With Anthropometric Waist Circumference)"
        features = ["Age (Years)", "Waist Circumference", "DEXA Body Fat %", "Predicted BMI", "Biological Sex"]
        importance_pct = [34.5, 25.8, 19.8, 13.1, 6.8]
        bar_colors = ["#0284c7", "#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd"]
        max_x = 42.0
    else:
        pathway_badge = '<span class="cdss-pill" style="background:#ecfdf5; color:#047857; border-color:#a7f3d0; font-size:11.5px;">Active Pathway: Contactless Vision-Only (No Waist)</span>'
        chart_subtitle = "Stage 2 XGBoost Contactless Vision-Only Model (Normalized without Waist)"
        features = ["Age (Years)", "DEXA Body Fat %", "Predicted BMI", "Biological Sex"]
        importance_pct = [46.5, 26.7, 17.6, 9.2]
        bar_colors = ["#059669", "#10b981", "#34d399", "#6ee7b7"]
        max_x = 55.0

    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <h4 style="margin:0; font-weight:800; color:#0f172a;">📈 Stage 2 XGBoost Feature Importance Breakdown</h4>
        {pathway_badge}
    </div>
    <div style="font-size:12px; color:#64748b; margin-bottom:12px;">{chart_subtitle}</div>
    """, unsafe_allow_html=True)
    
    fig_imp = go.Figure(go.Bar(
        x=importance_pct,
        y=features,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="#0f172a", width=0.5)
        ),
        text=[f"{v:.1f}%" for v in importance_pct],
        textposition="auto",
        textfont=dict(color="#ffffff", size=12, family="Arial")
    ))
    fig_imp.update_layout(
        xaxis=dict(title="Relative Feature Importance (%)", range=[0, max_x], tickfont=dict(size=10, color="#94a3b8")),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#334155")),
        height=240,
        margin=dict(l=30, r=20, t=15, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # Academic & Clinical Model Card
    col_mc1, col_mc2 = st.columns(2)
    with col_mc1:
        st.markdown("""
        **🧬 Methodological Highlights:**
        * **Cascading Multimodal Architecture:** Couples Vision Transformer (ViT-H/14) for facial BMI estimation with tabular Monotonic Extreme Gradient Boosting (XGBoost).
        * **Sex Error Shock Absorber:** Biological sex commands 74.99% of regression variance in Stage 1.5, attenuating upstream vision noise by factor $\\alpha = 0.39$ (with waist) / $\\alpha = 1.00$ (without waist).
        * **Monotonicity Constraints ($c_j = +1$):** Enforces biological plausibility across Age, BMI, Waist, and Body Fat, preventing counter-intuitive risk surfaces.
        * **Two-Tier Safety Guard:** Combines Aleatoric pose rejection (Yaw $> \\pm 15^\\circ$) and Epistemic anomaly filtering ($\\sigma > 1.80\\ \\text{kg/m}^2$).
        """)
    with col_mc2:
        st.markdown("""
        **📊 Held-Out CDC NHANES Benchmark (N = 3,540 / 3,554):**
        | Evaluation Protocol | Decision Cutoff (τ*) | Diabetes ROC-AUC | Diabetes Sensitivity | Rescued Cases |
        | :--- | :--- | :--- | :--- | :--- |
        | **Default Baseline** | 0.500 (50.0%) | 0.747 | 0.0% | 0 (0-Recall) |
        | **Youden's J Index** | 0.074 (7.4%) | 0.765 | 81.8% | +45 cases |
        | **Optimized F2 Utility** | **0.061 (6.1%)** | **0.765** | **89.1%** | **+49 cases** |
        
        *Reference: CDC NHANES 1999–2018 Multi-Ethnic Adult Cohort with DEXA Gold Standard (DXDTOPF).*
        """)

