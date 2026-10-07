# Face2Health: Non-Invasive Multimodal Health Risk Screening & Clinical Decision Support System (CDSS)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%2011.8)-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Monotonic%20Calibrated-brightgreen.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34+-FF4B4B.svg)](https://streamlit.io/)
[![CDC NHANES](https://img.shields.io/badge/Dataset-CDC%20NHANES%20Benchmark-informational.svg)](https://www.cdc.gov/nchs/nhanes/index.htm)
[![IEEE Conference Format](https://img.shields.io/badge/Publication-IEEE%20Format-00629B.svg)](reports/Face2Health_IEEE_Conference_Paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Face2Health** is an end-to-end, non-invasive, multi-modal clinical screening pipeline and decision support kiosk system for primary healthcare triage. It couples **Vision Transformers (ViT-H/14 with Monte Carlo Dropout)**, **Scale-Invariant Facial Morphometrics**, and **Tabular Monotonic Extreme Gradient Boosting (XGBoost)** calibrated against the gold-standard **CDC NHANES** multi-ethnic adult cohort ($N = 3,540$).

The platform addresses the silent progression of major Non-Communicable Diseases (NCDs) — **Type 2 Diabetes Mellitus (T2DM)** and **Essential Hypertension** — by resolving the traditional **"0-Recall Paradox"** in clinical screening through Platt scaling and utility-optimized $F_2$-score decision cutoffs.

---

## 📑 Table of Contents
1. [Key Features & Clinical Innovations](#-key-features--clinical-innovations)
2. [End-to-End System Inputs & Outputs](#-end-to-end-system-inputs--outputs)
3. [Pipeline Architecture & Theory](#-pipeline-architecture--theory)
   - [Two-Tier Safety Guard Protocol](#1-two-tier-safety-guard-protocol)
   - [Stage 1: ViT-H/14 Facial BMI & Morphometrics](#2-stage-1-face--bmi-via-vit-h14-mc-dropout--facial-morphometry)
   - [Stage 1.5: DEXA Body Fat & Sex Error Shock Absorber](#3-stage-15-dexa-body-fat-regression--sex-error-shock-absorber-alpha--039)
   - [Stage 2: Monotonic Calibrated XGBoost Classifiers](#4-stage-2-monotonic-xgboost-classifiers--platt-calibration)
   - [Resolving the 0-Recall Paradox via F2-Optimization](#5-resolving-the-0-recall-paradox-via-f_2-score-utility-optimization)
4. [4-Tier Clinical Triage Protocol](#-4-tier-clinical-triage-protocol)
5. [Repository Structure & File Inventory](#-repository-structure--file-inventory)
6. [Installation & Environment Setup](#-installation--environment-setup)
7. [Execution Guide & Runnable Entry Points](#-execution-guide--runnable-entry-points)
   - [Entry Point 1: CDSS Kiosk Simulation Dashboard](#entry-point-1-cdss-kiosk-simulation-dashboard-run_dashboardbat)
   - [Entry Point 2: Production Health Screening Application](#entry-point-2-production-health-screening-app-run_appbat)
   - [Entry Point 3: CLI Prediction Pipeline & Sanity Check](#entry-point-3-cli-pipeline-inference--sanity-checks)
   - [Entry Point 4: IEEE Academic Paper Compilation](#entry-point-4-ieee-academic-paper-compilation)
   - [Entry Point 5: Full Pipeline Retraining & ETL](#entry-point-5-data-etl--model-retraining-pipeline)
8. [Held-Out Benchmark Evaluation](#-held-out-benchmark-evaluation)
9. [Medical & Regulatory Disclaimer](#-medical--regulatory-disclaimer)
10. [Acknowledgements & Citation](#-acknowledgements--citation)

---

## 🌟 Key Features & Clinical Innovations

- **🛡️ Two-Tier Safety Guard Protocol:** Protects against algorithmic hallucination by coupling real-time geometric pose inspection ($|\text{Yaw}| \le 15^\circ$, $|\text{Roll}| \le 15^\circ$, inter-cheek symmetry ratio $\le 0.50$) with epistemic uncertainty rejection ($\sigma \le 1.80\text{ kg/m}^2$).
- **🧬 Scale-Invariant Facial Morphometry:** Extracts anthropometric ratios correlated with visceral and subcutaneous adiposity:
  - **LFWR** (Lower-Face Width to Face Height Ratio) — Visceral adiposity proxy (*Lee & Kim 2014*).
  - **CJWR** (Cheek-to-Jaw Width Ratio) — Buccal/Jowl fat pad indicator (*Coetzee 2009*).
  - **PAR** (Perimeter-to-Area Ratio) — Mandibular circularity (*Wen & Guo 2013*).
  - **FWHR** (Facial Width-to-Height Ratio) — Mid-facial skeletal ratio.
- **🛡️ Biological Sex Error Shock Absorber ($\alpha = 0.39$):** Biological sex commands $74.99\%$ of variance in dual-energy X-ray absorptiometry (DEXA) body fat regression. Upstream facial BMI estimation noise is naturally attenuated by factor $\alpha = 0.39$ (with waist) / $\alpha = 1.00$ (without waist), preventing downstream error propagation.
- **⚖️ Monotonicity Constraints ($c_j = +1$):** Enforces biologically plausible risk surfaces across Age, BMI, Waist Circumference, and Body Fat, eliminating counter-intuitive step-function risk dips.
- **📈 Elimination of the 0-Recall Paradox:** Traditional ML thresholds ($\tau = 0.50$) yield $0.0\%$ Sensitivity due to population prevalence limits ($\sim 10.3\%$). By combining Platt sigmoid scaling with validation-tuned $F_2$-score screening cutoffs ($\tau^* = 0.061$ for T2DM, $\tau^* = 0.158$ for HTN), the system achieves **$89.1\%$ Sensitivity** in diabetes and **$88.4\%$ Sensitivity** in hypertension, rescuing $49/55$ and $70/91$ false-negative cases respectively.
- **🏥 Two Dedicated Application Modes:**
  1. **Production Full-Stack Application (`scripts/app.py`):** Loads PyTorch ViT-H/14 weights, webcam input, and exports bilingual clinical PDF reports.
  2. **Standalone CDSS Kiosk Dashboard (`dashboard/app.py`):** Zero-lag mathematical simulation surrogate ($< 0.05\text{s}$) tailored for nurse stations, outpatient kiosks, and live clinical demonstrations.

---

## 🎯 End-to-End System Inputs & Outputs

### System Inputs

| Input Parameter | Data Format | Requirement | Description & Clinical Bounds |
|---|---|:---:|---|
| **Facial Photograph** | `.jpg`, `.jpeg`, `.png`, `.webp`, or Live Webcam | **Mandatory** | Frontal face portrait. Verified by Two-Tier Safety Guard for pose, blur, and illumination. |
| **Chronological Age** | Integer (12–99 in UI; 10–100 in backend) | **Mandatory** | Calibrated against CDC NHANES adult cohort protocols. |
| **Biological Sex** | Binary (`Male` / `Female`) | **Mandatory** | Encoded as `GENDER_NUM` (Male = 1, Female = 0). Governs DEXA shock absorber and body fat cutoffs. |
| **Physical Activity Level** | Categorical (`Sedentary`, `Normal`, `Active`) | **Mandatory** | Mapped to `LIFESTYLE_LEVEL` (0 = Sedentary, 1 = Normal, 2 = Active). Trained as an empirical feature. |
| **Waist Circumference** | Floating point (50.0–160.0 cm) | *Optional* | Activates **Dual-Route Mode** (`with_waist` vs `no_waist`) for visceral adiposity refinement. |
| **Face Guard Bypass** | Boolean Checkbox | *Optional* | User-permitted override allowing processing of sub-optimal photos with explicit clinical advisories. |

### Expected Pipeline Outputs

| Pipeline Stage | Output Metric | Units / Format | Clinical Interpretation & Strata |
|---|---|---|---|
| **Stage 1 (ViT-H/14)** | Predicted BMI ($\mu$) | $\text{kg/m}^2$ (Float) | WHO Asian Classification: Underweight ($<18.5$), Normal ($18.5–23$), Overweight ($23–25$), Obese ($\ge 25$). |
| **Stage 1 (Uncertainty)** | Epistemic Uncertainty ($\sigma$) | $\pm\text{SD}$ (Float) | 25-pass Monte Carlo Dropout. Guard intercept triggers if $\sigma > 1.80\text{ kg/m}^2$. |
| **Stage 1 (Morphometry)** | Facial Ratios (LFWR, CJWR, PAR, FWHR) | Float Ratios | Scale-invariant anatomical proxies for visceral and buccal fat accumulation. |
| **Stage 1.5 (XGBoost)** | Total Body Fat % | % (Float) | Calibrated against DEXA gold standard. Sex-stratified: Lean, Fit, Acceptable, Obese. |
| **Stage 2 (Classifiers)** | Calibrated T2DM Risk ($P$) | % Probability (0–100%) | Platt-calibrated posterior probability of Type 2 Diabetes Mellitus. |
| **Stage 2 (Classifiers)** | Calibrated HTN Risk ($P$) | % Probability (0–100%) | Platt-calibrated posterior probability of Essential Hypertension. |
| **Stage 3 (Triage)** | Clinical Triage Stratum | 4 Tiers | **Low Risk** (🟢), **Watchful** (🟡), **Screen Positive** (🟠), **Urgent Risk** (🔴). |
| **Export Engines** | PDF / IEEE Paper | `.pdf` / `.docx` | Patient summary report with clinical confirmatory recommendations or IEEE conference publication. |

---

## 🔄 Pipeline Architecture & Theory

```
                                 [ Facial Selfie / Webcam ]
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Safety Layer 1: Aleatoric Face Guard      │
                      │  - Face Detection (MediaPipe FaceMesh/Haar)  │
                      │  - Pose Tolerance: |Yaw| ≤ 15°, |Roll| ≤ 15° │
                      │  - Laplacian Blur Variance ≥ 18.0            │
                      └──────────────────────┬───────────────────────┘
                                             │ PASS
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Stage 1: Vision Transformer (ViT-H/14)    │
                      │  - Backbone: 632M Parameters (SWAG Pretrain) │
                      │  - Head: 5-Layer MLP Head with MC Dropout    │
                      │  - Forward Passes: n = 25-30 Iterations      │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │  Safety Layer 2: Epistemic Anomaly Guard     │
                      │  - Variance Check: σ ≤ 1.80 kg/m²            │
                      │  - Scale-Invariant Morphometry (LFWR, CJWR)  │
                      └──────────────────────┬───────────────────────┘
                                             │ PASS: Predicted BMI (ŷ₁)
                                             ▼
    [ Demographics: Age, Biological Sex ] ───┼─── [ Lifestyle Level (0, 1, 2) ]
    [ Optional: Waist Circumference (cm)] ───┤
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ Stage 1.5: DEXA Body Fat Regression          │
                      │  - Dual Route: With Waist vs Without Waist   │
                      │  - Biological Sex Error Shock Absorber       │
                      │    (Attenuation Factor α = 0.39 / 1.00)      │
                      └──────────────────────┬───────────────────────┘
                                             │ Predicted Body Fat % (ŷ₂)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ Stage 2: Monotonic XGBoost Classifiers       │
                      │  - Monotonicity Constraints (+1 on all adiposity)
                      │  - Platt Scaling Sigmoid Calibration (CV = 5)│
                      │  - Targets: Type 2 Diabetes & Hypertension   │
                      └──────────────────────┬───────────────────────┘
                                             │ Posterior Probabilities P(Y=1|X)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │ Stage 3: 4-Tier Clinical Triage Engine       │
                      │  - Utility-Optimized F2 Screening Cutoffs    │
                      │    (T2DM τ* = 0.061 | HTN τ* = 0.158)        │
                      │  - Triage Strata: Low, Watch, Screen+, Urgent│
                      └──────────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
            [ CDSS Interactive Kiosk UI ]           [ Bilingual PDF Clinical Export ]
```

### 1. Two-Tier Safety Guard Protocol
- **Safety Layer 1 (Aleatoric Guard — Pose & Illumination):** Non-frontal perspective introduces parallax distortion. MediaPipe FaceMesh verifies that head yaw angle $|\text{Yaw}| \le 15^\circ$, roll angle $|\text{Roll}| \le 15^\circ$, and facial symmetry asymmetry $\Delta_{\text{sym}} \le 50\%$. The image must achieve Laplacian variance $\text{Var}(\nabla^2 I) \ge 18.0$ and nominal brightness ($55 \le \bar{I} \le 212$).
- **Safety Layer 2 (Epistemic Guard — Out-of-Distribution Rejection):** Monte Carlo Dropout samples 25 stochastic forward passes. If epistemic standard deviation $\sigma > 1.80\text{ kg/m}^2$, the facial structure is flagged as out-of-distribution (OOD) for the Vision Transformer, safely blocking downstream screening to prevent algorithmic hallucination.

### 2. Stage 1: Face → BMI via ViT-H/14 MC Dropout & Facial Morphometry
- **ViT-H/14 Backbone:** Evaluates $518 \times 518$ bicubic RGB tensors using ImageNet-1K SWAG pre-trained weights (`IMAGENET1K_SWAG_E2E_V1`).
- **Regression Architecture (`BMIHead`):**
  $$\text{Input (1280)} \xrightarrow{\text{FC}} 640 \xrightarrow{\text{GELU}} \text{Dropout}(0.5) \xrightarrow{\text{FC}} 320 \xrightarrow{\text{GELU}} 160 \xrightarrow{\text{GELU}} 80 \xrightarrow{\text{GELU}} 1$$
- **Morphometric Indicators:**
  $$\text{CJWR} = \frac{\text{Bizygomatic Width}}{\text{Bigonial Width}}, \quad \text{LFWR} = \frac{\text{Bigonial Width}}{\text{Lower Face Height}}, \quad \text{PAR} = \frac{\text{Jawline Perimeter}}{\sqrt{\text{Jawline Area}}}$$

### 3. Stage 1.5: DEXA Body Fat Regression & Sex Error Shock Absorber ($\alpha = 0.39$)
- **Dual Route Formulation:**
  - Route A (`with_waist`): Features = `['BMI', 'AGE', 'GENDER_NUM', 'WAIST_CM', 'LIFESTYLE_LEVEL']`
  - Route B (`no_waist`): Features = `['BMI', 'AGE', 'GENDER_NUM', 'LIFESTYLE_LEVEL']`
- **Error Shock Absorber Effect:** Biological sex commands $74.99\%$ of regression variance in DEXA total body fat ($R^2 = 0.78$, $\text{MAE} = 2.45\%$). The gradient of predicted body fat with respect to upstream BMI is bounded:
  $$\frac{\partial \hat{y}_{\text{fat}}}{\partial \hat{y}_{\text{BMI}}} \approx \alpha = 0.39 \quad (\text{with waist}), \quad \alpha = 1.00 \quad (\text{without waist})$$
  This dampens vision estimation variance by over $60\%$, insulating Stage 2 from error cascading.

### 4. Stage 2: Monotonic XGBoost Classifiers & Platt Calibration
- **Monotonicity Enforcement ($c_j = +1$):**
  $$\frac{\partial P(\text{NCD})}{\partial \text{Age}} \ge 0, \quad \frac{\partial P(\text{NCD})}{\partial \text{BMI}} \ge 0, \quad \frac{\partial P(\text{NCD})}{\partial \text{Waist}} \ge 0, \quad \frac{\partial P(\text{NCD})}{\partial \text{BodyFat}} \ge 0$$
  Implemented via XGBoost `monotone_constraints = (1, 0, 1, 1, 1)`.
- **Platt Sigmoid Scaling:** Uncalibrated tree ensembles produce distorted sigmoid-shaped reliability diagrams. Platt scaling via `CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv=5)` maps decision margins to true posterior probabilities $P(Y=1|X) = \frac{1}{1 + \exp(A \cdot f(X) + B)}$, yielding Brier scores of **$0.0881$** (T2DM) and **$0.1870$** (HTN).

### 5. Resolving the 0-Recall Paradox via $F_2$-Score Utility Optimization
In clinical epidemiology, True Population Disease Prevalence bounds posterior probabilities. With T2DM prevalence around $\sim 10.3\%$, Platt-calibrated probabilities rarely exceed $0.38$. Consequently, the standard machine learning default cutoff ($\tau = 0.50$) produces catastrophic **0.0% Sensitivity (0 out of 55 positive patients detected)**.

To prioritize clinical recall over precision (as missing an undiagnosed patient incurs severe medical consequences), decision cutoffs are optimized on a held-out validation cohort ($15\%$ split, zero test leakage) maximizing the $F_\beta$ metric with $\beta = 2.0$:
$$F_2 = (1 + 2^2) \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}}$$

- **Type 2 Diabetes:** $\tau^* = 0.061$ ($6.1\%$) with waist / $\tau^* = 0.064$ ($6.4\%$) without waist $\rightarrow$ **Recall rises from 0.0% to 89.1% (+49 cases rescued)**.
- **Essential Hypertension:** $\tau^* = 0.158$ ($15.8\%$) $F_2$ cutoff / $\tau^* = 0.330$ ($33.0\%$) Youden cutoff $\rightarrow$ **Recall rises from 49.7% to 88.4% (+70 cases rescued)**.

---

## 🩺 4-Tier Clinical Triage Protocol

Calibrated posterior probabilities are mapped into actionable clinical pathways synchronized via `weights/thresholds.json`:

| Triage Stratum | Badge & Color | T2DM Posterior Probability ($P$) | HTN Posterior Probability ($P$) | Clinical Protocol & Referral Action |
|---|:---:|:---:|:---:|---|
| **Low Risk** | 🟢 Green | $P < 4.5\%$ | $P < 20.0\%$ | Standard annual medical checkup; encourage routine healthy nutrition and physical activity. |
| **Watchful** | 🟡 Yellow | $4.5\% \le P < 6.1\%$ | $20.0\% \le P < 33.0\%$ | Pre-disease lifestyle counseling; reduce refined sugars/sodium, body weight regulation, repeat screening in 6–12 months. |
| **Screen Positive / High Risk** | 🟠 Orange | $P \ge 6.1\%$ ($F_2$ Cutoff) | $P \ge 33.0\%$ (Youden) / $\ge 15.8\%$ ($F_2$) | Preliminary screen positive; recommend confirmatory laboratory phlebotomy (FPG $\ge 126\text{ mg/dL}$, HbA1c $\ge 6.5\%$) or calibrated sphygmomanometry. |
| **Urgent Risk** | 🔴 Red | $P \ge 12.0\%$ | $P \ge 55.0\%$ | Significant clinical risk; immediate physician consultation, metabolic panel, and diagnostic evaluation for micro/macrovascular complications. |

---

## 📂 Repository Structure & File Inventory

```
Project Face/
├── assets/
│   └── fonts/
│       ├── Sarabun-Bold.ttf          # Thai Sarabun font (Bold) for PDF clinical export
│       └── Sarabun-Regular.ttf       # Thai Sarabun font (Regular) for PDF clinical export
├── dashboard/
│   └── app.py                        # Standalone CDSS Kiosk Mockup Dashboard (< 0.05s Zero-Lag Engine)
├── data/
│   ├── processed/                    # Cleaned & merged NHANES tabular cohorts
│   │   ├── nhanes_cleaned_merged.csv
│   │   ├── nhanes_cleaned_merged_final.csv
│   │   ├── nhanes_data_converted_AndroidGynoid.csv
│   │   ├── nhanes_data_converted_BMX_C.csv
│   │   ├── nhanes_data_converted_DEMO_C.csv
│   │   ├── nhanes_data_converted_WholeBody.csv
│   │   └── nhanes_stage2.parquet
│   ├── raw/                          # CDC NHANES raw SAS transport files (.xpt)
│   │   ├── BMX_C.xpt, BMX_J.XPT, BPQ_C.xpt, BPQ_J.XPT, BPX_J.XPT
│   │   ├── DEMO_C.xpt, DEMO_J.XPT, DIQ_C.xpt, DIQ_J.XPT
│   │   ├── DXXAG_C.xpt, DXXAG_J.XPT, DXX_J.XPT, GHB_J.XPT, GLU_J.XPT, PAQ_C.xpt
│   ├── test_images/                  # Curated face test photographs (testpic01.png - testpic13.png)
│   ├── Images/                       # Face image dataset for ViT training (.bmp)
│   └── split_fixed_v2.csv            # Fixed reproducible Train/Val/Test cohort splits
├── reports/
│   ├── Face2Health_IEEE_Conference_Paper.pdf  # Compiled 4-page IEEE publication research paper
│   └── [ENG]-67076012-Short Paper.pdf        # Prior academic short paper
├── scripts/
│   ├── results/                      # Stage 2 calibration plots, ROC curves, and confusion matrices
│   │   ├── stage2_benchmark_calibration.png
│   │   ├── stage2_benchmark_cm.png
│   │   ├── stage2_benchmark_comparison.csv
│   │   ├── stage2_benchmark_comparison.md
│   │   ├── stage2_benchmark_roc.png
│   │   └── xgboost_train_summary.csv
│   ├── app.py                        # Full production Streamlit UI (ViT-H/14, Face Guard, PDF Export)
│   ├── build_dataset.py              # Modern multi-cycle NHANES ETL builder (.parquet)
│   ├── convert_xpt_to_csv.py         # Utility converting SAS .xpt files to .csv
│   ├── eval_vit.py                   # Stage 1 evaluation script on test split
│   ├── evaluateAll.py                # Comprehensive multi-model benchmark evaluation suite
│   ├── face_guard.py                 # Two-Tier Face Guard (MediaPipe/OpenCV, area ratio, blur, illumination)
│   ├── facial_concepts.py            # Scale-invariant facial morphometric extractor (LFWR, CJWR, PAR, FWHR)
│   ├── generate_academic_paper_docx.py # Automated IEEE research paper PDF compiler via PyMuPDF
│   ├── loader.py                     # Image augmentations, ViT transforms, and PyTorch DataLoaders
│   ├── models.py                     # PyTorch ViT-H/14 model architecture with BMIHead & MC Dropout
│   ├── pdf_export.py                 # Bilingual (Thai/English) medical PDF generator
│   ├── predict_pipeline.py           # Production inference pipeline engine with MC Dropout
│   ├── prepare_dataset.py            # NHANES data cleaning and merging pipeline
│   ├── stage2_train.py               # Basic Stage 2 Platt-calibrated XGBoost trainer
│   ├── stage2_train_optimized.py     # Production Monotonic XGBoost trainer with F2 threshold tuning
│   ├── train_vit_bmi.py              # PyTorch training loop for Stage 1 ViT-H/14 BMI model
│   └── train_xgboost.py              # Training script for Stage 1.5 DEXA body fat regressors
├── weights/
│   ├── vit_bmi_model.pt              # Trained ViT-H/14 BMI model weights (2.5 GB)
│   ├── xgboost_bodyfat_with_waist.pkl   # Stage 1.5 XGBoost regressor (with waist)
│   ├── xgboost_bodyfat_no_waist.pkl     # Stage 1.5 XGBoost regressor (no waist)
│   ├── xgb_classifier_diabetes.pkl      # Stage 2 Monotonic Calibrated XGBoost (Diabetes, with waist)
│   ├── xgb_classifier_diabetes_no_waist.pkl # Stage 2 Monotonic Calibrated XGBoost (Diabetes, no waist)
│   ├── xgb_classifier_hypertension.pkl  # Stage 2 Monotonic Calibrated XGBoost (HTN, with waist)
│   ├── xgb_classifier_hypertension_no_waist.pkl # Stage 2 Monotonic Calibrated XGBoost (HTN, no waist)
│   └── thresholds.json               # Persisted clinical cutoff metadata & test metrics
├── archive/                          # Historical pre-production experiments and residual models
├── environment.yml                   # Complete Conda environment specification (`face2bmi`)
├── requirements.txt                  # Python pip dependencies
├── run_app.bat                       # One-click Windows launcher for Production App
├── run_dashboard.bat                 # One-click Windows launcher for CDSS Kiosk Dashboard
├── final_intro.mp4                   # Demonstration video
├── LICENSE                           # MIT License
└── README.md                         # Master repository documentation
```

---

## ⚙️ Installation & Environment Setup

The system is tested on **Windows 11 / 10** and **Linux (Ubuntu 22.04 LTS)** with **Python 3.10** and optional NVIDIA CUDA 11.8+ GPU acceleration.

### 1. Clone the Repository
```bash
git clone https://github.com/Soulbazz/Project-Face.git
cd "Project Face"
```

### 2. Create the Conda Environment (Recommended)
```bash
conda env create -f environment.yml
conda activate face2bmi
```

*Alternatively, install via pip:*
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Verify Model Weights
Ensure required checkpoints are present in `weights/`:
```bash
python -c "from scripts.predict_pipeline import check_weights; missing = check_weights(); print('Missing:' if missing else 'All weights verified ✅', missing)"
```

---

## 🚀 Execution Guide & Runnable Entry Points

### Entry Point 1: CDSS Kiosk Simulation Dashboard (`run_dashboard.bat`)
Tailored for primary care nurse stations and real-time clinical presentations. Runs an instantaneous mathematical surrogate engine ($< 0.05\text{s}$) with 1-click clinical scenarios:
- **Preset 1:** Normal / Low Risk (Age 24, Female, BMI 21.0, Low Risk).
- **Preset 2 (Clinical Highlight):** Rescued by $F_2$ Threshold (Age 56, Male, BMI 27.8, Waist 98 cm, caught as Screen Positive).
- **Preset 3:** Safety Guard Interception (Camera Yaw 22°, Epistemic $\sigma = 2.45$, automated screening safely blocked).

```bash
# Windows 1-Click:
run_dashboard.bat

# Or via Command Line:
streamlit run dashboard/app.py
```

### Entry Point 2: Production Health Screening App (`run_app.bat`)
The full production application with camera input, real-time Face Guard, ViT-H/14 point and uncertainty estimation, explainable facial morphometry gauges, and bilingual PDF report generation.

```bash
# Windows 1-Click:
run_app.bat

# Or via Command Line (from repository root or scripts/):
streamlit run scripts/app.py
```

### Entry Point 3: CLI Pipeline Inference & Sanity Checks
Executes the full multi-stage inference pipeline with sample portraits from `data/test_images/`:

```bash
# Runs sanity tests, full mode (with waist), and basic mode (no waist + MC Dropout n=30):
python scripts/predict_pipeline.py
```

### Entry Point 4: IEEE Academic Paper Compilation
Compiles the publication-ready 4-page academic research paper directly into `reports/Face2Health_IEEE_Conference_Paper.pdf` using PyMuPDF:

```bash
python scripts/generate_academic_paper_docx.py
```

### Entry Point 5: Data ETL & Model Retraining Pipeline
To reproduce all model weights from raw CDC NHANES data:

```bash
# 1. Clean and merge raw NHANES demographic, examination, and laboratory tables:
python scripts/prepare_dataset.py

# 2. Train Stage 1.5 DEXA Total Body Fat % Regressors (with & without waist):
python scripts/train_xgboost.py

# 3. Train Stage 2 Monotonic Platt-Calibrated XGBoost & Tune F2 Screening Cutoffs:
python scripts/stage2_train_optimized.py

# 4. Optional: Train Stage 1 Vision Transformer (requires data/Images/ dataset):
python scripts/train_vit_bmi.py --augmented

# 5. Evaluate ViT-H/14 on held-out test split:
python scripts/eval_vit.py
```

---

## 📊 Held-Out Benchmark Evaluation

Evaluated on the held-out CDC NHANES test cohort ($N = 531$, representing an isolated $15\%$ test split with zero threshold tuning leakage):

### Type 2 Diabetes Mellitus (T2DM) Screening

| Model Strategy | Decision Cutoff ($\tau^*$) | ROC-AUC | Brier Score | Sensitivity (Recall) | Specificity | Rescued Cases (FN Saved) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Default ML Baseline** | 0.500 ($50.0\%$) | 0.7468 | 0.0881 | **0.0%** | 100.0% | 0 (0-Recall Paradox) |
| **Youden's $J$ Index** | 0.0741 ($7.4\%$) | 0.7645 | 0.0881 | **81.8%** | 59.2% | +45 Patients |
| **Optimized $F_2$ Screening** | **0.0610 ($6.1\%$)** | **0.7645** | **0.0881** | **89.1%** | 49.0% | **+49 Patients Rescued** |

### Essential Hypertension Screening

| Model Strategy | Decision Cutoff ($\tau^*$) | ROC-AUC | Brier Score | Sensitivity (Recall) | Specificity | Rescued Cases (FN Saved) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Default ML Baseline** | 0.500 ($50.0\%$) | 0.7602 | 0.1870 | **46.9%** | 83.5% | Baseline |
| **Youden's $J$ Index** | 0.3312 ($33.1\%$) | 0.7548 | 0.1870 | **68.5%** | 71.9% | +34 Patients |
| **Optimized $F_2$ Screening** | **0.1608 ($16.1\%$)** | **0.7548** | **0.1870** | **87.8%** | 42.9% | **+70 Patients Rescued** |

*Artifacts generated and saved under `scripts/results/` (`stage2_benchmark_roc.png`, `stage2_benchmark_calibration.png`, `stage2_benchmark_cm.png`).*

---

## ⚠️ Medical & Regulatory Disclaimer

> **IMPORTANT MEDICAL NOTICE:**  
> Face2Health is an academic research demonstration and **opportunistic primary care screening tool**. It is **NOT** a diagnostic medical device (not cleared by FDA, CE-IVD, or Thai FDA) and does **not** provide definitive medical diagnoses.
>
> Automated screening cannot substitute for clinical venous blood testing (such as Fasting Plasma Glucose or Glycated Hemoglobin HbA1c) or calibrated medical sphygmomanometry. Individuals identified in the **Screen Positive** or **Urgent Risk** strata must seek confirmatory evaluation from licensed healthcare professionals.

---

## 👥 Acknowledgements & Citation

1. **Facial BMI Foundation:** Core Vision Transformer architecture for facial feature extraction was adapted from the open-source research by Liujie Zheng (*face-to-bmi-vit*), under MIT License.
2. **Clinical Cohort:** Benchmark data derived from the **Centers for Disease Control and Prevention (CDC) National Health and Nutrition Examination Survey (NHANES)** (1999–2018).
3. **Facial Anthropometry References:**
   - Lee & Kim (2014), *"A facial anthropometric predictor of visceral fat in Asian adults"*.
   - Coetzee et al. (2009), *"Facial adiposity: A reliable measure of health and attractiveness"*.
   - Wen & Guo (2013), *"Computational methods for facial age and health estimation"*.

### Citation
```bibtex
@inproceedings{face2health2026,
  title={Face2Health: A Multimodal Non-Invasive Health Risk Screening Pipeline Using Vision Transformer and Monotonic Calibrated XGBoost},
  author={Hemvut, Phalathip and Chaivivatporn, Eua-Unggul},
  booktitle={IEEE Conference Proceedings on Biomedical Health Informatics},
  year={2026}
}
```

---
**License:** Distributed under the [MIT License](LICENSE). Copyright (c) 2026 Face2Health Contributors.