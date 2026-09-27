"""
stage2_train_optimized.py
Production-ready Stage 2 Training with Automated Threshold Tuning & Regularization
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE_DIR)
WEIGHTS_DIR = os.path.join(ROOT, 'weights')
DATA_DIR = os.path.join(ROOT, 'data', 'processed')
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MAIN_DATA_PATH = os.path.join(DATA_DIR, 'nhanes_cleaned_merged_final.csv')
DIAB_RAW_PATH = os.path.join(RAW_DIR, 'DIQ_C.XPT')
HYP_RAW_PATH = os.path.join(RAW_DIR, 'BPQ_C.XPT')

print(f"[*] Loading dataset: {MAIN_DATA_PATH}")
df_main = pd.read_csv(MAIN_DATA_PATH)
df_main = df_main[df_main['AGE'] >= 20].copy()

if 'GENDER_NUM' not in df_main.columns and 'GENDER' in df_main.columns:
    df_main['GENDER_NUM'] = df_main['GENDER'].map({'Male': 1, 'Female': 0})

# Merge Diabetes & Hypertension Labels
df_diab_raw = pd.read_sas(DIAB_RAW_PATH)[['SEQN', 'DIQ010']]
df_diab_raw = df_diab_raw[df_diab_raw['DIQ010'].isin([1, 2])].copy()
df_diab_raw['TARGET_DIABETES'] = (df_diab_raw['DIQ010'] == 1).astype(int)
df_diab = pd.merge(df_main, df_diab_raw[['SEQN', 'TARGET_DIABETES']], on='SEQN', how='inner')

df_hyp_raw = pd.read_sas(HYP_RAW_PATH)[['SEQN', 'BPQ020']]
df_hyp_raw = df_hyp_raw[df_hyp_raw['BPQ020'].isin([1, 2])].copy()
df_hyp_raw['TARGET_HYPERTENSION'] = (df_hyp_raw['BPQ020'] == 1).astype(int)
df_hyp = pd.merge(df_main, df_hyp_raw[['SEQN', 'TARGET_HYPERTENSION']], on='SEQN', how='inner')


def optimize_threshold(y_true, y_prob, method="youden", beta=2.0):
    """Calculates optimal threshold on validation set without test leakage."""
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])
    elif method == "f_beta":
        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f_scores = (1 + beta**2) * (prec[:-1] * rec[:-1]) / ((beta**2 * prec[:-1]) + rec[:-1] + 1e-10)
        best_idx = np.argmax(f_scores)
        return float(thresholds[best_idx]), float(f_scores[best_idx])
    raise ValueError(f"Unknown method: {method}")


def evaluate_at_threshold(y_true, y_prob, threshold, label="Model"):
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (prec * sens) / (prec + sens + 1e-10)
    
    print(f"\n--- {label} (Threshold: {threshold:.4f}) ---")
    print(f"Sensitivity (Recall): {sens*100:.2f}% | Specificity: {spec*100:.2f}% | Precision: {prec*100:.2f}% | F1: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return {"threshold": round(threshold, 4), "sensitivity": round(sens, 4), 
            "specificity": round(spec, 4), "precision": round(prec, 4), "f1": round(f1, 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def train_calibrate_and_tune(data, target_col, feature_cols, model_filename, global_thresholds_dict):
    print("\n" + "="*60)
    print(f"Training: {model_filename} | Target: {target_col}")
    print(f"Features: {feature_cols}")
    
    clean_data = data.dropna(subset=feature_cols + [target_col]).copy()
    X = clean_data[feature_cols]
    y = clean_data[target_col]
    
    # 1. 3-Way Split: Train 70% | Validation 15% | Test 15%
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    val_ratio_of_train_val = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_of_train_val, random_state=42, stratify=y_train_val
    )
    
    print(f"Cohort splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 2. Monotonicity: Age, BMI, Waist, Body Fat must monotonically increase risk
    monotone_dict = {}
    for col in feature_cols:
        if col in ['AGE', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT']:
            monotone_dict[col] = 1
        else:
            monotone_dict[col] = 0
    monotone_tuple = tuple(monotone_dict[c] for c in feature_cols)
    
    # Regularized Base Estimator
    base_xgb = XGBClassifier(
        n_estimators=120,
        max_depth=3,                  # Shallow depth prevents volatile step-functions
        learning_rate=0.03,
        subsample=0.8,                # Bagging subsample
        colsample_bytree=0.75,        # Mitigates BMI & Waist collinearity
        reg_alpha=0.5,                # L1 regularization
        reg_lambda=2.0,               # L2 regularization
        min_child_weight=5,
        monotone_constraints=monotone_tuple,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 3. Platt Scaling Calibration (5-fold internal CV)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method='sigmoid',
        cv=5
    )
    calibrated_model.fit(X_train, y_train)
    
    # 4. Search Optimal Threshold on Validation Set (Zero Test Leakage)
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    th_youden, j_stat = optimize_threshold(y_val, val_probs, method="youden")
    th_f2, f2_stat = optimize_threshold(y_val, val_probs, method="f_beta", beta=2.0)
    
    print(f"Validation Tuning: Optimal Youden Threshold = {th_youden:.4f} (J = {j_stat:.4f})")
    print(f"Validation Tuning: Optimal F2-Score Threshold = {th_f2:.4f} (F2 = {f2_stat:.4f})")
    
    # 5. Evaluate on Unseen Held-out Test Set
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probs)
    brier = brier_score_loss(y_test, test_probs)
    print(f"\n[Test Set Performance] AUC-ROC: {auc:.4f} | Brier Score: {brier:.4f}")
    
    metrics_default = evaluate_at_threshold(y_test, test_probs, 0.50, label="Default Threshold 0.5")
    metrics_youden = evaluate_at_threshold(y_test, test_probs, th_youden, label="Optimized (Youden's J)")
    metrics_f2 = evaluate_at_threshold(y_test, test_probs, th_f2, label="Optimized (F2 Screening)")
    
    # 6. Save Artifacts
    save_path = os.path.join(WEIGHTS_DIR, model_filename)
    joblib.dump(calibrated_model, save_path)
    print(f"Saved model: {save_path}")
    
    # Record threshold metadata
    global_thresholds_dict[model_filename] = {
        "features": feature_cols,
        "default_threshold": 0.5,
        "youden_threshold": th_youden,
        "f2_threshold": th_f2,
        "test_metrics": {
            "auc": round(auc, 4),
            "brier": round(brier, 4),
            "at_0.5": metrics_default,
            "at_youden": metrics_youden,
            "at_f2": metrics_f2
        }
    }


if __name__ == "__main__":
    thresholds_config = {}
    feats_waist = ['AGE', 'GENDER_NUM', 'BMI', 'WAIST_CM', 'TOTAL_BODY_FAT_PCT']
    feats_nowaist = ['AGE', 'GENDER_NUM', 'BMI', 'TOTAL_BODY_FAT_PCT']
    
    train_calibrate_and_tune(df_diab, 'TARGET_DIABETES', feats_waist, 'xgb_classifier_diabetes.pkl', thresholds_config)
    train_calibrate_and_tune(df_diab, 'TARGET_DIABETES', feats_nowaist, 'xgb_classifier_diabetes_no_waist.pkl', thresholds_config)
    train_calibrate_and_tune(df_hyp, 'TARGET_HYPERTENSION', feats_waist, 'xgb_classifier_hypertension.pkl', thresholds_config)
    train_calibrate_and_tune(df_hyp, 'TARGET_HYPERTENSION', feats_nowaist, 'xgb_classifier_hypertension_no_waist.pkl', thresholds_config)
    
    # Persist thresholds config for pipeline & evaluation scripts
    th_path = os.path.join(WEIGHTS_DIR, 'thresholds.json')
    with open(th_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds_config, f, indent=4)
    print(f"\n[OK] Successfully wrote threshold configuration: {th_path}")