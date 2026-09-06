# =============================================================
# evaluateAll.py — Full Evaluation Suite (Fixed & Self-Healing)
# -------------------------------------------------------------
# แก้ครบ:
#   ✔ Path อิง ROOT อัตโนมัติ (ไม่มี scripts/data ซ้ำอีก)
#   ✔ Feature order ดึงจากตัวโมเดลเอง -> ไม่มี mismatch
#   ✔ Auto-detect คอลัมน์โรค (DIABETES / HYPERTENSION)
#   ✔ Auto-detect ไฟล์ .pkl ใน weights/ ถ้าชื่อไม่ตรง
#   ✔ Auto-detect ViT checkpoint
#   ✔ ไม่พังทั้งไฟล์เมื่อ stage ใด stage หนึ่ง error
# รันครั้งเดียว -> ได้ทุกอย่างใน scripts/results/
# =============================================================

import os, sys, glob, warnings, traceback
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss, roc_curve
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# =============================================================
# PATHS — อิงตำแหน่งไฟล์นี้ ไม่ขึ้นกับว่า cd จากไหน
# =============================================================
HERE     = os.path.dirname(os.path.abspath(__file__))          # .../scripts
ROOT     = os.path.dirname(HERE)                                # .../Project Face
DATA_DIR = os.path.join(ROOT, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
WEIGHTS  = os.path.join(ROOT, "weights")
RESULTS  = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

CFG = {
    "run_stage1":  True,
    "run_stage15": True,
    "run_stage2":  True,
    "run_sens":    True,

    "vit_ckpt":     None,   # None = ค้นหาอัตโนมัติใน weights/
    "vit_test_csv": os.path.join(DATA_DIR, "test_faces.csv"),
    "img_size": 224,
    "batch_size": 32,
    "vit_arch": "vit_base_patch16_224",

    "bf_test_csv":  os.path.join(DATA_DIR, "test_tabular.csv"),
    "ncd_test_csv": os.path.join(DATA_DIR, "test_tabular.csv"),

    "col_age":    "AGE",
    "col_gender": "GENDER_NUM",
    "col_bmi":    "BMI",
    "col_waist":  "WAIST_CM",
    "col_bf":     "TOTAL_BODY_FAT_PCT",
    "male_value": 1,
}

# ลำดับ feature ตั้งต้น (ใช้เมื่อโมเดลไม่ได้เก็บชื่อไว้)
BF_FEATS_WAIST   = ["BMI", "AGE", "GENDER_NUM", "WAIST_CM"]
BF_FEATS_NOWAIST = ["BMI", "AGE", "GENDER_NUM"]
NCD_FEATS_WAIST   = ["BMI", "AGE", "GENDER_NUM", "WAIST_CM", "TOTAL_BODY_FAT_PCT"]
NCD_FEATS_NOWAIST = ["BMI", "AGE", "GENDER_NUM", "TOTAL_BODY_FAT_PCT"]

REPORT = []

def log(msg=""):
    s = str(msg)
    print(s)
    REPORT.append(s)

def save_fig(name):
    p = os.path.join(RESULTS, name)
    plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
    print(f"   [saved] {os.path.relpath(p, ROOT)}")


# =============================================================
# AUTO-DISCOVERY
# =============================================================
def find_pkl(*keywords, exclude=(), prefer=("xgboost", "xgb")):
    """หาไฟล์ .pkl ที่ตรง keyword — ให้ priority กับชื่อที่มี prefer"""
    hits = []
    for path in glob.glob(os.path.join(WEIGHTS, "**", "*.pkl"), recursive=True):
        low = os.path.basename(path).lower()
        if all(k in low for k in keywords) and not any(x in low for x in exclude):
            score = sum(2 for p in prefer if p in low)
            if "custom" in low or "baseline" in low or "randomforest" in low:
                score -= 1
            hits.append((score, os.path.getmtime(path), path))
    if not hits:
        return None
    hits.sort(key=lambda t: (t[0], t[1]), reverse=True)   # คะแนนสูงสุด แล้วใหม่สุด
    if len(hits) > 1:
        others = [os.path.basename(h[2]) for h in hits[1:]]
        print(f"   [info] เจอหลายตัว {keywords} → เลือก {os.path.basename(hits[0][2])} "
              f"(ตัวอื่น: {others})")
    return hits[0][2]

def find_vit_ckpt():
    if CFG["vit_ckpt"] and os.path.exists(CFG["vit_ckpt"]):
        return CFG["vit_ckpt"]
    cands = []
    for ext in ("*.pt", "*.pth"):
        cands += glob.glob(os.path.join(WEIGHTS, "**", ext), recursive=True)
    if not cands:
        return None
    for kw in ("best", "final", "aug"):
        for c in cands:
            if kw in os.path.basename(c).lower():
                return c
    return max(cands, key=os.path.getmtime)

def model_features(model, fallback):
    """ดึงลำดับ feature ที่โมเดลจำไว้ตอนเทรน"""
    for attr in ("feature_names_in_", "feature_name_"):
        v = getattr(model, attr, None)
        if v is not None and len(v):
            return list(v)
    try:
        v = model.get_booster().feature_names
        if v:
            return list(v)
    except Exception:
        pass
    return list(fallback)

def align(df, model, fallback):
    """จัดคอลัมน์ให้ตรงลำดับที่โมเดลต้องการเป๊ะ"""
    feats = model_features(model, fallback)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"ขาดคอลัมน์ {missing} (โมเดลต้องการ {feats})")
    return df[feats], feats

def find_binary_col(df, keywords):
    """หาคอลัมน์ที่เป็น binary target และชื่อมี keyword"""
    best = None
    for col in df.columns:
        low = col.lower()
        if not any(k in low for k in keywords):
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        u = set(pd.unique(s))
        if u.issubset({0, 1, 0.0, 1.0, True, False}) and 0 < s.mean() < 1:
            return col
        if best is None and len(u) <= 5:
            best = col
    return best


# =============================================================
# REGRESSION REPORT
# =============================================================
def regression_report(y_true, y_pred, title, prefix):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) < 3:
        log(f"> ⚠️ ข้อมูลน้อยเกินไปสำหรับ {title}")
        return None

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    w2   = float(np.mean(np.abs(y_pred - y_true) <= 2) * 100)
    w3   = float(np.mean(np.abs(y_pred - y_true) <= 3) * 100)

    log(f"\n### {title}\n")
    log("| Metric | Value |")
    log("|---|---|")
    log(f"| N (test) | {len(y_true)} |")
    log(f"| MAE | {mae:.3f} |")
    log(f"| RMSE | {rmse:.3f} |")
    log(f"| R² | {r2:.3f} |")
    log(f"| Bias (mean error) | {bias:+.3f} |")
    log(f"| คลาดเคลื่อน ≤ 2 | {w2:.1f}% |")
    log(f"| คลาดเคลื่อน ≤ 3 | {w3:.1f}% |")

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_true, y_pred, s=14, alpha=0.4, edgecolors="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect prediction")
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"{title}\nMAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}", fontsize=10)
    plt.legend(); plt.grid(alpha=0.25)
    save_fig(f"{prefix}_scatter.png")

    mv, dv = (y_true + y_pred) / 2, y_pred - y_true
    md, sd = np.mean(dv), np.std(dv)
    plt.figure(figsize=(6, 4.5))
    plt.scatter(mv, dv, s=14, alpha=0.4, edgecolors="none")
    plt.axhline(md, color="r", ls="--", label=f"Mean diff = {md:+.2f}")
    plt.axhline(md + 1.96*sd, color="gray", ls=":", label=f"+1.96 SD = {md+1.96*sd:+.2f}")
    plt.axhline(md - 1.96*sd, color="gray", ls=":", label=f"-1.96 SD = {md-1.96*sd:+.2f}")
    plt.xlabel("Mean of actual & predicted"); plt.ylabel("Predicted - Actual")
    plt.title(f"Bland-Altman: {title}", fontsize=10)
    plt.legend(fontsize=8); plt.grid(alpha=0.25)
    save_fig(f"{prefix}_bland_altman.png")

    return {"n": len(y_true), "mae": round(mae, 3), "rmse": round(rmse, 3),
            "r2": round(r2, 3), "bias": round(bias, 3)}


# =============================================================
# STAGE 1 — ViT Face → BMI
# =============================================================
def eval_stage1():
    log("\n---\n")
    log("## Stage 1 — Face-to-BMI (Vision Transformer)")

    csv = CFG["vit_test_csv"]
    if not os.path.exists(csv):
        log(f"\n> ⚠️ ข้าม: ไม่พบ `{os.path.relpath(csv, ROOT)}`")
        log("> รัน `make_test_faces.py` ก่อนเพื่อสร้างไฟล์นี้")
        return None

    ckpt = find_vit_ckpt()
    if ckpt is None:
        log(f"\n> ⚠️ ข้าม: ไม่พบไฟล์ .pt/.pth ใน `weights/`")
        return None
    log(f"\n> ใช้ checkpoint: `{os.path.basename(ckpt)}`")

    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms as T

    df = pd.read_csv(csv)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    if len(df) == 0:
        log("> ⚠️ ไม่พบไฟล์ภาพเลยสักไฟล์")
        return None
    log(f"> จำนวนภาพทดสอบ: {len(df)}")

    tf = T.Compose([
        T.Resize((CFG["img_size"], CFG["img_size"])),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DS(Dataset):
        def __len__(self): return len(df)
        def __getitem__(self, i):
            r = df.iloc[i]
            return tf(Image.open(r["image_path"]).convert("RGB")), float(r["bmi"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    obj = torch.load(ckpt, map_location=dev, weights_only=False)

    if isinstance(obj, dict):
        sd = obj.get("state_dict", obj.get("model_state_dict", obj))
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        try:
            import timm
            model = timm.create_model(CFG["vit_arch"], pretrained=False, num_classes=1)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if len(missing) > 20:
                log(f"> ⚠️ โหลด weight ไม่ตรงสถาปัตยกรรม (missing {len(missing)} keys)")
                log(f"> ลองเปลี่ยน `vit_arch` ใน CFG ให้ตรงกับตอนเทรน")
                return None
        except ImportError:
            log("> ⚠️ ต้องติดตั้ง timm ก่อน: `pip install timm`")
            return None
    else:
        model = obj

    model.to(dev).eval()

    preds, gts = [], []
    with torch.no_grad():
        for x, y in DataLoader(DS(), batch_size=CFG["batch_size"], num_workers=0):
            out = model(x.to(dev))
            out = out.logits if hasattr(out, "logits") else out
            preds.extend(np.asarray(out.squeeze(-1).cpu()).ravel())
            gts.extend(np.asarray(y).ravel())

    gts, preds = np.array(gts, float), np.array(preds, float)
    res = regression_report(gts, preds, "Stage 1: Face → BMI (Test Set)", "stage1_bmi")
    if res is None:
        return None

    base_mae = mean_absolute_error(gts, np.full_like(gts, gts.mean()))
    log(f"\n**Baseline (ทายค่าเฉลี่ยเสมอ):** MAE = {base_mae:.3f}")
    log(f"**โมเดลของเรา:** MAE = {res['mae']:.3f} → "
        f"ดีขึ้น {(1 - res['mae']/base_mae)*100:.1f}%")

    labels = ["ผอม", "ปกติ", "ท้วม", "อ้วน I", "อ้วน II+"]
    grp = pd.cut(gts, bins=[0, 18.5, 23, 25, 30, 100], labels=labels)
    rows = []
    for g in labels:
        m = (grp == g).to_numpy()
        if m.sum() >= 3:
            rows.append({"ช่วง BMI": g, "N": int(m.sum()),
                         "MAE": round(mean_absolute_error(gts[m], preds[m]), 3),
                         "Bias": round(float(np.mean(preds[m] - gts[m])), 3)})
    if rows:
        sub = pd.DataFrame(rows)
        log("\n**ผลแยกตามช่วง BMI:**\n")
        log(sub.to_markdown(index=False))
        sub.to_csv(os.path.join(RESULTS, "stage1_by_bmi_group.csv"), index=False)

    pd.DataFrame({"actual": gts, "predicted": preds}).to_csv(
        os.path.join(RESULTS, "stage1_predictions.csv"), index=False)
    return res


# =============================================================
# STAGE 1.5 — Body Fat %
# =============================================================
def deurenberg(bmi, age, gender_num):
    """Deurenberg et al. (1991): BF% = 1.20*BMI + 0.23*Age - 10.8*Sex - 5.4"""
    sex = (np.asarray(gender_num) == CFG["male_value"]).astype(float)
    return 1.20*np.asarray(bmi, float) + 0.23*np.asarray(age, float) - 10.8*sex - 5.4

def eval_stage15():
    log("\n---\n")
    log("## Stage 1.5 — Body Fat % (XGBoost vs Baseline)")

    csv = CFG["bf_test_csv"]
    if not os.path.exists(csv):
        log(f"\n> ⚠️ ข้าม: ไม่พบ `{os.path.relpath(csv, ROOT)}` — รัน `train_xgboost.py` ก่อน")
        return
    df = pd.read_csv(csv).dropna(subset=[CFG["col_bf"]])
    log(f"\n> ชุดทดสอบ: {len(df)} แถว")

    summary = []

    yb = deurenberg(df[CFG["col_bmi"]], df[CFG["col_age"]], df[CFG["col_gender"]])
    r = regression_report(df[CFG["col_bf"]].values, yb,
                          "Baseline: Deurenberg Formula (1991)", "stage15_deurenberg")
    if r: summary.append({"Model": "Deurenberg (1991)", **r})

    for label, key, fb in [
        ("XGBoost (+waist)",  ("bodyfat", "waist"), BF_FEATS_WAIST),
        ("XGBoost (no waist)", ("bodyfat", "no_waist"), BF_FEATS_NOWAIST),
    ]:
        path = (find_pkl(*key) if "no_waist" in key
                else find_pkl(*key, exclude=("no_waist", "nowaist")))
        if not path:
            log(f"\n> ⚠️ ไม่พบไฟล์โมเดล `{label}` ใน weights/")
            continue
        try:
            m = joblib.load(path)
            d = df.dropna(subset=[c for c in fb if c in df.columns])
            X, feats = align(d, m, fb)
            log(f"\n> `{os.path.basename(path)}` → features: `{feats}`")
            r = regression_report(d[CFG["col_bf"]].values, m.predict(X),
                                  f"Body Fat — {label}",
                                  f"stage15_{'waist' if 'no' not in label else 'nowaist'}")
            if r: summary.append({"Model": label, **r})
        except Exception as e:
            log(f"\n> ❌ `{label}` error: {e}")

    if len(summary) >= 2:
        tab = pd.DataFrame(summary)
        log("\n### 📊 ตารางเปรียบเทียบ Body Fat\n")
        log(tab.to_markdown(index=False))
        tab.to_csv(os.path.join(RESULTS, "stage15_comparison.csv"), index=False)
        best = tab.loc[tab["mae"].idxmin(), "Model"]
        log(f"\n> **ข้อสรุป:** โมเดลที่แม่นที่สุดคือ **{best}**")
        d_row = tab[tab["Model"].str.contains("Deurenberg")]
        if not d_row.empty and "Deurenberg" not in best:
            imp = (d_row["mae"].values[0] - tab["mae"].min()) / d_row["mae"].values[0] * 100
            log(f"> Machine Learning ลด error ได้ **{imp:.1f}%** เทียบกับสูตรสำเร็จปี 1991")


# =============================================================
# STAGE 2 — Classifiers
# =============================================================
def eval_classifier(model, X, y, name, prefix):
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc   = roc_auc_score(y, proba)
    prec  = precision_score(y, pred, zero_division=0)
    rec   = recall_score(y, pred, zero_division=0)
    f1    = f1_score(y, pred, zero_division=0)
    brier = brier_score_loss(y, proba)
    cm    = confusion_matrix(y, pred, labels=[0, 1])

    log(f"\n### {name}\n")
    log("| Metric | Value |")
    log("|---|---|")
    log(f"| N | {len(y)} |")
    log(f"| Positive rate | {y.mean()*100:.1f}% |")
    log(f"| AUC-ROC | {auc:.3f} |")
    log(f"| Precision | {prec:.3f} |")
    log(f"| Recall (Sensitivity) | {rec:.3f} |")
    log(f"| F1-score | {f1:.3f} |")
    log(f"| Brier score (ต่ำ = ดี) | {brier:.4f} |")
    log(f"\n**Confusion Matrix:** TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    log("```")
    log(classification_report(y, pred, digits=3, zero_division=0))
    log("```")

    plt.figure(figsize=(4.2, 3.8))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.xticks([0, 1], ["Pred: No", "Pred: Yes"], fontsize=8)
    plt.yticks([0, 1], ["True: No", "True: Yes"], fontsize=8)
    plt.title(f"Confusion Matrix\n{name}", fontsize=9)
    save_fig(f"{prefix}_confusion.png")

    fpr, tpr, _ = roc_curve(y, proba)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.500)")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {name}", fontsize=10); plt.legend(); plt.grid(alpha=0.25)
    save_fig(f"{prefix}_roc.png")

    try:
        nb = min(10, max(3, len(y)//40))
        pt, pp = calibration_curve(y, proba, n_bins=nb, strategy="quantile")
        plt.figure(figsize=(5, 4.8))
        plt.plot(pp, pt, "o-", lw=2, label="โมเดลของเรา")
        plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfectly calibrated")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title(f"Calibration — {name}\nBrier = {brier:.4f}", fontsize=10)
        plt.legend(); plt.grid(alpha=0.25)
        save_fig(f"{prefix}_calibration.png")
        gap = float(np.mean(np.abs(pt - pp)))
        verdict = ("✅ เชื่อถือได้ — ตัวเลข % ที่โชว์ผู้ใช้ใกล้ความจริง" if gap < 0.10
                   else "⚠️ ยังเพี้ยน — ควรทำ Platt scaling หรือ Isotonic regression")
        log(f"\n**Calibration gap เฉลี่ย = {gap:.3f}** → {verdict}")
    except Exception as e:
        log(f"\n> ข้าม calibration curve: {e}")

    return {"model": name, "n": len(y), "auc": round(auc, 3),
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(f1, 3), "brier": round(brier, 4)}


def eval_stage2():
    log("\n---\n")
    log("## Stage 2 — NCD Risk Classifiers")

    csv = CFG["ncd_test_csv"]
    if not os.path.exists(csv):
        log(f"\n> ⚠️ ข้าม: ไม่พบ `{os.path.relpath(csv, ROOT)}`")
        return None
    df = pd.read_csv(csv)
    log(f"\n> คอลัมน์ในไฟล์ test: `{list(df.columns)}`")

    c_diab = find_binary_col(df, ["diab", "diq", "_dm", "dm_"])
    c_hyp  = find_binary_col(df, ["hyper", "bpq", "htn", "blood_press"])
    log(f"> ตรวจพบ: diabetes = `{c_diab}` | hypertension = `{c_hyp}`")

    if c_diab is None and c_hyp is None:
        log("\n> ⚠️ **ไม่พบคอลัมน์โรคในไฟล์นี้**")
        log("> แปลว่า `Stage2.py` เทรนจากไฟล์ CSV คนละตัวกับ `train_xgboost.py`")
        log("> วิธีแก้: เปิด `Stage2.py` ดูว่าอ่านไฟล์ไหน แล้วให้สคริปต์นั้น")
        log("> เซฟ test set ของตัวเองออกมา แล้วชี้ `ncd_test_csv` มาที่ไฟล์นั้น")
        return None

    rows = []
    for disease, target, kws in [("diabetes", c_diab, ("diabet",)),
                                 ("hypertension", c_hyp, ("hyperten",))]:
        if target is None:
            log(f"\n> ⚠️ ไม่พบ target ของ {disease} — ข้าม")
            continue
        for variant, fb in [("waist", NCD_FEATS_WAIST), ("nowaist", NCD_FEATS_NOWAIST)]:
            path = (find_pkl(*kws, "no_waist") if variant == "nowaist"
                    else find_pkl(*kws, exclude=("no_waist", "nowaist")))
            if not path:
                log(f"\n> ⚠️ ไม่พบโมเดล {disease}/{variant} ใน weights/")
                continue
            try:
                m = joblib.load(path)
                feats = model_features(m, fb)
                d = df.dropna(subset=[c for c in feats if c in df.columns] + [target])
                if len(d) < 20:
                    log(f"\n> ⚠️ {disease}/{variant}: เหลือ {len(d)} แถว น้อยเกินไป")
                    continue
                X, feats = align(d, m, fb)
                nice = f"{disease.capitalize()} ({'มีรอบเอว' if variant=='waist' else 'ไม่มีรอบเอว'})"
                log(f"\n> `{os.path.basename(path)}` → features: `{feats}`")
                rows.append(eval_classifier(m, X, d[target].astype(int).values,
                                            nice, f"stage2_{disease}_{variant}"))
            except Exception as e:
                log(f"\n> ❌ {disease}/{variant} error: {e}")

    if rows:
        tab = pd.DataFrame(rows)
        log("\n### 📊 สรุป Stage 2 ทุกโมเดล\n")
        log(tab.to_markdown(index=False))
        tab.to_csv(os.path.join(RESULTS, "stage2_summary.csv"), index=False)
        log("\n> **ประเด็นเขียนในรายงาน:** ส่วนต่าง AUC ระหว่าง `มีรอบเอว` กับ "
            "`ไม่มีรอบเอว` คือ *มูลค่าเชิงตัวเลขของการถามรอบเอว* — ตอบคำถามว่า "
            "\"ทำไมต้องมี 2 โมเดล\" ด้วยหลักฐาน ไม่ใช่ความรู้สึก")
    return df


# =============================================================
# SENSITIVITY ANALYSIS
# =============================================================
def eval_sensitivity(df):
    log("\n---\n")
    log("## Sensitivity Analysis — Error Propagation")
    log("\n> **จุดประสงค์:** Stage 1 มี MAE ระดับหนึ่ง — คำถามคือ error นั้นถูก")
    log("> *ขยาย (amplify)* หรือถูก *กลืน (absorb)* เมื่อไหลไปถึงความเสี่ยงโรคที่ Stage 2")

    if df is None:
        log("\n> ⚠️ ข้าม: ไม่มีข้อมูล Stage 2")
        return

    path = find_pkl("diabet", exclude=("no_waist", "nowaist")) or find_pkl("diabet")
    if not path:
        log("\n> ⚠️ ข้าม: ไม่พบโมเดล diabetes")
        return

    try:
        m = joblib.load(path)
        feats = model_features(m, NCD_FEATS_WAIST)
        d = df.dropna(subset=[c for c in feats if c in df.columns]).copy()
        if len(d) < 20:
            log("\n> ⚠️ ข้อมูลน้อยเกินไป"); return

        base = m.predict_proba(d[feats])[:, 1] * 100
        deltas, rows = [-4, -2, -1, 0, 1, 2, 4], []
        for dv in deltas:
            dd = d.copy()
            dd[CFG["col_bmi"]] = dd[CFG["col_bmi"]] + dv
            p = m.predict_proba(dd[feats])[:, 1] * 100
            diff = p - base
            rows.append({
                "BMI error": f"{dv:+.0f}",
                "ความเสี่ยงเฉลี่ย (%)": round(float(p.mean()), 2),
                "เปลี่ยนแปลงเฉลี่ย (pp)": round(float(diff.mean()), 2),
                "เปลี่ยนแปลงสูงสุด (pp)": round(float(np.abs(diff).max()), 2),
                "อัตราขยาย error": round(float(np.abs(diff).mean()/abs(dv)), 2) if dv else 0.0,
            })

        tab = pd.DataFrame(rows)
        log("")
        log(tab.to_markdown(index=False))
        tab.to_csv(os.path.join(RESULTS, "sensitivity_analysis.csv"), index=False)

        plt.figure(figsize=(6.5, 4.5))
        plt.plot(deltas, tab["ความเสี่ยงเฉลี่ย (%)"], "o-", lw=2, color="crimson")
        plt.axvline(0, color="gray", ls="--", lw=1)
        plt.xlabel("BMI input error (units)")
        plt.ylabel("Mean predicted diabetes risk (%)")
        plt.title("Error Propagation: BMI error → Diabetes risk")
        plt.grid(alpha=0.25)
        save_fig("sensitivity_curve.png")

        a = float(tab[tab["BMI error"] == "+2"]["อัตราขยาย error"].values[0])
        log(f"\n> **อ่านผล:** BMI คลาด 1 หน่วย → ความเสี่ยงเปลี่ยน ~{a:.2f} percentage point")
        log(f"> {'✅ ระบบทนต่อ error ได้ดี (absorb)' if a < 2 else '⚠️ error ถูกขยาย — ควรระบุเป็นข้อจำกัดในรายงาน'}")
    except Exception as e:
        log(f"\n> ❌ Sensitivity error: {e}")


# =============================================================
# MAIN
# =============================================================
def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        log(f"\n> ❌ {fn.__name__} error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    log("# Evaluation Report — Face-to-BMI Health Risk System")
    log(f"\n_Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}_")
    log("\n> ผลทั้งหมดคำนวณบน **held-out test set** ที่โมเดลไม่เคยเห็นระหว่างการฝึก")
    log(f"\n> Root: `{ROOT}`")

    print("\n" + "="*60)
    print("ไฟล์ .pkl ที่พบใน weights/:")
    pkls = glob.glob(os.path.join(WEIGHTS, "**", "*.pkl"), recursive=True)
    for p in pkls: print("  -", os.path.basename(p))
    if not pkls: print("  (ไม่พบเลย)")
    print("="*60)

    if CFG["run_stage1"]:  safe(eval_stage1)
    if CFG["run_stage15"]: safe(eval_stage15)
    df2 = safe(eval_stage2) if CFG["run_stage2"] else None
    if CFG["run_sens"]:    safe(eval_sensitivity, df2)

    log("\n---\n")
    log("## ข้อจำกัดของงาน (Limitations)")
    log("- ข้อมูลฝึกมาจาก NHANES (ประชากรสหรัฐฯ) อาจเกิด **domain shift** เมื่อใช้กับคนไทย")
    log("- ทำนายจากภาพใบหน้าเพียงอย่างเดียว ไม่ครอบคลุมพันธุกรรมและพฤติกรรม")
    log("- ผลลัพธ์เป็นเครื่องมือ **คัดกรองเบื้องต้น** ไม่ใช่การวินิจฉัยทางการแพทย์")

    out = os.path.join(RESULTS, "EVALUATION_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT))

    print("\n" + "="*60)
    print(f"✅ เสร็จแล้ว — เปิดดูที่: {os.path.relpath(out, ROOT)}")
    print(f"   รูปทั้งหมดอยู่ใน: {os.path.relpath(RESULTS, ROOT)}/")
    print("="*60)