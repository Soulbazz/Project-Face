"""Stage1_data_profiling.py — ต้องวางไว้ในโฟลเดอร์ scripts/ และรันจากที่นั่น"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")               # กันหน้าต่างกราฟค้าง
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from loader import get_dataloaders   # ← ของจริงจาก loader.py
from models import get_model         # ← ของจริงจาก models.py

DEV  = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "../weights/aug_epoch_7_backup.pt"      # Old Model (MAE 3.28)
OUT  = Path("../outputs/stage1"); OUT.mkdir(parents=True, exist_ok=True)

# ---------- 1) โหลดโมเดล (ก๊อปวิธีเดียวกับ OLDNEWCHECK.py) ----------
model = get_model().float().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()
print(f"loaded {CKPT} on {DEV}")

# ---------- 2) test loader (พารามิเตอร์เดิมเป๊ะ = ได้ split เดิม) ----------
_, test_loader, _ = get_dataloaders(16, augmented=False,
                                    vit_transformed=True, show_sample=False)

# ---------- 3) inference ----------
y_true, y_pred = [], []
with torch.no_grad():
    for X, y in test_loader:
        X = X.to(DEV)
        with autocast(device_type=DEV, enabled=(DEV == "cuda")):
            preds = model(X).squeeze().float().cpu().numpy()
        y_pred.extend(np.atleast_1d(preds))      # กันกรณี batch เหลือ 1 รูป
        y_true.extend(np.atleast_1d(y.numpy()))

y_true = np.asarray(y_true, dtype=float)
y_pred = np.asarray(y_pred, dtype=float)

df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
df["resid"]     = df.y_pred - df.y_true          # + ทายเกิน / - ทายต่ำ
df["abs_error"] = df.resid.abs()
df.to_csv(OUT / "stage1_residuals.csv", index=False)

# ---------- 4) metrics รวม ----------
m = {
    "n":           int(len(df)),
    "MAE":         float(mean_absolute_error(y_true, y_pred)),
    "RMSE":        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    "R2":          float(r2_score(y_true, y_pred)),
    "bias":        float(df.resid.mean()),
    "resid_std":   float(df.resid.std()),
    "p90_abs_err": float(df.abs_error.quantile(0.90)),
    "within_2":    float((df.abs_error <= 2).mean()),
    "within_3":    float((df.abs_error <= 3).mean()),
}
json.dump(m, open(OUT / "stage1_metrics.json", "w"), indent=2)
print(json.dumps(m, indent=2))

# ---------- 5) แยกตามช่วง BMI ----------
bins   = [0, 18.5, 25, 30, 35, 40, 100]
labels = ["Under", "Normal", "Overweight", "Obese I", "Obese II", "Severe"]
df["bin"] = pd.cut(df.y_true, bins, labels=labels)
tbl = (df.groupby("bin", observed=True)
         .agg(n=("resid", "size"), mean_bias=("resid", "mean"),
              std=("resid", "std"), MAE=("abs_error", "mean"))
         .round(2))
tbl.to_csv(OUT / "stage1_binned_error.csv")
print("\n", tbl)

# ---------- 6) กราฟ 3 รูป ----------
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].hist(df.resid, bins=40, color="#4C9F70", edgecolor="k", alpha=.85)
ax[0].axvline(0, ls="--", c="k"); ax[0].axvline(m["bias"], c="r")
ax[0].set_title(f"Residual Distribution (bias={m['bias']:.2f})")
ax[0].set_xlabel("Predicted - True (BMI units)")

ax[1].scatter(df.y_true, df.resid, s=14, alpha=.45, color="#4C9F70")
z  = np.polyfit(df.y_true, df.resid, 1)
xs = np.linspace(df.y_true.min(), df.y_true.max(), 100)
ax[1].plot(xs, np.polyval(z, xs), "r-", lw=2, label=f"slope={z[0]:.3f}")
ax[1].axhline(0, ls="--", c="k"); ax[1].legend()
ax[1].set_title("Residual vs True BMI")
ax[1].set_xlabel("True BMI"); ax[1].set_ylabel("Residual")

mean_  = (df.y_true + df.y_pred) / 2
lo, hi = m["bias"] - 1.96*m["resid_std"], m["bias"] + 1.96*m["resid_std"]
ax[2].scatter(mean_, df.resid, s=14, alpha=.45, color="#4C9F70")
for v, c in [(m["bias"], "r"), (lo, "b"), (hi, "b")]:
    ax[2].axhline(v, ls="--", c=c)
    ax[2].text(mean_.max(), v, f" {v:.2f}", va="bottom")
ax[2].set_title("Bland-Altman Plot")
ax[2].set_xlabel("Mean of True & Predicted BMI")

plt.tight_layout()
plt.savefig(OUT / "stage1_error_analysis.png", dpi=150)
print("\nsaved ->", OUT.resolve())