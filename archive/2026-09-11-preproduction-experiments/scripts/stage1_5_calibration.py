"""Stage1_5_calibration.py — แก้ regression to the mean"""
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.amp import autocast
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from loader import get_dataloaders
from models import get_model

DEV  = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = "../weights/aug_epoch_7_backup.pt"
OUT  = Path("../outputs/stage1"); OUT.mkdir(parents=True, exist_ok=True)

model = get_model().float().to(DEV)
model.load_state_dict(torch.load(CKPT, map_location=DEV))
model.eval()

train_loader, test_loader, val_loader = get_dataloaders(
    16, augmented=False, vit_transformed=True, show_sample=False)

def infer(loader):
    yt, yp = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEV)
            with autocast(device_type=DEV, enabled=(DEV == "cuda")):
                p = model(X).squeeze().float().cpu().numpy()
            yp.extend(np.atleast_1d(p)); yt.extend(np.atleast_1d(y.numpy()))
    return np.asarray(yt, float), np.asarray(yp, float)

# 1) fit บน val
v_true, v_pred = infer(val_loader)
a, b = np.polyfit(v_pred, v_true, 1)          # true ≈ a*pred + b
print(f"calibration:  BMI_cal = {a:.4f} * pred + {b:.4f}   (n_val={len(v_true)})")

# 2) ทดสอบบน test
t_true, t_pred = infer(test_loader)
t_cal = a * t_pred + b

def report(name, yt, yp):
    r = yp - yt
    return {"model": name,
            "MAE":  round(float(mean_absolute_error(yt, yp)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "R2":   round(float(r2_score(yt, yp)), 4),
            "within_2": round(float((np.abs(r) <= 2).mean()), 4),
            "within_3": round(float((np.abs(r) <= 3).mean()), 4)}

res = [report("raw", t_true, t_pred), report("calibrated", t_true, t_cal)]
print(pd.DataFrame(res).to_string(index=False))

# 3) bias แยกช่วง ก่อน/หลัง
bins   = [0, 18.5, 25, 30, 35, 40, 100]
labels = ["Under", "Normal", "Overweight", "Obese I", "Obese II", "Severe"]
df = pd.DataFrame({"y_true": t_true, "raw": t_pred, "cal": t_cal})
df["bin"] = pd.cut(df.y_true, bins, labels=labels)
df["bias_raw"] = df.raw - df.y_true
df["bias_cal"] = df.cal - df.y_true
tbl = (df.groupby("bin", observed=True)
         .agg(n=("y_true","size"),
              bias_raw=("bias_raw","mean"), bias_cal=("bias_cal","mean"),
              mae_raw=("bias_raw", lambda s: s.abs().mean()),
              mae_cal=("bias_cal", lambda s: s.abs().mean())).round(2))
print("\n", tbl)

json.dump({"a": float(a), "b": float(b), "results": res},
          open(OUT / "calibration.json", "w"), indent=2)
df.to_csv(OUT / "stage1_calibrated.csv", index=False)
tbl.to_csv(OUT / "stage1_bin_before_after.csv")

# 4) กราฟเทียบ
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for i, (col, ttl) in enumerate([("bias_raw", "Before"), ("bias_cal", "After")]):
    ax[i].scatter(df.y_true, df[col], s=14, alpha=.4, color="#4C9F70")
    z = np.polyfit(df.y_true, df[col], 1)
    xs = np.linspace(df.y_true.min(), df.y_true.max(), 100)
    ax[i].plot(xs, np.polyval(z, xs), "r-", lw=2, label=f"slope={z[0]:.3f}")
    ax[i].axhline(0, ls="--", c="k"); ax[i].legend()
    ax[i].set_title(f"{ttl} Calibration"); ax[i].set_xlabel("True BMI")
ax[0].set_ylabel("Residual")
plt.tight_layout(); plt.savefig(OUT / "stage1_calibration.png", dpi=150)
print("\nsaved ->", OUT.resolve())