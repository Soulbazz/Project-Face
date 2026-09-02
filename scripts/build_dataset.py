"""build_dataset.py — download + merge + label NHANES for Stage 2"""
import functools, urllib.request
from pathlib import Path
import pandas as pd, numpy as np

RAW = Path("data/raw"); OUT = Path("data/processed")
RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{yr}/DataFiles/{f}.XPT"
CYCLES = {"2017": "_J"}          # เพิ่ม "2015":"_I", "2013":"_H", "2011":"_G" ได้
STEMS  = ["DEMO","BMX","DXX","DXXAG","DIQ","BPQ","GHB","GLU","BPX"]

KEEP = ["SEQN","RIDAGEYR","RIAGENDR","RIDRETH3","BMXBMI","BMXWAIST","BMXHT",
        "BMXWT","DXDTOPF","DXXAPFAT","DXXGPFAT","DIQ010","BPQ020","BPQ080",
        "LBXGH","LBXGLU"] + [f"BPXSY{i}" for i in range(1,5)] \
      + [f"BPXDI{i}" for i in range(1,5)]

def fetch(yr, stem, suf):
    name = f"{stem}{suf}"; p = RAW / f"{name}.XPT"
    if not p.exists():
        url = BASE.format(yr=yr, f=name)
        print("↓", url)
        try: urllib.request.urlretrieve(url, p)
        except Exception as e: print("  skip:", e); return None
    return pd.read_sas(p, format="xport")

frames = []
for yr, suf in CYCLES.items():
    dfs = [d for d in (fetch(yr, s, suf) for s in STEMS) if d is not None]
    m = functools.reduce(lambda l, r: l.merge(r, on="SEQN", how="left"), dfs)
    m["CYCLE"] = yr
    frames.append(m)

df = pd.concat(frames, ignore_index=True)
df = df[[c for c in KEEP if c in df.columns] + ["CYCLE"]]

# ---------- Labels ----------
sbp = df[[c for c in df if c.startswith("BPXSY")]].replace(0, np.nan).mean(axis=1)
dbp = df[[c for c in df if c.startswith("BPXDI")]].replace(0, np.nan).mean(axis=1)
df["SBP"], df["DBP"] = sbp, dbp

df["dm"] = (((df.DIQ010 == 1) | (df.LBXGH >= 6.5) |
             (df.LBXGLU >= 126)).astype(int))
df["htn"] = (((df.BPQ020 == 1) | (df.BPQ080 == 1) |
              (sbp >= 130) | (dbp >= 80)).astype(int))

# ---------- Filter ----------
core = ["BMXBMI","BMXWAIST","DXDTOPF","RIDAGEYR","RIAGENDR"]
df = df[(df.RIDAGEYR >= 18) & (df.RIDAGEYR <= 59)].dropna(subset=core)

print(df.shape)
print("missing %:\n", (df.isna().mean() * 100).round(1))
print("prevalence:\n", df[["dm","htn"]].mean().round(3))
df.to_parquet(OUT / "nhanes_stage2.parquet", index=False)