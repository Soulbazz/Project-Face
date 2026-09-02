import numpy as np, pandas as pd, os
from pathlib import Path

SEED = 42
SRC  = Path("../data/data.csv")
IMG  = Path("../data/Images")
OUT  = Path("../data/split_fixed.csv")

# 1. โหลดแล้วกรองเหมือน BMIDataset เป๊ะๆ
df = pd.read_csv(SRC)
images = os.listdir(IMG)
df = df[df['name'].isin(images)]
df.reset_index(drop=True, inplace=True)

# 2. ค่อยแบ่ง split บน df ที่กรองแล้ว
n = len(df)
idx = np.arange(n)
np.random.default_rng(SEED).shuffle(idx)

n_val, n_test = int(0.10 * n), int(0.20 * n)
split = np.empty(n, dtype=object)
split[idx[:n - n_val - n_test]] = "train"
split[idx[n - n_val - n_test : n - n_test]] = "val"
split[idx[n - n_test:]] = "test"

df["split"] = split
df.to_csv(OUT, index=False)
print(f"สร้างไฟล์สำเร็จ! จำนวนแถวรวม = {len(df)}")