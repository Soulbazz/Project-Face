import numpy as np, pandas as pd, os
from pathlib import Path

SEED = 42
SRC  = Path("../data/data.csv")
IMG  = Path("../data/Images")
OUT  = Path("../data/split_fixed_v2.csv")   # ตั้งชื่อใหม่ อย่าทับของเก่า จะได้เทียบกันได้

# 1) โหลด + กรองเหมือน BMIDataset เป๊ะๆ (เหมือนเดิม)
df = pd.read_csv(SRC)
images = os.listdir(IMG)
df = df[df['name'].isin(images)].reset_index(drop=True)

# 2) สร้าง pseudo-identity จาก (bmi, gender) ที่ตรงกันเป๊ะ
df['pseudo_id'] = df.groupby(['bmi', 'gender']).ngroup()

# 3) สุ่มระดับ "กลุ่ม" ไม่ใช่ระดับ "รูป"
rng = np.random.default_rng(SEED)
groups = df['pseudo_id'].unique()
rng.shuffle(groups)

# แบ่งกลุ่มแบบสะสมจำนวนรูปให้ได้สัดส่วนใกล้ 70/10/20
group_sizes = df['pseudo_id'].value_counts()
n_total = len(df)
targets = {'train': 0.70 * n_total, 'val': 0.10 * n_total}

assign, counts = {}, {'train': 0, 'val': 0, 'test': 0}
for g in groups:
    size = group_sizes[g]
    if counts['train'] + size <= targets['train']:
        assign[g] = 'train'
    elif counts['val'] + size <= targets['val']:
        assign[g] = 'val'
    else:
        assign[g] = 'test'
    counts[assign[g]] += size

df['split'] = df['pseudo_id'].map(assign)

# 4) ตรวจสอบตัวเองก่อนเซฟ — ห้ามมีกลุ่มไหนข้าม split
leak = df.groupby('pseudo_id')['split'].nunique()
assert (leak == 1).all(), "ยังมี identity รั่วข้าม split!"

df.drop(columns='pseudo_id').to_csv(OUT, index=False)
print(f"สร้างสำเร็จ! สัดส่วนจริง:\n{df['split'].value_counts(normalize=True).round(3)}")
print(f"จำนวนรูป: {counts}")
