import os, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, 'data', 'split_fixed.csv'))

test = df[df['split'].astype(str).str.lower().str.strip() == 'test'].copy()

IMG_DIR = os.path.join(ROOT, 'data', 'Images')
test['image_path'] = test['name'].apply(
    lambda p: os.path.join(IMG_DIR, os.path.basename(str(p))))
test = test[test['image_path'].apply(os.path.exists)]

for grp, tag in [(0, 'A_clean'), (1, 'B_suspect')]:
    sub = test[test['is_training'] == grp][['image_path', 'bmi']]
    dst = os.path.join(ROOT, 'data', f'test_faces_{tag}.csv')
    sub.to_csv(dst, index=False)
    print(f"✅ {tag}: {len(sub)} แถว | BMI mean={sub['bmi'].mean():.2f} "
          f"sd={sub['bmi'].std():.2f} → {os.path.basename(dst)}")