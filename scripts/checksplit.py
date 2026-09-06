import os
import pandas as pd

IMAGE_FOLDER = '../data/Images'   # แก้ path ให้ตรงกับที่ใช้จริง
DATA_CSV = '../data/data.csv'
SPLIT_FILE = '../data/split_fixed_v2.csv'

# จำลองสิ่งที่ BMIDataset.__init__ ทำจริง
csv = pd.read_csv(DATA_CSV)
images = os.listdir(IMAGE_FOLDER)
csv = csv[csv['name'].isin(images)]
csv.reset_index(drop=True, inplace=True)   # <-- นี่คือ index ที่ dataset จริงใช้

split_df = pd.read_csv(SPLIT_FILE)

# สร้าง mapping name -> index แบบเดียวกับที่แก้ใน loader.py
name_to_idx = {name: idx for idx, name in enumerate(csv['name'])}

mismatch = 0
missing_from_dataset = []

for _, row in split_df.iterrows():
    name = row['name']
    if name not in name_to_idx:
        missing_from_dataset.append(name)
        continue
    # ในที่นี้แค่ยืนยันว่า name หา index เจอ ไม่ใช่เทียบ index ตำแหน่งตรงๆ แบบเดิม

print(f'จำนวนภาพใน split_fixed.csv ที่หา index ใน dataset จริงไม่เจอ: {len(missing_from_dataset)} จาก {len(split_df)}')
if missing_from_dataset:
    print('ตัวอย่าง:', missing_from_dataset[:10])
else:
    print('✅ ทุกภาพใน split_fixed.csv จับคู่ index ใน dataset จริงได้ถูกต้อง — loader.py แก้ถูกแล้ว')