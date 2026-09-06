import pandas as pd
import numpy as np

# 1. โหลดไฟล์ PAQ_C
paq = pd.read_sas('../data/raw/PAQ_C.xpt', format='xport')
print(paq['PAQ180'].unique()) 
# # 2. เลือกคอลัมน์ที่ต้องการ + ใส่ .copy() กัน SettingWithCopy/recursion bug
# cols_needed = ['SEQN', 'PAQ180', 'PAD200', 'PAD320', 'PAD590', 'PAD600']
# paq = paq[cols_needed].copy()          # <-- เพิ่ม .copy() ตรงนี้

# # 3. ทำความสะอาด missing codes -> ใช้ np.nan แทน pd.NA
# missing_codes = [7, 9, 77, 99]
# for col in ['PAQ180', 'PAD200', 'PAD320']:
#     paq[col] = paq[col].replace(missing_codes, np.nan)   # <-- เปลี่ยน pd.NA เป็น np.nan

# paq['PAD590'] = paq['PAD590'].replace([77, 99], np.nan)
# paq['PAD600'] = paq['PAD600'].replace([77, 99], np.nan)

# # 4. Merge เข้ากับ dataset เดิม
# df_main = pd.read_csv('../data/processed/nhanes_cleaned_merged_final.csv') 
# df_merged = df_main.merge(paq, on='SEQN', how='left')

# # 5. Impute missing values
# df_merged['PAQ180'] = df_merged['PAQ180'].fillna(df_merged['PAQ180'].median())
# df_merged['PAD590'] = df_merged['PAD590'].fillna(df_merged['PAD590'].median())
# df_merged['PAD600'] = df_merged['PAD600'].fillna(df_merged['PAD600'].median())

# # 6. แปลง Yes/No (1/2) เป็น 1/0
# df_merged['PAD200'] = df_merged['PAD200'].map({1: 1, 2: 0}).fillna(0)
# df_merged['PAD320'] = df_merged['PAD320'].map({1: 1, 2: 0}).fillna(0)

# df_merged.to_csv('../data/data_with_lifestyle.csv', index=False)
# print(f"Merge สำเร็จ: {df_merged.shape[0]} แถว, คอลัมน์ใหม่: {cols_needed[1:]}")
# print(df_merged[cols_needed[1:]].describe())