import pandas as pd

# 1. โหลดข้อมูล (อย่าลืมแก้เลข _4, _5 ต่อท้ายชื่อไฟล์ให้ตรงกับที่คุณมีในโฟลเดอร์นะครับ)
demo = pd.read_csv('nhanes_data_converted_DEMO_C.csv')
bmx = pd.read_csv('nhanes_data_converted_BMX_C.csv')
dxa_ag = pd.read_csv('nhanes_data_converted_AndroidGynoid.csv')
dxa_wb = pd.read_csv('nhanes_data_converted_WholeBody.csv')

# 2. เลือกคอลัมน์ที่จำเป็น (มี BMXWAIST และ RIDRETH1 ครบถ้วน)
demo_cols = demo[['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1']]
bmx_cols = bmx[['SEQN', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST']]
ag_cols = dxa_ag[['SEQN', 'DXXAPFAT', 'DXXGPFAT']]

# 3. จัดการไฟล์ WholeBody โดยหาค่าเฉลี่ยของไขมันรวม (DXDTOPF)
wb_cols = dxa_wb[['SEQN', 'DXDTOPF']].groupby('SEQN').mean().reset_index()
wb_cols.rename(columns={'DXDTOPF': 'TOTAL_BODY_FAT_PCT'}, inplace=True)

# 4. รวมไฟล์ทั้งหมดเข้าด้วยกัน
df = demo_cols.merge(bmx_cols, on='SEQN', how='inner')
df = df.merge(ag_cols, on='SEQN', how='inner')
df = df.merge(wb_cols, on='SEQN', how='inner')

# 5. เปลี่ยนชื่อคอลัมน์ให้อ่านง่ายและเป็นสากล
df.rename(columns={
    'RIAGENDR': 'GENDER',
    'RIDAGEYR': 'AGE',
    'RIDRETH1': 'RACE',
    'BMXWT': 'WEIGHT_KG',
    'BMXHT': 'HEIGHT_CM',
    'BMXBMI': 'BMI',
    'BMXWAIST': 'WAIST_CM',       # <--- รอบเอวมาแล้ว!
    'DXXAPFAT': 'ANDROID_FAT_PCT',
    'DXXGPFAT': 'GYNOID_FAT_PCT'
}, inplace=True)

# 6. แปลงตัวเลขเป็นข้อความ
df['GENDER'] = df['GENDER'].map({1: 'Male', 2: 'Female'})
df['RACE'] = df['RACE'].map({
    1: 'Mexican American',
    2: 'Other Hispanic',
    3: 'Non-Hispanic White',
    4: 'Non-Hispanic Black',
    5: 'Other Race'
})

# 7. คลีนข้อมูล: ตัดแถวที่แหว่ง และ กรองค่า 7777 (ปฏิเสธ), 9999 (ไม่ทราบ) ของรอบเอวทิ้ง
df_clean = df.dropna().copy()
df_clean = df_clean[(df_clean['WAIST_CM'] != 7777) & (df_clean['WAIST_CM'] != 9999)]

# 8. เซฟไฟล์ฉบับ Final
output_name = 'nhanes_cleaned_merged_final.csv'
df_clean.to_csv(output_name, index=False)

print(f"✅ ทำความสะอาดเสร็จสมบูรณ์! จำนวนประชากรที่พร้อมใช้งาน: {len(df_clean)} คน")
print(f"เซฟไฟล์ชื่อ: {output_name} เรียบร้อยแล้ว")