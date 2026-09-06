import pandas as pd

df = pd.read_csv('../data/split_fixed.csv')

# หา BMI ที่ซ้ำกันเป๊ะ (proxy ของ "คนเดียวกัน")
dup = df[df.duplicated(subset=['bmi', 'gender'], keep=False)]

# กลุ่ม bmi+gender เดียวกัน ที่กระจายอยู่มากกว่า 1 split = leakage
cross = dup.groupby(['bmi', 'gender'])['split'].nunique()
leaky_groups = cross[cross > 1]

print(f"กลุ่มที่สงสัยว่าเป็นคนเดียวกัน: {dup.groupby(['bmi','gender']).ngroups} กลุ่ม")
print(f"กลุ่มที่รั่วข้าม split: {len(leaky_groups)} กลุ่ม")

n_leaky_imgs = dup.set_index(['bmi','gender']).loc[leaky_groups.index].shape[0]
print(f"จำนวนรูปที่เกี่ยวข้องกับการรั่ว: {n_leaky_imgs} / {len(df)} รูป")
