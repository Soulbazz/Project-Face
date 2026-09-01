# import pandas as pd

# df = pd.read_csv('../data/data.csv')
# print(f"รวมทั้งหมด: {len(df):,} แถว | คอลัมน์: {df.columns.tolist()}\n")

# # --- 1) สัดส่วนเพศ ---
# print("=== สัดส่วนเพศ ===")
# g = pd.DataFrame({
#     'count': df['gender'].value_counts(),
#     'pct': (df['gender'].value_counts(normalize=True) * 100).round(2)
# })
# print(g, "\n")

# # --- 2) คนที่ BMI ต่ำกว่า 18 ---
# low = df[df['bmi'] < 18]
# print("=== BMI < 18 ===")
# print(f"จำนวน: {len(low):,} ({len(low)/len(df)*100:.2f}%)")
# print(low['gender'].value_counts(), "\n")

# # --- 3) แจกแจงตามเกณฑ์ WHO x เพศ ---
# bins   = [0, 18.5, 25, 30, 35, 40, 999]
# labels = ['<18.5 ผอม', '18.5-25 ปกติ', '25-30 ท้วม',
#           '30-35 อ้วน I', '35-40 อ้วน II', '40+ อ้วน III']
# df['bmi_class'] = pd.cut(df['bmi'], bins=bins, labels=labels)
# print("=== BMI Class x Gender ===")
# print(pd.crosstab(df['bmi_class'], df['gender'], margins=True), "\n")

# # --- 4) สถิติ BMI แยกเพศ ---
# print("=== สถิติ BMI แยกเพศ ===")
# print(df.groupby('gender')['bmi'].describe().round(2), "\n")

# # --- 5) เจาะช่วงที่โมเดลพัง (เคส Selena) ---
# print("=== ช่วงวิกฤต ===")
# print(f"BMI 14-16 ทั้งหมด      : {len(df[(df.bmi>=14)&(df.bmi<16)]):,} ภาพ")
# print(f"หญิง + BMI < 20        : {len(df[(df.gender=='Female')&(df.bmi<20)]):,} ภาพ")
# print(f"หญิง + BMI 14-16       : {len(df[(df.gender=='Female')&(df.bmi>=14)&(df.bmi<16)]):,} ภาพ")

# # --- 6) เช็กว่า is_training ใช้ได้ไหม ---
# print(f"\nis_training distribution:\n{df['is_training'].value_counts()}")

"""
ตรวจสอบ feature importance ของ XGBoost ทั้ง 2 โมเดล
วิธีรัน: cd scripts แล้ว python check_importance.py
"""
import os
import joblib
import pandas as pd

# ---------- ตั้งค่า path ----------
MODELS = {
    "Full (มีรอบเอว)":  "../weights/xgboost_bodyfat_with_waist.pkl",
    "Basic (ไม่มีเอว)": "../weights/xgboost_bodyfat_no_waist.pkl",
}
IMPORTANCE_TYPES = ["gain", "weight", "cover", "total_gain", "total_cover"]


def inspect(name, path):
    print("\n" + "=" * 60)
    print(f"  {name}")
    print("=" * 60)

    if not os.path.exists(path):
        print(f"[!] ไม่พบไฟล์: {path}")
        return

    model = joblib.load(path)

    # --- 1) ลำดับ feature ตอนเทรน ---
    try:
        print(f"feature_names_in_ : {list(model.feature_names_in_)}")
    except AttributeError:
        print("feature_names_in_ : (ไม่มี — เทรนด้วย numpy array)")

    booster = model.get_booster()
    print(f"booster.features  : {booster.feature_names}")

    # --- 2) รวม importance ทุกแบบเป็นตารางเดียว ---
    scores = {}
    for t in IMPORTANCE_TYPES:
        scores[t] = booster.get_score(importance_type=t)

    df = pd.DataFrame(scores)
    df = df.reindex(booster.feature_names)      # เรียงตามลำดับ feature จริง
    df = df.fillna(0).round(2)

    # --- 3) เพิ่มคอลัมน์ % ของ gain เพื่ออ่านง่าย ---
    df["gain_%"] = (df["gain"] / df["gain"].sum() * 100).round(1)
    df["total_gain_%"] = (df["total_gain"] / df["total_gain"].sum() * 100).round(1)

    print("\n" + df.to_string())

    # --- 4) สรุปอันดับตาม total_gain (น่าเชื่อถือกว่า gain เดี่ยว) ---
    rank = df["total_gain"].sort_values(ascending=False)
    print("\nอันดับตาม total_gain:")
    for i, (feat, val) in enumerate(rank.items(), 1):
        print(f"  {i}. {feat:<12} {val:>12,.0f}  ({df.loc[feat,'total_gain_%']}%)")


if __name__ == "__main__":
    for name, path in MODELS.items():
        inspect(name, path)
    print("\n✅ เสร็จแล้ว")