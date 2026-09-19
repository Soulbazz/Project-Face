import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


# ---------- PATH กลาง แก้ที่เดียวจบ ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
WEIGHTS = os.path.join(ROOT, "weights")
RESULTS = os.path.join(ROOT, "scripts", "results")

PAQ_PATH = os.path.join(RAW_DIR, "PAQ_C.XPT")
MERGED_PATH = os.path.join(
    PROC_DIR,
    "nhanes_cleaned_merged_final.csv"
)

for directory in (WEIGHTS, RESULTS):
    os.makedirs(directory, exist_ok=True)


# ---------- Feature และ Target ----------
# ลำดับ Feature ต้องตรงกับ predict_pipeline.py ทุกประการ
FEATS_FULL = [
    "BMI",
    "AGE",
    "GENDER_NUM",
    "WAIST_CM",
    "LIFESTYLE_LEVEL",
]

FEATS_BASIC = [
    "BMI",
    "AGE",
    "GENDER_NUM",
    "LIFESTYLE_LEVEL",
]

TARGET = "TOTAL_BODY_FAT_PCT"

print("กำลังโหลดข้อมูลและฝึกสอน XGBoost...")


# ============================================================
# 1. โหลดข้อมูล NHANES หลัก
# ============================================================
if not os.path.exists(MERGED_PATH):
    raise FileNotFoundError(
        f"ไม่พบไฟล์ข้อมูลหลัก: {MERGED_PATH}"
    )

df = pd.read_csv(MERGED_PATH)

if "SEQN" not in df.columns:
    raise KeyError(
        "ไม่พบคอลัมน์ SEQN ใน nhanes_cleaned_merged_final.csv"
    )

print(f"ข้อมูลหลักก่อน Merge: {len(df):,} แถว")


# ============================================================
# 2. โหลดข้อมูล Physical Activity จาก PAQ_C.XPT
# ============================================================
if not os.path.exists(PAQ_PATH):
    raise FileNotFoundError(
        f"ไม่พบไฟล์ Physical Activity: {PAQ_PATH}"
    )

paq = pd.read_sas(PAQ_PATH, format="xport")

required_paq_columns = ["SEQN", "PAD200", "PAD320"]
missing_paq_columns = [
    column
    for column in required_paq_columns
    if column not in paq.columns
]

if missing_paq_columns:
    raise KeyError(
        "ไม่พบคอลัมน์ใน PAQ_C.XPT: "
        + ", ".join(missing_paq_columns)
    )

paq = paq[required_paq_columns].copy()

print(f"ข้อมูล PAQ ก่อนกรอง: {len(paq):,} แถว")


# ============================================================
# 3. ทำความสะอาดข้อมูล PAQ
# ============================================================
for column in required_paq_columns:
    paq[column] = pd.to_numeric(
        paq[column],
        errors="coerce",
    )

# ตัดแถวที่ PAD200 หรือ PAD320 มีรหัส:
# 7 = Refused
# 9 = Don't know
invalid_response = (
    paq["PAD200"].isin([7, 9])
    | paq["PAD320"].isin([7, 9])
)

invalid_count = int(invalid_response.sum())
paq = paq.loc[~invalid_response].copy()

# ไม่ควรมี SEQN ว่าง เพราะใช้เป็นคีย์ Merge
missing_seqn_count = int(paq["SEQN"].isna().sum())
paq = paq.dropna(subset=["SEQN"]).copy()

# ทำให้ SEQN เป็นชนิดเดียวกันก่อน Merge
paq["SEQN"] = paq["SEQN"].astype("int64")

print(f"ตัด Refused/Don't know: {invalid_count:,} แถว")
print(f"ตัดแถวที่ไม่มี SEQN: {missing_seqn_count:,} แถว")
print(f"ข้อมูล PAQ หลังกรอง: {len(paq):,} แถว")


# ============================================================
# 4. สร้าง LIFESTYLE_LEVEL
# ============================================================
# ค่าเริ่มต้น:
# 0 = Sedentary
paq["LIFESTYLE_LEVEL"] = 0

# PAD320 == 1:
# 1 = Normal
paq.loc[
    paq["PAD320"].eq(1),
    "LIFESTYLE_LEVEL",
] = 1

# PAD200 == 1:
# 2 = Active
# ทำทีหลังเพื่อให้ Active มีลำดับความสำคัญสูงกว่า Normal
paq.loc[
    paq["PAD200"].eq(1),
    "LIFESTYLE_LEVEL",
] = 2

paq["LIFESTYLE_LEVEL"] = paq[
    "LIFESTYLE_LEVEL"
].astype("int8")

# ตรวจสอบว่า PAQ มี SEQN ซ้ำหรือไม่
duplicate_seqn_count = int(paq["SEQN"].duplicated().sum())

if duplicate_seqn_count > 0:
    raise ValueError(
        f"พบ SEQN ซ้ำใน PAQ_C.XPT จำนวน "
        f"{duplicate_seqn_count:,} แถว "
        "จึงยังไม่ทำ Merge เพื่อป้องกันจำนวนแถวเพิ่มผิดปกติ"
    )

print("\nการกระจายของ LIFESTYLE_LEVEL:")
print(
    paq["LIFESTYLE_LEVEL"]
    .value_counts()
    .sort_index()
    .rename(index={
        0: "0 - Sedentary",
        1: "1 - Normal",
        2: "2 - Active",
    })
)


# ============================================================
# 5. Merge LIFESTYLE_LEVEL เข้ากับข้อมูลหลัก
# ============================================================
# รองรับกรณีเคยรันสคริปต์นี้แล้ว
if "LIFESTYLE_LEVEL" in df.columns:
    print("\nพบ LIFESTYLE_LEVEL เดิม กำลังแทนที่ด้วยข้อมูลจาก PAQ_C.XPT")
    df = df.drop(columns=["LIFESTYLE_LEVEL"])

# ทำ SEQN ให้เป็น numeric ก่อน
df["SEQN"] = pd.to_numeric(
    df["SEQN"],
    errors="coerce",
)

# ใช้ left merge เพื่อรักษาข้อมูลหลักไว้ทั้งหมด
# ผู้ที่จับคู่ PAQ ไม่ได้จะมี LIFESTYLE_LEVEL เป็น NaN
df = df.merge(
    paq[["SEQN", "LIFESTYLE_LEVEL"]],
    on="SEQN",
    how="left",
    validate="many_to_one",
)

matched_count = int(df["LIFESTYLE_LEVEL"].notna().sum())
unmatched_count = int(df["LIFESTYLE_LEVEL"].isna().sum())

print("\nผลการ Merge:")
print(f"จับคู่ PAQ สำเร็จ: {matched_count:,} แถว")
print(f"จับคู่ PAQ ไม่ได้: {unmatched_count:,} แถว")
print(f"ข้อมูลรวมทั้งหมด: {len(df):,} แถว")

# บันทึก LIFESTYLE_LEVEL กลับเข้าไฟล์ processed
# ใช้ไฟล์ชั่วคราวก่อน แล้วจึงแทนที่ไฟล์จริง
temporary_path = MERGED_PATH + ".tmp"

df.to_csv(
    temporary_path,
    index=False,
)

os.replace(
    temporary_path,
    MERGED_PATH,
)

print(f"✅ อัปเดตข้อมูลแล้ว: {MERGED_PATH}")


# ============================================================
# 6. เตรียมข้อมูลสำหรับเทรน
# ============================================================
if "GENDER" not in df.columns:
    raise KeyError("ไม่พบคอลัมน์ GENDER ในข้อมูลหลัก")

df["GENDER_NUM"] = df["GENDER"].map({
    "Male": 1,
    "Female": 0,
})

required_model_columns = list(
    dict.fromkeys(FEATS_FULL + FEATS_BASIC + [TARGET])
)

missing_model_columns = [
    column
    for column in required_model_columns
    if column not in df.columns
]

if missing_model_columns:
    raise KeyError(
        "ไม่พบคอลัมน์ที่จำเป็นสำหรับเทรน: "
        + ", ".join(missing_model_columns)
    )

# แปลง Feature และ Target เป็นตัวเลข
for column in required_model_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# ใช้ cohort เดียวกันสำหรับทั้งสองโมเดล
# เพื่อให้การเปรียบเทียบ MAE มีความยุติธรรม
df_model = df.dropna(
    subset=FEATS_FULL + [TARGET]
).reset_index(drop=True)

if len(df_model) == 0:
    raise ValueError(
        "ไม่มีข้อมูลที่สมบูรณ์เพียงพอสำหรับเทรนโมเดล "
        "โปรดตรวจสอบผลการ Merge และคอลัมน์ Feature"
    )

print(
    f"\nข้อมูลที่ใช้เทรนได้: "
    f"{len(df_model):,} / {len(df):,} แถว"
)

print("\nLifestyle ในชุดข้อมูลสำหรับเทรน:")
print(
    df_model["LIFESTYLE_LEVEL"]
    .astype(int)
    .value_counts()
    .sort_index()
    .rename(index={
        0: "0 - Sedentary",
        1: "1 - Normal",
        2: "2 - Active",
    })
)


# ============================================================
# 7. แบ่ง Train/Test
# ============================================================
train_df, test_df = train_test_split(
    df_model,
    test_size=0.2,
    random_state=42,
)

y_train = train_df[TARGET]
y_test = test_df[TARGET]

print(
    f"\nTrain: {len(train_df):,} | "
    f"Test: {len(test_df):,}"
)

# บันทึกข้อมูล Train/Test
train_path = os.path.join(DATA_DIR, "train_tabular.csv")
test_path = os.path.join(DATA_DIR, "test_tabular.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ บันทึก Train: {train_path}")
print(f"✅ บันทึก Test: {test_path}")


# ============================================================
# 8. โมเดลมีรอบเอว
# ============================================================
xgb_full = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
)

xgb_full.fit(
    train_df[FEATS_FULL],
    y_train,
)

pred_full = xgb_full.predict(
    test_df[FEATS_FULL]
)

mae_full = mean_absolute_error(
    y_test,
    pred_full,
)

full_model_path = os.path.join(
    WEIGHTS,
    "xgboost_bodyfat_with_waist.pkl",
)

joblib.dump(
    xgb_full,
    full_model_path,
)

print("\n=== XGBoost: มีรอบเอว ===")
print(f"Features: {FEATS_FULL}")
print(f"MAE: {mae_full:.3f} %")
print(f"✅ บันทึกโมเดล: {full_model_path}")


# ============================================================
# 9. โมเดลไม่มีรอบเอว
# ============================================================
xgb_basic = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    n_jobs=-1,
)

xgb_basic.fit(
    train_df[FEATS_BASIC],
    y_train,
)

pred_basic = xgb_basic.predict(
    test_df[FEATS_BASIC]
)

mae_basic = mean_absolute_error(
    y_test,
    pred_basic,
)

basic_model_path = os.path.join(
    WEIGHTS,
    "xgboost_bodyfat_no_waist.pkl",
)

joblib.dump(
    xgb_basic,
    basic_model_path,
)

print("\n=== XGBoost: ไม่มีรอบเอว ===")
print(f"Features: {FEATS_BASIC}")
print(f"MAE: {mae_basic:.3f} %")
print(f"✅ บันทึกโมเดล: {basic_model_path}")


# ============================================================
# 10. สรุปผล
# ============================================================
gain = mae_basic - mae_full

relative_gain = (
    (gain / mae_basic) * 100
    if mae_basic != 0
    else 0.0
)

print("\n" + "=" * 60)
print(f"MAE มีรอบเอว      : {mae_full:.3f} %")
print(f"MAE ไม่มีรอบเอว    : {mae_basic:.3f} %")
print(
    f"รอบเอวช่วยลด error: {gain:.3f} % "
    f"({relative_gain:.1f}% relative)"
)
print("=" * 60)

summary = pd.DataFrame([
    {
        "model": "with_waist",
        "features": ", ".join(FEATS_FULL),
        "mae": round(mae_full, 3),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "paq_matched": matched_count,
        "paq_unmatched": unmatched_count,
    },
    {
        "model": "no_waist",
        "features": ", ".join(FEATS_BASIC),
        "mae": round(mae_basic, 3),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "paq_matched": matched_count,
        "paq_unmatched": unmatched_count,
    },
])

summary_path = os.path.join(
    RESULTS,
    "xgboost_train_summary.csv",
)

summary.to_csv(
    summary_path,
    index=False,
)

print(f"\n✅ บันทึกผลสรุป: {summary_path}")
print("✅ เทรนและบันทึกโมเดลใหม่เรียบร้อย")