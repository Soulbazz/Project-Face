import os, glob, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src  = os.path.join(ROOT, 'data', 'split_fixed.csv')

if not os.path.exists(src):
    raise SystemExit(f"ไม่พบ {src}")

df = pd.read_csv(src)
print("คอลัมน์:", list(df.columns))
print(df.head(3).to_string(), "\n")


def pick(*cands):
    for c in cands:
        for col in df.columns:
            if col.lower().strip() == c:
                return col
    return None


# เพิ่ม 'name' เข้าไปแล้ว (คือคอลัมน์ของคุณ)
col_split = pick('split', 'set', 'fold', 'subset', 'phase')
col_img   = pick('name', 'image_path', 'path', 'filename',
                 'image', 'img_path', 'file', 'img')
col_bmi   = pick('bmi', 'bmi_true', 'target', 'label', 'y')

print(f"ใช้: split={col_split} | image={col_img} | bmi={col_bmi}")
if col_img is None or col_bmi is None:
    raise SystemExit("หาคอลัมน์ไม่เจอ — ส่งรายชื่อคอลัมน์ข้างบนมาให้ผมดูครับ")


# ---------- เลือกเฉพาะ split == test ----------
print("ค่าใน split:", df[col_split].unique())
test = df[df[col_split].astype(str).str.lower().str.strip()
          .isin(['test', 'testing'])]
print(f"แถวที่เป็น test: {len(test)}")

out = test[[col_img, col_bmi]].rename(
    columns={col_img: 'image_path', col_bmi: 'bmi'}).dropna().reset_index(drop=True)


# ---------- ค้นหาโฟลเดอร์ภาพอัตโนมัติ ----------
sample = str(out['image_path'].iloc[0])
print(f"\nกำลังค้นหาโฟลเดอร์ที่มีไฟล์ '{sample}' ...")

IMG_DIR = None
for dirpath, dirnames, filenames in os.walk(ROOT):
    if sample in filenames:
        IMG_DIR = dirpath
        break

if IMG_DIR is None:
    print("⚠️ หาไฟล์ภาพไม่เจอเลย — ลองแสดงโฟลเดอร์ที่มี .bmp/.jpg มากที่สุด:")
    counts = {}
    for dirpath, _, filenames in os.walk(ROOT):
        n = sum(1 for f in filenames
                if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png')))
        if n > 0:
            counts[dirpath] = n
    for d, n in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   {n:>6} ไฟล์  |  {os.path.relpath(d, ROOT)}")
    raise SystemExit("→ บอกผมว่าโฟลเดอร์ไหนคือภาพใบหน้า แล้วผมจะปรับให้ครับ")

print(f"✅ เจอโฟลเดอร์ภาพ: {os.path.relpath(IMG_DIR, ROOT)}")


# ---------- แปลงเป็น absolute path ----------
out['image_path'] = out['image_path'].apply(
    lambda p: os.path.join(IMG_DIR, os.path.basename(str(p))))

ok = out['image_path'].apply(os.path.exists)
print(f"\nไฟล์ภาพที่หาเจอ: {ok.sum()} / {len(out)}")
if not ok.all():
    print("ตัวอย่างที่หาไม่เจอ:",
          out.loc[~ok, 'image_path'].head(3).tolist())

out = out[ok].reset_index(drop=True)
dst = os.path.join(ROOT, 'data', 'test_faces.csv')
out.to_csv(dst, index=False)

print(f"\n✅ เซฟ {os.path.relpath(dst, ROOT)} ({len(out)} แถว)")
print(f"   BMI range: {out['bmi'].min():.1f} – {out['bmi'].max():.1f}"
      f" | mean = {out['bmi'].mean():.2f}")
print(out.head(3).to_string())