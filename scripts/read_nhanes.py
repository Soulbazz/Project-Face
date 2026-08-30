import pandas as pd

# 1. ระบุชื่อไฟล์ .XPT (อย่าลืมเปลี่ยนชื่อในเครื่องหมายคำพูดให้ตรงกับไฟล์ที่คุณโหลดมานะครับ)
file_name = "dxx_C.xpt" 

print("กำลังอ่านไฟล์ XPT...")
# 2. ใช้คำสั่งอเนกประสงค์ของ pandas เพื่ออ่านไฟล์นามสกุล SAS
df = pd.read_sas(file_name)

# 3. แสดงตัวอย่างข้อมูล 5 บรรทัดแรกบน Terminal
print(df.head())

# 4. แปลงร่างและบันทึกเป็นไฟล์ .csv
output_name = "nhanes_data_converted_WholeBody.csv"
df.to_csv(output_name, index=False)

print(f"\n✅ แปลงไฟล์สำเร็จ! ข้อมูลถูกเซฟไว้ที่: {output_name}")