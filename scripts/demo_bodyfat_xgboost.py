import joblib
import os
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import warnings

# นำเข้า tkinter สำหรับทำ GUI
import tkinter as tk
from tkinter import filedialog, messagebox

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms

def test_and_show(img_dir, weight_dir, age, sex, waist):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # open and transform image for vit
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ToTensor()(image)
    image_vit = vit_transforms(image)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    # get model and predict
    model = get_model()
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    bmi = pred.item()

    # -----------------------------------------
    # เงื่อนไขตรวจสอบการใช้รอบเอว (Waist) อัตโนมัติ
    # -----------------------------------------

    if waist > 0:
        # เปลี่ยนชื่อไฟล์มาดึง XGBoost
        fat_model = joblib.load('xgboost_bodyfat_with_waist.pkl')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body_fat = fat_model.predict([[bmi, age, sex, waist]])[0]
        
        plot_msg = f"Mode: XGBoost Full (Waist {waist} cm)"
        status_msg = f"คำนวณโดยใช้รอบเอว {waist} ซม. (XGBoost ความแม่นยำสูงสุด)"
    else:
        # เปลี่ยนชื่อไฟล์มาดึง XGBoost
        fat_model = joblib.load('xgboost_bodyfat_no_waist.pkl')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body_fat = fat_model.predict([[bmi, age, sex]])[0]
            
        plot_msg = "Mode: XGBoost Basic (No Waist Data)"
        status_msg = "ไม่ได้ใช้ข้อมูลรอบเอวในการคำนวณ (XGBoost ข้อมูลพื้นฐาน)"

    # if waist > 0:
    #     fat_model = joblib.load('custom_bodyfat_model_with_waist.pkl')
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         body_fat = fat_model.predict([[bmi, age, sex, waist]])[0]
        
    #     plot_msg = f"Mode: Full (Waist {waist} cm used)"
    #     status_msg = f"คำนวณโดยใช้รอบเอว {waist} ซม. (มีความแม่นยำสูง)"
    # else:
    #     fat_model = joblib.load('custom_bodyfat_model_no_waist.pkl')
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         body_fat = fat_model.predict([[bmi, age, sex]])[0]
            
    #     plot_msg = "Mode: Basic (No Waist Data)"
    #     status_msg = "ไม่ได้ใช้ข้อมูลรอบเอวในการคำนวณ (ใช้ข้อมูลพื้นฐาน)"

    # plot
    plt.figure(figsize=(7, 7)) 
    plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title(f"Predicted BMI: {bmi:.2f} | Body Fat: {body_fat:.2f}%\n[{plot_msg}]", fontsize=12, color="blue")
    plt.show()

    return bmi, body_fat, status_msg


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# ==========================================
# ส่วนของการสร้างหน้าต่าง GUI (Tkinter)
# ==========================================
def run_gui():
    # ฟังก์ชันเลือกไฟล์รูปภาพ
    def browse_image():
        filepath = filedialog.askopenfilename(
            initialdir="../data", 
            title="เลือกรูปภาพใบหน้า",
            filetypes=(("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*"))
        )
        if filepath:
            img_path_var.set(filepath)

    # ฟังก์ชันเมื่อกดปุ่ม "คำนวณ"
    def start_prediction():
        img_path = img_path_var.get()
        if not img_path:
            messagebox.showerror("ข้อผิดพลาด", "กรุณาเลือกรูปภาพก่อนครับ!")
            return
            
        try:
            # ดึงค่าจากช่อง Input
            user_age = int(age_entry.get())
            user_waist = float(waist_entry.get())
            user_sex = 1 if sex_var.get() == "Male" else 0
            
            # โหลด Weight ถ้ายังไม่มี
            weight_dir = "../weights/aug_epoch_7.pt"
            if not os.path.exists("../weights"):
                os.makedirs("../weights")
                url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
                print("downloading weights...")
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
                    urllib.request.urlretrieve(url, weight_dir, reporthook=t.update_to)

            # รันฟังก์ชันหลัก
            print("กำลังวิเคราะห์รูปภาพ...")
            bmi, body_fat, status_msg = test_and_show(img_path, weight_dir, user_age, user_sex, user_waist)
            
            # ปริ้นผลลัพธ์ใน Terminal
            print('\n=========================================')
            print(f'AI Predicted BMI: {bmi:.2f}')
            print(f'AI Predicted Body Fat: {body_fat:.2f}%')
            print(f'Status: {status_msg}')
            print('=========================================')
            
        except ValueError:
            messagebox.showerror("ข้อผิดพลาด", "กรุณากรอกตัวเลขให้ถูกต้องครับ (เช่น อายุ 25, รอบเอว 80.5 หรือ 0)")

    # ตั้งค่าหน้าต่างโปรแกรม
    root = tk.Tk()
    root.title("Face-to-BMI & Body Fat Predictor")
    root.geometry("400x350")
    root.eval('tk::PlaceWindow . center') # ให้อยู่กลางจอ

    # ฟอนต์เริ่มต้น
    font_style = ("Arial", 10)

    # --- ส่วนเลือกรูปภาพ ---
    tk.Label(root, text="1. เลือกรูปภาพใบหน้า:", font=font_style).pack(pady=(15, 0))
    img_path_var = tk.StringVar()
    tk.Entry(root, textvariable=img_path_var, width=40, state='readonly').pack(pady=5)
    tk.Button(root, text="Browse...", command=browse_image).pack()

    # --- ส่วนเลือกอายุ และ เพศ ---
    frame_info = tk.Frame(root)
    frame_info.pack(pady=15)

    tk.Label(frame_info, text="อายุ (ปี):", font=font_style).grid(row=0, column=0, padx=5, sticky="e")
    age_entry = tk.Entry(frame_info, width=10)
    age_entry.insert(0, "20")
    age_entry.grid(row=0, column=1, padx=5, sticky="w")

    tk.Label(frame_info, text="เพศ:", font=font_style).grid(row=1, column=0, padx=5, pady=10, sticky="e")
    sex_var = tk.StringVar(value="Male")
    tk.Radiobutton(frame_info, text="ชาย (Male)", variable=sex_var, value="Male").grid(row=1, column=1, sticky="w")
    tk.Radiobutton(frame_info, text="หญิง (Female)", variable=sex_var, value="Female").grid(row=1, column=2, sticky="w")

    # --- ส่วนระบุรอบเอว ---
    tk.Label(root, text="รอบเอว (เซนติเมตร):", font=font_style).pack()
    waist_entry = tk.Entry(root, width=15)
    waist_entry.insert(0, "0")
    waist_entry.pack(pady=5)
    tk.Label(root, text="* หากไม่ทราบข้อมูล ให้ใส่เลข 0", fg="gray", font=("Arial", 9)).pack()

    # --- ปุ่มรัน ---
    tk.Button(root, text="🔍 วิเคราะห์ข้อมูล", font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", command=start_prediction).pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    run_gui()