import os
import urllib.request
import warnings
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from tqdm import tqdm

from loader import vit_transforms
from models import get_model


def test_and_show_rf(img_dir, weight_dir, age, sex, waist=0):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # 1. โหลดและแปลงรูปภาพสำหรับโมเดล ViT
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = ToTensor()(image)
    image_vit = vit_transforms(image_tensor)
    image_vit = image_vit.unsqueeze(0).to(device)

    # 2. ทำนายค่า BMI จากใบหน้าด้วย Vision Transformer
    model = get_model().to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    bmi = pred.item()

    # 3. นำค่า BMI ไปทำนาย Body Fat ต่อด้วย Random Forest
    if waist > 0:
        rf_model = joblib.load('../weights/randomforest_bodyfat_with_waist.pkl')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body_fat = rf_model.predict([[bmi, age, sex, waist]])[0]
        mode_label = f"Random Forest Full (Waist {waist} cm)"
    else:
        rf_model = joblib.load('../weights/randomforest_bodyfat_no_waist.pkl')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            body_fat = rf_model.predict([[bmi, age, sex]])[0]
        mode_label = "Random Forest Basic (No Waist)"

    # 4. แสดงผลรูปภาพและค่าที่ทำนายได้
    plt.figure(figsize=(6, 6))
    plt.imshow(image_tensor.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title(f"Predicted BMI: {bmi:.2f} | Body Fat: {body_fat:.2f}%\n[{mode_label}]", fontsize=11, color="darkgreen")
    plt.show()

    return bmi, body_fat


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    # ตรวจสอบและดาวน์โหลดน้ำหนักโมเดล ViT หากยังไม่มี
    weight_dir = "../weights/aug_epoch_7.pt"
    if not os.path.exists(weight_dir):
        os.makedirs("../weights", exist_ok=True)
        url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
        print("Downloading ViT weights...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, weight_dir, reporthook=t.update_to)

    # ----------------------------------------------------
    # กำหนดค่าตัวแปรสำหรับทดสอบ
    # ----------------------------------------------------
    user_age = 25
    user_sex = 1      # 1 = ชาย (Male), 0 = หญิง (Female)
    user_waist = 80.0 # รอบเอว (ซม.) | ถ้าไม่ทราบให้ใส่ 0
    image_path = '../data/test_images/testpic02.png'

    bmi, body_fat = test_and_show_rf(image_path, weight_dir, user_age, user_sex, user_waist)

    print("\n" + "=" * 45)
    print(f"Predicted BMI      : {bmi:.2f}")
    print(f"Predicted Body Fat : {body_fat:.2f}%")
    print("=" * 45)