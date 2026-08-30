import joblib
import os
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms


def test_and_show(img_dir, weight_dir, age, sex):
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
    # Deurenberg Formula สำหรับคำนวณ Body Fat (%)
    # -----------------------------------------
    body_fat = (1.20 * bmi) + (0.23 * age) - (10.8 * sex) - 5.4

    # plot
    plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    # แสดงค่าแบบทศนิยม 2 ตำแหน่งบนหัวรูปภาพ
    plt.title(f"Predicted BMI: {bmi:.2f} | Body Fat: {body_fat:.2f}%") 
    plt.show()

    return bmi, body_fat


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    if not os.path.exists("../weights"):
        os.makedirs("../weights")
        weight_dir = "../weights/aug_epoch_7.pt"
        url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
        print("dowloading weights...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, weight_dir, reporthook=t.update_to)

    # -----------------------------------------
    # ตัวแปรข้อมูลส่วนบุคคลสำหรับการคำนวณ
    # -----------------------------------------
    user_age = 41
    user_sex = 1  # กำหนดค่าเพศ: 1 สำหรับผู้ชาย, 0 สำหรับผู้หญิง
    
    # ⚠️ อย่าลืมแก้ชื่อไฟล์ในวงเล็บนี้ให้ตรงกับรูปของคุณเป๊ะๆ
    image_path = '../data/testpic02.png' 

    # เรียกใช้ฟังก์ชันและรับค่าที่ทำนายกลับมา
    bmi, body_fat = test_and_show(image_path, '../weights/aug_epoch_7.pt', user_age, user_sex)
    
    # พิมพ์ผลลัพธ์ลงใน Terminal ด้านล่าง
    print(f'Predicted BMI: {bmi:.2f}')
    print(f'Estimated Body Fat: {body_fat:.2f}%')