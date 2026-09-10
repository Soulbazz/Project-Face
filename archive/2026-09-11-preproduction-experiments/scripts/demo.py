import os
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms

import xgboost as xgb
import pandas as pd

model_resid = xgb.XGBRegressor()
model_resid.load_model('../models/residual_corrector.json')

def test_and_show(img_dir, weight_dir):
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

    # plot
    plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
    plt.axis("off")
    plt.title(f"Predicted BMI: {pred.item():>5f}")
    plt.show()

    return pred.item()



class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


if __name__ == "__main__":
    if not os.path.exists("../weights"):
        os.makedirs("../weights")
    weight_dir = "../weights/aug_epoch_7_backup.pt"
    # ถ้ายังไม่มีไฟล์ weights ให้ดาวน์โหลดก่อน (บรรทัดที่ 72 ของเดิม)
    if not os.path.exists(weight_dir):
        url = "https://face-to-bmi-weights.s3.us-east.cloud-object-storage.appdomain.cloud/aug_epoch_7.pt"
        print("downloading weights...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, weight_dir, reporthook=t.update_to)

    # 1. ทายค่าจาก ViT
    pred_vit = test_and_show('../data/test_images/testpic11.png', weight_dir)

    # 2. ให้ XGBoost ทายค่าความคลาดเคลื่อน
    residual_guess = model_resid.predict(pd.DataFrame({'y_pred': [pred_vit]}))

    # 3. รวมผล
    final_bmi = pred_vit + residual_guess[0]

    print(f"ค่าก่อนแก้ (ViT): {pred_vit:.4f}")
    print(f"ค่าหลังแก้ (Corrected): {final_bmi:.4f}")