import torch
from models import get_model
from loader import get_dataloaders  # นำเข้าฟังก์ชันนี้จาก loader.py

# 1. โหลดโมเดล
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model().to(device)
# ระบุ path ให้ถูกต้องตามโครงสร้างโฟลเดอร์ของคุณ
model.load_state_dict(torch.load('../weights/aug_epoch_7.pt', map_location=device))
model.eval()

# 2. ดึง DataLoader (เรียกใช้ฟังก์ชันของคุณ โดยตั้งค่าแบบไม่ต้องทำ Augmentation ในช่วง Test)
_, test_loader, _ = get_dataloaders(batch_size=16, augmented=False, vit_transformed=True)

# 3. ทำ Inference
preds, targets = [], []
with torch.no_grad():
    for images, bmis in test_loader:
        images = images.to(device)
        outputs = model(images)
        # ใช้ .flatten() เพื่อให้แน่ใจว่าเป็น 1D array
        preds.extend(outputs.cpu().numpy().flatten())
        targets.extend(bmis.numpy().flatten())

# 4. คำนวณ Residual และบันทึกเป็น CSV
import pandas as pd
import numpy as np
df = pd.DataFrame({'y_true': targets, 'y_pred': preds})
df['resid'] = df['y_true'] - df['y_pred']
df.to_csv('stage1_residuals.csv', index=False)

print("Saved stage1_residuals.csv successfully!")
print(f"MAE on test set: {np.mean(np.abs(df['resid'])):.4f}")