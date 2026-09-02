import torch
from loader import get_dataloaders
from models import get_model
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 1. โหลด Data (ใช้ test_loader)
_, test_loader, _ = get_dataloaders(12, augmented=True, vit_transformed=True) # ปรับ augmented ให้ตรงกับตอนเทรน

# 2. โหลดโมเดลที่เทรนเสร็จแล้ว
model = get_model().to(device)
model.load_state_dict(torch.load('../weights/aug_epoch_7.pt', map_location=device))
model.eval()

# 3. รัน Test
test_loss_mse = 0
test_loss_mae = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device).unsqueeze(1).float()
        pred = model(X)
        test_loss_mse += nn.MSELoss()(pred, y).item()
        test_loss_mae += nn.L1Loss()(pred, y).item()

print(f"ผลการทดสอบบน Test Set:")
print(f"MSE Loss: {test_loss_mse / len(test_loader):.4f}")
print(f"MAE Loss: {test_loss_mae / len(test_loader):.4f}")