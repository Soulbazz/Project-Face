import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from loader import get_dataloaders
from models import get_model

import numpy as np
import argparse
import time
from datetime import datetime


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train(train_loader, model, loss_fn, optimizer, scaler):
    device = get_device()
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast():
            pred = model(X)
            if batch == 0:
                print(f">> Autocast status: {torch.is_autocast_enabled()}")
                print(f">> Prediction Tensor dtype: {pred.dtype}")
            y = y.unsqueeze(1).float()
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 50 == 0:
            loss_val, current = loss.item(), batch * len(X)
            print(f"train loss: {loss_val:>7f} [{current:>5d}/{len(train_loader.dataset):>5d}]")


def validate(val_loader, model):
    device = get_device()
    model.eval()
    val_loss_mse = 0.0
    val_loss_mae = 0.0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(val_loader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            y = y.unsqueeze(1).float()
            val_loss_mse += nn.MSELoss()(pred, y).item()
            val_loss_mae += nn.L1Loss()(pred, y).item()

    val_loss_mse /= len(val_loader)
    val_loss_mae /= len(val_loader)
    print(f"val mse loss: {val_loss_mse:>7f}, val mae loss: {val_loss_mae:>7f}")
    return val_loss_mae


def test(test_loader, model):
    device = get_device()
    model.eval()
    test_loss_mse = 0.0
    test_loss_mae = 0.0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            y = y.unsqueeze(1).float()
            test_loss_mse += nn.MSELoss()(pred, y).item()
            test_loss_mae += nn.L1Loss()(pred, y).item()

    test_loss_mse /= len(test_loader)
    test_loss_mae /= len(test_loader)
    print(f"\n==========================================")
    print(f"TEST FINAL RESULT:")
    print(f"test mse loss: {test_loss_mse:>7f}, test mae loss: {test_loss_mae:>7f}")
    print(f"==========================================\n")
    return test_loss_mse, test_loss_mae


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path="../weights/vit_head_split_v2.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving checkpoint...")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss


if __name__ == "__main__":
    device = get_device()
    print(f"Execution Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--augmented", action="store_true", help="use augmented dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for dataloader")
    args = parser.parse_args()

    # 1. โหลด Dataloader ลำดับ (Train, Val, Test)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        augmented=args.augmented,
        vit_transformed=True,
        show_sample=False,
    )

    CKPT = "../weights/vit_head_split_v2.pt"

    # 2. โหลดโมเดล ViT
    model = get_model().float().to(device)

    # 3. Freeze Backbone ทั้งหมดก่อน
    for param in model.parameters():
        param.requires_grad = False

    # 4. ปลดล็อคเฉพาะ Prediction Head / MLP ท้ายสุด
    head_found = False
    for candidate_name in ["heads", "head", "mlp", "fc", "classifier"]:
        if hasattr(model, candidate_name):
            head_module = getattr(model, candidate_name)
            for param in head_module.parameters():
                param.requires_grad = True
            head_found = True
            print(f">> Successfully unfrozen Prediction Head: model.{candidate_name}")
            break

    # Fallback กรณีโครงสร้างเป็น Sequential หรือ custom module
    if not head_found:
        child_list = list(model.named_children())
        if child_list:
            last_name, last_module = child_list[-1]
            for param in last_module.parameters():
                param.requires_grad = True
            print(f">> Unfrozen last child module: model.{last_name}")
        else:
            # หากไม่มี child module ให้ปลดล็อคเฉพาะพารามิเตอร์ชุดท้ายสุด
            params = list(model.parameters())
            for p in params[-2:]:
                p.requires_grad = True
            print(">> Unfrozen last 2 parameter tensors.")

    # ตรวจสอบจำนวนพารามิเตอร์ที่เปิดให้เทรน
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f">> Total parameters: {total_params:,} | Trainable parameters: {trainable_count:,}")

    # 5. ตั้งค่า Optimizer, Scaler, และ Loss
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True, save_path=CKPT)

    total_start_time = time.time()
    epochs = 50

    for t in range(epochs):
        epoch_start_time = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\nEpoch {t + 1}/{epochs} | Started at: {current_time}\n" + "-" * 35)

        train(train_loader, model, loss_fn, optimizer, scaler)
        val_loss = validate(val_loader, model)
        early_stopping(val_loss, model)

        epoch_duration = (time.time() - epoch_start_time) / 60
        print(f">> Epoch completed in: {epoch_duration:.2f} minutes")

        if early_stopping.early_stop:
            print(f">> Early stopping triggered at epoch {t + 1}!")
            break

    total_duration = (time.time() - total_start_time) / 60
    print(f"\nTotal Training Duration: {total_duration:.2f} minutes")

    # 6. โหลดโมเดลที่ดีที่สุดมาประเมินบน Test Set
    model.load_state_dict(torch.load(CKPT, map_location=device))
    test(test_loader, model)
    print("Training process finished successfully.")