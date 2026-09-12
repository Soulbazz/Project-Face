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
    """คำนวณให้เจอที่ใช้ device เดียวกันแน่นอน (แก้บั๊กเดิมที่ main block เคยชน mps)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# train one epoch
def train(train_loader, model, loss_fn, optimizer, scaler):
    device = get_device()
    model.train()
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        with autocast():                        # << เปิดกลับมาใช้จริง
            pred = model(X)
            y = y.unsqueeze(1).float()
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if batch % 100 == 0:
            loss_val, current = loss.item(), batch * len(X)
            print(f"train loss: {loss_val:>7f} [{current:>5d}/{len(train_loader.dataset):>5d}]")


def validate(val_loader, model):
    device = get_device()
    model.eval()
    val_loss_mse = 0
    val_loss_mae = 0
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
    test_loss_mse = 0
    test_loss_mae = 0
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
    print(f"test mse loss: {test_loss_mse:>7f}, test mae loss: {test_loss_mae:>7f}")
    return test_loss_mse, test_loss_mae


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '../weights/vit_bmi_model.pt')
        self.val_loss_min = val_loss



if __name__ == "__main__":
    device = get_device()

    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--augmented', action='store_true', help='use augmented dataset')
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(
        24, augmented=args.augmented, vit_transformed=True, show_sample=False    # << 12 -> 24
    )

    CKPT = '../weights/vit_bmi_model.pt'

    model = get_model().float().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=5, verbose=True)

    total_start_time = time.time()

    for t in range(50):
        epoch_start_time = time.time()
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\nEpoch {t + 1} | Started at: {current_time}\n-------------------------------")

        train(train_loader, model, loss_fn, optimizer, scaler)
        val_loss = validate(val_loader, model)
        early_stopping(val_loss, model)

        epoch_duration = (time.time() - epoch_start_time) / 60
        print(f">> Epoch completed in: {epoch_duration:.2f} minutes")

        if early_stopping.early_stop:
            print("Early stopping triggered. Training complete!")
            break

    total_duration = (time.time() - total_start_time) / 60
    print(f"Total Training Time: {total_duration:.2f} minutes")

    model.load_state_dict(torch.load(CKPT, map_location=device))
    test(test_loader, model)
    print("Done!")