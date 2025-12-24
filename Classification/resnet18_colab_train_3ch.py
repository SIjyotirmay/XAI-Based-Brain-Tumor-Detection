#!/usr/bin/env python3
"""
ResNet-18 training script (3-channel: grayscale image + mask + gradcam)
Includes automatic evaluation at the end of training:
 - classification report (precision/recall/f1/support)
 - confusion matrix image saved to <checkpoint_dir>/confusion_matrix.png
 - per-image predictions CSV saved to <checkpoint_dir>/val_predictions.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

# -------------------------
# Dataset
# -------------------------
class MaskGradcamDataset(Dataset):
    def __init__(self, df, transforms=None, img_size=256):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def _read_gray(self, p):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {p}")
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        return (img.astype(np.float32) / 255.0)

    def _read_mask(self, p):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Mask not found: {p}")
        m = cv2.resize(m, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        if m.max() > 1:
            m = m.astype(np.float32) / 255.0
        else:
            m = m.astype(np.float32)
        return m

    def _read_gcam(self, p):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise FileNotFoundError(f"Gradcam not found: {p}")
        g = cv2.resize(g, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        g = g.astype(np.float32)
        gmax = g.max()
        if gmax > 0:
            g = g / gmax
        return g

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = self._read_gray(row['image_path'])
        mask = self._read_mask(row['mask_path'])
        gcam = self._read_gcam(row['gradcam_path'])
        stacked = np.stack([img, mask, gcam], axis=2)  # H,W,3

        if self.transforms:
            t = self.transforms(image=stacked)
            return t['image'], int(row['label'])
        else:
            stacked = stacked.transpose(2,0,1)
            return torch.tensor(stacked, dtype=torch.float32), int(row['label'])

# -------------------------
# Transforms
# -------------------------
def get_train_transforms(img_size=256):
    # If you want to remove ShiftScaleRotate warning, replace with A.Affine(...)
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.12),
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])

def get_valid_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2()
    ])

# -------------------------
# Model helper (conv1 adapt)
# -------------------------
def make_resnet18(num_classes, in_channels=3, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    if in_channels != 3:
        old = model.conv1
        new_conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            old_w = old.weight.data
            # copy existing 3 channels, fill others with mean
            new_w = torch.zeros((old_w.shape[0], in_channels, *old_w.shape[2:]), dtype=old_w.dtype)
            new_w[:, :3, :, :] = old_w
            if in_channels > 3:
                mean_w = old_w.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_w[:, c:c+1, :, :] = mean_w
            new_conv.weight.data = new_w
        model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# -------------------------
# Utils
# -------------------------
def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total

def make_weighted_sampler(df):
    counts = df['label'].value_counts().to_dict()
    class_weights = {k: 1.0 / v for k, v in counts.items()}
    sample_weights = df['label'].map(class_weights).values
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def compute_class_weights_tensor(df, device):
    counts = df['label'].value_counts().sort_index().values.astype(np.float32)
    weights = (counts.sum() / (counts + 1e-9))
    weights = weights / weights.sum()
    return torch.tensor(weights, dtype=torch.float).to(device)

# -------------------------
# Train / Validate
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    loop = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in loop:
        imgs = imgs.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        c, t = accuracy_from_logits(outputs, labels)
        running_correct += c
        running_total += t
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate_collect(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    all_preds = []
    all_labels = []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device).float()
        labels = labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        running_total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / running_total, running_correct / running_total, np.array(all_preds), np.array(all_labels)

# -------------------------
# Final evaluation: save confusion matrix + predictions CSV
# -------------------------
def final_evaluation_and_save(model, val_loader, val_df, device, classes, ckpt_dir):
    model.eval()
    preds_list = []
    labels_list = []
    img_paths = []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="FinalEval", leave=False):
            imgs = imgs.to(device).float()
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            for i in range(len(preds)):
                preds_list.append(int(preds[i]))
                labels_list.append(int(labels[i].item()))
    # metrics
    print("\n=== Classification Report ===")
    print(classification_report(labels_list, preds_list, target_names=classes, digits=4))
    cm = confusion_matrix(labels_list, preds_list)
    print("Confusion matrix:\n", cm)
    # save confusion matrix image
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cm_path = ckpt_dir / "confusion_matrix.png"
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(str(cm_path)); plt.close()
    print("Saved confusion matrix to:", str(cm_path))
    # Save per-image predictions CSV (map using val_df order)
    # Note: val_loader used shuffle=False so order matches val_df
    out_rows = []
    idx = 0
    # re-run to collect confidences per sample (efficient enough)
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device).float()
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            batch = probs.shape[0]
            for i in range(batch):
                row = val_df.iloc[idx]
                out_rows.append({
                    'image_path': row['image_path'],
                    'true_label': int(row['label']),
                    'pred_label': int(preds[i]),
                    'pred_confidence': float(probs[i, preds[i]])
                })
                idx += 1
    pd.DataFrame(out_rows).to_csv(str(ckpt_dir / "val_predictions.csv"), index=False)
    print("Saved per-image predictions to:", str(ckpt_dir / "val_predictions.csv"))

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-csv', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--weighted-sampler', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    train_df = df[df['split']=='train'].reset_index(drop=True)
    val_df = df[df['split']=='val'].reset_index(drop=True)
    print(f"Train samples: {len(train_df)}  Val samples: {len(val_df)}")

    classes = sorted(train_df['class_name'].unique())
    print("Classes:", classes)

    train_ds = MaskGradcamDataset(train_df, transforms=get_train_transforms(args.img_size), img_size=args.img_size)
    val_ds = MaskGradcamDataset(val_df, transforms=get_valid_transforms(args.img_size), img_size=args.img_size)

    if args.weighted_sampler:
        print("Using WeightedRandomSampler for train loader")
        sampler = make_weighted_sampler(train_df)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    num_classes = int(df['label'].max()) + 1
    model = make_resnet18(num_classes=num_classes, in_channels=3, pretrained=True).to(device)

    if args.weighted_sampler:
        class_weights = compute_class_weights_tensor(train_df, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using class weights:", class_weights.cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    scaler = torch.cuda.amp.GradScaler() if (args.use_amp and device == 'cuda') else None

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        ckpt = Path(args.resume)
        if ckpt.exists():
            print("Resuming from", ckpt)
            data = torch.load(str(ckpt), map_location=device)
            if 'model_state_dict' in data:
                model.load_state_dict(data['model_state_dict'])
                optimizer.load_state_dict(data.get('optimizer_state_dict', optimizer.state_dict()))
                start_epoch = data.get('epoch', 0) + 1
                best_val_acc = data.get('val_acc', 0.0)
            else:
                model.load_state_dict(data)
                print("Loaded model state dict.")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs} -----------------------------")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                          use_amp=(scaler is not None), scaler=scaler)
        val_loss, val_acc, _, _ = validate_collect(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch} | Train loss {tr_loss:.4f} acc {tr_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} | time {(time.time()-t0):.1f}s")

        # save epoch checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, str(ckpt_dir / f"resnet18_epoch{epoch}.pth"))

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(ckpt_dir / "best_resnet18.pth"))
            print("Saved best model to", str(ckpt_dir / "best_resnet18.pth"))

    # Final evaluation & saving confusion matrix + predictions
    print("\nRunning final evaluation on validation set...")
    final_evaluation_and_save(model, val_loader, val_df, device, classes, ckpt_dir)
    print("Done. Best val acc:", best_val_acc)

if __name__ == '__main__':
    main()
