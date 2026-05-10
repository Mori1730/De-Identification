"""
step2_attack.py
---------------
Step 2 of HW3: train a CNN to re-identify the original person from
de-identified (pixelized / blurred) face images.

This is intentionally a small CNN -- the AT&T dataset is tiny (400 images,
40 classes) so a deep model would overfit and is also unnecessary to make
the homework's point: "even mild obfuscation is broken by a learned model."

Run:
    python src/step2_attack.py
"""

from pathlib import Path
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from utils import load_att_dataset, ensure_dir, RESULTS
from step1_obfuscate import pixelize, gaussian_blur, PIX_SIZES, BLUR_KERNELS


# Reproducibility -- same train/test split every run
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset wrapper: applies an obfuscation function on the fly
class ATTFaceDataset(Dataset):
    """AT&T faces with optional obfuscation applied per item."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform   # callable: img -> img, or None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        # Normalize to [0,1] and add channel dim -> (1, H, W)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        return img, self.labels[idx]


# Tiny CNN -- 3 conv blocks + 2 FC layers
class TinyCNN(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56x46
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 28x23
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 14x11
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 11, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Train + evaluate one configuration (e.g., "pix_b=4")
def train_eval(images, labels, transform, name, epochs=50, batch=64):
    """Train a fresh CNN on `transform`-obfuscated data, return top-1 accuracy."""
    dataset = ATTFaceDataset(images, labels, transform=transform)
    n_test = len(dataset) // 5                 # 80/20 split
    n_train = len(dataset) - n_test
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch)

    model = TinyCNN(num_classes=40).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

    # Evaluate top-1 accuracy on held-out images
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"  {name:>14s}  top-1 = {acc:5.2f}%")
    return acc

def main():
    images, labels = load_att_dataset()
    print(f"Device: {DEVICE}", end="")
    if DEVICE == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" -- WARNING: CUDA not detected, falling back to CPU.")
    print(f"Dataset: {len(images)} images, {len(set(labels))} classes")

    results = {"baseline_random": 100.0 / 40}    # 2.5% chance per class

    # Original (no obfuscation) -- upper bound on attacker accuracy
    results["original"] = train_eval(images, labels, transform=None,
                                     name="original")

    # Pixelization
    for b in PIX_SIZES:
        results[f"pix_b{b}"] = train_eval(
            images, labels,
            transform=lambda im, b=b: pixelize(im, b),
            name=f"pix_b={b}",
        )

    # Gaussian blur
    for k in BLUR_KERNELS:
        results[f"blur_k{k}"] = train_eval(
            images, labels,
            transform=lambda im, k=k: gaussian_blur(im, k),
            name=f"blur_k={k}",
        )

    out_dir = ensure_dir(RESULTS / "step2_metrics")
    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'accuracy.json'}")


if __name__ == "__main__":
    main()
