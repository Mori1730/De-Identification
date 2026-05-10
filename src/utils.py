"""
utils.py
--------
Shared helpers used by step1, step2, step3:
    - File / dataset IO for AT&T face images
    - Image quality metrics (MSE, SSIM)
    - Plot helpers
"""

import os
from pathlib import Path
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


# -----------------------------------------------------------------------------
# Paths (resolved relative to the project root, not the cwd)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_OBF = PROJECT_ROOT / "data" / "obfuscated"
RESULTS = PROJECT_ROOT / "results"


def ensure_dir(p: Path) -> Path:
    """Create directory `p` (and parents) if missing. Returns the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# Dataset loading (AT&T / ORL face database)
# -----------------------------------------------------------------------------
def load_att_dataset(root: Path = DATA_RAW):
    """
    Load the AT&T face dataset.

    Expects the layout:
        root/s1/1.pgm, ..., root/s1/10.pgm
        root/s2/1.pgm, ...
        ...
        root/s40/1.pgm, ...

    Returns
    -------
    images : list[np.ndarray]   each grayscale, shape (112, 92)
    labels : list[int]          person id, 0..39
    """
    images, labels = [], []
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Download AT&T faces and unzip into data/raw/ "
            "(should produce s1..s40 subfolders)."
        )

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("s"):
            continue
        # Person id from folder name "s1" -> 0, "s2" -> 1, ...
        person_id = int(person_dir.name[1:]) - 1
        for img_path in sorted(person_dir.glob("*.pgm")):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(person_id)
    return images, labels


# -----------------------------------------------------------------------------
# Quality metrics
# -----------------------------------------------------------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error between two images (same shape, uint8 or float)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """Structural Similarity Index. Higher = more similar (max 1.0)."""
    # data_range = max possible pixel value - min possible (255 for uint8)
    return float(ssim(a, b, data_range=255))
