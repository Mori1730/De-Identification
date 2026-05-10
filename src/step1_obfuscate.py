"""
step1_obfuscate.py
------------------
Step 1 of HW3: Face de-identification by two methods:
    (A) Pixelization with block size b
    (B) Gaussian blur with kernel size k

For each method we apply several parameter values and save side-by-side
comparison images so the report can show: smaller b / smaller k => better
visual quality, larger value => stronger obfuscation.

Run:
    python src/step1_obfuscate.py
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import load_att_dataset, ensure_dir, DATA_OBF, RESULTS


# Obfuscation functions
def pixelize(img: np.ndarray, b: int) -> np.ndarray:
    """
    Pixelization: replace each b x b block with its mean color.

    We do it the standard way: shrink the image to (W/b, H/b) using AREA
    interpolation (which averages each block), then upscale back with
    NEAREST interpolation (which makes the chunky-block look).

    Parameters
    ----------
    img : np.ndarray         input grayscale image, shape (H, W)
    b   : int                block size in pixels (>=1)

    Returns
    -------
    np.ndarray  same shape and dtype as `img`
    """
    h, w = img.shape[:2]
    # Shrink to one pixel per block (averages each block)
    small = cv2.resize(img, (max(1, w // b), max(1, h // b)),
                       interpolation=cv2.INTER_AREA)
    # Upscale back to original size with NEAREST -> visible chunky blocks
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def gaussian_blur(img: np.ndarray, k: int) -> np.ndarray:
    """
    Gaussian blur: each pixel is replaced by the Gaussian-weighted average
    of its k x k neighborhood. Larger k => stronger blur.

    OpenCV requires k to be odd, so we round up to the next odd number.

    Parameters
    ----------
    img : np.ndarray         input grayscale image, shape (H, W)
    k   : int                kernel size (any positive int; will be made odd)

    Returns
    -------
    np.ndarray  same shape and dtype as `img`
    """
    if k < 1:
        k = 1
    if k % 2 == 0:           # GaussianBlur kernel must be odd
        k += 1
    # sigma=0 means "auto, computed from kernel size"
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)


# Driver: generate obfuscated dataset + a comparison figure for the report
PIX_SIZES   = [4, 8, 16]            # b values to evaluate
BLUR_KERNELS = [15, 45, 99]         # k values to evaluate


def main():
    images, labels = load_att_dataset()
    print(f"Loaded {len(images)} images, {len(set(labels))} people.")

    # 1) Obfuscate the WHOLE dataset and save to disk -- step2 will read these
    for b in PIX_SIZES:
        out_dir = ensure_dir(DATA_OBF / f"pix_b{b}")
        for i, (img, lab) in enumerate(zip(images, labels)):
            cv2.imwrite(str(out_dir / f"{lab:02d}_{i:04d}.png"), pixelize(img, b))
        print(f"  saved pixelized (b={b}) -> {out_dir}")

    for k in BLUR_KERNELS:
        out_dir = ensure_dir(DATA_OBF / f"blur_k{k}")
        for i, (img, lab) in enumerate(zip(images, labels)):
            cv2.imwrite(str(out_dir / f"{lab:02d}_{i:04d}.png"),
                        gaussian_blur(img, k))
        print(f"  saved gaussian blur (k={k}) -> {out_dir}")

    # 2) Build a comparison figure for the report (one face, all parameters)
    sample = images[0]                              # first image of person s1
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    axes[0, 0].imshow(sample, cmap="gray"); axes[0, 0].set_title("(a) original")
    for col, b in enumerate(PIX_SIZES, start=1):
        axes[0, col].imshow(pixelize(sample, b), cmap="gray")
        axes[0, col].set_title(f"pix b={b}")

    axes[1, 0].imshow(sample, cmap="gray"); axes[1, 0].set_title("(a) original")
    for col, k in enumerate(BLUR_KERNELS, start=1):
        axes[1, col].imshow(gaussian_blur(sample, k), cmap="gray")
        axes[1, col].set_title(f"blur k={k}")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()

    out_fig = ensure_dir(RESULTS / "step1_images") / "comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"  saved comparison figure -> {out_fig}")


if __name__ == "__main__":
    main()
