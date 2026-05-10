"""
step3_defense.py
----------------
Step 3 of HW3: defend against the Step 2 attacker by adding *differential
privacy* noise to the obfuscation.

Approach (matches the TPDP2019 / Fan paper referenced in the homework):
    DP-Pix(img, b, eps, m):
        1. Pixelize the image with block size b   (each block = mean of b*b pixels)
        2. Add Laplace noise to each block with scale = (255 * m / b^2) / eps
           - sensitivity for the mean of b*b pixels in [0,255] is 255*m/b^2,
             where m = number of pixels that may differ between two
             neighboring images.
           - We default to m = b^2 (block-level privacy: a whole block is
             allowed to change between neighbors).  This is what gives a
             non-trivial amount of noise; m=1 (per-pixel) gives noise of
             magnitude < 1 grayscale level for b>=16, which is invisible
             to the attacker.
           - smaller eps  ->  larger noise  ->  stronger privacy

    DP-Blur(img, k, eps, m): same idea on the Gaussian-smoothed image.

IMPORTANT design note:
    The noise must be sampled ONCE per image and then frozen.  An earlier
    version of this file passed `dp_pixelize` as the DataLoader's transform,
    which re-sampled fresh Laplace noise on every fetch -- effectively
    training the attacker with noise-injection data augmentation, which
    *increased* its accuracy.  We now precompute the noisy dataset and pass
    it in directly with transform=None.

For each (method, eps) we measure:
    * MSE  vs original  (utility -- smaller is better)
    * SSIM vs original  (utility -- larger  is better)
    * CNN re-identification accuracy (privacy -- smaller is better),
      averaged over N_RUNS trainings to smooth out seed noise.

Run:
    python src/step3_defense.py
"""

from pathlib import Path
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import load_att_dataset, ensure_dir, RESULTS, mse, ssim_score
from step1_obfuscate import pixelize, gaussian_blur
from step2_attack import train_eval


# Reproducibility -- step2_attack already seeds at import time, but we want
# the noise draws here to be deterministic too.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Differentially-private obfuscation
def dp_pixelize(img: np.ndarray, b: int, eps: float,
                rng: np.random.Generator,
                m: int | None = None) -> np.ndarray:
    """Pixelization + Laplace noise (per the Fan TPDP2019 formulation).

    Sensitivity = 255 * m / b**2.  Default m = b**2 protects an entire block
    against arbitrary change between neighboring images.
    """
    if m is None:
        m = b * b
    base = pixelize(img, b).astype(np.float32)
    sensitivity = 255.0 * m / (b * b)
    scale = sensitivity / eps                        # Laplace scale parameter
    noise = rng.laplace(loc=0.0, scale=scale, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def dp_blur(img: np.ndarray, k: int, eps: float,
            rng: np.random.Generator,
            m: int | None = None) -> np.ndarray:
    """Gaussian blur + Laplace noise (parallel to dp_pixelize)."""
    if m is None:
        m = k * k
    base = gaussian_blur(img, k).astype(np.float32)
    sensitivity = 255.0 * m / (k * k)
    scale = sensitivity / eps
    noise = rng.laplace(loc=0.0, scale=scale, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# Driver
EPSILONS = [0.1, 0.5, 1.0]      # smaller = more privacy
PIX_B    = 16                   # match the table on slide 9 of the homework
BLUR_K   = 99
N_RUNS   = 3                    # average attacker accuracy over this many trainings


def utility_metrics(images_orig, images_obf):
    """Average MSE & SSIM between two pre-computed image lists."""
    mses  = [mse(a, b)        for a, b in zip(images_orig, images_obf)]
    ssims = [ssim_score(a, b) for a, b in zip(images_orig, images_obf)]
    return float(np.mean(mses)), float(np.mean(ssims))


def repeated_eval(images, labels, name, n_runs=N_RUNS):
    """Train n_runs models on the SAME (frozen) images and average top-1."""
    accs = []
    for r in range(n_runs):
        a = train_eval(images, labels, transform=None,
                       name=f"{name} [{r+1}/{n_runs}]")
        accs.append(a)
    mean_a = float(np.mean(accs))
    print(f"  {name:>22s}  mean top-1 = {mean_a:5.2f}%   runs={[f'{x:.2f}' for x in accs]}")
    return mean_a


def main():
    images, labels = load_att_dataset()
    out_dir = ensure_dir(RESULTS / "step3_metrics")

    summary = {"pix": {}, "blur": {}}

    # ---- Pixelization (NP-Pix baseline + DP-Pix at each epsilon) -----------
    print(f"\n=== DP-Pix (b={PIX_B}) ===")
    np_pix_imgs = [pixelize(im, PIX_B) for im in images]
    np_mse, np_ssim = utility_metrics(images, np_pix_imgs)
    np_acc = repeated_eval(np_pix_imgs, labels, name=f"NP-Pix b={PIX_B}")
    summary["pix"]["NP"] = {"mse": np_mse, "ssim": np_ssim, "acc": np_acc}

    for eps in EPSILONS:
        # Per-eps RNG so different eps draws are independent but reproducible.
        rng = np.random.default_rng(SEED + int(round(eps * 1000)))
        dp_pix_imgs = [dp_pixelize(im, PIX_B, eps, rng) for im in images]
        m_, s_ = utility_metrics(images, dp_pix_imgs)
        a_ = repeated_eval(dp_pix_imgs, labels, name=f"DP-Pix eps={eps}")
        summary["pix"][f"DP_eps{eps}"] = {"mse": m_, "ssim": s_, "acc": a_}

    # ---- Gaussian blur (NP-Blur baseline + DP-Blur) ------------------------
    print(f"\n=== DP-Blur (k={BLUR_K}) ===")
    np_blur_imgs = [gaussian_blur(im, BLUR_K) for im in images]
    np_mse, np_ssim = utility_metrics(images, np_blur_imgs)
    np_acc = repeated_eval(np_blur_imgs, labels, name=f"NP-Blur k={BLUR_K}")
    summary["blur"]["NP"] = {"mse": np_mse, "ssim": np_ssim, "acc": np_acc}

    for eps in EPSILONS:
        rng = np.random.default_rng(SEED + int(round(eps * 1000)) + 99_991)
        dp_blur_imgs = [dp_blur(im, BLUR_K, eps, rng) for im in images]
        m_, s_ = utility_metrics(images, dp_blur_imgs)
        a_ = repeated_eval(dp_blur_imgs, labels, name=f"DP-Blur eps={eps}")
        summary["blur"][f"DP_eps{eps}"] = {"mse": m_, "ssim": s_, "acc": a_}

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved metrics -> {out_dir / 'summary.json'}")

    # ---- Plots: MSE & SSIM vs epsilon (slide 10) ---
    for method, label in [("pix", "DP-Pix"), ("blur", "DP-Blur")]:
        eps_vals = EPSILONS
        np_m  = summary[method]["NP"]["mse"]
        np_s  = summary[method]["NP"]["ssim"]
        dp_m  = [summary[method][f"DP_eps{e}"]["mse"]  for e in eps_vals]
        dp_s  = [summary[method][f"DP_eps{e}"]["ssim"] for e in eps_vals]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(eps_vals, dp_m, "o-", label=label)
        ax1.axhline(np_m, ls="--", label=f"NP ({label.split('-')[1]})")
        ax1.set_xlabel("epsilon"); ax1.set_ylabel("MSE")
        ax1.set_title(f"MSE vs eps ({label})"); ax1.legend()

        ax2.plot(eps_vals, dp_s, "o-", label=label)
        ax2.axhline(np_s, ls="--", label=f"NP ({label.split('-')[1]})")
        ax2.set_xlabel("epsilon"); ax2.set_ylabel("SSIM")
        ax2.set_title(f"SSIM vs eps ({label})"); ax2.legend()

        plt.tight_layout()
        plt.savefig(out_dir / f"utility_{method}.png", dpi=150)
        plt.close()
        print(f"  plot saved -> {out_dir / f'utility_{method}.png'}")

    # ---- Privacy plot: attacker accuracy vs epsilon ------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label, color in [("pix", "DP-Pix", "tab:blue"),
                                 ("blur", "DP-Blur", "tab:orange")]:
        accs = [summary[method][f"DP_eps{e}"]["acc"] for e in EPSILONS]
        ax.plot(EPSILONS, accs, "o-", color=color, label=label)
        ax.axhline(summary[method]["NP"]["acc"], ls="--", color=color,
                   alpha=0.6, label=f"NP-{label.split('-')[1]}")
    ax.set_xlabel("epsilon (smaller = more privacy)")
    ax.set_ylabel("Attacker top-1 accuracy (%)")
    ax.set_title("Re-identification accuracy vs epsilon")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "privacy_vs_eps.png", dpi=150)
    plt.close()
    print(f"  plot saved -> {out_dir / 'privacy_vs_eps.png'}")


if __name__ == "__main__":
    main()
