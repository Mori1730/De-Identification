# HW3 — Face De-Identification & Its Attacks and Defenses

## Project structure
```
HW3-DeIdentification/
├── environment.yml          # Anaconda environment definition
├── README.md                # This file
├── data/
│   ├── raw/                 # Put the AT&T face dataset here (40 folders s1..s40)
│   └── obfuscated/          # Auto-generated: pixelated/blurred outputs
├── src/
│   ├── step1_obfuscate.py   # Pixelization + Gaussian blur
│   ├── step2_attack.py      # CNN attacker (re-identification)
│   ├── step3_defense.py     # Differential privacy defense
│   └── utils.py             # Shared helpers (MSE, SSIM, IO)
├── results/
│   ├── step1_images/        # Side-by-side comparison images
│   ├── step2_metrics/       # Accuracy tables / plots
│   └── step3_metrics/       # MSE/SSIM vs epsilon plots
└── report/
    └── Report.docx          # Final report (team responsibility table inside)
```

## Setup (Anaconda)

```bash
# 1. Create the environment from the yml file
conda env create -f environment.yml

# 2. Activate it
conda activate hw3-deid

# 3. Verify
python -c "import cv2, torch, skimage; print('OpenCV', cv2.__version__, '| Torch', torch.__version__)"
```

If you have an NVIDIA GPU, remove the `cpuonly` line in `environment.yml` before creating the env, and conda will install the CUDA build of PyTorch.

## Dataset

Download the AT&T (ORL) face dataset — 40 people × 10 images, ~5 MB:
- https://cam-orl.co.uk/facedatabase.html (or search "AT&T ORL faces")
- Unzip into `data/raw/` so it looks like `data/raw/s1/1.pgm`, `data/raw/s1/2.pgm`, ...

## How to run

```bash
conda activate hw3-deid

# Step 1: Generate obfuscated faces (pixelization + blur at multiple parameters)
python src/step1_obfuscate.py

# Step 2: Train CNN attacker, evaluate re-identification accuracy
python src/step2_attack.py

# Step 3: Apply differential privacy, re-evaluate, plot MSE/SSIM vs epsilon
python src/step3_defense.py
```
