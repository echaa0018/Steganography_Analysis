# Steganography Research

Code and experiments for two connected research papers on image steganography:

1. **Imperceptibility and Payload Capacity Analysis of LSB and DCT Steganography on Digital Images**
   — measures how visually undetectable the embedding is (PSNR / SSIM / MSE) and how much data each technique can carry (payload capacity in bits and bits-per-pixel).

2. **Chi-Square and Histogram-Based Steganalysis Detection of LSB and DCT Steganographic Images**
   — attacks the stego images with statistical steganalysis (chi-square test, histogram analysis) and reports detection performance.

## Project structure

```
steganography-research/
├── data/cover_images/        # input cover images (you supply these)
├── src/
│   ├── lsb.py                # LSB embed / extract
│   ├── dct.py                # DCT embed / extract
│   ├── metrics.py            # PSNR, SSIM, MSE, payload capacity (paper 1)
│   ├── steganalysis.py       # chi-square test, histogram analysis (paper 2)
│   └── utils.py              # shared image I/O + helpers
├── stego_images/lsb/         # generated LSB stego images (git-ignored)
├── stego_images/dct/         # generated DCT stego images (git-ignored)
├── experiments/
│   ├── run_paper1.py         # imperceptibility + capacity experiment
│   └── run_paper2.py         # steganalysis detection experiment
├── results/                  # generated tables + figures (git-ignored)
└── notebooks/                # exploratory notebooks
```

## Setup

```bash
python -m venv venv
# Windows (PowerShell): venv\Scripts\Activate.ps1
# macOS / Linux:        source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## How to run

```bash
# Paper 1 — imperceptibility & payload capacity
python experiments/run_paper1.py

# Paper 2 — steganalysis detection
python experiments/run_paper2.py
```

Both scripts read cover images from `data/cover_images/`, write stego images to `stego_images/` and write CSV tables and figures to `results/`. If no images are found, the scripts uses a small synthetic test image so the script can be run without custom files.

## Reproducibility

Random operations are seeded so experiment runs are reproducible.
