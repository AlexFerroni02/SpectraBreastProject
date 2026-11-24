# SPECTRA-BREAST — Raman Foundation Models (FP & High-Wavenumber CH)

Modular PyTorch research framework for **Self-Supervised Pre-training** (`SMAE`, `Barlow Twins ResNet-34 1D`) and **Downstream Clinical Evaluation** across **Fingerprint (`FP`)** and **High-Wavenumber (`CH`)** Raman spectral regions.

## Repository Structure

```text
SPECTRA-BREAST/
├── configs/                  # YAML recipes: dataset preprocessing, regions, SSL & downstream runs
├── data/
│   ├── 00_raw/               # Read-only Data Lake (separated into pretrain/ and downstream/)
│   ├── 01_interim/           # Cleaned spectra (post-Despike/SavGol/ALS, pre-crop & pre-norm)
│   └── 02_ready/             # Uniform .npz tensors sliced by region & normalization
├── src/spectrabreast/        # Core Python package (preprocessing, augmentations, models, trainers)
├── scripts/                  # CLI entrypoints for Local, Google Colab & HPC (SLURM)
├── notebooks/                # Lightweight exploratory & visualization notebooks
├── experiments/              # Checkpoints, metrics, and plots organized by experiment run
└── docs/                     # Project deliverables, dataset audit reports & key references
```

## Quick Start

1. **Audit all raw datasets (ranges, spectral counts, preprocessing state)**:
   ```bash
   python scripts/audit_datasets.py
   ```
2. **Prepare standardized datasets for any region & normalization**:
   ```bash
   python scripts/prepare_data.py --region ch --bounds 2800 3050 --points 500 --norm minmax
   ```
3. **Launch SSL Pre-training (`SMAE` or `Barlow Twins`)**:
   ```bash
   python scripts/train_ssl.py --config configs/pretrain/smae_ch.yaml
   ```
