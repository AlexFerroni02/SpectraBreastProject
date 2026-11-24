"""
High-level dataset loaders, statistical accumulators, and publication-grade
plotting utilities for Step 1 (`notebooks/01_raw_datasets_atlas.ipynb`).

Keeping the heavy file-parsing and matplotlib layout logic in this module allows
the Jupyter notebook cells to remain concise, modular, and easy to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

__all__ = [
    "accumulate_h5_stats",
    "compute_matrix_stats",
    "add_spectral_bands",
    "plot_single_profile",
    "load_fp_raw_cohorts",
    "load_hw_raw_cohorts",
    "load_fr_raw_cohorts",
    "plot_coverage_gantt",
    "plot_fp_raw_grid",
    "plot_hw_raw_grid",
    "plot_fr_raw_grid",
]


def accumulate_h5_stats(h5_files: list[Path]) -> dict[str, Any]:
    """Compute exact global mean, std, min, max across a list of FAIRaman .h5 maps."""
    total_n = 0
    sum_x: np.ndarray | None = None
    sum_x2: np.ndarray | None = None
    g_min = np.inf
    g_max = -np.inf
    wn_ref: np.ndarray | None = None

    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            wn = np.array(f["/ENTRY/data/raman_shift"], dtype=np.float64).flatten()
            intens = np.array(f["/ENTRY/data/intensity"], dtype=np.float64)
            flat = intens.reshape(-1, wn.size)
            if wn[0] > wn[-1]:
                wn = wn[::-1]
                flat = flat[:, ::-1]
            valid = ~np.isnan(flat).any(axis=1)
            flat = flat[valid]
            if wn_ref is None:
                wn_ref = wn
                sum_x = np.zeros_like(wn_ref, dtype=np.float64)
                sum_x2 = np.zeros_like(wn_ref, dtype=np.float64)
            elif flat.shape[1] != wn_ref.size or abs(wn[0] - wn_ref[0]) > 2.0:
                flat = np.array([np.interp(wn_ref, wn, row) for row in flat], dtype=np.float64)

            total_n += flat.shape[0]
            sum_x += flat.sum(axis=0)
            sum_x2 += (flat * flat).sum(axis=0)
            g_min = min(g_min, float(flat.min()))
            g_max = max(g_max, float(flat.max()))

    assert wn_ref is not None and sum_x is not None and sum_x2 is not None
    mean = sum_x / total_n
    var = np.maximum(sum_x2 / total_n - mean * mean, 0.0)
    return {
        "wn": wn_ref,
        "mean": mean,
        "std": np.sqrt(var),
        "n_spectra": int(total_n),
        "n_points": int(wn_ref.size),
        "min": float(g_min),
        "max": float(g_max),
    }


def compute_matrix_stats(wn: np.ndarray, mat: np.ndarray) -> dict[str, Any]:
    """Compute exact global mean, std, min, max across all valid non-NaN rows."""
    mat64 = np.asarray(mat, dtype=np.float64)
    wn64 = np.asarray(wn, dtype=np.float64).flatten()
    if wn64[0] > wn64[-1]:
        wn64 = wn64[::-1]
        mat64 = mat64[:, ::-1]
    valid = ~np.isnan(mat64).any(axis=1) & (np.ptp(mat64, axis=1) > 0)
    vmat = mat64[valid]
    return {
        "wn": wn64,
        "mean": vmat.mean(axis=0),
        "std": vmat.std(axis=0),
        "n_spectra": int(vmat.shape[0]),
        "n_points": int(wn64.size),
        "min": float(vmat.min()),
        "max": float(vmat.max()),
    }


def add_spectral_bands(ax: plt.Axes, wn_min: float, wn_max: float) -> None:
    """Highlight FP (650-1750), CH Core (2800-3045), and HW Extended (2765-3250) windows."""
    if wn_min <= 1750 and wn_max >= 650:
        lo, hi = max(650.0, wn_min), min(1750.0, wn_max)
        if hi > lo:
            ax.axvspan(lo, hi, color="#0072B2", alpha=0.06, zorder=0)
    if wn_min <= 3250 and wn_max >= 2765:
        lo_hw, hi_hw = max(2765.0, wn_min), min(3250.0, wn_max)
        if hi_hw > lo_hw:
            ax.axvspan(lo_hw, hi_hw, color="#E69F00", alpha=0.08, zorder=0)
    if wn_min <= 3045 and wn_max >= 2800:
        lo_ch, hi_ch = max(2800.0, wn_min), min(3045.0, wn_max)
        if hi_ch > lo_ch:
            ax.axvspan(lo_ch, hi_ch, color="#D55E00", alpha=0.13, zorder=0)


def plot_single_profile(
    ax: plt.Axes,
    stats: dict[str, Any],
    title: str,
    subtitle: str,
    state_badge: str,
    color: str,
    ylabel: str = "Intensity (a.u.)",
) -> None:
    """Render a single Global Mean +/- 1 SD panel with +30% top headroom for zero overlap."""
    wn, mean, std = stats["wn"], stats["mean"], stats["std"]
    n_sp, n_pts = stats["n_spectra"], stats["n_points"]

    ax.plot(wn, mean, color=color, lw=1.85, label=f"Mean ($n={n_sp:,}$)")
    ax.fill_between(wn, mean - std, mean + std, color=color, alpha=0.24, label="$\\pm 1$ SD ($\\sigma$)")
    add_spectral_bands(ax, float(wn[0]), float(wn[-1]))

    y_lo, y_hi = float(np.min(mean - std)), float(np.max(mean + std))
    span = max(y_hi - y_lo, 1e-9)
    ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.30 * span)
    ax.set_xlim(float(wn[0]), float(wn[-1]))
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.4, fontweight="bold", pad=5)
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.6)
    ax.set_ylabel(ylabel, fontsize=8.6)

    info = (
        f"State: {state_badge}\n"
        f"Range: {wn[0]:.1f}–{wn[-1]:.1f} cm$^{{-1}}$ ({n_pts} pts)\n"
        f"Y-Range: [{stats['min']:.3g}, {stats['max']:.3g}]"
    )
    ax.text(
        0.02,
        0.96,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.6,
        bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": color, "alpha": 0.93},
    )
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)


def load_fp_raw_cohorts(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all 8 Fingerprint (FP) cohorts from `data/00_raw/{pretrain,downstream}/FP/`."""
    deepr_dir = raw_dir / "pretrain" / "FP" / "deepr"
    deepr_wn = np.loadtxt(deepr_dir / "axis.txt").flatten()
    tr_in = scipy.io.loadmat(str(deepr_dir / "Train_Inputs.mat"))["Train_Inputs"]
    te_in = scipy.io.loadmat(str(deepr_dir / "Test_Inputs.mat"))["Test_Inputs"]
    tr_out = scipy.io.loadmat(str(deepr_dir / "Train_Outputs.mat"))["Train_Outputs"]
    te_out = scipy.io.loadmat(str(deepr_dir / "Test_Outputs.mat"))["Test_Outputs"]

    rmae_dir = raw_dir / "pretrain" / "FP" / "raman_mae"
    m_cells = scipy.io.loadmat(str(next(rmae_dir.glob("*231CTCLM*.mat"))))
    cell_mask = m_cells["labels_table_status_cell1_bg0"].flatten() == 1
    m_twist = scipy.io.loadmat(str(next(rmae_dir.glob("*TWIST*.mat"))))
    m_imm = scipy.io.loadmat(str(next(rmae_dir.glob("*immuno*.mat"))))

    sbfp_files = sorted([p for p in (raw_dir / "pretrain" / "FP" / "spectrabreast_fp").glob("*.h5") if not p.name.startswith("._")])

    df_ibd = pd.read_excel(raw_dir / "downstream" / "FP" / "ibd" / "Dataset Raman IBD Final ICSM.xlsx", sheet_name="Foglio2")
    ibd_cols = [c for c in df_ibd.columns if isinstance(c, (int, float))]

    bac_dir = raw_dir / "downstream" / "FP" / "bacteria_id"
    bac_wn = np.load(bac_dir / "wavenumbers.npy").flatten()
    bac_sp = np.vstack([np.load(bac_dir / "X_finetune.npy"), np.load(bac_dir / "X_test.npy")])

    return {
        "deepr_in": compute_matrix_stats(deepr_wn, np.vstack([tr_in, te_in])),
        "deepr_out": compute_matrix_stats(deepr_wn, np.vstack([tr_out, te_out])),
        "cells": compute_matrix_stats(m_cells["wn"].flatten(), np.nan_to_num(m_cells["All_filt"][cell_mask], nan=0.0)),
        "twist": compute_matrix_stats(m_twist["x_red"].flatten(), m_twist["All_filt"]),
        "immuno": compute_matrix_stats(m_imm["x_red"].flatten(), m_imm["All_filt"]),
        "spectrabreast_fp": accumulate_h5_stats(sbfp_files),
        "ibd": compute_matrix_stats(np.array(ibd_cols, dtype=np.float64), df_ibd[ibd_cols].to_numpy(dtype=np.float64)),
        "bacteria_id": compute_matrix_stats(bac_wn, bac_sp),
    }


def load_hw_raw_cohorts(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all 9 High-Wavenumber (HW) cohorts from `data/00_raw/{pretrain,downstream}/HW/`."""
    sbhw_files = sorted([p for p in (raw_dir / "pretrain" / "HW" / "spectrabreast_hw").glob("*.h5") if not p.name.startswith("._")])
    g1_files, g2_files, als_files = [], [], []
    for fp in sbhw_files:
        with h5py.File(fp, "r") as f:
            wn_0 = float(np.array(f["/ENTRY/data/raman_shift"]).flatten().min())
        if "SB015HT 2" in fp.name:
            als_files.append(fp)
        elif wn_0 < 2600.0:
            g1_files.append(fp)
        else:
            g2_files.append(fp)

    pdo_files = sorted([p for p in (raw_dir / "pretrain" / "HW" / "pdo_hw_maps").glob("*.h5") if not p.name.startswith("._")])
    meat_files = sorted([p for p in (raw_dir / "pretrain" / "HW" / "riverd_meat_hw").glob("*.h5") if not p.name.startswith("._")])

    oral_dir = raw_dir / "downstream" / "HW" / "riverd_oral_hw"
    oral_wn = pd.read_excel(oral_dir / "RamanMetaData.xlsx", sheet_name="wvn", header=None).to_numpy(dtype=np.float64).flatten()

    return {
        "sbhw_g1": accumulate_h5_stats(g1_files),
        "sbhw_g2": accumulate_h5_stats(g2_files),
        "sbhw_als": accumulate_h5_stats(als_files),
        "pdo_raw": accumulate_h5_stats([p for p in pdo_files if "PDO2" not in p.name]),
        "pdo_als": accumulate_h5_stats([p for p in pdo_files if "PDO2" in p.name]),
        "meat": accumulate_h5_stats(meat_files),
        "oral_c": compute_matrix_stats(oral_wn, pd.read_excel(oral_dir / "RamanData_c.xlsx", header=None).to_numpy(dtype=np.float64).T),
        "oral_pc": compute_matrix_stats(oral_wn, pd.read_excel(oral_dir / "RamanData_pc.xlsx", header=None).to_numpy(dtype=np.float64).T),
        "oral_bc": compute_matrix_stats(oral_wn, pd.read_excel(oral_dir / "RamanData_bc.xlsx", header=None).to_numpy(dtype=np.float64).T),
    }


def load_fr_raw_cohorts(raw_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all 3 Full-Range (FR) cohorts from `data/00_raw/{pretrain,downstream}/FR/`."""
    tr_dir = raw_dir / "pretrain" / "FR" / "trophy_jena"
    meta_tr = pd.read_csv(tr_dir / "Pixelwise metadata.csv")
    fg_mask = {str(k).strip(): (g["fg_mask"].to_numpy() == 1) & (g["is_burnt"].to_numpy() == 0) for k, g in meta_tr.groupby("Map name")}

    def _read_tr(sub: str) -> dict[str, Any]:
        rows, wn_ref = [], None
        for mf in sorted((tr_dir / "Raman maps" / sub).glob("*.mat")):
            d = scipy.io.loadmat(str(mf))
            wn_ref = d["x"].flatten().astype(np.float64)
            cube = d["data"].astype(np.float64).reshape(-1, wn_ref.size)
            if mf.stem.strip() in fg_mask:
                cube = cube[fg_mask[mf.stem.strip()]]
            rows.append(cube)
        return compute_matrix_stats(wn_ref, np.vstack(rows))

    df_rz = pd.read_csv(raw_dir / "downstream" / "FR" / "dataset_renzo" / "def_dataset_final.csv", low_memory=False)
    wn_rz = df_rz.iloc[7:, 0].to_numpy(dtype=np.float64)
    sp_rz = df_rz.iloc[7:, 1:].to_numpy(dtype=np.float64).T

    return {
        "trophy_raw": _read_tr("Raw"),
        "trophy_prep": _read_tr("Preprocessed"),
        "renzo": compute_matrix_stats(wn_rz, sp_rz),
    }


def plot_coverage_gantt(save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 1: Global Spectral Coverage across all 11 Spontaneous Raman datasets."""
    bars = [
        ("ibd (Downstream FP, 228 Patients)", 403.9, 1754.3, 228, "downstream/FP", "#56B4E9"),
        ("bacteria_id (Downstream FP, 3k+3k / 78.5k)", 382.0, 1792.4, 6000, "downstream/FP", "#56B4E9"),
        ("spectrabreast_fp (Pretrain FP, 4 Maps)", 660.2, 1759.4, 52992, "pretrain/FP", "#0072B2"),
        ("raman_mae: TWIST (Pretrain FP)", 650.9, 1749.7, 16496, "pretrain/FP", "#0072B2"),
        ("raman_mae: IMMUNO (Pretrain FP)", 597.3, 1798.9, 11099, "pretrain/FP", "#0072B2"),
        ("raman_mae: CELLS (Pretrain FP)", 600.5, 1963.2, 15429, "pretrain/FP", "#0072B2"),
        ("deepr (Pretrain FP, Paired 0.1s/1.0s)", 498.3, 1799.9, 172312, "pretrain/FP", "#0072B2"),
        ("dataset_renzo (Downstream FR, 47 Mice)", 380.0, 3049.0, 3200, "downstream/FR", "#66C2A5"),
        ("trophy_jena [Preprocessed] (Pretrain FR)", 648.7, 3050.4, 115913, "pretrain/FR", "#009E73"),
        ("trophy_jena [Raw] (Pretrain FR)", 200.0, 3200.0, 115913, "pretrain/FR", "#009E73"),
        ("riverd_oral_hw (Downstream HW, 186 Samples)", 2600.0, 4000.0, 8422, "downstream/HW", "#E69F00"),
        ("riverd_meat_hw (Pretrain HW, 60 Maps)", 2520.0, 4000.0, 1200, "pretrain/HW", "#D55E00"),
        ("pdo_hw_maps (Pretrain HW, 6 Maps)", 2603.3, 3915.4, 194444, "pretrain/HW", "#D55E00"),
        ("spectrabreast_hw Grating 2 (Pretrain HW, 8 Maps)", 2763.9, 3487.6, 81991, "pretrain/HW", "#D55E00"),
        ("spectrabreast_hw Grating 1 (Pretrain HW, 7 Maps)", 2488.9, 3258.9, 54810, "pretrain/HW", "#D55E00"),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    for i, (_, w0, w1, n_sp, folder, col) in enumerate(bars):
        ax.barh(i, w1 - w0, left=w0, height=0.64, color=col, edgecolor="black", linewidth=0.6, alpha=0.88)
        ax.text(w1 + 40, i, f"[{w0:.0f}–{w1:.0f} cm⁻¹]  •  {n_sp:,} spectra  ({folder})", va="center", ha="left", fontsize=8.5, fontweight="bold")
    ax.axvspan(650, 1750, color="#0072B2", alpha=0.10, label="FP Target Window (650–1750 cm$^{-1}$)")
    ax.axvspan(2615, 3900, color="#CC79A7", alpha=0.07, label="HW Broad C–H + O–H Window (2615–3900 cm$^{-1}$)")
    ax.axvspan(2765, 3250, color="#E69F00", alpha=0.14, label="HW Extended Overlap (2765–3250 cm$^{-1}$)")
    ax.axvspan(2800, 3045, color="#D55E00", alpha=0.22, label="CH Core Window (2800–3045 cm$^{-1}$)")
    ax.set_yticks(np.arange(len(bars)))
    ax.set_yticklabels([b[0] for b in bars], fontsize=9.2, fontweight="bold")
    ax.set_xlim(0, 5350)
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=11, fontweight="bold")
    ax.set_title("SPECTRA-BREAST Data Lake — Global Spectral Coverage Across All 11 Spontaneous Raman Datasets (FP, FR, HW)", fontsize=12.2, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8.8, framealpha=0.95)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig


def plot_fp_raw_grid(fp_cohorts: dict[str, dict[str, Any]], save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 2: 4x2 grid of all 8 Fingerprint (FP) cohorts."""
    fig, axes = plt.subplots(4, 2, figsize=(16.5, 15.0))
    panels = [
        (axes[0, 0], fp_cohorts["deepr_in"], "1. pretrain/FP/deepr (Inputs 0.1s)", "Low-SNR Acquisition (Train + Test Inputs)", "Scaled / Low-SNR", "#0072B2", "Normalized Intensity"),
        (axes[0, 1], fp_cohorts["deepr_out"], "2. pretrain/FP/deepr (Outputs 1.0s)", "High-SNR Clean Reference (Train + Test Outputs)", "Pre-cleaned / Scaled", "#005282", "Normalized Intensity"),
        (axes[1, 0], fp_cohorts["cells"], "3. pretrain/FP/raman_mae — CELLS", "Breast Cancer Cell Lines (15,429 Valid Cell Spectra)", "Baseline-Subtracted + L2 Norm", "#009E73", "L2-Normalized Intensity"),
        (axes[1, 1], fp_cohorts["twist"], "4. pretrain/FP/raman_mae — TWIST", "TWIST1 EMT Mammary Epithelial Cohort (16,496 Spectra)", "Baseline-Subtracted + L2 Norm", "#009E73", "L2-Normalized Intensity"),
        (axes[2, 0], fp_cohorts["immuno"], "5. pretrain/FP/raman_mae — IMMUNO", "Lymphocyte Immune Subtypes (B / T CD4+ / T CD8+, 11,099 Spectra)", "Baseline-Subtracted + L2 Norm", "#009E73", "L2-Normalized Intensity"),
        (axes[2, 1], fp_cohorts["spectrabreast_fp"], "6. pretrain/FP/spectrabreast_fp", "4 Breast Tissue Maps (SB001TT–SB004TT, 785 nm, 52,992 Spectra)", "RAW CCD Counts (Unprocessed)", "#D55E00", "Detector Counts"),
        (axes[3, 0], fp_cohorts["ibd"], "7. downstream/FP/ibd (ICSM Clinical)", "228 Patient-Averaged Spectra (77 HC vs 151 IBD)", "Baseline-Corrected", "#CC79A7", "Intensity (a.u.)"),
        (axes[3, 1], fp_cohorts["bacteria_id"], "8. downstream/FP/bacteria_id (Active Split)", "Fine-Tune (3,000) + Test (3,000) across 30 Bacterial Isolates", "Pre-cleaned + Scaled [0, 1]", "#56B4E9", "Scaled Intensity [0, 1]"),
    ]
    for ax, st, t, sub, bdg, col, ylab in panels:
        plot_single_profile(ax, st, t, sub, bdg, col, ylab)
    fig.suptitle("Folder FP (Fingerprint: pretrain/FP & downstream/FP) — Initial Raw Dataset Profiles (Global Mean ± 1 SD)", fontsize=13.5, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig


def plot_hw_raw_grid(hw_cohorts: dict[str, dict[str, Any]], save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 3: 3x3 grid of all 9 High-Wavenumber (HW) cohorts."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 13.0))
    panels = [
        (axes[0, 0], hw_cohorts["sbhw_g1"], "1. pretrain/HW/spectrabreast_hw (Grating 1)", "7 RAW Breast Tissue Maps (2488.9–3258.9 cm⁻¹, 785 nm)", "RAW CCD Counts (Unprocessed)", "#D55E00", "Detector Counts"),
        (axes[0, 1], hw_cohorts["sbhw_g2"], "2. pretrain/HW/spectrabreast_hw (Grating 2)", "7 RAW Breast Tissue Maps (2763.9–3487.6 cm⁻¹, 785 nm)", "RAW CCD Counts (Unprocessed)", "#E69F00", "Detector Counts"),
        (axes[0, 2], hw_cohorts["sbhw_als"], "3. pretrain/HW/spectrabreast_hw (SB015HT 2)", "1 Pre-Cleaned Breast Map (2765.0–3487.6 cm⁻¹, 785 nm)", "Post-ALS Baseline Subtracted", "#0072B2", "Baseline-Subtracted Counts"),
        (axes[1, 0], hw_cohorts["pdo_raw"], "4. pretrain/HW/pdo_hw_maps (5 RAW Maps)", "Patient-Derived Organoids (2603.3–3915.4 cm⁻¹, 532 nm)", "RAW CCD Counts (Unprocessed)", "#D55E00", "Detector Counts"),
        (axes[1, 1], hw_cohorts["pdo_als"], "5. pretrain/HW/pdo_hw_maps (CMA64 PDO2)", "1 Confocal Organoid Map (2614.7–3915.4 cm⁻¹, 532 nm)", "Post-ALS Baseline Subtracted", "#0072B2", "Baseline-Subtracted Counts"),
        (axes[1, 2], hw_cohorts["meat"], "6. pretrain/HW/riverd_meat_hw (60 Maps)", "Reference Meat Tissue Maps (2520.0–4000.0 cm⁻¹, 671 nm)", "RAW Detector Counts", "#009E73", "Detector Counts"),
        (axes[2, 0], hw_cohorts["oral_c"], "7. downstream/HW/riverd_oral_hw (RamanData_c)", "Stage 1: Wavenumber/Instrument Calibrated (186 Samples)", "Calibrated (Raw Background)", "#CC79A7", "Intensity (a.u.)"),
        (axes[2, 1], hw_cohorts["oral_pc"], "8. downstream/HW/riverd_oral_hw (RamanData_pc)", "Stage 2: Fiber-Optic Probe Corrected (186 Samples)", "Probe Background Subtracted", "#56B4E9", "Intensity (a.u.)"),
        (axes[2, 2], hw_cohorts["oral_bc"], "9. downstream/HW/riverd_oral_hw (RamanData_bc)", "Stage 3: Baseline Corrected (5,899 Normal / 2,523 Tumor)", "Baseline-Corrected (Clean)", "#009E73", "Intensity (a.u.)"),
    ]
    for ax, st, t, sub, bdg, col, ylab in panels:
        plot_single_profile(ax, st, t, sub, bdg, col, ylab)
    fig.suptitle("Folder HW (High-Wavenumber: pretrain/HW & downstream/HW) — Initial Raw Dataset Profiles (Global Mean ± 1 SD)", fontsize=13.5, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig


def plot_fr_raw_grid(fr_cohorts: dict[str, dict[str, Any]], save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 4: 3x3 grid of Full-Range (FR) cohorts with FP and CH/HW regional zooms."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 12.2))
    rows = [
        (0, fr_cohorts["trophy_raw"], "1. pretrain/FR/trophy_jena (Raw/ Folder)", "10 Breast Maps (2 Patients, 200–3200 cm⁻¹, 1024 pts)", "RAW Counts (Un-normalized)", "#D55E00", (650.0, 1750.0), (2750.0, 3200.0)),
        (1, fr_cohorts["trophy_prep"], "2. pretrain/FR/trophy_jena (Preprocessed/ Folder)", "10 Breast Maps (2 Patients, 648.7–3050.4 cm⁻¹, 820 pts)", "Baseline-Subtracted + Normalized", "#009E73", (650.0, 1750.0), (2750.0, 3050.4)),
        (2, fr_cohorts["renzo"], "3. downstream/FR/dataset_renzo (Murine Bone)", "47 Mice Samples (3,200 Spectra, 380–3049 cm⁻¹, 2670 pts)", "Baseline-Subtracted + Normalized", "#0072B2", (650.0, 1750.0), (2750.0, 3049.0)),
    ]
    for r_idx, st, name, sub, bdg, col, fp_win, hw_win in rows:
        plot_single_profile(axes[r_idx, 0], st, f"{name} — [A] Full Range", sub, bdg, col)
        wn = st["wn"]
        m_fp = (wn >= fp_win[0]) & (wn <= fp_win[1])
        st_fp = {**st, "wn": wn[m_fp], "mean": st["mean"][m_fp], "std": st["std"][m_fp], "min": float((st["mean"][m_fp] - st["std"][m_fp]).min()), "max": float((st["mean"][m_fp] + st["std"][m_fp]).max())}
        plot_single_profile(axes[r_idx, 1], st_fp, f"[B] Zoom FP Window ({fp_win[0]:.0f}–{fp_win[1]:.0f} cm⁻¹)", f"{name.split(' (')[0]} — Fingerprint Region", bdg, col)
        m_hw = (wn >= hw_win[0]) & (wn <= hw_win[1])
        st_hw = {**st, "wn": wn[m_hw], "mean": st["mean"][m_hw], "std": st["std"][m_hw], "min": float((st["mean"][m_hw] - st["std"][m_hw]).min()), "max": float((st["mean"][m_hw] + st["std"][m_hw]).max())}
        plot_single_profile(axes[r_idx, 2], st_hw, f"[C] Zoom CH/HW Window ({hw_win[0]:.0f}–{hw_win[1]:.0f} cm⁻¹)", f"{name.split(' (')[0]} — High-Wavenumber Region", bdg, col)
    fig.suptitle("Folder FR (Full-Range: pretrain/FR & downstream/FR) — Full Spectrum vs FP Zoom vs CH/HW Zoom (Global Mean ± 1 SD)", fontsize=13.5, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig
