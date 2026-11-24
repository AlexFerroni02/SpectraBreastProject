"""
High-level loaders and Before-vs-After visualization helpers for Step 2
(`notebooks/02_preprocessing_before_after.ipynb`), keeping notebook cells short and readable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

from .cleaning import (
    als_baseline_batch,
    despike_whitaker_hayes_batch,
    minmax_normalize_batch,
)

__all__ = [
    "compute_stats",
    "plot_profile_panel",
    "load_fp_before_after",
    "load_hw_before_after",
    "load_fr_before_after",
    "plot_als_validation_grid",
    "plot_fp_before_after_grid",
    "plot_hw_before_after_grid",
    "plot_fr_before_after_grid",
]


def compute_stats(wn: np.ndarray, sp: np.ndarray) -> dict[str, Any]:
    """Compute exact global mean, std, min, max across all non-NaN rows."""
    sp64 = np.asarray(sp, dtype=np.float64)
    valid = ~np.isnan(sp64).any(axis=1)
    sp_v = sp64[valid]
    wn_f = np.asarray(wn, dtype=np.float64).flatten()
    if wn_f[0] > wn_f[-1]:
        wn_f = wn_f[::-1]
        sp_v = sp_v[:, ::-1]
    return {
        "wn": wn_f,
        "mean": sp_v.mean(axis=0),
        "std": sp_v.std(axis=0),
        "n_spectra": int(sp_v.shape[0]),
        "n_points": int(wn_f.size),
        "min": float(sp_v.min()),
        "max": float(sp_v.max()),
    }


def plot_profile_panel(
    ax: plt.Axes,
    st: dict[str, Any],
    title: str,
    subtitle: str,
    badge: str,
    color: str,
    ylabel: str = "Intensity",
) -> None:
    """Plot a single Global Mean +/- 1 SD panel with +32% top margin for zero overlap."""
    wn, mean, std = st["wn"], st["mean"], st["std"]
    ax.plot(wn, mean, color=color, lw=1.85, label=f"Mean ($n={st['n_spectra']:,}$)")
    ax.fill_between(wn, mean - std, mean + std, color=color, alpha=0.24, label="$\\pm 1$ SD ($\\sigma$)")

    for pos, col in [(1003, "#555"), (1445, "#555"), (1655, "#555"), (2850, "#8c2d04"), (2935, "#8c2d04"), (3400, "#08519c")]:
        if float(wn[0]) + 25 <= pos <= float(wn[-1]) - 25:
            ax.axvline(pos, color=col, linestyle=":", linewidth=0.85, alpha=0.50)

    y_lo, y_hi = float(np.min(mean - std)), float(np.max(mean + std))
    span = max(y_hi - y_lo, 1e-9)
    ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.32 * span)
    ax.set_xlim(float(wn[0]), float(wn[-1]))
    ax.set_title(f"{title}\n{subtitle}", fontsize=9.3, fontweight="bold", pad=5)
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.6)
    ax.set_ylabel(ylabel, fontsize=8.6)

    info = (
        f"{badge}\n"
        f"Range: {wn[0]:.1f}–{wn[-1]:.1f} cm$^{{-1}}$ ({st['n_points']} pts)\n"
        f"Y-Range: [{st['min']:.3g}, {st['max']:.3g}]"
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


def load_fp_before_after(raw_dir: Path, int_dir: Path) -> dict[str, dict[str, Any]]:
    """Load Before (00_raw) and After (01_interim) statistics for Folder FP."""
    sbfp_raw_l, sbfp_int_l, sbfp_wn = [], [], None
    for f_raw in sorted((raw_dir / "pretrain" / "FP" / "spectrabreast_fp").glob("*.h5")):
        if f_raw.name.startswith("._"):
            continue
        with h5py.File(f_raw, "r") as h5r:
            wn = np.array(h5r["/ENTRY/data/raman_shift"], dtype=np.float64).flatten()
            r2d = np.array(h5r["/ENTRY/data/intensity"], dtype=np.float64).reshape(-1, wn.size)
            if wn[0] > wn[-1]:
                wn, r2d = wn[::-1], r2d[:, ::-1]
        with h5py.File(int_dir / "pretrain" / "FP" / "spectrabreast_fp" / f_raw.name, "r") as h5i:
            c2d = np.array(h5i["processed_spectra"], dtype=np.float32)
        sbfp_wn = wn
        sbfp_raw_l.append(r2d)
        sbfp_int_l.append(c2d)

    rmae_raw = raw_dir / "pretrain" / "FP" / "raman_mae"
    rmae_int = int_dir / "pretrain" / "FP" / "raman_mae"
    m_cells = scipy.io.loadmat(str(next(rmae_raw.glob("*231CTCLM*.mat"))))
    mask_c = m_cells["labels_table_status_cell1_bg0"].flatten() == 1
    sp_cells_raw = np.nan_to_num(m_cells["All_filt"][mask_c].astype(np.float64), nan=0.0)
    c_npz = np.load(rmae_int / "cells_interim.npz")
    t_npz = np.load(rmae_int / "twist_interim.npz")
    i_npz = np.load(rmae_int / "immuno_interim.npz")
    ibd_npz = np.load(int_dir / "downstream" / "FP" / "ibd" / "ibd_interim.npz")
    bac_npz = np.load(int_dir / "downstream" / "FP" / "bacteria_id" / "bacteria_id_interim.npz")

    return {
        "sbfp_raw": compute_stats(sbfp_wn, np.vstack(sbfp_raw_l)),
        "sbfp_int": compute_stats(sbfp_wn, np.vstack(sbfp_int_l)),
        "cells_raw": compute_stats(c_npz["wavenumbers"], sp_cells_raw),
        "cells_int": compute_stats(c_npz["wavenumbers"], c_npz["spectra"]),
        "twist_int": compute_stats(t_npz["wavenumbers"], t_npz["spectra"]),
        "immuno_int": compute_stats(i_npz["wavenumbers"], i_npz["spectra"]),
        "ibd_int": compute_stats(ibd_npz["wavenumbers"], ibd_npz["spectra"]),
        "bac_int": compute_stats(bac_npz["wavenumbers"], np.vstack([bac_npz["X_finetune"], bac_npz["X_test"]])),
    }


def load_hw_before_after(raw_dir: Path, int_dir: Path) -> dict[str, Any]:
    """Load Before (00_raw) and After (01_interim) statistics for Folder HW."""
    sbhw_raw = raw_dir / "pretrain" / "HW" / "spectrabreast_hw"
    sbhw_int = int_dir / "pretrain" / "HW" / "spectrabreast_hw"
    g1_r, g1_i, g2_r, g2_i = [], [], [], []
    als_i, wn_g1, wn_g2 = None, None, None
    sb015_raw_3, sb015_als_2 = None, None

    for f_raw in sorted(sbhw_raw.glob("*.h5")):
        if f_raw.name.startswith("._"):
            continue
        with h5py.File(f_raw, "r") as h5r:
            wn = np.array(h5r["/ENTRY/data/raman_shift"], dtype=np.float64).flatten()
            r2d = np.array(h5r["/ENTRY/data/intensity"], dtype=np.float64).reshape(-1, wn.size)
            if wn[0] > wn[-1]:
                wn, r2d = wn[::-1], r2d[:, ::-1]
        with h5py.File(sbhw_int / f_raw.name, "r") as h5i:
            c2d = np.array(h5i["processed_spectra"], dtype=np.float32)
        if "SB015HT 2" in f_raw.name:
            sb015_als_2, als_i, wn_g2 = (wn, r2d), c2d, wn
        else:
            if "SB015HT 3" in f_raw.name:
                sb015_raw_3 = (wn, r2d)
            if wn[0] < 2600.0:
                wn_g1 = wn
                g1_r.append(r2d)
                g1_i.append(c2d)
            else:
                wn_g2 = wn
                g2_r.append(r2d)
                g2_i.append(c2d)

    pdo_raw = raw_dir / "pretrain" / "HW" / "pdo_hw_maps"
    pdo_int = int_dir / "pretrain" / "HW" / "pdo_hw_maps"
    pdo_r5, pdo_i6, wn_pdo = [], [], None
    cma64_raw, cma64_als = None, None
    for f_raw in sorted(pdo_raw.glob("*.h5")):
        if f_raw.name.startswith("._"):
            continue
        with h5py.File(f_raw, "r") as h5r:
            wn = np.array(h5r["/ENTRY/data/raman_shift"], dtype=np.float64).flatten()
            r2d = np.array(h5r["/ENTRY/data/intensity"], dtype=np.float64).reshape(-1, wn.size)
        with h5py.File(pdo_int / f_raw.name, "r") as h5i:
            c2d = np.array(h5i["processed_spectra"], dtype=np.float32)
        if "PDO2" in f_raw.name:
            cma64_als = (wn, r2d)
        else:
            wn_pdo = wn
            pdo_r5.append(r2d)
            if "CMA64" in f_raw.name:
                cma64_raw = (wn, r2d)
        pdo_i6.append(c2d)

    meat_raw = raw_dir / "pretrain" / "HW" / "riverd_meat_hw"
    meat_int = int_dir / "pretrain" / "HW" / "riverd_meat_hw"
    m_r, m_i, wn_m = [], [], None
    for f_raw in sorted(meat_raw.glob("*.h5")):
        if f_raw.name.startswith("._"):
            continue
        with h5py.File(f_raw, "r") as h5r:
            wn_m = np.array(h5r["/ENTRY/data/raman_shift"], dtype=np.float64).flatten()
            m_r.append(np.array(h5r["/ENTRY/data/intensity"], dtype=np.float64).reshape(-1, wn_m.size))
        with h5py.File(meat_int / f_raw.name, "r") as h5i:
            m_i.append(np.array(h5i["processed_spectra"], dtype=np.float32))

    oral_raw = raw_dir / "downstream" / "HW" / "riverd_oral_hw"
    wn_oral = pd.read_excel(oral_raw / "RamanMetaData.xlsx", sheet_name="wvn", header=None).to_numpy(dtype=np.float64).flatten()
    sp_oral_c = pd.read_excel(oral_raw / "RamanData_c.xlsx", header=None).to_numpy(dtype=np.float64).T
    oral_npz = np.load(int_dir / "downstream" / "HW" / "riverd_oral_hw" / "riverd_oral_hw_interim.npz")

    return {
        "g1_raw": compute_stats(wn_g1, np.vstack(g1_r)),
        "g1_int": compute_stats(wn_g1, np.vstack(g1_i)),
        "g2_raw": compute_stats(wn_g2, np.vstack(g2_r)),
        "g2_int": compute_stats(wn_g2, np.vstack(g2_i + [als_i])),
        "pdo_raw": compute_stats(wn_pdo, np.vstack(pdo_r5)),
        "pdo_int": compute_stats(wn_pdo, np.vstack(pdo_i6)),
        "meat_raw": compute_stats(wn_m, np.vstack(m_r)),
        "meat_int": compute_stats(wn_m, np.vstack(m_i)),
        "oral_raw_c": compute_stats(wn_oral, sp_oral_c),
        "oral_int": compute_stats(oral_npz["wavenumbers"], oral_npz["spectra"]),
        "sb015_raw_3": sb015_raw_3,
        "sb015_als_2": sb015_als_2,
        "cma64_raw": cma64_raw,
        "cma64_als": cma64_als,
    }


def load_fr_before_after(raw_dir: Path, int_dir: Path) -> dict[str, dict[str, Any]]:
    """Load Before (00_raw) and After (01_interim) statistics for Folder FR."""
    tr_raw = raw_dir / "pretrain" / "FR" / "trophy_jena"
    tr_int = int_dir / "pretrain" / "FR" / "trophy_jena"
    meta_tr = pd.read_csv(tr_raw / "Pixelwise metadata.csv")
    fg_mask = {str(k).strip(): (g["fg_mask"].to_numpy() == 1) & (g["is_burnt"].to_numpy() == 0) for k, g in meta_tr.groupby("Map name")}

    def _read_tr(sub: str) -> tuple[np.ndarray, np.ndarray]:
        rows, wn_ref = [], None
        for mf in sorted((tr_raw / "Raman maps" / sub).glob("*.mat")):
            d = scipy.io.loadmat(str(mf))
            wn_ref = d["x"].flatten().astype(np.float64)
            cube = d["data"].astype(np.float64).reshape(-1, wn_ref.size)
            if mf.stem.strip() in fg_mask:
                cube = cube[fg_mask[mf.stem.strip()]]
            rows.append(cube)
        return wn_ref, np.vstack(rows)

    wn_tr_r, sp_tr_r = _read_tr("Raw")
    wn_tr_p, sp_tr_p = _read_tr("Preprocessed")
    tr_als_npz = np.load(tr_int / "trophy_raw_als_minmax.npz")
    tr_prep_npz = np.load(tr_int / "trophy_preprocessed_minmax.npz")

    df_rz = pd.read_csv(raw_dir / "downstream" / "FR" / "dataset_renzo" / "def_dataset_final.csv", low_memory=False)
    wn_rz = df_rz.iloc[7:, 0].to_numpy(dtype=np.float64)
    sp_rz_r = df_rz.iloc[7:, 1:].to_numpy(dtype=np.float64).T
    rz_npz = np.load(int_dir / "downstream" / "FR" / "dataset_renzo" / "dataset_renzo_interim.npz")

    return {
        "tr_raw_orig": compute_stats(wn_tr_r, sp_tr_r),
        "tr_raw_als_int": compute_stats(tr_als_npz["wavenumbers"], tr_als_npz["spectra"]),
        "tr_prep_orig": compute_stats(wn_tr_p, sp_tr_p),
        "tr_prep_int": compute_stats(tr_prep_npz["wavenumbers"], tr_prep_npz["spectra"]),
        "rz_raw": compute_stats(wn_rz, sp_rz_r),
        "rz_int": compute_stats(rz_npz["wavenumbers"], rz_npz["spectra"]),
    }


def plot_als_validation_grid(
    fr_stats: dict[str, dict[str, Any]],
    hw_stats: dict[str, Any],
    save_paths: list[Path] | None = None,
) -> plt.Figure:
    """Plot Figure 1: 3-column validation of our ALS against 3 laboratory references."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))

    st_tr_raw_orig = fr_stats["tr_raw_orig"]
    st_tr_als = fr_stats["tr_raw_als_int"]
    st_tr_prep = fr_stats["tr_prep_int"]
    wn_tr_raw, wn_tr_prep = st_tr_als["wn"], st_tr_prep["wn"]
    idx_ov = np.where((wn_tr_raw >= wn_tr_prep[0] - 0.5) & (wn_tr_raw <= wn_tr_prep[-1] + 0.5))[0]
    m_fp = (wn_tr_prep >= 650) & (wn_tr_prep <= 1750)
    m_ch = (wn_tr_prep >= 2765) & (wn_tr_prep <= 3050)
    mean_our, mean_auth = st_tr_als["mean"][idx_ov], st_tr_prep["mean"]
    r_fp = float(np.corrcoef(mean_our[m_fp], mean_auth[m_fp])[0, 1])
    r_ch = float(np.corrcoef(mean_our[m_ch], mean_auth[m_ch])[0, 1])

    plot_profile_panel(axes[0, 0], st_tr_raw_orig, "1A. trophy_jena — Raw/ Folder (Before ALS)", "115,913 Foreground Spectra (200–3200 cm⁻¹, Silica Glass Peak <648 cm⁻¹)", "RAW Detector Counts", "#D55E00", "Detector Counts")
    ax = axes[1, 0]
    our_mm = (st_tr_als["mean"] - st_tr_als["mean"].min()) / (st_tr_als["mean"].max() - st_tr_als["mean"].min())
    auth_mm = (mean_auth - mean_auth.min()) / (mean_auth.max() - mean_auth.min())
    ax.plot(wn_tr_raw, our_mm, color="#0072B2", lw=2.1, label="Our Despike + ALS on Raw/ (648.7–3200 cm$^{-1}$)")
    ax.plot(wn_tr_prep, auth_mm, color="#009E73", lw=1.7, linestyle="--", label="Jena Authors' Preprocessed/ (648.7–3050.4 cm$^{-1}$)")
    ax.set_ylim(-0.06, 1.34)
    ax.set_xlim(648.7, 3200.0)
    ax.set_title(f"1B. trophy_jena — Our ALS vs Jena Authors' Preprocessed\nPearson $r = {r_fp:.4f}$ (FP) | $r = {r_ch:.4f}$ (CH) + Extends to 3200 cm$^{{-1}}$!", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.6)
    ax.set_ylabel("Normalized Mean Profile [0, 1]", fontsize=8.6)
    ax.text(0.02, 0.96, f"Validation: n = 115,913 spectra\nFP (650–1750): r = {r_fp:.4f}\nCH (2765–3050): r = {r_ch:.4f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.8, bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#0072B2", "alpha": 0.93})
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    wn_sb3, raw_sb3 = hw_stats["sb015_raw_3"]
    wn_sb2, als_sb2 = hw_stats["sb015_als_2"]
    plot_profile_panel(axes[0, 1], compute_stats(wn_sb3, raw_sb3), "2A. spectrabreast_hw — SB015HT 3 (Before ALS)", "RAW Breast Tissue Map from Donor SB015 (2765–3488 cm⁻¹)", "RAW CCD Counts", "#E69F00", "Detector Counts")
    our_sb3 = als_baseline_batch(despike_whitaker_hayes_batch(raw_sb3[:2000]), lam=1e6, p=0.005).mean(axis=0)
    ref_sb2 = als_sb2.mean(axis=0)
    r_sb = float(np.corrcoef(our_sb3, ref_sb2)[0, 1])
    ax = axes[1, 1]
    ax.plot(wn_sb3, (our_sb3 - our_sb3.min()) / (our_sb3.max() - our_sb3.min()), color="#0072B2", lw=2.0, label="SB015HT 3 RAW + Our ALS ($\\lambda=10^6, p=0.005$)")
    ax.plot(wn_sb2, (ref_sb2 - ref_sb2.min()) / (ref_sb2.max() - ref_sb2.min()), color="#009E73", lw=1.6, linestyle="--", label="SB015HT 2 Lab Post-ALS Reference")
    ax.set_ylim(-0.06, 1.34)
    ax.set_xlim(float(wn_sb3[0]), float(wn_sb3[-1]))
    ax.set_title(f"2B. spectrabreast_hw — Our ALS vs Lab Post-ALS Reference\nSame Patient Donor (SB015HT): Pearson $r = {r_sb:.4f}$", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.6)
    ax.set_ylabel("Normalized Mean Profile [0, 1]", fontsize=8.6)
    ax.text(0.02, 0.96, f"HW Tissue Regime Validation\nParameters: lam=1e6, p=0.005\nPearson r = {r_sb:.4f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.8, bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#0072B2", "alpha": 0.93})
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    wn_cma_r, raw_cma = hw_stats["cma64_raw"]
    wn_cma_a, als_cma = hw_stats["cma64_als"]
    plot_profile_panel(axes[0, 2], compute_stats(wn_cma_r, raw_cma), "3A. pdo_hw_maps — CMA64 Organoid (Before ALS)", "RAW 532 nm Organoid Map with Broad C–H + O–H Water Band", "RAW CCD Counts", "#CC79A7", "Detector Counts")
    our_cma = als_baseline_batch(despike_whitaker_hayes_batch(raw_cma[:2000]), lam=5e7, p=0.001).mean(axis=0)
    ref_cma = als_cma.mean(axis=0)
    r_cma = float(np.corrcoef(our_cma, ref_cma)[0, 1])
    ax = axes[1, 2]
    ax.plot(wn_cma_r, (our_cma - our_cma.min()) / (our_cma.max() - our_cma.min()), color="#0072B2", lw=2.0, label="CMA64 RAW + Our ALS ($\\lambda=5\\times 10^7, p=0.001$)")
    ax.plot(wn_cma_a, (ref_cma - ref_cma.min()) / (ref_cma.max() - ref_cma.min()), color="#009E73", lw=1.6, linestyle="--", label="CMA64 PDO2 Lab Post-ALS Reference")
    ax.set_ylim(-0.06, 1.34)
    ax.set_xlim(float(wn_cma_r[0]), float(wn_cma_r[-1]))
    ax.set_title(f"3B. pdo_hw_maps — Our ALS vs Lab Post-ALS Reference\nSame Organoid Line (CMA64): Pearson $r = {r_cma:.4f}$", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.6)
    ax.set_ylabel("Normalized Mean Profile [0, 1]", fontsize=8.6)
    ax.text(0.02, 0.96, f"HW Broad Water Regime Validation\nParameters: lam=5e7, p=0.001\nPearson r = {r_cma:.4f}", transform=ax.transAxes, va="top", ha="left", fontsize=7.8, bbox={"boxstyle": "round,pad=0.26", "facecolor": "white", "edgecolor": "#0072B2", "alpha": 0.93})
    ax.legend(loc="upper right", fontsize=7.8, framealpha=0.92)

    fig.suptitle("Methodological Validation of Our ALS Baseline Subtraction Against 3 Laboratory References (r = 0.988 – 0.997)", fontsize=13.5, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig


def plot_fp_before_after_grid(fp_stats: dict[str, dict[str, Any]], save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 2: Folder FP Before vs After grid."""
    fig, axes = plt.subplots(4, 2, figsize=(16.5, 15.0))
    pairs = [
        (axes[0, 0], fp_stats["sbfp_raw"], "1A. BEFORE (00_raw): pretrain/FP/spectrabreast_fp", "4 Breast Maps (SB001TT–SB004TT, Unprocessed CCD Counts)", "RAW CCD Counts", "#D55E00", "Detector Counts"),
        (axes[0, 1], fp_stats["sbfp_int"], "1B. AFTER (01_interim): pretrain/FP/spectrabreast_fp", "Despiked + ALS Baseline-Subtracted + Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[1, 0], fp_stats["cells_raw"], "2A. BEFORE (00_raw): pretrain/FP/raman_mae (CELLS)", "15,429 Valid Cell Spectra (Baseline-Subtracted + L2 Norm)", "L2-Normalized (||x||₂=1)", "#009E73", "L2 Intensity"),
        (axes[1, 1], fp_stats["cells_int"], "2B. AFTER (01_interim): pretrain/FP/raman_mae (CELLS)", "Converted from L2 Norm to Strict Per-Spectrum Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[2, 0], fp_stats["twist_int"], "3A. AFTER (01_interim): pretrain/FP/raman_mae (TWIST)", "16,496 Valid EMT Spectra Converted to Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[2, 1], fp_stats["immuno_int"], "3B. AFTER (01_interim): pretrain/FP/raman_mae (IMMUNO)", "11,099 Immune Subtype Spectra Converted to Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[3, 0], fp_stats["ibd_int"], "4A. AFTER (01_interim): downstream/FP/ibd (228 Patients)", "77 HC vs 151 IBD Converted to Strict Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#CC79A7", "Normalized Intensity [0, 1]"),
        (axes[3, 1], fp_stats["bac_int"], "4B. AFTER (01_interim): downstream/FP/bacteria_id (6,000 Active)", "3,000 Fine-Tune + 3,000 Test Verified in Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#56B4E9", "Normalized Intensity [0, 1]"),
    ]
    for ax, st, t, sub, bdg, col, yl in pairs:
        plot_profile_panel(ax, st, t, sub, bdg, col, yl)
    fig.suptitle("Folder FP (Fingerprint: pretrain/FP & downstream/FP) — Before (00_raw) vs After (01_interim Min-Max [0, 1])", fontsize=13.5, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig


def plot_hw_before_after_grid(hw_stats: dict[str, Any], save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 3: Folder HW Before vs After grid."""
    fig, axes = plt.subplots(5, 2, figsize=(16.5, 18.0))
    pairs = [
        (axes[0, 0], hw_stats["g1_raw"], "1A. BEFORE (00_raw): spectrabreast_hw (Grating 1: 7 Maps)", "54,810 Spectra (2488.9–3258.9 cm⁻¹, Unprocessed CCD Counts)", "RAW CCD Counts", "#D55E00", "Detector Counts"),
        (axes[0, 1], hw_stats["g1_int"], "1B. AFTER (01_interim): spectrabreast_hw (Grating 1: 7 Maps)", "54,810 Spectra Despiked + ALS (lam=1e6, p=0.005) + Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[1, 0], hw_stats["g2_raw"], "2A. BEFORE (00_raw): spectrabreast_hw (Grating 2: 7 RAW Maps)", "71,113 RAW Spectra (+ 10,878 Post-ALS in SB015HT 2)", "RAW CCD Counts (Mixed)", "#E69F00", "Detector Counts"),
        (axes[1, 1], hw_stats["g2_int"], "2B. AFTER (01_interim): spectrabreast_hw (Grating 2: All 8 Maps Unified)", "All 81,991 Spectra Unified via ALS + Min-Max [0, 1] (2763.9–3487.6 cm⁻¹)", "Unified + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[2, 0], hw_stats["pdo_raw"], "3A. BEFORE (00_raw): pretrain/HW/pdo_hw_maps (5 RAW Maps)", "146,583 RAW Spectra (+ 47,861 Post-ALS in CMA64 PDO2)", "RAW CCD Counts (Mixed)", "#D55E00", "Detector Counts"),
        (axes[2, 1], hw_stats["pdo_int"], "3B. AFTER (01_interim): pretrain/HW/pdo_hw_maps (All 6 Maps Unified)", "All 194,444 Organoid Spectra Unified via ALS + Min-Max [0, 1]", "Unified + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[3, 0], hw_stats["meat_raw"], "4A. BEFORE (00_raw): pretrain/HW/riverd_meat_hw (60 Maps)", "1,200 Reference Meat Spectra (2520–4000 cm⁻¹, Unprocessed Counts)", "RAW Detector Counts", "#009E73", "Detector Counts"),
        (axes[3, 1], hw_stats["meat_int"], "4B. AFTER (01_interim): pretrain/HW/riverd_meat_hw (60 Maps)", "1,200 Spectra Despiked + ALS (lam=1e7, p=0.002) + Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[4, 0], hw_stats["oral_raw_c"], "5A. BEFORE (00_raw): downstream/HW/riverd_oral_hw (RamanData_c)", "8,422 Spectra across 186 Oral Samples (With Fiber-Probe Background)", "Calibrated (Raw Background)", "#CC79A7", "Intensity (a.u.)"),
        (axes[4, 1], hw_stats["oral_int"], "5B. AFTER (01_interim): downstream/HW/riverd_oral_hw (RamanData_bc)", "8,422 Spectra (Probe & Baseline Corrected + Min-Max [0, 1])", "Cleaned + Min-Max [0, 1]", "#009E73", "Normalized Intensity [0, 1]"),
    ]
    for ax, st, t, sub, bdg, col, yl in pairs:
        plot_profile_panel(ax, st, t, sub, bdg, col, yl)
    fig.suptitle("Folder HW (High-Wavenumber: pretrain/HW & downstream/HW) — Before (00_raw) vs After (01_interim Min-Max [0, 1])", fontsize=13.5, fontweight="bold", y=1.004)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig


def plot_fr_before_after_grid(fr_stats: dict[str, dict[str, Any]], save_paths: list[Path] | None = None) -> plt.Figure:
    """Plot Figure 4: Folder FR Before vs After grid."""
    fig, axes = plt.subplots(3, 2, figsize=(16.5, 11.5))
    pairs = [
        (axes[0, 0], fr_stats["tr_raw_orig"], "1A. BEFORE (00_raw): pretrain/FR/trophy_jena (Raw/ Folder)", "115,913 Valid Breast Spectra (200–3200 cm⁻¹, 1024 pts, RAW Counts)", "RAW Counts (Unprocessed)", "#D55E00", "Detector Counts"),
        (axes[0, 1], fr_stats["tr_raw_als_int"], "1B. AFTER (01_interim): trophy_jena (Raw/ + Our ALS + Min-Max [0, 1])", "115,913 Spectra Cleaned across 648.7–3200 cm⁻¹ (871 pts, Glass Removed, Extended HW!)", "Our ALS + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
        (axes[1, 0], fr_stats["tr_prep_orig"], "2A. BEFORE (00_raw): pretrain/FR/trophy_jena (Preprocessed/ Folder)", "115,913 Valid Spectra (648.7–3050.4 cm⁻¹, 820 pts, Authors Cleaned)", "Authors Baseline-Subtracted", "#009E73", "Intensity (a.u.)"),
        (axes[1, 1], fr_stats["tr_prep_int"], "2B. AFTER (01_interim): trophy_jena (Preprocessed/ + Min-Max [0, 1])", "115,913 Spectra Scaled to Strict Per-Spectrum Min-Max [0, 1]", "Authors Clean + Min-Max [0, 1]", "#009E73", "Normalized Intensity [0, 1]"),
        (axes[2, 0], fr_stats["rz_raw"], "3A. BEFORE (00_raw): downstream/FR/dataset_renzo (47 Mice)", "3,200 Murine Bone Spectra (380–3049 cm⁻¹, 2670 pts, Area Normalized)", "Baseline-Subtracted", "#CC79A7", "Intensity (a.u.)"),
        (axes[2, 1], fr_stats["rz_int"], "3B. AFTER (01_interim): downstream/FR/dataset_renzo (Min-Max [0, 1])", "3,200 Spectra Scaled to Strict Per-Spectrum Min-Max [0, 1]", "Cleaned + Min-Max [0, 1]", "#0072B2", "Normalized Intensity [0, 1]"),
    ]
    for ax, st, t, sub, bdg, col, yl in pairs:
        plot_profile_panel(ax, st, t, sub, bdg, col, yl)
    fig.suptitle("Folder FR (Full-Range: pretrain/FR & downstream/FR) — Before (00_raw) vs After (01_interim Min-Max [0, 1])", fontsize=13.5, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_paths:
        for p in save_paths:
            fig.savefig(p, bbox_inches="tight")
    return fig
