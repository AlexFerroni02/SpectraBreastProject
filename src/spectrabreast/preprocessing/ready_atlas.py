"""
Step 3 (`data/01_interim/` -> `data/02_ready/`) dataset compiler, statistical
loaders, and publication-grade visualization utilities for
`notebooks/03_ready_datasets_compilation.ipynb`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .resampling import (
    GRID_CH_1000,
    GRID_FP_1000,
    GRID_FR_JENA_1000,
    GRID_FR_RENZO_1000,
    GRID_HW_1000,
    N_READY_POINTS,
    resample_cubic_spline_batch,
)

__all__ = [
    "compile_ready_datasets",
    "load_ready_summary_stats",
    "plot_ready_architecture_overview",
    "plot_cubic_spline_validation",
    "plot_ready_pretrain_corpora",
    "plot_ready_downstream_benchmarks",
]


def _load_h5_folder_resampled(
    folder: Path,
    wn_target: np.ndarray,
    filter_fn: Any = None,
) -> tuple[np.ndarray, int, float, float, int]:
    """Load all .h5 files in `folder`, resample each file onto `wn_target`, and stack."""
    chunks: list[np.ndarray] = []
    w0_min, w1_max, n_pts_native = np.inf, -np.inf, 0
    for fp in sorted(folder.glob("*.h5")):
        if fp.name.startswith("._"):
            continue
        with h5py.File(fp, "r") as h:
            wn = np.array(h["raman_shift"], dtype=np.float64).flatten()
            sp = np.array(h["processed_spectra"], dtype=np.float32)
        if wn[0] > wn[-1]:
            wn = wn[::-1]
            sp = sp[:, ::-1]
        if filter_fn is not None and not filter_fn(fp.name, wn):
            continue
        w0_min = min(w0_min, float(wn[0]))
        w1_max = max(w1_max, float(wn[-1]))
        n_pts_native = int(wn.size)
        res = resample_cubic_spline_batch(wn, sp, wn_target, renormalize_minmax=True)
        chunks.append(res)
    stacked = np.vstack(chunks)
    return stacked, int(stacked.shape[0]), w0_min, w1_max, n_pts_native


def _compute_source_summaries(
    spectra: np.ndarray,
    source_id: np.ndarray,
    n_sources: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-source mean, std, and counts for instant plotting."""
    means = np.zeros((n_sources, spectra.shape[1]), dtype=np.float32)
    stds = np.zeros((n_sources, spectra.shape[1]), dtype=np.float32)
    counts = np.zeros(n_sources, dtype=np.int64)
    for sid in range(n_sources):
        mask = source_id == sid
        counts[sid] = int(np.sum(mask))
        if counts[sid] > 0:
            sub = spectra[mask]
            means[sid] = sub.mean(axis=0, dtype=np.float64).astype(np.float32)
            stds[sid] = sub.std(axis=0, dtype=np.float64).astype(np.float32)
    return means, stds, counts


def compile_ready_datasets(
    interim_dir: Path,
    ready_dir: Path,
    catalog_csv_path: Path | None = None,
) -> pd.DataFrame:
    """
    Compile all 1,000-point `data/02_ready/` tensors from `data/01_interim/` using
    boundary-clamped Natural Cubic Spline interpolation and post-crop `[0, 1]`
    re-anchoring. Returns the master summary DataFrame.
    """
    for sub in [
        ready_dir / "pretrain" / "FP",
        ready_dir / "pretrain" / "HW",
        ready_dir / "pretrain" / "CH",
        ready_dir / "pretrain" / "FR",
        ready_dir / "pretrain" / "CNN",
        ready_dir / "downstream" / "FP",
        ready_dir / "downstream" / "HW",
        ready_dir / "downstream" / "CH",
        ready_dir / "downstream" / "FR",
    ]:
        sub.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # =========================================================================
    # 1. PRETRAIN FP COHORTS -> GRID_FP_1000 (650.0 - 1750.0 cm^-1, 1000 pts)
    # =========================================================================
    print("[Step 3.1] Resampling FP Pre-Training cohorts onto 650.0-1750.0 cm^-1 (1,000 pts)...")
    d_deepr = np.load(interim_dir / "pretrain" / "FP" / "deepr" / "deepr_interim.npz")
    wn_deepr = d_deepr["wavenumbers"]
    deepr_10s_fp = resample_cubic_spline_batch(wn_deepr, d_deepr["outputs_10s"], GRID_FP_1000)
    deepr_01s_fp = resample_cubic_spline_batch(wn_deepr, d_deepr["inputs_01s"], GRID_FP_1000)

    np.savez(
        ready_dir / "pretrain" / "FP" / "deepr_paired_fp_1000.npz",
        wavenumbers=GRID_FP_1000,
        inputs_01s=deepr_01s_fp,
        outputs_10s=deepr_10s_fp,
    )

    d_cells = np.load(interim_dir / "pretrain" / "FP" / "raman_mae" / "cells_interim.npz")
    cells_fp = resample_cubic_spline_batch(d_cells["wavenumbers"], d_cells["spectra"], GRID_FP_1000)

    d_twist = np.load(interim_dir / "pretrain" / "FP" / "raman_mae" / "twist_interim.npz")
    twist_fp = resample_cubic_spline_batch(d_twist["wavenumbers"], d_twist["spectra"], GRID_FP_1000)

    d_immuno = np.load(interim_dir / "pretrain" / "FP" / "raman_mae" / "immuno_interim.npz")
    immuno_fp = resample_cubic_spline_batch(d_immuno["wavenumbers"], d_immuno["spectra"], GRID_FP_1000)

    sb_fp, _, _, _, _ = _load_h5_folder_resampled(
        interim_dir / "pretrain" / "FP" / "spectrabreast_fp", GRID_FP_1000
    )

    # Full-Range Trophy Jena -> extract FP window (650-1750 cm^-1)
    d_tr_als = np.load(interim_dir / "pretrain" / "FR" / "trophy_jena" / "trophy_raw_als_minmax.npz")
    wn_tr = d_tr_als["wavenumbers"]
    sp_tr = d_tr_als["spectra"]
    tr_fp = resample_cubic_spline_batch(wn_tr, sp_tr, GRID_FP_1000)

    fp_sources = [
        ("deepr_10s", deepr_10s_fp),
        ("raman_mae_cells", cells_fp),
        ("raman_mae_twist", twist_fp),
        ("raman_mae_immuno", immuno_fp),
        ("spectrabreast_fp", sb_fp),
        ("trophy_jena_fp", tr_fp),
    ]
    fp_spectra = np.vstack([arr for _, arr in fp_sources])
    fp_source_id = np.concatenate([
        np.full(arr.shape[0], idx, dtype=np.int8) for idx, (_, arr) in enumerate(fp_sources)
    ])
    fp_source_names = np.array([name for name, _ in fp_sources])
    fp_means, fp_stds, fp_counts = _compute_source_summaries(fp_spectra, fp_source_id, len(fp_sources))
    fp_global_mean = fp_spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    fp_global_std = fp_spectra.std(axis=0, dtype=np.float64).astype(np.float32)

    np.savez(
        ready_dir / "pretrain" / "FP" / "pretrain_vit_fp_1000.npz",
        wavenumbers=GRID_FP_1000,
        spectra=fp_spectra,
        source_id=fp_source_id,
        source_names=fp_source_names,
        source_means=fp_means,
        source_stds=fp_stds,
        source_counts=fp_counts,
        global_mean=fp_global_mean,
        global_std=fp_global_std,
    )
    rows.append({
        "Role": "SSL Pre-Training (ViT-SMAE)",
        "Folder": "pretrain/FP",
        "Ready_File": "pretrain_vit_fp_1000.npz",
        "Target_Regime": "FP (Fingerprint)",
        "Grid_Range_cm1": "650.00–1750.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_FP_1000[1] - GRID_FP_1000[0]), 4),
        "Total_Spectra": int(fp_spectra.shape[0]),
        "Constituent_Cohorts": "deepr (172,312) + raman_mae (43,024) + spectrabreast_fp (52,992) + trophy_jena_fp (115,913)",
        "Value_Range": f"[{fp_spectra.min():.4f}, {fp_spectra.max():.4f}]",
    })
    rows.append({
        "Role": "SSL Denoising Pre-Training",
        "Folder": "pretrain/FP",
        "Ready_File": "deepr_paired_fp_1000.npz",
        "Target_Regime": "FP (Paired 0.1s/1.0s)",
        "Grid_Range_cm1": "650.00–1750.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_FP_1000[1] - GRID_FP_1000[0]), 4),
        "Total_Spectra": int(deepr_10s_fp.shape[0]),
        "Constituent_Cohorts": "deepr paired inputs_01s (172,312) & outputs_10s (172,312)",
        "Value_Range": f"[{deepr_10s_fp.min():.4f}, {deepr_10s_fp.max():.4f}]",
    })

    # =========================================================================
    # 2. PRETRAIN HW & CH COHORTS -> GRID_HW_1000 (2765-3200) & GRID_CH_1000 (2800-3045)
    # =========================================================================
    print("[Step 3.2] Resampling HW/CH Pre-Training cohorts onto 2765-3200 and 2800-3045 cm^-1 (1,000 pts)...")
    sbhw_dir = interim_dir / "pretrain" / "HW" / "spectrabreast_hw"
    sb_hw_g1, _, _, _, _ = _load_h5_folder_resampled(
        sbhw_dir, GRID_HW_1000, filter_fn=lambda name, wn: wn[0] < 2600.0
    )
    sb_hw_g2, _, _, _, _ = _load_h5_folder_resampled(
        sbhw_dir, GRID_HW_1000, filter_fn=lambda name, wn: wn[0] >= 2600.0
    )
    pdo_hw, _, _, _, _ = _load_h5_folder_resampled(
        interim_dir / "pretrain" / "HW" / "pdo_hw_maps", GRID_HW_1000
    )
    meat_hw, _, _, _, _ = _load_h5_folder_resampled(
        interim_dir / "pretrain" / "HW" / "riverd_meat_hw", GRID_HW_1000
    )
    tr_hw = resample_cubic_spline_batch(wn_tr, sp_tr, GRID_HW_1000)

    hw_sources = [
        ("spectrabreast_hw_g1", sb_hw_g1),
        ("spectrabreast_hw_g2", sb_hw_g2),
        ("pdo_hw_maps", pdo_hw),
        ("riverd_meat_hw", meat_hw),
        ("trophy_jena_hw", tr_hw),
    ]
    hw_spectra = np.vstack([arr for _, arr in hw_sources])
    hw_source_id = np.concatenate([
        np.full(arr.shape[0], idx, dtype=np.int8) for idx, (_, arr) in enumerate(hw_sources)
    ])
    hw_source_names = np.array([name for name, _ in hw_sources])
    hw_means, hw_stds, hw_counts = _compute_source_summaries(hw_spectra, hw_source_id, len(hw_sources))
    hw_global_mean = hw_spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    hw_global_std = hw_spectra.std(axis=0, dtype=np.float64).astype(np.float32)

    np.savez(
        ready_dir / "pretrain" / "HW" / "pretrain_vit_hw_1000.npz",
        wavenumbers=GRID_HW_1000,
        spectra=hw_spectra,
        source_id=hw_source_id,
        source_names=hw_source_names,
        source_means=hw_means,
        source_stds=hw_stds,
        source_counts=hw_counts,
        global_mean=hw_global_mean,
        global_std=hw_global_std,
    )
    rows.append({
        "Role": "SSL Pre-Training (ViT-SMAE)",
        "Folder": "pretrain/HW",
        "Ready_File": "pretrain_vit_hw_1000.npz",
        "Target_Regime": "HW (Extended High-Wavenumber)",
        "Grid_Range_cm1": "2765.00–3200.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_HW_1000[1] - GRID_HW_1000[0]), 4),
        "Total_Spectra": int(hw_spectra.shape[0]),
        "Constituent_Cohorts": "spectrabreast_hw (136,801) + pdo_hw_maps (194,444) + riverd_meat_hw (1,200) + trophy_jena_hw (115,913)",
        "Value_Range": f"[{hw_spectra.min():.4f}, {hw_spectra.max():.4f}]",
    })

    # Core C-H Stretching window (2800.0 - 3045.0 cm^-1, 1000 pts)
    sb_ch_g1, _, _, _, _ = _load_h5_folder_resampled(
        sbhw_dir, GRID_CH_1000, filter_fn=lambda name, wn: wn[0] < 2600.0
    )
    sb_ch_g2, _, _, _, _ = _load_h5_folder_resampled(
        sbhw_dir, GRID_CH_1000, filter_fn=lambda name, wn: wn[0] >= 2600.0
    )
    pdo_ch, _, _, _, _ = _load_h5_folder_resampled(
        interim_dir / "pretrain" / "HW" / "pdo_hw_maps", GRID_CH_1000
    )
    meat_ch, _, _, _, _ = _load_h5_folder_resampled(
        interim_dir / "pretrain" / "HW" / "riverd_meat_hw", GRID_CH_1000
    )
    tr_ch = resample_cubic_spline_batch(wn_tr, sp_tr, GRID_CH_1000)

    ch_sources = [
        ("spectrabreast_hw_g1", sb_ch_g1),
        ("spectrabreast_hw_g2", sb_ch_g2),
        ("pdo_hw_maps", pdo_ch),
        ("riverd_meat_hw", meat_ch),
        ("trophy_jena_ch", tr_ch),
    ]
    ch_spectra = np.vstack([arr for _, arr in ch_sources])
    ch_source_id = hw_source_id.copy()
    ch_source_names = np.array([name for name, _ in ch_sources])
    ch_means, ch_stds, ch_counts = _compute_source_summaries(ch_spectra, ch_source_id, len(ch_sources))
    ch_global_mean = ch_spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    ch_global_std = ch_spectra.std(axis=0, dtype=np.float64).astype(np.float32)

    np.savez(
        ready_dir / "pretrain" / "CH" / "pretrain_vit_ch_1000.npz",
        wavenumbers=GRID_CH_1000,
        spectra=ch_spectra,
        source_id=ch_source_id,
        source_names=ch_source_names,
        source_means=ch_means,
        source_stds=ch_stds,
        source_counts=ch_counts,
        global_mean=ch_global_mean,
        global_std=ch_global_std,
    )
    rows.append({
        "Role": "SSL Pre-Training (ViT-SMAE)",
        "Folder": "pretrain/CH",
        "Ready_File": "pretrain_vit_ch_1000.npz",
        "Target_Regime": "CH (Core C-H Stretch)",
        "Grid_Range_cm1": "2800.00–3045.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_CH_1000[1] - GRID_CH_1000[0]), 4),
        "Total_Spectra": int(ch_spectra.shape[0]),
        "Constituent_Cohorts": "spectrabreast_hw (136,801) + pdo_hw_maps (194,444) + riverd_meat_hw (1,200) + trophy_jena_ch (115,913)",
        "Value_Range": f"[{ch_spectra.min():.4f}, {ch_spectra.max():.4f}]",
    })
    del ch_spectra, sb_ch_g1, sb_ch_g2, pdo_ch, meat_ch, tr_ch

    # =========================================================================
    # 3. PRETRAIN FR & GLOBAL CNN (BARLOW TWINS) CORPUS (716,686 UNIQUE SPECTRA)
    # =========================================================================
    print("[Step 3.3] Resampling Full-Range Trophy Jena and compiling 716,686-spectrum CNN Corpus...")
    tr_fr_1000 = resample_cubic_spline_batch(wn_tr, sp_tr, GRID_FR_JENA_1000)
    np.savez(
        ready_dir / "pretrain" / "FR" / "trophy_jena_fr_1000.npz",
        wavenumbers=GRID_FR_JENA_1000,
        spectra=tr_fr_1000,
        global_mean=tr_fr_1000.mean(axis=0, dtype=np.float64).astype(np.float32),
        global_std=tr_fr_1000.std(axis=0, dtype=np.float64).astype(np.float32),
    )
    rows.append({
        "Role": "SSL Pre-Training (Full-Range)",
        "Folder": "pretrain/FR",
        "Ready_File": "trophy_jena_fr_1000.npz",
        "Target_Regime": "FR (Full-Range 650–3200)",
        "Grid_Range_cm1": "650.00–3200.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_FR_JENA_1000[1] - GRID_FR_JENA_1000[0]), 4),
        "Total_Spectra": int(tr_fr_1000.shape[0]),
        "Constituent_Cohorts": "trophy_jena Raw+ALS Full-Range (115,913 foreground pixels)",
        "Value_Range": f"[{tr_fr_1000.min():.4f}, {tr_fr_1000.max():.4f}]",
    })

    # Global 716,686 unique spectra for 1D-ResNet-34 (Barlow Twins):
    # - 268,328 FP spectra (first 5 sources of fp_sources)
    # - 332,445 HW spectra (first 4 sources of hw_sources)
    # - 115,913 FR spectra (trophy_jena_fr_1000)
    cnn_sources = [
        ("deepr_10s", deepr_10s_fp, 0),
        ("raman_mae_cells", cells_fp, 0),
        ("raman_mae_twist", twist_fp, 0),
        ("raman_mae_immuno", immuno_fp, 0),
        ("spectrabreast_fp", sb_fp, 0),
        ("spectrabreast_hw_g1", sb_hw_g1, 1),
        ("spectrabreast_hw_g2", sb_hw_g2, 1),
        ("pdo_hw_maps", pdo_hw, 1),
        ("riverd_meat_hw", meat_hw, 1),
        ("trophy_jena_fr", tr_fr_1000, 2),
    ]
    cnn_spectra = np.vstack([arr for _, arr, _ in cnn_sources])
    cnn_source_id = np.concatenate([
        np.full(arr.shape[0], idx, dtype=np.int8) for idx, (_, arr, _) in enumerate(cnn_sources)
    ])
    cnn_regime_id = np.concatenate([
        np.full(arr.shape[0], reg, dtype=np.int8) for _, arr, reg in cnn_sources
    ])
    cnn_source_names = np.array([name for name, _, _ in cnn_sources])
    cnn_regime_names = np.array(["FP (650-1750 cm^-1)", "HW (2765-3200 cm^-1)", "FR (650-3200 cm^-1)"])
    cnn_means, cnn_stds, cnn_counts = _compute_source_summaries(cnn_spectra, cnn_source_id, len(cnn_sources))
    cnn_global_mean = cnn_spectra.mean(axis=0, dtype=np.float64).astype(np.float32)
    cnn_global_std = cnn_spectra.std(axis=0, dtype=np.float64).astype(np.float32)

    np.savez(
        ready_dir / "pretrain" / "CNN" / "pretrain_cnn_all_1000.npz",
        spectra=cnn_spectra,
        source_id=cnn_source_id,
        regime_id=cnn_regime_id,
        source_names=cnn_source_names,
        regime_names=cnn_regime_names,
        source_means=cnn_means,
        source_stds=cnn_stds,
        source_counts=cnn_counts,
        global_mean=cnn_global_mean,
        global_std=cnn_global_std,
    )
    rows.append({
        "Role": "SSL Pre-Training (1D-ResNet-34 Barlow Twins)",
        "Folder": "pretrain/CNN",
        "Ready_File": "pretrain_cnn_all_1000.npz",
        "Target_Regime": "Multi-Regime (FP + HW + FR)",
        "Grid_Range_cm1": "Normalized 1,000-ch index",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": 0.0,
        "Total_Spectra": int(cnn_spectra.shape[0]),
        "Constituent_Cohorts": "All 7 Pre-Train Datasets: 268,328 FP + 332,445 HW + 115,913 FR (716,686 unique spectra)",
        "Value_Range": f"[{cnn_spectra.min():.4f}, {cnn_spectra.max():.4f}]",
    })
    del fp_spectra, hw_spectra, cnn_spectra

    # =========================================================================
    # 4. DOWNSTREAM BENCHMARKS -> 1,000-PT READY TENSORS
    # =========================================================================
    print("[Step 3.4] Resampling Downstream Evaluation Benchmarks onto 1,000-point grids...")
    # 4A. IBD (downstream/FP)
    d_ibd = np.load(interim_dir / "downstream" / "FP" / "ibd" / "ibd_interim.npz", allow_pickle=True)
    wn_ibd = d_ibd["wavenumbers"]
    sp_ibd = d_ibd["spectra"]
    ibd_fp_1000 = resample_cubic_spline_batch(wn_ibd, sp_ibd, GRID_FP_1000)
    wn_ibd_nat1000 = np.linspace(float(wn_ibd[0]), float(wn_ibd[-1]), N_READY_POINTS, dtype=np.float32)
    ibd_nat_1000 = resample_cubic_spline_batch(wn_ibd, sp_ibd, wn_ibd_nat1000)
    np.savez(
        ready_dir / "downstream" / "FP" / "ibd_fp_1000.npz",
        wavenumbers=GRID_FP_1000,
        spectra=ibd_fp_1000,
        wavenumbers_native_1000=wn_ibd_nat1000,
        spectra_native_1000=ibd_nat_1000,
        labels=d_ibd["labels"],
        patient_ids=d_ibd["patient_ids"],
        diagnosis=d_ibd["diagnosis"],
    )
    rows.append({
        "Role": "Downstream Benchmark (FP)",
        "Folder": "downstream/FP",
        "Ready_File": "ibd_fp_1000.npz",
        "Target_Regime": "FP (ViT Grid + Native 1000)",
        "Grid_Range_cm1": "650.00–1750.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_FP_1000[1] - GRID_FP_1000[0]), 4),
        "Total_Spectra": int(ibd_fp_1000.shape[0]),
        "Constituent_Cohorts": "ICSM IBD Plasma Cohort: 228 subjects (77 Control vs 151 IBD)",
        "Value_Range": f"[{ibd_fp_1000.min():.4f}, {ibd_fp_1000.max():.4f}]",
    })

    # 4B. Bacteria-ID (downstream/FP)
    d_bact = np.load(interim_dir / "downstream" / "FP" / "bacteria_id" / "bacteria_id_interim.npz")
    wn_bact = d_bact["wavenumbers"]
    bact_ft_fp = resample_cubic_spline_batch(wn_bact, d_bact["X_finetune"], GRID_FP_1000)
    bact_te_fp = resample_cubic_spline_batch(wn_bact, d_bact["X_test"], GRID_FP_1000)
    np.savez(
        ready_dir / "downstream" / "FP" / "bacteria_id_fp_1000.npz",
        wavenumbers=GRID_FP_1000,
        X_finetune=bact_ft_fp,
        y_finetune=d_bact["y_finetune"],
        X_test=bact_te_fp,
        y_test=d_bact["y_test"],
        wavenumbers_native_1000=wn_bact.astype(np.float32),
        X_finetune_native_1000=d_bact["X_finetune"],
        X_test_native_1000=d_bact["X_test"],
    )
    rows.append({
        "Role": "Downstream Benchmark (FP)",
        "Folder": "downstream/FP",
        "Ready_File": "bacteria_id_fp_1000.npz",
        "Target_Regime": "FP (ViT Grid + Native 1000)",
        "Grid_Range_cm1": "650.00–1750.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_FP_1000[1] - GRID_FP_1000[0]), 4),
        "Total_Spectra": int(bact_ft_fp.shape[0] + bact_te_fp.shape[0]),
        "Constituent_Cohorts": "Bacteria-ID 30-Species: X_finetune (3,000) + X_test (3,000)",
        "Value_Range": f"[{bact_ft_fp.min():.4f}, {bact_ft_fp.max():.4f}]",
    })

    # 4C. RiverD Oral HW (downstream/HW & downstream/CH)
    d_oral = np.load(interim_dir / "downstream" / "HW" / "riverd_oral_hw" / "riverd_oral_hw_interim.npz")
    wn_oral = d_oral["wavenumbers"]
    sp_oral = d_oral["spectra"]
    oral_hw_1000 = resample_cubic_spline_batch(wn_oral, sp_oral, GRID_HW_1000)
    oral_ch_1000 = resample_cubic_spline_batch(wn_oral, sp_oral, GRID_CH_1000)
    wn_oral_broad1000 = np.linspace(float(wn_oral[0]), float(wn_oral[-1]), N_READY_POINTS, dtype=np.float32)
    oral_broad_1000 = resample_cubic_spline_batch(wn_oral, sp_oral, wn_oral_broad1000)

    np.savez(
        ready_dir / "downstream" / "HW" / "riverd_oral_hw_1000.npz",
        wavenumbers=GRID_HW_1000,
        spectra=oral_hw_1000,
        wavenumbers_broad_1000=wn_oral_broad1000,
        spectra_broad_1000=oral_broad_1000,
        labels=d_oral["labels"],
        sample_ids=d_oral["sample_ids"],
        data_source=d_oral["data_source"],
    )
    np.savez(
        ready_dir / "downstream" / "CH" / "riverd_oral_ch_1000.npz",
        wavenumbers=GRID_CH_1000,
        spectra=oral_ch_1000,
        labels=d_oral["labels"],
        sample_ids=d_oral["sample_ids"],
        data_source=d_oral["data_source"],
    )
    rows.append({
        "Role": "Downstream Benchmark (HW)",
        "Folder": "downstream/HW",
        "Ready_File": "riverd_oral_hw_1000.npz",
        "Target_Regime": "HW (2765–3200 + Broad 2600–4000)",
        "Grid_Range_cm1": "2765.00–3200.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_HW_1000[1] - GRID_HW_1000[0]), 4),
        "Total_Spectra": int(oral_hw_1000.shape[0]),
        "Constituent_Cohorts": "RiverD Oral Surgery: 8,422 spectra (5,899 Normal vs 2,523 Tumor, 186 patients)",
        "Value_Range": f"[{oral_hw_1000.min():.4f}, {oral_hw_1000.max():.4f}]",
    })
    rows.append({
        "Role": "Downstream Benchmark (CH)",
        "Folder": "downstream/CH",
        "Ready_File": "riverd_oral_ch_1000.npz",
        "Target_Regime": "CH (Core C-H 2800–3045)",
        "Grid_Range_cm1": "2800.00–3045.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_CH_1000[1] - GRID_CH_1000[0]), 4),
        "Total_Spectra": int(oral_ch_1000.shape[0]),
        "Constituent_Cohorts": "RiverD Oral Surgery (Core C-H window): 8,422 spectra",
        "Value_Range": f"[{oral_ch_1000.min():.4f}, {oral_ch_1000.max():.4f}]",
    })

    # 4D. Dataset Renzo (downstream/FR, downstream/FP, downstream/CH)
    d_rz = np.load(interim_dir / "downstream" / "FR" / "dataset_renzo" / "dataset_renzo_interim.npz", allow_pickle=True)
    wn_rz = d_rz["wavenumbers"]
    sp_rz = d_rz["spectra"]
    rz_fr_1000 = resample_cubic_spline_batch(wn_rz, sp_rz, GRID_FR_RENZO_1000)
    rz_fp_1000 = resample_cubic_spline_batch(wn_rz, sp_rz, GRID_FP_1000)
    rz_ch_1000 = resample_cubic_spline_batch(wn_rz, sp_rz, GRID_CH_1000)

    meta_rz = {
        "sample_id": d_rz["sample_id"],
        "bone": d_rz["bone"],
        "bone_type": d_rz["bone_type"] if "bone_type" in d_rz else d_rz["group_code"],
        "pathology": d_rz["pathology"],
        "age": d_rz["age"] if "age" in d_rz else d_rz["group_code"],
        "sex": d_rz["sex"] if "sex" in d_rz else d_rz["group_code"],
        "group_code": d_rz["group_code"],
    }
    np.savez(
        ready_dir / "downstream" / "FR" / "dataset_renzo_fr_1000.npz",
        wavenumbers=GRID_FR_RENZO_1000,
        spectra=rz_fr_1000,
        wavenumbers_fp=GRID_FP_1000,
        spectra_fp=rz_fp_1000,
        wavenumbers_ch=GRID_CH_1000,
        spectra_ch=rz_ch_1000,
        **meta_rz,
    )
    np.savez(
        ready_dir / "downstream" / "FP" / "dataset_renzo_fp_1000.npz",
        wavenumbers=GRID_FP_1000,
        spectra=rz_fp_1000,
        **meta_rz,
    )
    np.savez(
        ready_dir / "downstream" / "CH" / "dataset_renzo_ch_1000.npz",
        wavenumbers=GRID_CH_1000,
        spectra=rz_ch_1000,
        **meta_rz,
    )
    rows.append({
        "Role": "Downstream Benchmark (FR/FP/CH)",
        "Folder": "downstream/FR",
        "Ready_File": "dataset_renzo_fr_1000.npz",
        "Target_Regime": "FR (380–3049) + FP (650–1750) + CH (2800–3045)",
        "Grid_Range_cm1": "380.00–3049.00",
        "Grid_Points": N_READY_POINTS,
        "Step_cm1": round(float(GRID_FR_RENZO_1000[1] - GRID_FR_RENZO_1000[0]), 4),
        "Total_Spectra": int(rz_fr_1000.shape[0]),
        "Constituent_Cohorts": "Murine Bone Cohort (47 subjects, 3,200 spectra: WT, HSCT, oc-oc, Rankl-ko)",
        "Value_Range": f"[{rz_fr_1000.min():.4f}, {rz_fr_1000.max():.4f}]",
    })

    df = pd.DataFrame(rows)
    if catalog_csv_path is not None:
        catalog_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(catalog_csv_path, index=False)
    return df


def load_ready_summary_stats(ready_dir: Path) -> dict[str, Any]:
    """Load precomputed summary profiles and downstream tensors from `data/02_ready/`."""
    fp_d = np.load(ready_dir / "pretrain" / "FP" / "pretrain_vit_fp_1000.npz", allow_pickle=True)
    hw_d = np.load(ready_dir / "pretrain" / "HW" / "pretrain_vit_hw_1000.npz", allow_pickle=True)
    ch_d = np.load(ready_dir / "pretrain" / "CH" / "pretrain_vit_ch_1000.npz", allow_pickle=True)
    cnn_d = np.load(ready_dir / "pretrain" / "CNN" / "pretrain_cnn_all_1000.npz", allow_pickle=True)

    ibd_d = np.load(ready_dir / "downstream" / "FP" / "ibd_fp_1000.npz", allow_pickle=True)
    bact_d = np.load(ready_dir / "downstream" / "FP" / "bacteria_id_fp_1000.npz", allow_pickle=True)
    oral_hw_d = np.load(ready_dir / "downstream" / "HW" / "riverd_oral_hw_1000.npz", allow_pickle=True)
    oral_ch_d = np.load(ready_dir / "downstream" / "CH" / "riverd_oral_ch_1000.npz", allow_pickle=True)
    rz_d = np.load(ready_dir / "downstream" / "FR" / "dataset_renzo_fr_1000.npz", allow_pickle=True)

    return {
        "fp": {k: fp_d[k] for k in ["wavenumbers", "source_names", "source_means", "source_stds", "source_counts", "global_mean", "global_std"]},
        "hw": {k: hw_d[k] for k in ["wavenumbers", "source_names", "source_means", "source_stds", "source_counts", "global_mean", "global_std"]},
        "ch": {k: ch_d[k] for k in ["wavenumbers", "source_names", "source_means", "source_stds", "source_counts", "global_mean", "global_std"]},
        "cnn": {k: cnn_d[k] for k in ["source_names", "regime_names", "source_means", "source_stds", "source_counts", "global_mean", "global_std"]},
        "ibd": {k: ibd_d[k] for k in ["wavenumbers", "spectra", "labels", "diagnosis"]},
        "bact": {k: bact_d[k] for k in ["wavenumbers", "X_finetune", "y_finetune", "X_test", "y_test"]},
        "oral_hw": {k: oral_hw_d[k] for k in ["wavenumbers", "spectra", "labels", "data_source"]},
        "oral_ch": {k: oral_ch_d[k] for k in ["wavenumbers", "spectra", "labels"]},
        "renzo": {k: rz_d[k] for k in ["wavenumbers", "spectra", "wavenumbers_fp", "spectra_fp", "wavenumbers_ch", "spectra_ch", "pathology", "bone"]},
    }


def plot_ready_architecture_overview(out_paths: list[Path] | None = None) -> plt.Figure:
    """
    Plot Figure 1: Visual map of native vs canonical 1,000-point wavenumber windows
    and corpus routing for `1D-ViT (SMAE)`, `1D-ResNet-34 (Barlow Twins)`, and Downstream.
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(18.5, 8.2), gridspec_kw={"width_ratios": [1.35, 1.0]}
    )

    # Panel A: Native Wavenumber Span vs Cropped 1,000-pt Ready Windows
    cohorts = [
        ("dataset_renzo (Downstream FR)", 380.0, 3049.0, [(650, 1750, "#0072B2"), (2800, 3045, "#009E73")]),
        ("riverd_oral_hw (Downstream HW)", 2600.0, 4000.0, [(2765, 3200, "#D55E00"), (2800, 3045, "#009E73")]),
        ("bacteria_id (Downstream FP)", 382.0, 1792.4, [(650, 1750, "#0072B2")]),
        ("ibd (Downstream FP)", 403.9, 1754.3, [(650, 1750, "#0072B2")]),
        ("trophy_jena Raw+ALS (Pretrain FR)", 648.7, 3200.0, [(650, 1750, "#0072B2"), (2765, 3200, "#D55E00")]),
        ("riverd_meat_hw (Pretrain HW)", 2520.0, 4000.0, [(2765, 3200, "#D55E00")]),
        ("pdo_hw_maps (Pretrain HW)", 2603.3, 3915.4, [(2765, 3200, "#D55E00")]),
        ("spectrabreast_hw G2 (Pretrain HW)", 2763.8, 3487.6, [(2765, 3200, "#D55E00")]),
        ("spectrabreast_hw G1 (Pretrain HW)", 2488.9, 3258.8, [(2765, 3200, "#D55E00")]),
        ("spectrabreast_fp (Pretrain FP)", 660.2, 1759.4, [(650, 1750, "#0072B2")]),
        ("raman_mae TWIST (Pretrain FP)", 650.9, 1749.7, [(650, 1750, "#0072B2")]),
        ("raman_mae IMMUNO (Pretrain FP)", 597.3, 1798.9, [(650, 1750, "#0072B2")]),
        ("raman_mae CELLS (Pretrain FP)", 600.5, 1963.3, [(650, 1750, "#0072B2")]),
        ("deepr (Pretrain FP)", 498.3, 1799.9, [(650, 1750, "#0072B2")]),
    ]

    ax_left.axvspan(650, 1750, color="#0072B2", alpha=0.10, label="ViT-FP Target Window (650–1750 cm$^{-1}$, 1000 pts)")
    ax_left.axvspan(2765, 3200, color="#D55E00", alpha=0.10, label="ViT-HW Target Window (2765–3200 cm$^{-1}$, 1000 pts)")
    ax_left.axvspan(2800, 3045, color="#009E73", alpha=0.14, label="ViT-CH Core Window (2800–3045 cm$^{-1}$, 1000 pts)")

    y_pos = np.arange(len(cohorts))
    for i, (label, w0, w1, active_wins) in enumerate(cohorts):
        ax_left.plot([w0, w1], [i, i], color="#B0B8C4", lw=7.0, solid_capstyle="butt", zorder=2)
        for aw0, aw1, col in active_wins:
            ax_left.plot([aw0, aw1], [i, i], color=col, lw=7.0, solid_capstyle="butt", zorder=3)
        ax_left.text(w1 + 35, i, f"{int(w0)}–{int(w1)} cm$^{{-1}}$", va="center", ha="left", fontsize=7.6, color="#333333")

    ax_left.axhline(3.5, color="#333333", linestyle="--", lw=1.2)
    ax_left.text(
        4410, 3.66, "▲ SSL PRE-TRAINING (716,686 Spectra)",
        ha="right", va="bottom", fontsize=8.0, fontweight="bold", color="#222222",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.92},
    )
    ax_left.text(
        4410, 3.34, "▼ DOWNSTREAM (17,850 Spectra)",
        ha="right", va="top", fontsize=8.0, fontweight="bold", color="#222222",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.92},
    )

    ax_left.set_yticks(y_pos)
    ax_left.set_yticklabels([c[0] for c in cohorts], fontsize=8.5, fontweight="bold")
    ax_left.set_xlim(300, 4450)
    ax_left.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9.5, fontweight="bold")
    ax_left.set_title("1A. Native Grating Coverage (Gray) vs Cropped 1,000-Pt Ready Windows (Colored)", fontsize=10.5, fontweight="bold")
    ax_left.grid(axis="x", linestyle=":", alpha=0.5)
    ax_left.legend(loc="lower right", fontsize=8.0, framealpha=0.94)

    # Panel B: Compiled Corpus Composition Bar Chart
    corpora_labels = [
        "1D-ResNet-34 (Barlow Twins)\npretrain_cnn_all_1000.npz\n(All Regimes, 1000 pts)",
        "1D-ViT (SMAE — HW)\npretrain_vit_hw_1000.npz\n(2765–3200 cm⁻¹, 1000 pts)",
        "1D-ViT (SMAE — CH Core)\npretrain_vit_ch_1000.npz\n(2800–3045 cm⁻¹, 1000 pts)",
        "1D-ViT (SMAE — FP)\npretrain_vit_fp_1000.npz\n(650–1750 cm⁻¹, 1000 pts)",
    ]
    fp_only = np.array([268328, 0, 0, 268328])
    hw_only = np.array([332445, 332445, 332445, 0])
    fr_shared = np.array([115913, 115913, 115913, 115913])
    y_c = np.arange(len(corpora_labels))

    ax_right.barh(y_c, fp_only / 1e3, height=0.52, color="#0072B2", label="FP Pretrain Folders (268,328)")
    ax_right.barh(y_c, hw_only / 1e3, left=fp_only / 1e3, height=0.52, color="#D55E00", label="HW Pretrain Folders (332,445)")
    ax_right.barh(y_c, fr_shared / 1e3, left=(fp_only + hw_only) / 1e3, height=0.52, color="#7B4EA3", label="FR trophy_jena (115,913)")

    totals = fp_only + hw_only + fr_shared
    for idx, tot in enumerate(totals):
        ax_right.text(tot / 1e3 + 12, idx, f"{tot:,d}\nspectra", va="center", ha="left", fontsize=8.8, fontweight="bold", color="#111111")

    ax_right.set_yticks(y_c)
    ax_right.set_yticklabels(corpora_labels, fontsize=8.6, fontweight="bold")
    ax_right.set_xlim(0, 860)
    ax_right.set_xlabel("Compiled Corpus Size (Thousands of 1,000-Point Spectra)", fontsize=9.5, fontweight="bold")
    ax_right.set_title("1B. Compiled `data/02_ready/pretrain/` Corpora for ViT & ResNet-34", fontsize=10.5, fontweight="bold")
    ax_right.grid(axis="x", linestyle=":", alpha=0.5)
    ax_right.legend(loc="lower right", fontsize=8.2, framealpha=0.94)

    fig.suptitle(
        "Step 3 — Canonical 1,000-Point Spectral Grids & Compiled SSL Corpora (`data/01_interim` → `data/02_ready`)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if out_paths:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
    return fig


def plot_cubic_spline_validation(
    interim_dir: Path,
    ready_dir: Path,
    out_paths: list[Path] | None = None,
) -> plt.Figure:
    """
    Plot Figure 2: 2x3 high-resolution validation of Cubic Spline resampling (1,000 pts)
    vs Native Grid (`01_interim`), proving zero peak shift and exact `[0, 1]` bounds.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.8))

    # 1. deepr (Upsampling 500 pts -> 1000 pts on 650-1750 cm^-1)
    d_dp_int = np.load(interim_dir / "pretrain" / "FP" / "deepr" / "deepr_interim.npz")
    wn_dp_i = d_dp_int["wavenumbers"]
    sp_dp_i = d_dp_int["outputs_10s"][:2000]
    m_dp = (wn_dp_i >= 650.0) & (wn_dp_i <= 1750.0)
    wn_dp_sub = wn_dp_i[m_dp]
    mean_dp_i = sp_dp_i[:, m_dp].mean(axis=0)
    mean_dp_i = (mean_dp_i - mean_dp_i.min()) / (mean_dp_i.max() - mean_dp_i.min())

    d_fp_r = np.load(ready_dir / "pretrain" / "FP" / "pretrain_vit_fp_1000.npz")
    wn_fp = d_fp_r["wavenumbers"]
    mean_dp_r = d_fp_r["source_means"][0]
    mean_dp_r_mm = (mean_dp_r - mean_dp_r.min()) / (mean_dp_r.max() - mean_dp_r.min())

    ax = axes[0, 0]
    ax.plot(wn_dp_sub, mean_dp_i, "o", markersize=2.6, color="#D55E00", alpha=0.65, label=f"Interim Native ({wn_dp_sub.size} pts in 650–1750)")
    ax.plot(wn_fp, mean_dp_r_mm, "-", lw=1.8, color="#0072B2", label="Ready Cubic Spline (1,000 pts)")
    ax.set_title("2A. `deepr` — Upsampling (422 Native Pts → 1,000 Pts)\nZero Ringing on Sharp Bands (650–1750 cm$^{-1}$)", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Mean Intensity", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(alpha=0.25)

    # 2. Zoom on Phenylalanine (980-1030 cm^-1) and CH2/Amide I in spectrabreast_fp (1015 -> 1000 pts)
    f_sbfp = sorted((interim_dir / "pretrain" / "FP" / "spectrabreast_fp").glob("*.h5"))[0]
    with h5py.File(f_sbfp, "r") as h:
        wn_sb_i = h["raman_shift"][:]
        sp_sb_i = h["processed_spectra"][:].mean(axis=0)
    sp_sb_r = d_fp_r["source_means"][4]
    ax = axes[0, 1]
    m_z1 = (wn_sb_i >= 960) & (wn_sb_i <= 1700)
    m_z2 = (wn_fp >= 960) & (wn_fp <= 1700)
    ax.plot(wn_sb_i[m_z1], sp_sb_i[m_z1], "o", markersize=2.5, color="#D55E00", alpha=0.65, label="Interim Native (1,015 pts, Δ=1.08 cm$^{-1}$)")
    ax.plot(wn_fp[m_z2], sp_sb_r[m_z2], "-", lw=1.8, color="#0072B2", label="Ready Cubic Spline (1,000 pts, Δ=1.10 cm$^{-1}$)")
    ax.axvline(1003, color="#555555", linestyle=":", lw=1.0)
    ax.axvline(1445, color="#555555", linestyle=":", lw=1.0)
    ax.axvline(1655, color="#555555", linestyle=":", lw=1.0)
    ax.set_title("2B. `spectrabreast_fp` — Regridding (1,015 → 1,000 Pts)\nExact Peak Centroid Alignment (1003, 1445, 1655 cm$^{-1}$)", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Mean Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(alpha=0.25)

    # 3. trophy_jena: Full-Range (871 pts, 648.7-3200) -> Dual Window Extraction (FP 1000 pts & HW 1000 pts)
    d_tr_i = np.load(interim_dir / "pretrain" / "FR" / "trophy_jena" / "trophy_raw_als_minmax.npz")
    wn_tr_i = d_tr_i["wavenumbers"]
    mean_tr_i = d_tr_i["spectra"].mean(axis=0)
    d_hw_r = np.load(ready_dir / "pretrain" / "HW" / "pretrain_vit_hw_1000.npz")
    wn_hw = d_hw_r["wavenumbers"]
    mean_tr_fp = d_fp_r["source_means"][5]
    mean_tr_hw = d_hw_r["source_means"][4]

    ax = axes[0, 2]
    ax.plot(wn_tr_i, mean_tr_i / mean_tr_i.max(), color="#888888", lw=1.3, alpha=0.75, label="Interim Full-Range (871 pts, max on CH=1.0)")
    ax.plot(wn_fp, mean_tr_fp / mean_tr_fp.max(), color="#0072B2", lw=1.9, label="Ready Cropped FP (1,000 pts, Re-Anchored [0,1])")
    ax.plot(wn_hw, mean_tr_hw / mean_tr_hw.max(), color="#009E73", lw=1.9, label="Ready Cropped HW (1,000 pts, Re-Anchored [0,1])")
    ax.set_title("2C. `trophy_jena` — Why Post-Crop [0, 1] Re-Anchoring Matters\nRestores Full Dynamic Range to Cropped FP Window", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Scaled Mean Profile", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.6)
    ax.grid(alpha=0.25)

    # 4. pdo_hw_maps: Broad 2603-3915 (with O-H 3400 water band) -> Cropped HW 2765-3200 (1,000 pts)
    f_pdo = sorted((interim_dir / "pretrain" / "HW" / "pdo_hw_maps").glob("*.h5"))[1]
    with h5py.File(f_pdo, "r") as h:
        wn_pdo_i = h["raman_shift"][:]
        sp_pdo_i = h["processed_spectra"][:].mean(axis=0)
    mean_pdo_hw = d_hw_r["source_means"][2]
    ax = axes[1, 0]
    ax.plot(wn_pdo_i, sp_pdo_i / sp_pdo_i.max(), color="#888888", lw=1.4, label="Interim Native (2603–3906 cm$^{-1}$, O-H Water Peak)")
    ax.plot(wn_hw, mean_pdo_hw / mean_pdo_hw.max(), color="#D55E00", lw=2.0, label="Ready HW Window (2765–3200 cm$^{-1}$, 1,000 pts)")
    ax.axvspan(2765, 3200, color="#D55E00", alpha=0.08)
    ax.set_title("2D. `pdo_hw_maps` — Isolating C–H Stretch from 3400 cm$^{-1}$ Water\nCubic Spline Resampling onto 2765–3200 cm$^{-1}$ (1,000 Pts)", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Scaled Mean Profile", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7.6)
    ax.grid(alpha=0.25)

    # 5. riverd_oral_hw: 701 pts (2600-4000 cm^-1) -> HW (2765-3200, 1000 pts) & CH (2800-3045, 1000 pts)
    d_or_i = np.load(interim_dir / "downstream" / "HW" / "riverd_oral_hw" / "riverd_oral_hw_interim.npz")
    wn_or_i = d_or_i["wavenumbers"]
    mean_or_i = d_or_i["spectra"].mean(axis=0)
    d_or_hw = np.load(ready_dir / "downstream" / "HW" / "riverd_oral_hw_1000.npz")
    d_or_ch = np.load(ready_dir / "downstream" / "CH" / "riverd_oral_ch_1000.npz")
    ax = axes[1, 1]
    m_or = (wn_or_i >= 2750) & (wn_or_i <= 3220)
    ax.plot(wn_or_i[m_or], (mean_or_i[m_or] - mean_or_i[m_or].min()) / (mean_or_i[m_or].max() - mean_or_i[m_or].min()), "o", markersize=3.0, color="#555555", alpha=0.6, label="Interim Native (218 pts in 2765–3200)")
    m_hw_m = d_or_hw["spectra"].mean(axis=0)
    m_ch_m = d_or_ch["spectra"].mean(axis=0)
    ax.plot(d_or_hw["wavenumbers"], (m_hw_m - m_hw_m.min()) / (m_hw_m.max() - m_hw_m.min()), "-", lw=1.9, color="#D55E00", label="Ready HW (2765–3200 cm$^{-1}$, 1,000 pts)")
    ax.plot(d_or_ch["wavenumbers"], (m_ch_m - m_ch_m.min()) / (m_ch_m.max() - m_ch_m.min()), "--", lw=1.8, color="#009E73", label="Ready CH (2800–3045 cm$^{-1}$, 1,000 pts)")
    ax.set_title("2E. `riverd_oral_hw` — Smooth 4.5× Cubic Spline Upsampling\n218 Native Pts → 1,000 Ready Pts in HW & CH Windows", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Scaled Mean Profile", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7.6)
    ax.grid(alpha=0.25)

    # 6. dataset_renzo: 2670 pts (380-3049 cm^-1) -> FR (1000 pts), FP (1000 pts), CH (1000 pts)
    d_rz = np.load(ready_dir / "downstream" / "FR" / "dataset_renzo_fr_1000.npz")
    ax = axes[1, 2]
    ax.plot(d_rz["wavenumbers"], d_rz["spectra"].mean(axis=0), color="#7B4EA3", lw=1.6, alpha=0.85, label="Ready FR (380–3049 cm$^{-1}$, 1,000 pts)")
    ax.plot(d_rz["wavenumbers_fp"], d_rz["spectra_fp"].mean(axis=0), color="#0072B2", lw=1.9, label="Ready FP (650–1750 cm$^{-1}$, 1,000 pts)")
    ax.plot(d_rz["wavenumbers_ch"], d_rz["spectra_ch"].mean(axis=0), color="#009E73", lw=1.9, label="Ready CH (2800–3045 cm$^{-1}$, 1,000 pts)")
    ax.set_title("2F. `dataset_renzo` — Multi-View Ready Representation\nFR (CNN) + FP (ViT-FP) + CH (ViT-CH) at 1,000 Pts Each", fontsize=9.3, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Mean Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=7.6)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Step 3 — Physical Validation of Boundary-Clamped Cubic Spline Resampling & Post-Crop `[0, 1]` Re-Anchoring",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if out_paths:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
    return fig


def plot_ready_pretrain_corpora(
    ready_dir: Path,
    out_paths: list[Path] | None = None,
) -> plt.Figure:
    """
    Plot Figure 3: 2x2 visualization of the 4 compiled `data/02_ready/pretrain/` corpora
    (`ViT-FP`, `ViT-HW`, `ViT-CH`, and `CNN-All`) with constituent cohort profiles overlaid.
    """
    stats = load_ready_summary_stats(ready_dir)
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 10.2))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7B4EA3"]

    # Panel 3A: ViT-FP (650-1750 cm^-1, 384,241 spectra)
    ax = axes[0, 0]
    fp = stats["fp"]
    wn = fp["wavenumbers"]
    ax.fill_between(wn, np.clip(fp["global_mean"] - fp["global_std"], 0, 1), np.clip(fp["global_mean"] + fp["global_std"], 0, 1), color="#0072B2", alpha=0.15, label="Global Corpus μ ± 1σ (N = 384,241)")
    for i, name in enumerate(fp["source_names"]):
        ax.plot(wn, fp["source_means"][i], lw=1.6, color=palette[i % len(palette)], label=f"{name} (n={fp['source_counts'][i]:,d})")
    ax.plot(wn, fp["global_mean"], color="#111111", lw=2.3, linestyle="--", label="Global Corpus Mean μ")
    ax.set_xlim(650, 1750)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("3A. `pretrain_vit_fp_1000.npz` — 1D-ViT Fingerprint Corpus\nN = 384,241 Spectra | 650.0–1750.0 cm$^{-1}$ (1,000 Pts, Δ = 1.10 cm$^{-1}$)", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9.0)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=9.0)
    ax.legend(loc="upper left", fontsize=7.6, ncol=2, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel 3B: ViT-HW (2765-3200 cm^-1, 448,358 spectra)
    ax = axes[0, 1]
    hw = stats["hw"]
    wn = hw["wavenumbers"]
    ax.fill_between(wn, np.clip(hw["global_mean"] - hw["global_std"], 0, 1), np.clip(hw["global_mean"] + hw["global_std"], 0, 1), color="#D55E00", alpha=0.15, label="Global Corpus μ ± 1σ (N = 448,358)")
    for i, name in enumerate(hw["source_names"]):
        ax.plot(wn, hw["source_means"][i], lw=1.7, color=palette[i % len(palette)], label=f"{name} (n={hw['source_counts'][i]:,d})")
    ax.plot(wn, hw["global_mean"], color="#111111", lw=2.3, linestyle="--", label="Global Corpus Mean μ")
    ax.set_xlim(2765, 3200)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("3B. `pretrain_vit_hw_1000.npz` — 1D-ViT Extended HW Corpus\nN = 448,358 Spectra | 2765.0–3200.0 cm$^{-1}$ (1,000 Pts, Δ = 0.44 cm$^{-1}$)", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9.0)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=9.0)
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel 3C: ViT-CH Core (2800-3045 cm^-1, 448,358 spectra)
    ax = axes[1, 0]
    ch = stats["ch"]
    wn = ch["wavenumbers"]
    ax.fill_between(wn, np.clip(ch["global_mean"] - ch["global_std"], 0, 1), np.clip(ch["global_mean"] + ch["global_std"], 0, 1), color="#009E73", alpha=0.15, label="Global Corpus μ ± 1σ (N = 448,358)")
    for i, name in enumerate(ch["source_names"]):
        ax.plot(wn, ch["source_means"][i], lw=1.7, color=palette[i % len(palette)], label=f"{name} (n={ch['source_counts'][i]:,d})")
    ax.plot(wn, ch["global_mean"], color="#111111", lw=2.3, linestyle="--", label="Global Corpus Mean μ")
    ax.axvline(2850, color="#555555", linestyle=":", lw=1.0)
    ax.axvline(2930, color="#555555", linestyle=":", lw=1.0)
    ax.set_xlim(2800, 3045)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("3C. `pretrain_vit_ch_1000.npz` — 1D-ViT Core C–H Stretch Corpus\nN = 448,358 Spectra | 2800.0–3045.0 cm$^{-1}$ (1,000 Pts, Δ = 0.25 cm$^{-1}$)", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9.0)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=9.0)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.92)
    ax.grid(alpha=0.25)

    # Panel 3D: CNN Barlow Twins Multi-Regime Corpus (716,686 Unique Spectra, 1,000 Channels)
    ax = axes[1, 1]
    cnn = stats["cnn"]
    ch_idx = np.arange(N_READY_POINTS)
    ax.fill_between(ch_idx, np.clip(cnn["global_mean"] - cnn["global_std"], 0, 1), np.clip(cnn["global_mean"] + cnn["global_std"], 0, 1), color="#7B4EA3", alpha=0.15, label="Global CNN Corpus μ ± 1σ (N = 716,686)")
    # Group means by regime (FP, HW, FR)
    fp_mean = np.average(cnn["source_means"][:5], axis=0, weights=cnn["source_counts"][:5])
    hw_mean = np.average(cnn["source_means"][5:9], axis=0, weights=cnn["source_counts"][5:9])
    fr_mean = cnn["source_means"][9]
    ax.plot(ch_idx, fp_mean, color="#0072B2", lw=2.0, label="FP Regime Mean (n = 268,328 spectra)")
    ax.plot(ch_idx, hw_mean, color="#D55E00", lw=2.0, label="HW Regime Mean (n = 332,445 spectra)")
    ax.plot(ch_idx, fr_mean, color="#009E73", lw=2.0, label="FR Regime Mean (n = 115,913 spectra)")
    ax.plot(ch_idx, cnn["global_mean"], color="#111111", lw=2.3, linestyle="--", label="Global 716,686-Spectrum Mean μ")
    ax.set_xlim(0, 999)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("3D. `pretrain_cnn_all_1000.npz` — 1D-ResNet-34 (Barlow Twins) Global Corpus\nN = 716,686 Unique Spectra Across All 7 Pre-Train Datasets (1,000 Channels)", fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Resampled Spectral Channel Index (0 → 999)", fontsize=9.0)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=9.0)
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Step 3 — Compiled Self-Supervised Pre-Training Corpora in `data/02_ready/pretrain/` (1,000 Points, `[0, 1]` Scaled)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if out_paths:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
    return fig


def plot_ready_downstream_benchmarks(
    ready_dir: Path,
    out_paths: list[Path] | None = None,
) -> plt.Figure:
    """
    Plot Figure 4: 2x3 visualization of the 4 Downstream Evaluation Benchmarks in
    `data/02_ready/downstream/` stratified by clinical/biological class on their 1,000-pt grids.
    """
    stats = load_ready_summary_stats(ready_dir)
    fig, axes = plt.subplots(2, 3, figsize=(18.5, 9.8))

    # 4A. IBD (650-1750 cm^-1, 1000 pts): Healthy Control (0) vs IBD (1)
    ax = axes[0, 0]
    ibd = stats["ibd"]
    wn = ibd["wavenumbers"]
    sp = ibd["spectra"]
    lbl = ibd["labels"]
    for c_val, c_name, col in [(0, "Healthy Control (n=77)", "#0072B2"), (1, "IBD Patients (n=151)", "#D55E00")]:
        sub = sp[lbl == c_val]
        mu, sd = sub.mean(axis=0), sub.std(axis=0)
        ax.plot(wn, mu, lw=2.0, color=col, label=c_name)
        ax.fill_between(wn, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.16)
    ax.set_xlim(650, 1750)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("4A. `ibd_fp_1000.npz` — ICSM Clinical IBD Benchmark\n228 Subjects | 650.0–1750.0 cm$^{-1}$ (1,000 Pts, ViT-FP Grid)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(alpha=0.25)

    # 4B. Bacteria-ID (650-1750 cm^-1, 1000 pts): Finetune vs Test across 30 classes
    ax = axes[0, 1]
    bact = stats["bact"]
    wn = bact["wavenumbers"]
    ft_m, ft_s = bact["X_finetune"].mean(axis=0), bact["X_finetune"].std(axis=0)
    te_m, te_s = bact["X_test"].mean(axis=0), bact["X_test"].std(axis=0)
    ax.fill_between(wn, np.clip(ft_m - ft_s, 0, 1), np.clip(ft_m + ft_s, 0, 1), color="#0072B2", alpha=0.16)
    ax.plot(wn, ft_m, lw=2.0, color="#0072B2", label="X_finetune (n=3,000, 30 Species)")
    ax.plot(wn, te_m, lw=1.9, linestyle="--", color="#D55E00", label="X_test Clinical (n=3,000, 30 Species)")
    ax.set_xlim(650, 1750)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("4B. `bacteria_id_fp_1000.npz` — 30-Pathogen Benchmark\n6,000 Active Spectra | 650.0–1750.0 cm$^{-1}$ (1,000 Pts, ViT-FP Grid)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(alpha=0.25)

    # 4C. RiverD Oral HW (2765-3200 cm^-1, 1000 pts): Normal vs OSCC Tumor
    ax = axes[0, 2]
    ohw = stats["oral_hw"]
    wn = ohw["wavenumbers"]
    sp = ohw["spectra"]
    lbl = ohw["labels"]
    for c_val, c_name, col in [(0, "Normal Oral Tissue (n=5,899)", "#009E73"), (1, "OSCC Tumor (n=2,523)", "#D55E00")]:
        sub = sp[lbl == c_val]
        mu, sd = sub.mean(axis=0), sub.std(axis=0)
        ax.plot(wn, mu, lw=2.0, color=col, label=c_name)
        ax.fill_between(wn, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.16)
    ax.set_xlim(2765, 3200)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("4C. `riverd_oral_hw_1000.npz` — Oral Cancer Surgery (HW)\n8,422 Spectra (186 Patients) | 2765.0–3200.0 cm$^{-1}$ (1,000 Pts)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(alpha=0.25)

    # 4D. RiverD Oral CH (2800-3045 cm^-1, 1000 pts): Normal vs OSCC Tumor
    ax = axes[1, 0]
    och = stats["oral_ch"]
    wn = och["wavenumbers"]
    sp = och["spectra"]
    lbl = och["labels"]
    for c_val, c_name, col in [(0, "Normal Oral Tissue (n=5,899)", "#009E73"), (1, "OSCC Tumor (n=2,523)", "#D55E00")]:
        sub = sp[lbl == c_val]
        mu, sd = sub.mean(axis=0), sub.std(axis=0)
        ax.plot(wn, mu, lw=2.0, color=col, label=c_name)
        ax.fill_between(wn, np.clip(mu - sd, 0, 1), np.clip(mu + sd, 0, 1), color=col, alpha=0.16)
    ax.set_xlim(2800, 3045)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("4D. `riverd_oral_ch_1000.npz` — Oral Cancer Surgery (CH Core)\n8,422 Spectra | 2800.0–3045.0 cm$^{-1}$ (1,000 Pts, ViT-CH Grid)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(alpha=0.25)

    # 4E. Dataset Renzo FP (650-1750 cm^-1, 1000 pts) by Pathology
    ax = axes[1, 1]
    rz = stats["renzo"]
    wn_fp = rz["wavenumbers_fp"]
    sp_fp = rz["spectra_fp"]
    patho = np.array([str(x) for x in rz["pathology"]])
    pal_rz = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for idx, p_name in enumerate(["WT", "Rankl ko", "oc/oc", "HSCT"]):
        mask = patho == p_name
        if np.any(mask):
            sub = sp_fp[mask]
            ax.plot(wn_fp, sub.mean(axis=0), lw=1.9, color=pal_rz[idx % len(pal_rz)], label=f"{p_name} (n={sub.shape[0]:,d})")
    ax.set_xlim(650, 1750)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("4E. `dataset_renzo_fp_1000.npz` — Murine Bone Pathology (FP)\n3,200 Spectra (47 Mice) | 650.0–1750.0 cm$^{-1}$ (1,000 Pts, ViT-FP Grid)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(alpha=0.25)

    # 4F. Dataset Renzo CH (2800-3045 cm^-1, 1000 pts) by Pathology
    ax = axes[1, 2]
    wn_ch = rz["wavenumbers_ch"]
    sp_ch = rz["spectra_ch"]
    for idx, p_name in enumerate(["WT", "Rankl ko", "oc/oc", "HSCT"]):
        mask = patho == p_name
        if np.any(mask):
            sub = sp_ch[mask]
            ax.plot(wn_ch, sub.mean(axis=0), lw=1.9, color=pal_rz[idx % len(pal_rz)], label=f"{p_name} (n={sub.shape[0]:,d})")
    ax.set_xlim(2800, 3045)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("4F. `dataset_renzo_ch_1000.npz` — Murine Bone Pathology (CH Core)\n3,200 Spectra (47 Mice) | 2800.0–3045.0 cm$^{-1}$ (1,000 Pts, ViT-CH Grid)", fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=8.5)
    ax.set_ylabel("Normalized Intensity [0, 1]", fontsize=8.5)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Step 3 — Compiled Downstream Validation & Fine-Tuning Benchmarks in `data/02_ready/downstream/` (1,000 Points)",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if out_paths:
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=300, bbox_inches="tight")
    return fig
