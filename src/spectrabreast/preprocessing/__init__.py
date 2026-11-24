"""Preprocessing utilities for SPECTRA-BREAST Raman spectroscopy datasets."""

from .cleaning import (
    als_baseline_batch,
    clean_raw_spectra_batch,
    despike_whitaker_hayes_batch,
    minmax_normalize_batch,
)
from .interim_atlas import (
    compute_stats,
    load_fp_before_after,
    load_fr_before_after,
    load_hw_before_after,
    plot_als_validation_grid,
    plot_fp_before_after_grid,
    plot_fr_before_after_grid,
    plot_hw_before_after_grid,
    plot_profile_panel,
)
from .ready_atlas import (
    compile_ready_datasets,
    load_ready_summary_stats,
    plot_cubic_spline_validation,
    plot_ready_architecture_overview,
    plot_ready_downstream_benchmarks,
    plot_ready_pretrain_corpora,
)
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
    "als_baseline_batch",
    "clean_raw_spectra_batch",
    "despike_whitaker_hayes_batch",
    "minmax_normalize_batch",
    "compute_stats",
    "plot_profile_panel",
    "load_fp_before_after",
    "load_hw_before_after",
    "load_fr_before_after",
    "plot_als_validation_grid",
    "plot_fp_before_after_grid",
    "plot_hw_before_after_grid",
    "plot_fr_before_after_grid",
    "N_READY_POINTS",
    "GRID_FP_1000",
    "GRID_HW_1000",
    "GRID_CH_1000",
    "GRID_FR_JENA_1000",
    "GRID_FR_RENZO_1000",
    "resample_cubic_spline_batch",
    "compile_ready_datasets",
    "load_ready_summary_stats",
    "plot_ready_architecture_overview",
    "plot_cubic_spline_validation",
    "plot_ready_pretrain_corpora",
    "plot_ready_downstream_benchmarks",
]
