"""IO utilities for SPECTRA-BREAST."""

from spectrabreast.io.ch_maps import CHMap, CHMapMetadata, plot_ch_map, read_ch_map, scan_ch_maps
from spectrabreast.io.raw_atlas import (
    accumulate_h5_stats,
    add_spectral_bands,
    compute_matrix_stats,
    load_fp_raw_cohorts,
    load_fr_raw_cohorts,
    load_hw_raw_cohorts,
    plot_coverage_gantt,
    plot_fp_raw_grid,
    plot_fr_raw_grid,
    plot_hw_raw_grid,
    plot_single_profile,
)

__all__ = [
    "CHMap",
    "CHMapMetadata",
    "read_ch_map",
    "scan_ch_maps",
    "plot_ch_map",
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
