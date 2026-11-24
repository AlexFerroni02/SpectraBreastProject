"""
Data loaders, leakage-free patient/subject-grouped CV splits, temperature-balanced
cohort samplers, and physically grounded Raman augmentations / multi-block patch
masking for SPECTRA-BREAST (`Step 4`).
"""

from .augmentations import RamanPhysicalAugmentor, SpectralBlockMaskGenerator
from .dataloader_atlas import (
    plot_barlow_twins_physical_augmentations,
    plot_cohort_sampling_balance,
    plot_downstream_grouped_splits_audit,
    plot_smae_contiguous_block_masking,
)
from .datasets import DownstreamFoldLoader, SSLPretrainDataset
from .splits import (
    build_and_verify_downstream_splits,
    compute_temperature_sampling_weights,
)

__all__ = [
    "RamanPhysicalAugmentor",
    "SpectralBlockMaskGenerator",
    "compute_temperature_sampling_weights",
    "build_and_verify_downstream_splits",
    "SSLPretrainDataset",
    "DownstreamFoldLoader",
    "plot_cohort_sampling_balance",
    "plot_barlow_twins_physical_augmentations",
    "plot_smae_contiguous_block_masking",
    "plot_downstream_grouped_splits_audit",
]
