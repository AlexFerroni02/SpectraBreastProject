"""Self-supervised pre-training engines and thesis-grade diagnostic plots for SPECTRA-BREAST."""

from .plotting import (
    plot_barlow_twins_augmented_views,
    plot_barlow_twins_crosscorr,
    plot_barlow_twins_learning_curves,
    plot_smae_learning_curves,
    plot_smae_reconstructions,
    save_experiment_notebook,
)
from .pretrain_barlow_twins import TorchRamanAugmentor, train_barlow_twins_from_config
from .pretrain_smae import PretrainTensorDataset, train_smae_from_config

__all__ = [
    "PretrainTensorDataset",
    "train_smae_from_config",
    "TorchRamanAugmentor",
    "train_barlow_twins_from_config",
    "plot_smae_learning_curves",
    "plot_smae_reconstructions",
    "plot_barlow_twins_augmented_views",
    "plot_barlow_twins_learning_curves",
    "plot_barlow_twins_crosscorr",
    "save_experiment_notebook",
]
