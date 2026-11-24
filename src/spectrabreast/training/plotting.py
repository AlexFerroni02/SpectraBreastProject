"""
Publication-grade pre-training diagnostic plots for `1D-ViT (SMAE)` and
`1D-ResNet-34 (Barlow Twins)`, faithfully reproducing the visualizations from
`MAE_100_patches.ipynb`, `Barlow_Twins_Resnet_34.ipynb`, and Deliverable 3.2.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "plot_smae_learning_curves",
    "plot_smae_reconstructions",
    "plot_barlow_twins_augmented_views",
    "plot_barlow_twins_learning_curves",
    "plot_barlow_twins_crosscorr",
]


def plot_smae_learning_curves(
    history_df: pd.DataFrame,
    title: str = "SMAE Pre-Training Performance Metrics",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    2-panel SMAE pre-training dynamics matching `MAE_100_patches.ipynb` Cell 6:
    - Left : Masked Reconstruction Loss (MSE, red)
    - Right: Shape Alignment (Masked Cosine Similarity -> 1.0, green)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 4.9))
    epochs = history_df["epoch"].to_numpy()

    ax1.plot(epochs, history_df["train_mse"], color="red", lw=1.9, marker="o", markersize=3.2, label="Train Masked MSE")
    if "val_mse" in history_df.columns:
        ax1.plot(epochs, history_df["val_mse"], color="royalblue", lw=1.5, linestyle="--", label="Val Masked MSE")
    ax1.set_title("Reconstruction Loss", fontsize=12.0, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10.0)
    ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=10.0)
    ax1.grid(True, linestyle="--", alpha=0.65)
    ax1.legend(loc="upper right", fontsize=8.8)

    ax2.plot(epochs, history_df["train_cos"], color="green", lw=1.9, marker="s", markersize=3.2, label="Train Masked Cosine Sim.")
    if "val_cos" in history_df.columns:
        ax2.plot(epochs, history_df["val_cos"], color="#005A32", lw=1.5, linestyle="--", label="Val Masked Cosine Sim.")
    ax2.set_title("Shape Alignment (Cosine Similarity)", fontsize=12.0, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10.0)
    ax2.set_ylabel("Cosine Similarity (1.0 = Perfect)", fontsize=10.0)
    y_lo = max(0.0, float(min(history_df["train_cos"].min(), history_df.get("val_cos", history_df["train_cos"]).min())) - 0.03)
    ax2.set_ylim(y_lo, 1.01)
    ax2.grid(True, linestyle="--", alpha=0.65)
    ax2.legend(loc="lower right", fontsize=8.8)

    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_smae_reconstructions(
    wavenumbers: np.ndarray,
    originals: dict[str, np.ndarray],
    reconstructions: dict[str, np.ndarray],
    patch_masks: dict[str, np.ndarray],
    epoch: int,
    patch_size: int = 100,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Multi-source stacked spectral reconstruction plot matching `MAE_100_patches.ipynb` Cell 4:
    - Replaces ONLY masked patches with decoder predictions.
    - Blue solid  : Original Spectrum (`royalblue`)
    - Red dashed  : MAE Reconstruction (`red`)
    - Khaki shade : Masked Patches (`color='khaki', alpha=0.35`)
    """
    sources = list(originals.keys())
    n_rows = len(sources)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14.0, 2.9 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    wn = np.asarray(wavenumbers, dtype=np.float64)
    for row_idx, (ax, src_name) in enumerate(zip(axes, sources)):
        orig = np.asarray(originals[src_name], dtype=np.float64)
        pred_full = np.asarray(reconstructions[src_name], dtype=np.float64)
        pmask = np.asarray(patch_masks[src_name], dtype=np.float64)

        rec = orig.copy()
        n_patches = pmask.size
        first_shade = True
        for p_idx in range(n_patches):
            if pmask[p_idx] > 0.5:
                i0 = p_idx * patch_size
                i1 = min(wn.size, (p_idx + 1) * patch_size)
                rec[i0:i1] = pred_full[i0:i1]
                ax.axvspan(
                    wn[i0], wn[i1 - 1],
                    color="khaki",
                    alpha=0.35,
                    label="Masked Patch" if first_shade else None,
                )
                first_shade = False

        ax.plot(wn, orig, color="royalblue", lw=1.45, alpha=0.85, label="Original Spectrum")
        ax.plot(wn, rec, color="red", lw=1.5, linestyle="--", label="MAE Reconstruction")

        pt_mask = np.repeat(pmask > 0.5, patch_size)[: wn.size]
        num = float(np.dot(orig[pt_mask], rec[pt_mask]))
        den = float(np.linalg.norm(orig[pt_mask]) * np.linalg.norm(rec[pt_mask]) + 1e-8)
        cos_val = num / den
        mse_val = float(np.mean((orig[pt_mask] - rec[pt_mask]) ** 2))

        prefix = f"Reconstruction Progress – Epoch {epoch} | " if row_idx == 0 else ""
        ax.set_title(
            f"{prefix}Source: {src_name}  (Masked MSE = {mse_val:.5f} | Cosine = {cos_val:.4f})",
            fontsize=10.0,
            fontweight="bold",
        )
        ax.set_ylabel("Intensity (Min-Max [0, 1])", fontsize=9.0)
        ax.set_ylim(-0.05, 1.08)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.2, ncol=3, framealpha=0.92)

    axes[-1].set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9.8, fontweight="bold")
    axes[-1].set_xlim(float(wn[0]), float(wn[-1]))
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_barlow_twins_augmented_views(
    view_1: np.ndarray,
    view_2: np.ndarray,
    source_names: list[str] | None = None,
    n_examples: int = 3,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Preview correlated augmented views matching `Barlow_Twins_Resnet_34.ipynb` Cell 13
    (`Two augmented versions of same spectra`: Augmented Version 1 vs Version 2).
    """
    n_show = min(n_examples, view_1.shape[0])
    fig, axes = plt.subplots(n_show, 1, figsize=(13.0, 3.0 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    x_idx = np.arange(view_1.shape[1])
    for i, ax in enumerate(axes):
        ax.plot(x_idx, view_1[i], lw=1.4, color="#1F77B4", label="Augmented Version 1")
        ax.plot(x_idx, view_2[i], lw=1.4, color="#FF7F0E", alpha=0.85, label="Augmented Version 2")
        src = f" | Source: {source_names[i]}" if source_names and i < len(source_names) else ""
        ax.set_title(f"Two Augmented Versions of Same Spectrum #{i + 1}{src}", fontsize=10.2, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=9.0)
        ax.grid(linestyle=":", alpha=0.45)
        ax.legend(loc="upper right", fontsize=8.5)
    axes[-1].set_xlabel("Spectral Channel Index (0 → 999)", fontsize=9.5)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_barlow_twins_learning_curves(
    history_df: pd.DataFrame,
    title: str = "Barlow Twins (1D-ResNet-34 PyTorch) Pre-Training Dynamics",
    out_path: Path | None = None,
) -> plt.Figure:
    """3-panel Barlow Twins learning curves: Total Loss, Invariance vs Redundancy, Cross-Corr Stats."""
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    epochs = history_df["epoch"].to_numpy()

    axes[0].plot(epochs, history_df["train_loss"], color="#0072B2", lw=2.0, label="Train Total Loss")
    if "val_loss" in history_df.columns:
        axes[0].plot(epochs, history_df["val_loss"], color="#D55E00", lw=1.6, linestyle="--", label="Val Total Loss")
    axes[0].set_title("Training & Validation Loss", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Barlow Twins Loss")
    axes[0].grid(linestyle=":", alpha=0.55)
    axes[0].legend(fontsize=8.2)

    axes[1].plot(epochs, history_df["train_on_diag"], color="#009E73", lw=1.9, label="Invariance (On-Diag)")
    axes[1].plot(epochs, history_df["train_off_diag_weighted"], color="#CC79A7", lw=1.9, label=r"Redundancy ($\lambda \cdot$ Off-Diag)")
    axes[1].set_title("Invariance vs Redundancy Reduction", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component")
    axes[1].grid(linestyle=":", alpha=0.55)
    axes[1].legend(fontsize=8.2)

    axes[2].plot(epochs, history_df["train_diag_mean"], color="#2CA02C", lw=2.0, label=r"Mean Diag $C_{ii} \to 1.0$")
    axes[2].plot(epochs, history_df["train_off_abs_mean"], color="#D62728", lw=2.0, label=r"Mean $|C_{ij}|_{i \neq j} \to 0.0$")
    axes[2].axhline(1.0, color="#2CA02C", linestyle=":", lw=1.0, alpha=0.6)
    axes[2].axhline(0.0, color="#D62728", linestyle=":", lw=1.0, alpha=0.6)
    axes[2].set_ylim(-0.05, 1.08)
    axes[2].set_title("Empirical Cross-Correlation Convergence", fontsize=10.5, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Correlation")
    axes[2].grid(linestyle=":", alpha=0.55)
    axes[2].legend(loc="center right", fontsize=8.2)

    fig.suptitle(title, fontsize=12.0, fontweight="bold", y=1.02)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_barlow_twins_crosscorr(
    cross_corr: np.ndarray,
    emb_stds: np.ndarray,
    epoch: int,
    slice_dim: int = 64,
    out_path: Path | None = None,
) -> plt.Figure:
    """
    2-panel Barlow Twins representation collapse audit:
    - Left : `slice_dim x slice_dim` Empirical Cross-Correlation Matrix `C` (Target = Identity)
    - Right: Distribution of per-dimension embedding standard deviations (non-zero => no collapse)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    c_sub = np.asarray(cross_corr[:slice_dim, :slice_dim], dtype=np.float32)

    im = ax1.imshow(c_sub, cmap="viridis", vmin=-0.25, vmax=1.0, aspect="equal")
    ax1.set_title(f"Cross-Correlation Matrix C (Top {slice_dim}×{slice_dim}) — Epoch {epoch}", fontsize=10.2, fontweight="bold")
    ax1.set_xlabel("Embedding Dimension j (View B)")
    ax1.set_ylabel("Embedding Dimension i (View A)")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    stds = np.asarray(emb_stds, dtype=np.float32)
    ax2.hist(stds, bins=40, color="#0072B2", alpha=0.82, edgecolor="white")
    ax2.axvline(float(stds.mean()), color="#D55E00", lw=2.0, linestyle="--", label=f"Mean Std = {stds.mean():.3f}")
    ax2.set_title(f"Projector Embedding Std Distribution — Epoch {epoch}\n(Healthy Non-Zero Std Confirms Zero Representation Collapse)", fontsize=10.0, fontweight="bold")
    ax2.set_xlabel("Per-Dimension Std Across Validation Batch")
    ax2.set_ylabel("Count of Dimensions (D = 2,048)")
    ax2.grid(linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8.8)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig
