"""
Clinical evaluation figures and Pure-PyTorch Integrated Gradients (XAI) utilities
for Step 7 Downstream Fine-Tuning (`LP-FT` vs `Scratch Baseline`), faithfully
matching Deliverable 3.2 Figs. 6–11 and Report_Pre_Training Figs. 7–12.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
)

__all__ = [
    "compute_integrated_gradients",
    "plot_lpft_learning_curves",
    "plot_clinical_evaluation",
    "plot_probability_distribution",
    "plot_scratch_vs_pretrained",
    "plot_integrated_gradients_profiles",
]


def compute_integrated_gradients(
    model: nn.Module,
    X_np: np.ndarray,
    y_np: np.ndarray,
    wavenumbers: np.ndarray,
    class_names: list[str],
    device: torch.device,
    steps: int = 50,
    max_samples_per_class: int = 120,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Pure PyTorch Integrated Gradients (Sundararajan et al., ICML 2017) with zero
    external XAI dependencies (`steps = 50` linear path from zero baseline).
    Returns:
      - `ig_profiles`: dict mapping `class_name -> normalized mean |IG| profile (1000,)`
      - `peaks_df`: DataFrame of top-5 discriminative Raman shift peaks per class.
    """
    model.eval()
    rng = np.random.default_rng(42)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)[1:]  # (steps,)
    ig_profiles: dict[str, np.ndarray] = {}
    peak_rows: list[dict[str, Any]] = []
    wn = np.asarray(wavenumbers, dtype=np.float64)

    for c_idx, c_name in enumerate(class_names):
        cls_indices = np.where(y_np == c_idx)[0]
        if cls_indices.size == 0:
            continue
        if cls_indices.size > max_samples_per_class:
            cls_indices = rng.choice(cls_indices, size=max_samples_per_class, replace=False)

        ig_accum = np.zeros(X_np.shape[1], dtype=np.float64)
        for idx in cls_indices:
            x0 = torch.from_numpy(X_np[idx].astype(np.float32)).to(device)  # (L,)
            scaled = (alphas.unsqueeze(1) * x0.unsqueeze(0)).detach().requires_grad_(True)  # (steps, L)
            logits = model(scaled)
            target_score = logits[:, c_idx].sum()
            grads = torch.autograd.grad(target_score, scaled)[0]  # (steps, L)
            avg_grad = grads.mean(dim=0)
            ig = (x0 * avg_grad).abs().detach().cpu().numpy()
            ig_accum += ig

        mean_ig = ig_accum / float(len(cls_indices))
        mx = float(mean_ig.max())
        norm_ig = mean_ig / mx if mx > 1e-12 else mean_ig
        ig_profiles[c_name] = norm_ig

        # Extract top-5 peaks (min separation ~ 15 channels)
        p_idx, _ = find_peaks(norm_ig, distance=15, prominence=0.05)
        if p_idx.size > 0:
            top_order = p_idx[np.argsort(norm_ig[p_idx])[::-1]][:5]
            for rank, pi in enumerate(top_order, start=1):
                peak_rows.append({
                    "Class": c_name,
                    "Rank": rank,
                    "Raman_Shift_cm1": round(float(wn[pi]), 2),
                    "Channel_Index": int(pi),
                    "Normalized_IG_Score": round(float(norm_ig[pi]), 4),
                })

    return ig_profiles, pd.DataFrame(peak_rows)


def plot_lpft_learning_curves(
    history_df: pd.DataFrame,
    lp_epochs: int,
    title: str = "LP-FT Optimization Diagnostics",
    out_path: Path | None = None,
) -> plt.Figure:
    """3-row stacked LP-FT learning curves matching Deliverable 3.2 Fig. 9 & Cell 8."""
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 11.5), sharex=True)
    ep = history_df["epoch"].to_numpy()

    axes[0].plot(ep, history_df["train_loss"], color="#1F77B4", lw=1.8, label="Train Loss")
    axes[0].plot(ep, history_df["val_loss"], color="#FF7F0E", lw=1.8, label="Val Loss")
    if lp_epochs > 0:
        axes[0].axvline(lp_epochs, color="red", linestyle="--", lw=1.5, label="Start FT")
    axes[0].set_title("Cross-Entropy Loss", fontsize=11.0, fontweight="bold")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(ep, history_df["train_acc"], color="#1F77B4", lw=1.7, linestyle="--", label="Train Accuracy")
    axes[1].plot(ep, history_df["val_acc"], color="#FF7F0E", lw=1.9, label="Val Accuracy")
    if lp_epochs > 0:
        axes[1].axvline(lp_epochs, color="red", linestyle="--", lw=1.5, label="Start FT")
    axes[1].set_title("Accuracy", fontsize=11.0, fontweight="bold")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right")

    axes[2].plot(ep, history_df["train_f1"], color="#7B4EA3", lw=1.7, linestyle="--", label="Train Macro-F1")
    axes[2].plot(ep, history_df["val_f1"], color="#2CA02C", lw=1.9, label="Val Macro-F1")
    if lp_epochs > 0:
        axes[2].axvline(lp_epochs, color="red", linestyle="--", lw=1.5, label="Start FT")
    axes[2].set_title("Macro-F1", fontsize=11.0, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="lower right")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_clinical_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    model_label: str = "Pre-Trained LP-FT",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Clinical evaluation dashboard matching Deliverable 3.2 Figs. 6–8:
    - Binary tasks: Confusion Matrix + ROC Curve + Precision-Recall Curve.
    - Multiclass tasks: Confusion Matrix + Per-Class F1 Bar Chart.
    """
    n_classes = len(class_names)
    if n_classes == 2:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.2))
        cm = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues", ax=ax1, colorbar=False)
        ax1.set_title("Aggregated Out-of-Fold Confusion Matrix", fontsize=10.5, fontweight="bold")

        p1 = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
        RocCurveDisplay.from_predictions(y_true, p1, ax=ax2, color="darkorange", name=model_label)
        ax2.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Chance Level")
        ax2.set_title("ROC Curve", fontsize=10.5, fontweight="bold")
        ax2.grid(alpha=0.3)
        ax2.legend(loc="lower right")

        PrecisionRecallDisplay.from_predictions(y_true, p1, ax=ax3, color="purple", name=model_label)
        baseline = float(np.mean(y_true == 1))
        ax3.axhline(baseline, color="navy", linestyle="--", label=f"Baseline ({baseline:.2f})")
        ax3.set_title("Precision-Recall Curve", fontsize=10.5, fontweight="bold")
        ax3.grid(alpha=0.3)
        ax3.legend(loc="lower left")
    else:
        fig_w = 18.0 if n_classes > 10 else 14.5
        fig_h = 8.5 if n_classes > 10 else 5.5
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [1.35, 1.0]})
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names if n_classes <= 12 else None)
        disp.plot(cmap="Blues", ax=ax1, colorbar=True, include_values=(n_classes <= 12))
        ax1.set_title(f"Confusion Matrix ({n_classes} Classes)", fontsize=10.5, fontweight="bold")

        per_cls_f1 = f1_score(y_true, y_pred, average=None, labels=np.arange(n_classes), zero_division=0)
        y_pos = np.arange(n_classes)
        ax2.barh(y_pos, per_cls_f1, color="#0072B2", alpha=0.88)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(class_names, fontsize=7.5 if n_classes > 10 else 9.0)
        ax2.set_xlim(0.0, 1.08)
        ax2.set_xlabel("Per-Class F1-Score")
        ax2.set_title(f"Per-Class F1-Score (Macro-F1 = {per_cls_f1.mean():.4f})", fontsize=10.5, fontweight="bold")
        ax2.grid(axis="x", linestyle=":", alpha=0.5)

    fig.suptitle(f"Downstream Clinical Evaluation — {model_label}", fontsize=12.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    out_path: Path | None = None,
) -> plt.Figure:
    """Binary predicted probability histogram matching Deliverable 3.2 Fig. 7(b)."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    p1 = y_probs[:, 1] if y_probs.ndim == 2 else y_probs
    ax.hist(p1[y_true == 0], bins=25, color="#0072B2", alpha=0.58, label=f"True {class_names[0]}", edgecolor="white")
    ax.hist(p1[y_true == 1], bins=25, color="#D55E00", alpha=0.58, label=f"True {class_names[1]}", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle="--", lw=1.8, label="Default Threshold (0.5)")
    ax.set_title("Distribution of Predicted Probabilities (Out-of-Fold Test Set)", fontsize=11.0, fontweight="bold")
    ax.set_xlabel(f"Predicted Probability of {class_names[1]}")
    ax.set_ylabel("Spectrum Count")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_scratch_vs_pretrained(
    summary_df: pd.DataFrame,
    benchmark_name: str,
    model_name: str,
    out_path: Path | None = None,
) -> plt.Figure:
    """5-Fold Mean ± Std bar chart comparing Scratch Baseline vs Pre-Trained LP-FT."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    metric_labels = ["Accuracy", "Balanced Acc.", "Macro-F1", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.34

    for idx, (mode, col) in enumerate([("scratch", "#8E9AA8"), ("pretrained_lpft", "#0072B2")]):
        sub = summary_df[summary_df["mode"] == mode]
        if sub.empty:
            continue
        means = [float(sub[m].mean()) for m in metrics]
        stds = [float(sub[m].std()) for m in metrics]
        offset = (idx - 0.5) * width if len(summary_df["mode"].unique()) > 1 else 0.0
        lbl = "Scratch Baseline (Random Init)" if mode == "scratch" else f"Pre-Trained {model_name} (LP-FT)"
        bars = ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, color=col, alpha=0.9, label=lbl)
        for b, m_val, s_val in zip(bars, means, stds):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                m_val + s_val + 0.015,
                f"{m_val*100:.1f}%",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=9.8, fontweight="bold")
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("5-Fold Zero-Leakage Score", fontsize=9.8)
    ax.set_title(f"5-Fold Transfer Learning Gain — {model_name} on `{benchmark_name}`", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.55)
    ax.legend(loc="lower right", fontsize=9.0)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig


def plot_integrated_gradients_profiles(
    wavenumbers: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    ig_profiles: dict[str, np.ndarray],
    class_names: list[str],
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Class-averaged Integrated Gradients attribution profiles overlaid with mean spectra,
    matching Deliverable 3.2 Fig. 11 & Report_Pre_Training Fig. 11.
    """
    active_classes = [(i, c) for i, c in enumerate(class_names) if c in ig_profiles]
    if len(active_classes) > 4:
        active_classes = active_classes[:4]
    n_rows = len(active_classes)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13.5, 3.0 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    wn = np.asarray(wavenumbers, dtype=np.float64)
    colors = ["#0072B2", "#D62728", "#009E73", "#CC79A7"]

    for row, (ax, (c_idx, c_name)) in enumerate(zip(axes, active_classes)):
        ig = ig_profiles[c_name]
        col = colors[row % len(colors)]
        mean_sp = X_test[y_test == c_idx].mean(axis=0)
        mean_sp = (mean_sp - mean_sp.min()) / (mean_sp.max() - mean_sp.min() + 1e-8)

        ax.plot(wn, mean_sp, color="#444444", lw=1.2, alpha=0.55, label=f"Mean Spectrum ({c_name})")
        ax.plot(wn, ig, color=col, lw=1.8, label=f"Mean |Integrated Gradients| ({c_name})")
        ax.fill_between(wn, 0.0, ig, color=col, alpha=0.20)

        # Annotate top-3 peaks
        p_idx, _ = find_peaks(ig, distance=20, prominence=0.08)
        if p_idx.size > 0:
            for pi in p_idx[np.argsort(ig[p_idx])[::-1]][:3]:
                ax.annotate(
                    f"{wn[pi]:.0f}",
                    xy=(wn[pi], ig[pi]),
                    xytext=(wn[pi], min(1.06, ig[pi] + 0.08)),
                    ha="center", fontsize=8.0, fontweight="bold", color=col,
                )

        ax.set_title(f"Integrated Gradients Spectral Attribution — {c_name}", fontsize=10.2, fontweight="bold")
        ax.set_ylabel("Normalized Score [0, 1]", fontsize=8.8)
        ax.set_ylim(-0.02, 1.15)
        ax.grid(linestyle=":", alpha=0.45)
        ax.legend(loc="upper right", fontsize=8.2)

    axes[-1].set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=9.8, fontweight="bold")
    axes[-1].set_xlim(float(wn[0]), float(wn[-1]))
    fig.suptitle("Integrated Gradients (XAI) — Discriminative Raman Regions", fontsize=12.2, fontweight="bold", y=1.01)
    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig
