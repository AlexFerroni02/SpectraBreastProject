"""
Unified Zero-Leakage Downstream Fine-Tuning (`LP-FT` vs `Scratch Baseline`) and
Integrated Gradients XAI engine for both `1D-ViT (SpectralMAE)` and
`1D-ResNet-34 (Barlow Twins)` across all 4 SPECTRA-BREAST benchmarks:
- `riverd_oral_hw` / `riverd_oral_ch` (8,422 spectra, 186 surgical patients, 2 classes)
- `ibd` (228 clinical plasma subjects, 2 classes)
- `bacteria_id` (3,000 finetune + 3,000 clinical test, 30 pathogenic species)
- `dataset_renzo` (3,200 spectra, 47 mice, 4 genetic/therapeutic pathologies, FR/FP/CH)
"""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, TensorDataset

from spectrabreast.data.datasets import DownstreamFoldLoader
from spectrabreast.models.resnet_barlow_twins import ResNet1DBackbone, ResNet1DClassifier
from spectrabreast.models.vit_smae import SpectralMAE, SpectralMAEClassifier
from spectrabreast.training.finetune_plotting import (
    compute_integrated_gradients,
    plot_clinical_evaluation,
    plot_integrated_gradients_profiles,
    plot_lpft_learning_curves,
    plot_probability_distribution,
    plot_scratch_vs_pretrained,
)
from spectrabreast.training.plotting import save_experiment_notebook

__all__ = ["RamanFineTuneDataset", "run_downstream_finetuning"]


class RamanFineTuneDataset(Dataset):
    """
    Training-time spectral augmentation dataset faithfully matching
    `finetuning_SMAE_IBD.ipynb` & `finetuning_SMAE__Bacteria.ipynb`:
    - Random intensity scaling `[0.95, 1.05]`
    - Light Gaussian noise (`noise_std`)
    - Boundary-clamped wavenumber shift (`±max_shift` channels, no circular wrap)
    - Random contiguous 1D cutout (`p=0.5`, `max_mask_size` channels)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        max_shift: int = 3,
        max_mask_size: int = 40,
        noise_std: float = 0.002,
    ) -> None:
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.augment = augment
        self.max_shift = max_shift
        self.max_mask_size = max_mask_size
        self.noise_std = noise_std

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.augment:
            x = x * random.uniform(0.95, 1.05)
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.max_shift > 0 and random.random() < 0.5:
                sh = random.randint(-self.max_shift, self.max_shift)
                if sh > 0:
                    x = torch.cat([x[:1].expand(sh), x[:-sh]])
                elif sh < 0:
                    x = torch.cat([x[-sh:], x[-1:].expand(-sh)])
            if self.max_mask_size > 0 and random.random() < 0.5:
                m_len = random.randint(max(5, self.max_mask_size // 4), self.max_mask_size)
                s_idx = random.randint(0, x.shape[0] - m_len)
                x[s_idx : s_idx + m_len] = 0.0
        return x, self.y[idx]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_class_names(benchmark: str, n_classes: int, fold_dict: dict[str, Any]) -> list[str]:
    if benchmark == "ibd":
        return ["HC (0)", "IBD (1)"]
    if benchmark in ("riverd_oral_hw", "riverd_oral_ch"):
        return ["Normal (0)", "OSCC Tumor (1)"]
    if benchmark == "dataset_renzo" and "class_names" in fold_dict:
        return [str(x) for x in fold_dict["class_names"]]
    return [f"Class {i}" for i in range(n_classes)]


def _build_classifier(
    model_type: str,
    model_cfg: dict[str, Any],
    num_classes: int,
    checkpoint_path: Path | None,
    device: torch.device,
) -> nn.Module:
    if model_type == "smae":
        enc = SpectralMAE(
            spectrum_size=int(model_cfg.get("spectrum_size", 1000)),
            patch_size=int(model_cfg.get("patch_size", 100)),
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            depth=int(model_cfg.get("depth", 8)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            mlp_dim=int(model_cfg.get("mlp_dim", 512)),
            decoder_embed_dim=int(model_cfg.get("decoder_embed_dim", 128)),
            decoder_depth=int(model_cfg.get("decoder_depth", 1)),
            decoder_num_heads=int(model_cfg.get("decoder_num_heads", 8)),
            decoder_mlp_dim=int(model_cfg.get("decoder_mlp_dim", 512)),
        )
        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            enc.load_state_dict(sd, strict=True)
        clf = SpectralMAEClassifier(
            encoder=enc,
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("head_hidden_dim", 128)),
            dropout=float(model_cfg.get("head_dropout", 0.35)),
        )
    elif model_type == "barlow_twins":
        enc = ResNet1DBackbone(
            backbone=str(model_cfg.get("backbone", "resnet34")),
            in_channels=1,
            seq_len=int(model_cfg.get("seq_len", 1000)),
            pool_mode=str(model_cfg.get("pool_mode", "avgpool2_flatten")),
        )
        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(sd, dict) and "model_state_dict" in sd:
                sd = sd["model_state_dict"]
            # Handle both encoder-only (`best_resnet_encoder.pth`) and full (`best_barlow_twins_full.pth`)
            if any(k.startswith("encoder.") for k in sd.keys()):
                sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
            enc.load_state_dict(sd, strict=True)
        clf = ResNet1DClassifier(
            encoder=enc,
            num_classes=num_classes,
            hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
            dropout=float(model_cfg.get("head_dropout", 0.45)),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type!r}. Use 'smae' or 'barlow_twins'.")

    return clf.to(device)


def _freeze_batchnorm_layers(module: nn.Module) -> None:
    """Critical thesis trick (finetuning_BarlowTwins_IBD.ipynb): keep BN frozen during small-data FT."""
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.eval()
            for p in m.parameters():
                p.requires_grad = False


@torch.no_grad()
def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels, all_probs, losses = [], [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        losses.append(float(criterion(logits, yb).item()))
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(yb.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    n_classes = y_prob.shape[1]

    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        auc = float("nan")

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def _aggregate_subject_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray | None,
) -> dict[str, float]:
    """Compute patient/mouse-level majority/mean-probability metrics when subject IDs are available."""
    if groups is None or len(np.unique(groups)) == len(y_true):
        return {}
    uniq_g = np.unique(groups)
    g_true, g_pred, g_prob = [], [], []
    for g in uniq_g:
        m = groups == g
        # Subject true label = majority label among that subject's spectra
        vals, cnts = np.unique(y_true[m], return_counts=True)
        g_true.append(vals[np.argmax(cnts)])
        mean_p = y_prob[m].mean(axis=0)
        g_prob.append(mean_p)
        g_pred.append(int(np.argmax(mean_p)))
    g_true_arr = np.array(g_true)
    g_pred_arr = np.array(g_pred)
    g_prob_arr = np.vstack(g_prob)
    try:
        g_auc = float(
            roc_auc_score(g_true_arr, g_prob_arr[:, 1])
            if g_prob_arr.shape[1] == 2
            else roc_auc_score(g_true_arr, g_prob_arr, multi_class="ovr")
        )
    except ValueError:
        g_auc = float("nan")
    return {
        "subject_accuracy": float(accuracy_score(g_true_arr, g_pred_arr)),
        "subject_balanced_accuracy": float(balanced_accuracy_score(g_true_arr, g_pred_arr)),
        "subject_macro_f1": float(f1_score(g_true_arr, g_pred_arr, average="macro", zero_division=0)),
        "subject_roc_auc": g_auc,
    }


def _train_single_fold(
    model: nn.Module,
    model_type: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    t_cfg: dict[str, Any],
    is_pretrained: bool,
    device: torch.device,
) -> tuple[nn.Module, pd.DataFrame, int]:
    """Train one fold using Two-Stage LP-FT (if pre-trained) or full Scratch training."""
    n_classes = int(max(y_tr.max(), y_va.max()) + 1)
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
    w = w / np.maximum(w.sum(), 1e-8) * n_classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    batch_size = int(t_cfg.get("batch_size", 32))
    train_ds = RamanFineTuneDataset(
        X_tr, y_tr, augment=bool(t_cfg.get("use_augment", True)),
        max_shift=int(t_cfg.get("aug_max_shift", 3)),
        max_mask_size=int(t_cfg.get("aug_max_mask", 40)),
        noise_std=float(t_cfg.get("aug_noise_std", 0.002)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(X_va, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(y_va, dtype=np.int64)),
    )
    loader_tr = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_va = DataLoader(val_ds, batch_size=max(batch_size, 64), shuffle=False)

    weight_decay = float(t_cfg.get("weight_decay", 1e-4))
    patience = int(t_cfg.get("early_stopping_patience", 25))
    freeze_bn = bool(t_cfg.get("freeze_bn", model_type == "barlow_twins"))
    unfreeze_last_n = t_cfg.get("unfreeze_last_n_blocks", None)

    history: list[dict[str, float]] = []
    ep_counter = 0
    lp_epochs_used = 0

    # --- STAGE 1: Linear Probing (only for pre-trained encoders) ---
    if is_pretrained and int(t_cfg.get("lp_epochs", 30)) > 0:
        lp_epochs_used = int(t_cfg.get("lp_epochs", 30))
        model.freeze_encoder()
        if model_type == "barlow_twins":
            _freeze_batchnorm_layers(model.encoder)
        opt_lp = AdamW(model.head.parameters(), lr=float(t_cfg.get("lp_lr", 1e-3)), weight_decay=weight_decay)
        sch_lp = ReduceLROnPlateau(opt_lp, mode="min", factor=0.5, patience=8)

        for _ in range(lp_epochs_used):
            ep_counter += 1
            model.train()
            if model_type == "barlow_twins":
                _freeze_batchnorm_layers(model.encoder)
            tr_loss, tr_preds, tr_lbls = 0.0, [], []
            for xb, yb in loader_tr:
                xb, yb = xb.to(device), yb.to(device)
                opt_lp.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt_lp.step()
                tr_loss += float(loss.item())
                tr_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
                tr_lbls.append(yb.detach().cpu().numpy())

            tr_p, tr_l = np.concatenate(tr_preds), np.concatenate(tr_lbls)
            va_m = _evaluate_loader(model, loader_va, criterion, device)
            sch_lp.step(va_m["loss"])
            history.append({
                "epoch": ep_counter,
                "stage": 1,
                "train_loss": tr_loss / max(1, len(loader_tr)),
                "val_loss": va_m["loss"],
                "train_acc": float(accuracy_score(tr_l, tr_p)),
                "val_acc": va_m["accuracy"],
                "train_f1": float(f1_score(tr_l, tr_p, average="macro", zero_division=0)),
                "val_f1": va_m["macro_f1"],
            })

    # --- STAGE 2: Fine-Tuning (or Full Scratch Training) ---
    if is_pretrained:
        if model_type == "smae" and unfreeze_last_n is not None:
            model.unfreeze_encoder(n_last_blocks=int(unfreeze_last_n))
        else:
            model.unfreeze_encoder()
        if freeze_bn and model_type == "barlow_twins":
            _freeze_batchnorm_layers(model.encoder)
        ft_epochs = int(t_cfg.get("ft_epochs", 80))
        ft_lr = float(t_cfg.get("ft_lr", 1e-5))
    else:
        model.unfreeze_encoder()
        ft_epochs = int(t_cfg.get("scratch_epochs", 100))
        ft_lr = float(t_cfg.get("scratch_lr", 1e-4))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt_ft = AdamW(trainable_params, lr=ft_lr, weight_decay=weight_decay)
    sch_ft = ReduceLROnPlateau(opt_ft, mode="min", factor=0.5, patience=6)

    best_f1 = -1.0
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    for _ in range(ft_epochs):
        ep_counter += 1
        model.train()
        if is_pretrained and freeze_bn and model_type == "barlow_twins":
            _freeze_batchnorm_layers(model.encoder)
        tr_loss, tr_preds, tr_lbls = 0.0, [], []
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt_ft.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt_ft.step()
            tr_loss += float(loss.item())
            tr_preds.append(logits.argmax(dim=1).detach().cpu().numpy())
            tr_lbls.append(yb.detach().cpu().numpy())

        tr_p, tr_l = np.concatenate(tr_preds), np.concatenate(tr_lbls)
        va_m = _evaluate_loader(model, loader_va, criterion, device)
        sch_ft.step(va_m["loss"])
        history.append({
            "epoch": ep_counter,
            "stage": 2 if is_pretrained else 0,
            "train_loss": tr_loss / max(1, len(loader_tr)),
            "val_loss": va_m["loss"],
            "train_acc": float(accuracy_score(tr_l, tr_p)),
            "val_acc": va_m["accuracy"],
            "train_f1": float(f1_score(tr_l, tr_p, average="macro", zero_division=0)),
            "val_f1": va_m["macro_f1"],
        })

        if va_m["macro_f1"] > best_f1 or (va_m["macro_f1"] == best_f1 and va_m["loss"] < best_loss):
            best_f1 = va_m["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

        if va_m["loss"] < best_loss - 1e-5:
            best_loss = va_m["loss"]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), lp_epochs_used


def run_downstream_finetuning(
    config_path: str | Path,
    repo_root: str | Path | None = None,
    checkpoint_override: str | None = None,
    benchmark_override: str | None = None,
    regime_override: str | None = None,
    output_dir_override: str | None = None,
    run_scratch_override: bool | None = None,
    n_folds_override: int | None = None,
) -> dict[str, Any]:
    """
    Execute full 5-fold zero-leakage downstream evaluation (`Pre-Trained LP-FT`
    and optional `Scratch Baseline`) + Integrated Gradients XAI + auto-generated
    `notebook_eseguito.ipynb`.
    """
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if checkpoint_override is not None:
        cfg["model"]["pretrained_checkpoint"] = checkpoint_override
    if benchmark_override is not None:
        cfg["data"]["benchmark"] = benchmark_override
    if regime_override is not None:
        cfg["data"]["regime"] = regime_override
    if output_dir_override is not None:
        cfg["logging"]["output_dir"] = output_dir_override
    if run_scratch_override is not None:
        cfg["training"]["run_scratch_baseline"] = bool(run_scratch_override)

    root = Path(repo_root).resolve() if repo_root is not None else cfg_path.parents[2]
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = str(cfg["data"]["benchmark"])
    regime = str(cfg["data"].get("regime", "default"))
    model_type = str(cfg["model"]["type"])
    ckpt_rel = cfg["model"].get("pretrained_checkpoint")
    ckpt_path = (root / ckpt_rel) if ckpt_rel else None
    if ckpt_path is not None and not ckpt_path.exists():
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {ckpt_path}")

    # Auto-sync encoder hyperparameters from pre-training config_used.yaml if present
    model_cfg = dict(cfg["model"])
    if ckpt_path is not None and (ckpt_path.parent / "config_used.yaml").exists():
        pre_cfg = yaml.safe_load((ckpt_path.parent / "config_used.yaml").read_text(encoding="utf-8"))
        for k, v in pre_cfg.get("model", {}).items():
            model_cfg.setdefault(k, v)

    out_dir = root / cfg["logging"]["output_dir"]
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    loader = DownstreamFoldLoader(root / "data" / "02_ready")
    n_folds = int(n_folds_override or cfg["training"].get("n_folds", 5))
    modes = ["pretrained_lpft"]
    if bool(cfg["training"].get("run_scratch_baseline", True)):
        modes.append("scratch")

    print(f"[FineTune] Device={device} | Model={model_type} | Benchmark={benchmark} (regime={regime}) | Folds={n_folds}")
    print(f"[FineTune] Checkpoint={ckpt_path}")

    fold_rows: list[dict[str, Any]] = []
    oof_store: dict[str, dict[str, list[np.ndarray]]] = {
        m: {"y_true": [], "y_pred": [], "y_prob": [], "groups": []} for m in modes
    }
    first_fold_history: pd.DataFrame | None = None
    first_fold_lp_ep = 0
    best_pretrained_model: nn.Module | None = None
    best_pretrained_f1 = -1.0
    best_pretrained_fold_data: dict[str, Any] | None = None
    class_names: list[str] = []
    wavenumbers: np.ndarray | None = None

    for mode in modes:
        is_pre = mode == "pretrained_lpft"
        print(f"\n================ MODE: {mode.upper()} ================")
        for fold in range(n_folds):
            t0 = time.time()
            _set_seed(seed + fold)
            fdict = loader.get_fold(benchmark, fold=fold, regime=regime)
            wavenumbers = fdict["wavenumbers"]
            X_tr_full, y_tr_full = fdict["X_train"], fdict["y_train"]
            X_te, y_te = fdict["X_test"], fdict["y_test"]
            n_classes = int(max(y_tr_full.max(), y_te.max()) + 1)
            class_names = _get_class_names(benchmark, n_classes, fdict)

            # Carve internal validation split strictly from training fold (zero test leakage!)
            if "X_val" in fdict:
                X_tr, y_tr = X_tr_full, y_tr_full
                X_va, y_va = fdict["X_val"], fdict["y_val"]
            else:
                tr_groups = fdict.get("train_groups")
                if tr_groups is not None and len(np.unique(tr_groups)) < len(y_tr_full):
                    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold)
                    tr_i, va_i = next(sgkf.split(X_tr_full, y_tr_full, groups=tr_groups))
                else:
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
                    tr_i, va_i = next(skf.split(X_tr_full, y_tr_full))
                X_tr, y_tr = X_tr_full[tr_i], y_tr_full[tr_i]
                X_va, y_va = X_tr_full[va_i], y_tr_full[va_i]

            model = _build_classifier(
                model_type=model_type,
                model_cfg=model_cfg,
                num_classes=n_classes,
                checkpoint_path=ckpt_path if is_pre else None,
                device=device,
            )
            model, hist_df, lp_ep = _train_single_fold(
                model=model,
                model_type=model_type,
                X_tr=X_tr,
                y_tr=y_tr,
                X_va=X_va,
                y_va=y_va,
                t_cfg=cfg["training"],
                is_pretrained=is_pre,
                device=device,
            )

            test_ds = TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X_te, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y_te, dtype=np.int64)),
            )
            test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
            te_m = _evaluate_loader(model, test_loader, nn.CrossEntropyLoss(), device)
            subj_m = _aggregate_subject_metrics(te_m["y_true"], te_m["y_prob"], fdict.get("test_groups"))

            row = {
                "mode": mode,
                "fold": fold + 1,
                "accuracy": te_m["accuracy"],
                "balanced_accuracy": te_m["balanced_accuracy"],
                "macro_f1": te_m["macro_f1"],
                "roc_auc": te_m["roc_auc"],
                "epoch_count": len(hist_df),
                "elapsed_sec": round(time.time() - t0, 1),
                **subj_m,
            }
            fold_rows.append(row)
            oof_store[mode]["y_true"].append(te_m["y_true"])
            oof_store[mode]["y_pred"].append(te_m["y_pred"])
            oof_store[mode]["y_prob"].append(te_m["y_prob"])

            subj_str = f" | Subj-F1={subj_m['subject_macro_f1']:.4f} Subj-AUC={subj_m['subject_roc_auc']:.4f}" if subj_m else ""
            print(
                f"  [{mode}] Fold {fold+1}/{n_folds} | Acc={te_m['accuracy']:.4f} | "
                f"BalAcc={te_m['balanced_accuracy']:.4f} | Macro-F1={te_m['macro_f1']:.4f} | "
                f"AUC={te_m['roc_auc']:.4f}{subj_str} | {row['elapsed_sec']:.1f}s"
            )

            if is_pre and fold == 0:
                first_fold_history = hist_df
                first_fold_lp_ep = lp_ep
            if is_pre and te_m["macro_f1"] > best_pretrained_f1:
                best_pretrained_f1 = te_m["macro_f1"]
                best_pretrained_model = copy.deepcopy(model)
                best_pretrained_fold_data = {"X_test": X_te, "y_test": y_te}
                torch.save(model.state_dict(), out_dir / "best_finetuned_classifier.pth")

    results_df = pd.DataFrame(fold_rows)
    results_df.to_csv(out_dir / "fold_results.csv", index=False)

    # Print and save aggregated summary
    summary_lines = [f"=== 5-FOLD ZERO-LEAKAGE SUMMARY: {model_type.upper()} on {benchmark} ({regime}) ==="]
    summary_json: dict[str, Any] = {}
    for mode in modes:
        sub = results_df[results_df["mode"] == mode]
        summary_json[mode] = {}
        line = f"  {mode:16s} |"
        for metric in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]:
            mu, sd = float(sub[metric].mean()), float(sub[metric].std())
            summary_json[mode][metric] = {"mean": mu, "std": sd}
            line += f" {metric}={mu*100:.2f}% ± {sd*100:.2f}% |"
        if "subject_macro_f1" in sub.columns and sub["subject_macro_f1"].notna().any():
            smu, ssd = float(sub["subject_macro_f1"].mean()), float(sub["subject_macro_f1"].std())
            summary_json[mode]["subject_macro_f1"] = {"mean": smu, "std": ssd}
            line += f" Subj-F1={smu*100:.2f}% ± {ssd*100:.2f}% |"
        summary_lines.append(line)
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (out_dir / "summary_metrics.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    # Generate all clinical & XAI figures at 300 DPI
    saved_figs: list[Path] = []
    if first_fold_history is not None:
        p_lc = fig_dir / "ft_learning_curves_fold1.png"
        plot_lpft_learning_curves(first_fold_history, first_fold_lp_ep, title=f"LP-FT Learning Curves (Fold 1) — {model_type.upper()} on `{benchmark}`", out_path=p_lc)
        plt.close("all")
        saved_figs.append(p_lc)

    y_t_all = np.concatenate(oof_store["pretrained_lpft"]["y_true"])
    y_p_all = np.concatenate(oof_store["pretrained_lpft"]["y_pred"])
    y_pr_all = np.concatenate(oof_store["pretrained_lpft"]["y_prob"])

    p_eval = fig_dir / "ft_clinical_evaluation.png"
    plot_clinical_evaluation(y_t_all, y_p_all, y_pr_all, class_names, model_label=f"{model_type.upper()} LP-FT on {benchmark}", out_path=p_eval)
    plt.close("all")
    saved_figs.append(p_eval)

    if len(class_names) == 2:
        p_prob = fig_dir / "ft_probability_distribution.png"
        plot_probability_distribution(y_t_all, y_pr_all, class_names, out_path=p_prob)
        plt.close("all")
        saved_figs.append(p_prob)

    p_cmp = fig_dir / "ft_scratch_vs_pretrained_comparison.png"
    plot_scratch_vs_pretrained(results_df, benchmark_name=benchmark, model_name=model_type.upper(), out_path=p_cmp)
    plt.close("all")
    saved_figs.append(p_cmp)

    # Pure-PyTorch Integrated Gradients XAI on the best fold
    if bool(cfg["training"].get("run_integrated_gradients", True)) and best_pretrained_model is not None and wavenumbers is not None:
        ig_profiles, peaks_df = compute_integrated_gradients(
            model=best_pretrained_model,
            X_np=best_pretrained_fold_data["X_test"],
            y_np=best_pretrained_fold_data["y_test"],
            wavenumbers=wavenumbers,
            class_names=class_names,
            device=device,
            steps=int(cfg["training"].get("ig_steps", 50)),
        )
        peaks_df.to_csv(out_dir / "top_peaks_ig.csv", index=False)
        p_ig = fig_dir / "ft_integrated_gradients_xai.png"
        plot_integrated_gradients_profiles(
            wavenumbers=wavenumbers,
            X_test=best_pretrained_fold_data["X_test"],
            y_test=best_pretrained_fold_data["y_test"],
            ig_profiles=ig_profiles,
            class_names=class_names,
            out_path=p_ig,
        )
        plt.close("all")
        saved_figs.append(p_ig)

    # Save self-contained executed notebook inside the fine-tuning experiment folder
    nb_path = save_experiment_notebook(
        out_dir=out_dir,
        model_type=f"Downstream LP-FT ({model_type.upper()} on {benchmark})",
        cfg=cfg,
        history_df=results_df.rename(columns={"fold": "epoch"}),
        figure_paths=saved_figs,
        summary_text=summary_text,
    )
    print(f"[FineTune] Saved executed notebook -> {nb_path}")

    return {
        "output_dir": out_dir,
        "results_df": results_df,
        "summary": summary_json,
        "executed_notebook": nb_path,
    }
