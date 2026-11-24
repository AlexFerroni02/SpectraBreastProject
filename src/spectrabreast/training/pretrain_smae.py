"""
HPC-ready PyTorch Pre-Training engine for `1D-ViT (SpectralMAE)`.

Features:
- Loads YAML configs (`configs/train/smae_hw_1000.yaml` or `smae_fp_1000.yaml`).
- Temperature-balanced cohort sampling (`alpha = 0.5` or `1.0`) with zero CPU bottleneck.
- Automatic CUDA AMP (`bfloat16` on A100/H100, `float16` with GradScaler on V100/T4/GTX).
- Exact thesis metrics: Masked Reconstruction MSE + Masked Cosine Similarity.
- Automatic checkpointing (`best_smae_model.pth`, `best_encoder_only.pth`, `last_checkpoint.pth`)
  with seamless resume support (`resume=True`).
- Generates exact Deliverable 3.2 learning curves and multi-source reconstruction figures.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from spectrabreast.data.splits import compute_temperature_sampling_weights
from spectrabreast.models.vit_smae import SpectralMAE
from spectrabreast.training.plotting import plot_smae_learning_curves, plot_smae_reconstructions

__all__ = ["PretrainTensorDataset", "train_smae_from_config"]


class PretrainTensorDataset(Dataset):
    """In-RAM float32 tensor dataset for blazing-fast GPU pre-training."""

    def __init__(self, spectra: np.ndarray, source_id: np.ndarray) -> None:
        self.spectra = torch.from_numpy(np.ascontiguousarray(spectra, dtype=np.float32))
        self.source_id = torch.from_numpy(np.ascontiguousarray(source_id, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spectra[idx], self.source_id[idx]


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_amp_dtype(device: torch.device, use_amp: bool) -> tuple[bool, torch.dtype, bool]:
    if not use_amp or device.type != "cuda":
        return False, torch.float32, False
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16, False  # bf16 needs no GradScaler
    return True, torch.float16, True


def _cosine_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_smae_from_config(
    config_path: str | Path,
    repo_root: str | Path | None = None,
    resume: bool = True,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Run full `SpectralMAE` pre-training from a YAML config file."""
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(repo_root).resolve() if repo_root is not None else cfg_path.parents[2]
    _seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype, use_scaler = _select_amp_dtype(device, bool(cfg["training"].get("use_amp", True)))
    print(f"[SMAE] Device: {device} | AMP: {amp_enabled} ({amp_dtype})")

    # 1. Load compiled 1,000-pt ready pre-training archive
    npz_path = root / cfg["data"]["npz_path"]
    archive = np.load(npz_path, allow_pickle=True)
    spectra = archive["spectra"]
    source_id = archive["source_id"]
    source_names = [str(x) for x in archive["source_names"]]
    wavenumbers = archive["wavenumbers"] if "wavenumbers" in archive else np.arange(spectra.shape[1], dtype=np.float32)
    print(f"[SMAE] Loaded {npz_path.name}: {spectra.shape} across {len(source_names)} sources: {source_names}")

    # Stratified train/val split (deterministic)
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    val_frac = float(cfg["data"].get("val_fraction", 0.02))
    val_indices, train_indices = [], []
    recon_examples: dict[str, np.ndarray] = {}
    for sid, sname in enumerate(source_names):
        idx_s = np.where(source_id == sid)[0]
        rng.shuffle(idx_s)
        n_val = max(16, int(len(idx_s) * val_frac))
        val_indices.append(idx_s[:n_val])
        train_indices.append(idx_s[n_val:])
        recon_examples[sname] = spectra[idx_s[0]].astype(np.float32)

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    train_ds = PretrainTensorDataset(spectra[train_idx], source_id[train_idx])
    val_ds = PretrainTensorDataset(spectra[val_idx], source_id[val_idx])

    # Temperature-balanced sampler (alpha=0.5 recommended)
    alpha = float(cfg["data"].get("sampling_alpha", 0.5))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 4)) if device.type == "cuda" else 0
    samples_per_epoch = cfg["training"].get("samples_per_epoch")
    n_epoch_samples = int(samples_per_epoch) if samples_per_epoch else len(train_ds)

    if abs(alpha - 1.0) > 1e-6:
        sample_weights, _ = compute_temperature_sampling_weights(source_id[train_idx], alpha=alpha)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=n_epoch_samples,
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # 2. Build exact SpectralMAE model
    m_cfg = cfg["model"]
    model = SpectralMAE(
        spectrum_size=int(m_cfg["spectrum_size"]),
        patch_size=int(m_cfg["patch_size"]),
        embed_dim=int(m_cfg["embed_dim"]),
        depth=int(m_cfg["depth"]),
        num_heads=int(m_cfg["num_heads"]),
        mlp_dim=int(m_cfg["mlp_dim"]),
        decoder_embed_dim=int(m_cfg["decoder_embed_dim"]),
        decoder_depth=int(m_cfg["decoder_depth"]),
        decoder_num_heads=int(m_cfg["decoder_num_heads"]),
        decoder_mlp_dim=int(m_cfg["decoder_mlp_dim"]),
        dropout=float(m_cfg.get("dropout", 0.0)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SMAE] Initialized SpectralMAE ({n_params:,d} parameters | patch_size={m_cfg['patch_size']} | mask_ratio={m_cfg['mask_ratio']})")

    # 3. Optimizer & output directories
    t_cfg = cfg["training"]
    total_epochs = int(max_epochs_override or t_cfg["epochs"])
    base_lr = float(t_cfg["lr"])
    min_lr = float(t_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(t_cfg.get("warmup_epochs", 10))
    grad_clip = float(t_cfg.get("grad_clip_norm", 1.0))
    mask_ratio = float(m_cfg["mask_ratio"])
    mask_mode = str(m_cfg.get("mask_mode", "random"))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(t_cfg.get("weight_decay", 1e-4))
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_scaler)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_scaler)
    )

    out_dir = root / cfg["logging"]["output_dir"]
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # 4. Resume checkpoint if present
    start_epoch = 1
    best_val_mse = float("inf")
    history: list[dict[str, float]] = []
    last_ckpt_path = out_dir / "last_checkpoint.pth"
    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_mse = float(ckpt.get("best_val_mse", float("inf")))
        history = list(ckpt.get("history", []))
        print(f"[SMAE] Resumed from epoch {start_epoch - 1} (best_val_mse={best_val_mse:.6f})")

    plot_recon_epochs = set(int(e) for e in cfg["logging"].get("plot_recon_epochs", [1, total_epochs]))
    plot_recon_epochs.add(total_epochs)
    save_every = int(cfg["logging"].get("save_every_epochs", 50))

    # 5. Pre-Training Loop
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        lr_now = _cosine_lr(epoch - 1, total_epochs, base_lr, min_lr, warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        tr_mse_sum, tr_cos_sum, tr_steps = 0.0, 0.0, 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(batch_x, mask_ratio=mask_ratio, mask_mode=mask_mode)
                loss = out["loss"]
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            tr_mse_sum += float(loss.detach().item())
            tr_cos_sum += float(out["masked_cosine_sim"].detach().item())
            tr_steps += 1

        # Validation pass
        model.eval()
        val_mse_sum, val_cos_sum, val_steps = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out = model(batch_x, mask_ratio=mask_ratio, mask_mode=mask_mode)
                val_mse_sum += float(out["loss"].item())
                val_cos_sum += float(out["masked_cosine_sim"].item())
                val_steps += 1

        tr_mse = tr_mse_sum / max(1, tr_steps)
        tr_cos = tr_cos_sum / max(1, tr_steps)
        val_mse = val_mse_sum / max(1, val_steps)
        val_cos = val_cos_sum / max(1, val_steps)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_mse": tr_mse,
            "train_cos": tr_cos,
            "val_mse": val_mse,
            "val_cos": val_cos,
            "epoch_sec": elapsed,
        })
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "history.csv", index=False)

        if epoch == 1 or epoch % 10 == 0 or epoch == total_epochs:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | lr={lr_now:.2e} | "
                f"Train MSE={tr_mse:.6f} Cos={tr_cos:.4f} | "
                f"Val MSE={val_mse:.6f} Cos={val_cos:.4f} | {elapsed:.1f}s"
            )

        # Save best checkpoint
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), out_dir / "best_smae_model.pth")

        # Save resumable checkpoint
        if epoch % save_every == 0 or epoch == total_epochs or epoch == 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mse": best_val_mse,
                    "history": history,
                    "config": cfg,
                },
                last_ckpt_path,
            )

        # Save reconstruction visualization snapshot (exact Deliverable 3.2 Fig. 5 style)
        if epoch in plot_recon_epochs:
            model.eval()
            torch.manual_seed(42)  # Deterministic mask across epochs for fair visual progression
            recons, masks = {}, {}
            with torch.no_grad():
                for sname, sp_vec in recon_examples.items():
                    x_t = torch.from_numpy(sp_vec).unsqueeze(0).to(device)
                    out_s = model(x_t, mask_ratio=mask_ratio, mask_mode=mask_mode)
                    recons[sname] = out_s["pred_spectra"][0].cpu().numpy()
                    masks[sname] = out_s["mask"][0].cpu().numpy()
            plot_smae_reconstructions(
                wavenumbers=wavenumbers,
                originals=recon_examples,
                reconstructions=recons,
                patch_masks=masks,
                epoch=epoch,
                patch_size=int(m_cfg["patch_size"]),
                out_path=fig_dir / f"smae_reconstruction_epoch_{epoch:04d}.png",
            )
            plt.close("all")

    # Final learning curves figure
    hist_df = pd.DataFrame(history)
    plot_smae_learning_curves(
        hist_df,
        title=f"SMAE Pre-Training Performance Metrics ({cfg['data']['regime']} — {cfg['experiment_name']})",
        out_path=fig_dir / "smae_learning_curves.png",
    )
    plt.close("all")

    return {
        "output_dir": out_dir,
        "best_val_mse": best_val_mse,
        "history_df": hist_df,
        "best_checkpoint": out_dir / "best_smae_model.pth",
    }
