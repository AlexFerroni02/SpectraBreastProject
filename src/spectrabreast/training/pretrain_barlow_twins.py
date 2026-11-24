"""
HPC-ready Pure PyTorch Pre-Training engine for `1D-ResNet-34 (Barlow Twins)`.

Ports the thesis TensorFlow/Keras Barlow Twins pre-training (Deliverable 3.2 Sec. 4.1)
to Pure PyTorch with 100% GPU-vectorized physical Raman augmentations (`TorchRamanAugmentor`),
trained on all `716,686` unique spectra (`pretrain_cnn_all_1000.npz`).
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from spectrabreast.data.splits import compute_temperature_sampling_weights
from spectrabreast.models.resnet_barlow_twins import BarlowTwins1D
from spectrabreast.training.plotting import (
    plot_barlow_twins_crosscorr,
    plot_barlow_twins_learning_curves,
)
from spectrabreast.training.pretrain_smae import (
    PretrainTensorDataset,
    _cosine_lr,
    _seed_everything,
    _select_amp_dtype,
)

__all__ = ["TorchRamanAugmentor", "train_barlow_twins_from_config"]


class TorchRamanAugmentor(nn.Module):
    """
    GPU-vectorized physical Raman augmentation pipeline for `Barlow Twins`:
    1. Sub-pixel wavenumber calibration drift (`max_shift_pts`) via `F.grid_sample`.
    2. Smooth 2nd-order Legendre multiplicative optical envelope (`slope_scale`).
    3. Residual quadratic fluorescence baseline perturbation (`tilt_scale`).
    4. Heteroscedastic Poisson-Gaussian CCD shot + readout noise (`read_noise`, `shot_noise`).
    5. Random local spectral cutout/masking (`p_cutout`, `cutout_max_width`).
    6. Exact `[0, 1]` post-augmentation re-anchoring.
    """

    def __init__(
        self,
        seq_len: int = 1000,
        max_shift_pts: float = 5.0,
        blur_kernel_max: int = 5,
        zero_prob: float = 0.02,
        scale_min: float = 0.90,
        scale_max: float = 1.10,
        slope_scale: float = 0.06,
        tilt_scale: float = 0.025,
        read_noise: float = 0.010,
        shot_noise: float = 0.015,
        p_cutout: float = 0.25,
        cutout_max_width: int = 40,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.max_shift_pts = max_shift_pts
        self.blur_kernel_max = blur_kernel_max
        self.zero_prob = zero_prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.slope_scale = slope_scale
        self.tilt_scale = tilt_scale
        self.read_noise = read_noise
        self.shot_noise = shot_noise
        self.p_cutout = p_cutout
        self.cutout_max_width = cutout_max_width

        grid = torch.linspace(-1.0, 1.0, seq_len)
        self.register_buffer("p1", grid.unsqueeze(0))  # (1, L)
        self.register_buffer("p2", (0.5 * (3.0 * grid * grid - 1.0)).unsqueeze(0))

    @torch.no_grad()
    def augment_view(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GPU-vectorized Raman augmentations faithfully porting `RamanAugmentorTensor`."""
        B, L = x.shape
        device = x.device
        out = x.clone()

        # 1. Sub-pixel wavenumber shift via 1D-as-2D grid_sample (p=0.50)
        if self.max_shift_pts > 0:
            apply_sh = (torch.rand(B, 1, device=device) < 0.50).float()
            shifts = apply_sh * (torch.rand(B, 1, device=device) * 2.0 - 1.0) * (2.0 * self.max_shift_pts / float(L - 1))
            base_x = self.p1.expand(B, L) - shifts
            grid_2d = torch.stack([base_x, torch.zeros_like(base_x)], dim=-1).unsqueeze(1)
            out = F.grid_sample(
                out.unsqueeze(1).unsqueeze(2), grid_2d,
                mode="bilinear", padding_mode="border", align_corners=True,
            ).squeeze(1).squeeze(1)

        # 2. Random 1D average blur (k in [3, 5], p=0.40) matching `average_blur`
        if self.blur_kernel_max >= 3:
            apply_bl = torch.rand(B, device=device) < 0.40
            idx_b = torch.where(apply_bl)[0]
            if idx_b.numel() > 0:
                k = int(torch.randint(3, self.blur_kernel_max + 1, (1,)).item())
                if k % 2 == 0:
                    k = max(3, k - 1)
                blurred = F.avg_pool1d(out[idx_b].unsqueeze(1), kernel_size=k, stride=1, padding=k // 2).squeeze(1)
                out[idx_b] = blurred

        # 3. Random global intensity scaling [0.9, 1.1] + smooth optical envelope
        if self.scale_max > self.scale_min:
            scale = self.scale_min + torch.rand(B, 1, device=device) * (self.scale_max - self.scale_min)
            out = out * scale
        if self.slope_scale > 0:
            a1 = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * self.slope_scale
            a2 = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * (0.5 * self.slope_scale)
            out = out * (1.0 + a1 * self.p1 + a2 * self.p2)

        # 4. Residual quadratic baseline curvature
        if self.tilt_scale > 0:
            b1 = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * self.tilt_scale
            b2 = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * self.tilt_scale
            c0 = torch.rand(B, 1, device=device) * self.tilt_scale
            out = out + c0 + b1 * self.p1 + b2 * self.p2

        # 5. Heteroscedastic Poisson-Gaussian CCD noise
        if self.read_noise > 0 or self.shot_noise > 0:
            sigma = torch.sqrt(
                torch.tensor(self.read_noise**2, device=device)
                + (self.shot_noise**2) * out.clamp(min=0.0)
            )
            out = out + torch.randn_like(out) * sigma

        # 6. Random point zeroing (`random_zero`, p=0.50) & optional local cutout
        if self.zero_prob > 0:
            apply_z = (torch.rand(B, 1, device=device) < 0.50)
            z_mask = apply_z & (torch.rand(B, L, device=device) < self.zero_prob)
            out = out.masked_fill(z_mask, 0.0)
        if self.p_cutout > 0 and self.cutout_max_width > 0:
            apply_c = torch.rand(B, device=device) < self.p_cutout
            idx_c = torch.where(apply_c)[0]
            if idx_c.numel() > 0:
                widths = torch.randint(10, self.cutout_max_width + 1, (idx_c.numel(),), device=device)
                starts = torch.randint(0, L - self.cutout_max_width, (idx_c.numel(),), device=device)
                pos = torch.arange(L, device=device).unsqueeze(0)
                mask = (pos >= starts.unsqueeze(1)) & (pos < (starts + widths).unsqueeze(1))
                out[idx_c] = out[idx_c].masked_fill(mask, 0.0)

        # 7. Exact [0, 1] re-anchoring
        xmin = out.amin(dim=1, keepdim=True)
        xmax = out.amax(dim=1, keepdim=True)
        denom = (xmax - xmin).clamp(min=1e-6)
        return (out - xmin) / denom

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.augment_view(x), self.augment_view(x)


def train_barlow_twins_from_config(
    config_path: str | Path,
    repo_root: str | Path | None = None,
    resume: bool = True,
    max_epochs_override: int | None = None,
    output_dir_override: str | None = None,
    batch_size_override: int | None = None,
) -> dict[str, Any]:
    """Run Pure PyTorch `BarlowTwins1D` pre-training from a YAML config file."""
    cfg_path = Path(config_path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if output_dir_override is not None:
        cfg["logging"]["output_dir"] = output_dir_override
    if batch_size_override is not None:
        cfg["training"]["batch_size"] = int(batch_size_override)

    root = Path(repo_root).resolve() if repo_root is not None else cfg_path.parents[2]
    _seed_everything(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype, use_scaler = _select_amp_dtype(device, bool(cfg["training"].get("use_amp", True)))
    print(f"[BarlowTwins] Device: {device} | AMP: {amp_enabled} ({amp_dtype})")

    # 1. Load 716,686-spectrum Multi-Regime CNN corpus
    npz_path = root / cfg["data"]["npz_path"]
    archive = np.load(npz_path, allow_pickle=True)
    spectra = archive["spectra"]
    source_id = archive["source_id"]
    source_names = [str(x) for x in archive["source_names"]]
    print(f"[BarlowTwins] Loaded {npz_path.name}: {spectra.shape} across {len(source_names)} sources")

    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    val_frac = float(cfg["data"].get("val_fraction", 0.01))
    val_indices, train_indices = [], []
    for sid in range(len(source_names)):
        idx_s = np.where(source_id == sid)[0]
        rng.shuffle(idx_s)
        n_val = max(32, int(len(idx_s) * val_frac))
        val_indices.append(idx_s[:n_val])
        train_indices.append(idx_s[n_val:])

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    train_ds = PretrainTensorDataset(spectra[train_idx], source_id[train_idx])
    val_ds = PretrainTensorDataset(spectra[val_idx], source_id[val_idx])

    alpha = float(cfg["data"].get("sampling_alpha", 0.5))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 4)) if device.type == "cuda" else 0
    samples_per_epoch = cfg["training"].get("samples_per_epoch")
    n_epoch_samples = int(samples_per_epoch) if samples_per_epoch else len(train_ds)

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
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )

    # 2. Build Pure PyTorch BarlowTwins1D + GPU Augmentor
    m_cfg = cfg["model"]
    model = BarlowTwins1D(
        backbone=str(m_cfg["backbone"]),
        in_channels=int(m_cfg.get("in_channels", 1)),
        seq_len=int(m_cfg.get("seq_len", 1000)),
        pool_mode=str(m_cfg.get("pool_mode", "flatten")),
        projector_dims=tuple(int(d) for d in m_cfg["projector_dims"]),
        lambda_bt=float(m_cfg["lambda_bt"]),
    ).to(device)

    a_cfg = cfg["augmentations"]
    augmentor = TorchRamanAugmentor(
        seq_len=int(m_cfg.get("seq_len", 1000)),
        max_shift_pts=float(a_cfg.get("max_shift_pts", 5.0)),
        blur_kernel_max=int(a_cfg.get("blur_kernel_max", 5)),
        zero_prob=float(a_cfg.get("zero_prob", 0.02)),
        scale_min=float(a_cfg.get("scale_min", 0.90)),
        scale_max=float(a_cfg.get("scale_max", 1.10)),
        slope_scale=float(a_cfg.get("slope_scale", 0.06)),
        tilt_scale=float(a_cfg.get("tilt_scale", 0.025)),
        read_noise=float(a_cfg.get("read_noise", 0.010)),
        shot_noise=float(a_cfg.get("shot_noise", 0.015)),
        p_cutout=float(a_cfg.get("p_cutout", 0.25)),
        cutout_max_width=int(a_cfg.get("cutout_max_width", 40)),
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_tot = sum(p.numel() for p in model.parameters())
    print(f"[BarlowTwins] Backbone={m_cfg['backbone']} (pool={m_cfg['pool_mode']}, out_dim={model.encoder.out_dim}) | Encoder={n_enc:,d} params | Total={n_tot:,d} params")

    # 3. Optimizer & Output Dirs
    t_cfg = cfg["training"]
    total_epochs = int(max_epochs_override or t_cfg["epochs"])
    base_lr = float(t_cfg["lr"])
    min_lr = float(t_cfg.get("min_lr", 1e-6))
    warmup_epochs = int(t_cfg.get("warmup_epochs", 10))
    grad_clip = float(t_cfg.get("grad_clip_norm", 2.0))
    lambda_bt = float(m_cfg["lambda_bt"])

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

    # 4. Resume support
    start_epoch = 1
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    last_ckpt_path = out_dir / "last_checkpoint.pth"
    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        history = list(ckpt.get("history", []))
        print(f"[BarlowTwins] Resumed from epoch {start_epoch - 1} (best_val_loss={best_val_loss:.4f})")

    plot_cc_epochs = set(int(e) for e in cfg["logging"].get("plot_crosscorr_epochs", [1, total_epochs]))
    plot_cc_epochs.add(total_epochs)
    save_every = int(cfg["logging"].get("save_every_epochs", 25))

    # 5. Pre-Training Loop
    for epoch in range(start_epoch, total_epochs + 1):
        t0 = time.time()
        lr_now = _cosine_lr(epoch - 1, total_epochs, base_lr, min_lr, warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        model.train()
        sums = {"loss": 0.0, "on_diag": 0.0, "off_w": 0.0, "diag_m": 0.0, "off_m": 0.0, "std_m": 0.0}
        tr_steps = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            v1, v2 = augmentor(batch_x)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(v1, v2)
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

            sums["loss"] += float(loss.detach().item())
            sums["on_diag"] += float(out["on_diag_loss"].detach().item())
            sums["off_w"] += float(lambda_bt * out["off_diag_loss"].detach().item())
            sums["diag_m"] += float(out["diag_mean"].item())
            sums["off_m"] += float(out["off_abs_mean"].item())
            sums["std_m"] += float(out["emb_std_mean"].item())
            tr_steps += 1

        # Validation pass
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        last_cc, last_stds = None, None
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                v1, v2 = augmentor(batch_x)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out = model(v1, v2)
                val_loss_sum += float(out["loss"].item())
                val_steps += 1
                if last_cc is None:
                    last_cc = out["cross_corr"].cpu().numpy()
                    last_stds = out["z1"].float().std(dim=0, unbiased=False).cpu().numpy()

        val_loss = val_loss_sum / max(1, val_steps)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": sums["loss"] / max(1, tr_steps),
            "val_loss": val_loss,
            "train_on_diag": sums["on_diag"] / max(1, tr_steps),
            "train_off_diag_weighted": sums["off_w"] / max(1, tr_steps),
            "train_diag_mean": sums["diag_m"] / max(1, tr_steps),
            "train_off_abs_mean": sums["off_m"] / max(1, tr_steps),
            "emb_std_mean": sums["std_m"] / max(1, tr_steps),
            "epoch_sec": elapsed,
        }
        history.append(row)
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(out_dir / "history.csv", index=False)

        if epoch == 1 or epoch % 5 == 0 or epoch == total_epochs:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | lr={lr_now:.2e} | "
                f"Train Loss={row['train_loss']:.2f} | Val Loss={val_loss:.2f} | "
                f"Diag={row['train_diag_mean']:.3f} |Off|={row['train_off_abs_mean']:.4f} | "
                f"Std={row['emb_std_mean']:.3f} | {elapsed:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_barlow_twins_full.pth")
            torch.save(model.encoder.state_dict(), out_dir / "best_resnet_encoder.pth")

        if epoch % save_every == 0 or epoch == total_epochs or epoch == 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                    "config": cfg,
                },
                last_ckpt_path,
            )

        if epoch in plot_cc_epochs and last_cc is not None and last_stds is not None:
            plot_barlow_twins_crosscorr(
                cross_corr=last_cc,
                emb_stds=last_stds,
                epoch=epoch,
                out_path=fig_dir / f"barlow_twins_crosscorr_epoch_{epoch:04d}.png",
            )
            plt.close("all")

    hist_df = pd.DataFrame(history)
    plot_barlow_twins_learning_curves(
        hist_df,
        title=f"Barlow Twins ({m_cfg['backbone']} PyTorch) Pre-Training Dynamics",
        out_path=fig_dir / "barlow_twins_learning_curves.png",
    )
    plt.close("all")

    return {
        "output_dir": out_dir,
        "best_val_loss": best_val_loss,
        "history_df": hist_df,
        "best_encoder_checkpoint": out_dir / "best_resnet_encoder.pth",
    }
