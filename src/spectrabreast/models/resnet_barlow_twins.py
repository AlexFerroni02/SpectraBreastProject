"""
Pure PyTorch implementation of `Barlow Twins` with a `1D-ResNet-34` (or `1D-ResNet-18`)
backbone, faithfully porting the thesis TensorFlow/Keras Barlow Twins architecture
(Deliverable 3.2 Section 4.1) into PyTorch for HPC GPU pre-training and transfer learning.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = [
    "BasicBlock1D",
    "ResNet1DBackbone",
    "BarlowTwinsLoss",
    "BarlowTwins1D",
    "ResNet1DClassifier",
]


class BasicBlock1D(nn.Module):
    """Standard 1D residual block (`Conv1d(k=3) -> BN -> ReLU -> Conv1d(k=3) -> BN + skip`)."""

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1DBackbone(nn.Module):
    """
    1D Residual Convolutional Encoder (`resnet34`: `[3, 4, 6, 3]` or `resnet18`: `[2, 2, 2, 2]`).

    Matches Deliverable 3.2 Section 4.1:
    - Stem: `Conv1d(1, 64, k=7, s=2, p=3) -> BN -> ReLU -> MaxPool1d(k=3, s=2, p=1)`
    - Stages: `[64, 128, 256, 512]`
    - Pooling (`pool_mode`):
      * `"avgpool2_flatten"`: `AvgPool1d(2, stride=2, ceil_mode=True) + Flatten` -> `(B, 8192)`
        (exact 1-to-1 match to `ResNet34Old` in `Barlow_Twins_Resnet_34.ipynb`!).
      * `"flatten"`: preserves all local spatial feature maps `(B, 512, 32) -> (B, 16384)`.
      * `"adaptive_8"`: `AdaptiveAvgPool1d(8) + Flatten` -> `(B, 4096)`.
      * `"gap"`: `AdaptiveAvgPool1d(1) + Flatten` -> `(B, 512)`.
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        in_channels: int = 1,
        seq_len: int = 1000,
        pool_mode: str = "avgpool2_flatten",
    ) -> None:
        super().__init__()
        if backbone == "resnet34":
            layers = [3, 4, 6, 3]
        elif backbone == "resnet18":
            layers = [2, 2, 2, 2]
        else:
            raise ValueError(f"Unsupported backbone: {backbone!r}. Choose 'resnet34' or 'resnet18'.")

        self.backbone_name = backbone
        self.pool_mode = pool_mode
        self._cur_channels = 64

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_stage(64, layers[0], stride=1)
        self.layer2 = self._make_stage(128, layers[1], stride=2)
        self.layer3 = self._make_stage(256, layers[2], stride=2)
        self.layer4 = self._make_stage(512, layers[3], stride=2)

        if pool_mode == "avgpool2_flatten":
            self.pool: nn.Module = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
        elif pool_mode == "gap":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool_mode == "adaptive_8":
            self.pool = nn.AdaptiveAvgPool1d(8)
        elif pool_mode == "flatten":
            self.pool = nn.Identity()
        else:
            raise ValueError(f"Unsupported pool_mode: {pool_mode!r}")

        # Infer exact flattened feature dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, seq_len)
            self.out_dim = int(self.forward(dummy).shape[1])

        self._init_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock1D(self._cur_channels, out_channels, stride=stride)]
        self._cur_channels = out_channels
        for _ in range(1, num_blocks):
            blocks.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins Redundancy-Reduction Loss (Zbontar et al., ICML 2021):
        L_BT = sum_i (1 - C_ii)^2 + lambda_bt * sum_{i != j} C_ij^2
    where `C` is the empirical cross-correlation matrix between batch-normalized
    projector embeddings `(z_a, z_b)`.
    """

    def __init__(self, lambda_bt: float = 5e-3, eps: float = 1e-5) -> None:
        super().__init__()
        self.lambda_bt = lambda_bt
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, torch.Tensor]:
        B, D = z_a.shape
        # Standardize along batch dimension in fp32 for numerical stability under AMP
        z_a_f = z_a.float()
        z_b_f = z_b.float()
        z_a_norm = (z_a_f - z_a_f.mean(dim=0)) / (z_a_f.std(dim=0, unbiased=False) + self.eps)
        z_b_norm = (z_b_f - z_b_f.mean(dim=0)) / (z_b_f.std(dim=0, unbiased=False) + self.eps)

        c = (z_a_norm.T @ z_b_norm) / float(B)  # (D, D)
        diag = torch.diagonal(c)
        on_diag = (diag - 1.0).pow(2).sum()

        # Off-diagonal elements
        off_mask = ~torch.eye(D, dtype=torch.bool, device=c.device)
        off_vals = c[off_mask]
        off_diag = off_vals.pow(2).sum()

        loss = on_diag + self.lambda_bt * off_diag

        with torch.no_grad():
            diag_mean = diag.mean()
            off_abs_mean = off_vals.abs().mean()
            emb_std_mean = z_a_f.std(dim=0, unbiased=False).mean()

        return {
            "loss": loss,
            "on_diag_loss": on_diag,
            "off_diag_loss": off_diag,
            "diag_mean": diag_mean,
            "off_abs_mean": off_abs_mean,
            "emb_std_mean": emb_std_mean,
            "cross_corr": c.detach(),
        }


class BarlowTwins1D(nn.Module):
    """
    Full 1D Barlow Twins model (`ResNet1DBackbone` + 3-layer Projector MLP `[2048, 2048, 2048]`).
    """

    def __init__(
        self,
        backbone: str = "resnet34",
        in_channels: int = 1,
        seq_len: int = 1000,
        pool_mode: str = "avgpool2_flatten",
        projector_dims: Sequence[int] = (2048, 2048, 2048),
        lambda_bt: float = 5e-3,
    ) -> None:
        super().__init__()
        self.encoder = ResNet1DBackbone(
            backbone=backbone,
            in_channels=in_channels,
            seq_len=seq_len,
            pool_mode=pool_mode,
        )
        dims = [self.encoder.out_dim] + list(projector_dims)
        proj_layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_d, out_d = dims[idx], dims[idx + 1]
            is_last = idx == (len(dims) - 2)
            proj_layers.append(nn.Linear(in_d, out_d, bias=is_last))
            if not is_last:
                proj_layers.append(nn.BatchNorm1d(out_d))
                proj_layers.append(nn.ReLU(inplace=True))
        self.projector = nn.Sequential(*proj_layers)
        self.criterion = BarlowTwinsLoss(lambda_bt=lambda_bt)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor) -> dict[str, torch.Tensor]:
        h1 = self.encoder(view_1)
        h2 = self.encoder(view_2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        out = self.criterion(z1, z2)
        out["z1"] = z1.detach()
        return out


class ResNet1DClassifier(nn.Module):
    """Downstream classification wrapper reusing the pre-trained `ResNet1DBackbone`."""

    def __init__(
        self,
        encoder: ResNet1DBackbone,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.out_dim),
            nn.Linear(encoder.out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(spectra))
