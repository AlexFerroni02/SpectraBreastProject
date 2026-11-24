"""
Spectral Masked Autoencoder (`SpectralMAE` — 1D-ViT) and Downstream Classifier
(`SpectralMAEClassifier`) in PyTorch.

Architecture fidelity:
----------------------
Matches the exact winning Exp. 9 Deliverable 3.2 specification (`4,485,604`
parameters for `spectrum_size=1000, patch_size=100, embed_dim=256, depth=8,
num_heads=8, mlp_dim=512, decoder_embed_dim=128, decoder_depth=1,
decoder_num_heads=8, decoder_mlp_dim=512`), including exact state-dict key
compatibility with `smae_fp_1000pt_minmax_exp9.pth`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PatchEmbed1D",
    "MultiHeadSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "SpectralMAE",
    "SpectralMAEClassifier",
]


class PatchEmbed1D(nn.Module):
    """Non-overlapping 1D spectral patch embedding via strided Conv1d."""

    def __init__(self, spectrum_size: int = 1000, patch_size: int = 100, embed_dim: int = 256) -> None:
        super().__init__()
        if spectrum_size % patch_size != 0:
            raise ValueError(f"spectrum_size ({spectrum_size}) must be divisible by patch_size ({patch_size}).")
        self.spectrum_size = spectrum_size
        self.patch_size = patch_size
        self.num_patches = spectrum_size // patch_size
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) or (B, L) -> (B, num_patches, embed_dim)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.proj(x).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention with bias-free QKV projection."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, N, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use Flash/SDPA on PyTorch >= 2.0 when available
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)


class FeedForward(nn.Module):
    """Two-layer GELU MLP matching thesis state_dict keys (`ff.net.0`, `ff.net.3`)."""

    def __init__(self, dim: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block matching thesis checkpoint structure."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.shortcut = nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ls1 = nn.Identity()
        self.sd1 = nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim=mlp_dim, dropout=dropout)
        self.ls2 = nn.Identity()
        self.sd2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shortcut(x) + self.sd1(self.ls1(self.attn(self.norm1(x))))
        x = self.shortcut(x) + self.sd2(self.ls2(self.ff(self.norm2(x))))
        return x


class SpectralMAE(nn.Module):
    """
    1D Spectral Masked Autoencoder (`SMAE`) for Raman spectroscopy.

    Default parameters (`spectrum_size=1000, patch_size=100, embed_dim=256,
    depth=8, num_heads=8, mlp_dim=512, decoder_embed_dim=128, decoder_depth=1,
    decoder_num_heads=8, decoder_mlp_dim=512`) yield exactly `4,485,604` parameters.
    """

    def __init__(
        self,
        spectrum_size: int = 1000,
        patch_size: int = 100,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 8,
        mlp_dim: int = 512,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 1,
        decoder_num_heads: int = 8,
        decoder_mlp_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.spectrum_size = spectrum_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = spectrum_size // patch_size

        # Encoder
        self.patch_embed = PatchEmbed1D(spectrum_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_dim, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Asymmetric Lightweight Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_embed_dim, decoder_num_heads, decoder_mlp_dim, dropout=dropout)
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert `(B, L)` or `(B, 1, L)` spectra into `(B, P, patch_size)`."""
        if x.ndim == 3:
            x = x.squeeze(1)
        B, L = x.shape
        return x.reshape(B, self.num_patches, self.patch_size)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Convert `(B, P, patch_size)` back into `(B, L)`."""
        B = patches.shape[0]
        return patches.reshape(B, self.spectrum_size)

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        mask_mode: str = "random",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mask `int(P * mask_ratio)` patches per spectrum.
        Returns `(x_visible, mask, ids_restore)` where `mask` is `(B, P)` with `1=masked, 0=visible`.
        """
        B, P, D = x.shape
        len_keep = int(round(P * (1.0 - mask_ratio)))
        len_keep = max(1, min(P, len_keep))

        if mask_mode == "block" and len_keep < P:
            # Contiguous multi-block masking via circular random block scoring
            noise = torch.rand(B, P, device=x.device)
            # Smooth scores along patch axis to induce contiguous masked spans
            kernel = torch.ones(1, 1, min(3, P), device=x.device) / float(min(3, P))
            noise = F.conv1d(noise.unsqueeze(1), kernel, padding=min(3, P) // 2).squeeze(1)[:, :P]
        else:
            noise = torch.rand(B, P, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones((B, P), device=x.device, dtype=torch.float32)
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.50,
        mask_mode: str = "random",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode visible patches (or all patches when `mask_ratio == 0.0`)."""
        tokens = self.patch_embed(x) + self.pos_embed[:, 1:, :]

        if mask_ratio > 0.0:
            tokens, mask, ids_restore = self.random_masking(tokens, mask_ratio, mask_mode=mask_mode)
        else:
            B, P, _ = tokens.shape
            mask = torch.zeros((B, P), device=tokens.device, dtype=torch.float32)
            ids_restore = torch.arange(P, device=tokens.device).unsqueeze(0).expand(B, -1)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens, mask, ids_restore

    def forward_decoder(self, latent: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Reconstruct all patches from encoded visible tokens + learnable mask tokens."""
        x = self.decoder_embed(latent)
        B = x.shape[0]
        num_mask = ids_restore.shape[1] + 1 - x.shape[1]
        mask_tokens = self.mask_token.expand(B, num_mask, -1)

        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_no_cls = torch.gather(x_no_cls, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_no_cls], dim=1)
        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.decoder_pred(x[:, 1:, :])  # (B, P, patch_size)
        return pred

    def compute_metrics(
        self,
        spectra: torch.Tensor,
        pred_patches: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute exact thesis pre-training metrics strictly over MASKED patches:
        1. `loss_mse`: Mean Squared Error on masked patches.
        2. `masked_cosine_sim`: Shape alignment (Cosine Similarity -> 1.0) on masked patches.
        """
        target = self.patchify(spectra)  # (B, P, patch_size)
        mse_per_patch = (pred_patches - target).pow(2).mean(dim=-1)  # (B, P)
        denom = mask.sum().clamp(min=1.0)
        loss_mse = (mse_per_patch * mask).sum() / denom

        with torch.no_grad():
            # Exact global masked-point cosine similarity matching MAE_100_patches.ipynb Cell 4
            target_flat = spectra.squeeze(1) if spectra.ndim == 3 else spectra
            pred_flat = pred_patches.reshape(target_flat.shape[0], -1)
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, self.patch_size).reshape(target_flat.shape[0], -1).float()
            t_masked = target_flat * mask_expanded
            p_masked = pred_flat * mask_expanded
            dot = (t_masked * p_masked).sum(dim=1)
            cos_sim = dot / (t_masked.norm(dim=1) * p_masked.norm(dim=1) + 1e-8)
            masked_cosine_sim = cos_sim.mean()

        return loss_mse, masked_cosine_sim

    def forward(
        self,
        spectra: torch.Tensor,
        mask_ratio: float = 0.50,
        mask_mode: str = "random",
    ) -> dict[str, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(spectra, mask_ratio=mask_ratio, mask_mode=mask_mode)
        pred_patches = self.forward_decoder(latent, ids_restore)
        loss_mse, masked_cos = self.compute_metrics(spectra, pred_patches, mask)
        return {
            "loss": loss_mse,
            "masked_cosine_sim": masked_cos,
            "pred_patches": pred_patches,
            "pred_spectra": self.unpatchify(pred_patches),
            "mask": mask,
        }

    def encode_features(self, spectra: torch.Tensor, pool: str = "mean") -> torch.Tensor:
        """Extract frozen/trainable representations `(B, embed_dim)` with `mask_ratio=0.0`."""
        tokens, _, _ = self.forward_encoder(spectra, mask_ratio=0.0)
        if pool == "cls":
            return tokens[:, 0, :]
        # Exact Report_Pre_Training Eq. (1): mean of all patch tokens excluding CLS
        return tokens[:, 1:, :].mean(dim=1)


class SpectralMAEClassifier(nn.Module):
    """
    Downstream classification wrapper for `SpectralMAE` matching Report_Pre_Training Sec. 7.2:
    `LayerNorm(embed_dim) -> Linear(embed_dim, 128) -> GELU -> Dropout(0.3) -> Linear(128, num_classes)`.
    """

    def __init__(
        self,
        encoder: SpectralMAE,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        self.head = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Linear(encoder.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self, n_last_blocks: int | None = None) -> None:
        if n_last_blocks is None:
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            self.freeze_encoder()
            for blk in list(self.encoder.blocks)[-n_last_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.encoder.norm.parameters():
                p.requires_grad = True

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.encode_features(spectra, pool=self.pool)
        return self.head(feats)
