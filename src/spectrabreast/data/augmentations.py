"""
Physically grounded Raman spectral augmentations for Contrastive SSL
(`1D-ResNet-34 Barlow Twins`) and Contiguous Multi-Block Patch Masking for
Masked Autoencoding (`1D-ViT SMAE`).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RamanPhysicalAugmentor:
    """
    Physically motivated stochastic augmentation pipeline for 1D Raman spectra
    (`1,000` channels in `[0, 1]`), designed to generate two correlated views
    `(y_a, y_b)` for `1D-ResNet-34 (Barlow Twins)` without destroying biochemical
    peak identities.

    Transformations applied per view:
    ---------------------------------
    1. Sub-Pixel Wavenumber Calibration Drift (`max_shift_pts = 1.8`):
       Simulates slight monochromator / grating thermal drift across instruments
       via continuous linear interpolation with boundary clamping.
    2. Multiplicative Etalon / Chromatic Response Envelope (`slope_scale = 0.10`):
       Multiplies the spectrum by a smooth 2nd-order Legendre polynomial envelope
       `1 + a_1 P_1(x) + a_2 P_2(x)` to model subtle optical transmission differences.
    3. Residual Fluorescence / Baseline Curvature Perturbation (`tilt_scale = 0.035`):
       Adds a low-amplitude smooth quadratic baseline curve to enforce invariance
       against minor residual background variations.
    4. Heteroscedastic CCD Shot + Readout Noise (`read_noise = 0.012`, `shot_noise = 0.020`):
       Adds physical photon-counting noise whose standard deviation scales with
       square-root intensity `sqrt(sigma_read^2 + eta^2 * I(nu))`.
    5. Post-Augmentation `[0, 1]` Re-Anchoring:
       Guarantees every augmented view remains strictly within `[0.0000, 1.0000]`.
    """

    max_shift_pts: float = 1.8
    slope_scale: float = 0.10
    tilt_scale: float = 0.035
    read_noise: float = 0.012
    shot_noise: float = 0.020
    p_shift: float = 0.85
    p_envelope: float = 0.80
    p_tilt: float = 0.75
    p_noise: float = 0.90

    def augment_single_batch(
        self,
        spectra: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Apply stochastic physical Raman augmentations to a batch `(B, L)`."""
        if rng is None:
            rng = np.random.default_rng()

        X = np.asarray(spectra, dtype=np.float32).copy()
        if X.ndim == 1:
            X = X[np.newaxis, :]
        B, L = X.shape
        grid = np.linspace(-1.0, 1.0, L, dtype=np.float32)
        p1 = grid
        p2 = 0.5 * (3.0 * grid * grid - 1.0)

        # 1. Sub-pixel wavenumber calibration shift
        if self.max_shift_pts > 0:
            apply_sh = rng.random(B) < self.p_shift
            shifts = rng.uniform(-self.max_shift_pts, self.max_shift_pts, size=B).astype(np.float32)
            shifts = np.where(apply_sh, shifts, 0.0)
            base_idx = np.arange(L, dtype=np.float32)
            for i in np.where(apply_sh)[0]:
                q = np.clip(base_idx - shifts[i], 0.0, float(L - 1))
                idx0 = np.floor(q).astype(np.int32)
                idx1 = np.minimum(idx0 + 1, L - 1)
                frac = q - idx0
                X[i] = (1.0 - frac) * X[i, idx0] + frac * X[i, idx1]

        # 2. Smooth multiplicative instrument response envelope
        if self.slope_scale > 0:
            apply_env = (rng.random((B, 1)) < self.p_envelope).astype(np.float32)
            a1 = rng.uniform(-self.slope_scale, self.slope_scale, size=(B, 1)).astype(np.float32)
            a2 = rng.uniform(-0.5 * self.slope_scale, 0.5 * self.slope_scale, size=(B, 1)).astype(np.float32)
            env = 1.0 + apply_env * (a1 * p1[np.newaxis, :] + a2 * p2[np.newaxis, :])
            X = X * env

        # 3. Residual baseline tilt / fluorescence curvature
        if self.tilt_scale > 0:
            apply_t = (rng.random((B, 1)) < self.p_tilt).astype(np.float32)
            b1 = rng.uniform(-self.tilt_scale, self.tilt_scale, size=(B, 1)).astype(np.float32)
            b2 = rng.uniform(-self.tilt_scale, self.tilt_scale, size=(B, 1)).astype(np.float32)
            c0 = rng.uniform(0.0, self.tilt_scale, size=(B, 1)).astype(np.float32)
            tilt = apply_t * (c0 + b1 * p1[np.newaxis, :] + b2 * p2[np.newaxis, :])
            X = X + tilt

        # 4. Heteroscedastic Poisson-Gaussian CCD noise
        if self.read_noise > 0 or self.shot_noise > 0:
            apply_n = (rng.random((B, 1)) < self.p_noise).astype(np.float32)
            sigma = np.sqrt(
                self.read_noise**2 + (self.shot_noise**2) * np.maximum(X, 0.0)
            ).astype(np.float32)
            noise = rng.normal(0.0, 1.0, size=(B, L)).astype(np.float32) * sigma * apply_n
            X = X + noise

        # 5. Re-anchor each spectrum to [0.0000, 1.0000]
        xmin = X.min(axis=1, keepdims=True)
        xmax = X.max(axis=1, keepdims=True)
        denom = np.where((xmax - xmin) > 1e-8, xmax - xmin, 1.0)
        X = (X - xmin) / denom
        return X.astype(np.float32)

    def generate_barlow_twins_views(
        self,
        spectra: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate two independent physically augmented views `(view_1, view_2)`."""
        if rng is None:
            rng = np.random.default_rng()
        v1 = self.augment_single_batch(spectra, rng=rng)
        v2 = self.augment_single_batch(spectra, rng=rng)
        return v1, v2


@dataclass
class SpectralBlockMaskGenerator:
    """
    Patch-level mask generator for `1D-ViT (SMAE)` on `1,000`-point spectra.

    Why Contiguous Multi-Block Masking (`mode='block'`) over Random Patch Masking?
    ------------------------------------------------------------------------------
    With `seq_len = 1000` and `patch_size = 20` (`n_patches = 50`), each patch spans
    `22.0 cm^-1` in Fingerprint (`650-1750 cm^-1`) or `8.7 cm^-1` in Extended HW
    (`2765-3200 cm^-1`). Because biological Raman bands have FWHM ~ `15-45 cm^-1`
    (FP) and `80-250 cm^-1` (HW), masking isolated single patches allows a model to
    cheat via local interpolation from neighboring patches.
    Contiguous Multi-Block Masking groups the `38` masked patches (`75%` mask ratio)
    into contiguous blocks of `min_block_patches = 2` to `max_block_patches = 6`
    patches (`40-120` contiguous channels), completely occluding entire vibrational
    bands and forcing the encoder to learn long-range biochemical couplings across
    the spectrum.
    """

    seq_len: int = 1000
    patch_size: int = 20
    mask_ratio: float = 0.75
    min_block_patches: int = 2
    max_block_patches: int = 6

    @property
    def n_patches(self) -> int:
        return self.seq_len // self.patch_size

    @property
    def n_masked_patches(self) -> int:
        return int(round(self.n_patches * self.mask_ratio))

    def generate_masks(
        self,
        batch_size: int,
        mode: str = "block",
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return `(patch_mask, point_mask)` where:
        - `patch_mask`: `(B, n_patches)` boolean/int8 (`1 = masked`, `0 = visible`)
        - `point_mask`: `(B, seq_len)` boolean/int8 (`1 = masked`, `0 = visible`)
        """
        if rng is None:
            rng = np.random.default_rng()

        P = self.n_patches
        target_m = self.n_masked_patches
        patch_mask = np.zeros((batch_size, P), dtype=np.int8)

        if mode == "random":
            for b in range(batch_size):
                idx = rng.choice(P, size=target_m, replace=False)
                patch_mask[b, idx] = 1
        elif mode == "block":
            for b in range(batch_size):
                masked = 0
                attempts = 0
                while masked < target_m and attempts < 60:
                    attempts += 1
                    rem = target_m - masked
                    blk = int(rng.integers(self.min_block_patches, self.max_block_patches + 1))
                    blk = min(blk, rem)
                    start = int(rng.integers(0, max(1, P - blk + 1)))
                    for p_idx in range(start, min(P, start + blk)):
                        if patch_mask[b, p_idx] == 0 and masked < target_m:
                            patch_mask[b, p_idx] = 1
                            masked += 1
                if masked < target_m:
                    unmasked_idx = np.where(patch_mask[b] == 0)[0]
                    fill_idx = rng.choice(unmasked_idx, size=target_m - masked, replace=False)
                    patch_mask[b, fill_idx] = 1
        else:
            raise ValueError(f"Unsupported masking mode: {mode!r}")

        point_mask = np.repeat(patch_mask, self.patch_size, axis=1)
        return patch_mask, point_mask
