"""
Vectorized Cubic Spline resampling and canonical 1,000-point wavenumber grids
for Step 3 (`data/01_interim/` -> `data/02_ready/`).

Why boundary-clamped Natural Cubic Spline + post-crop [0, 1] re-anchoring?
--------------------------------------------------------------------------
1. Physical Peak Fidelity: Natural cubic spline interpolation (`bc_type='natural'`)
   is C^2-continuous and preserves the exact centroid, asymmetry, and FWHM of narrow
   Raman vibrational bands (e.g., 1003 cm^-1 Phenylalanine) when mapping between
   heterogeneous grating dispersions (500 to 2,670 native channels -> 1,000 points).
2. Zero Boundary Blow-Up: Evaluating an unguarded cubic polynomial even a few cm^-1
   outside a grating's native support (e.g., 650.0-660.4 cm^-1 for `spectrabreast_fp`)
   causes cubic runaway. By evaluating the spline strictly within `[wn_src[0], wn_src[-1]]`
   and clamping any edge points outside native support to `y[:, 0]` / `y[:, -1]`,
   boundary artifacts are completely eliminated.
3. Post-Crop [0, 1] Re-Anchoring: When cropping a sub-window (e.g., extracting `FP`
   650-1750 cm^-1 or `HW` 2765-3200 cm^-1 from Full-Range `trophy_jena`, or `CH`
   2800-3045 cm^-1 from `pdo_hw_maps` where the global maximum was on the 3400 cm^-1
   water band), the cropped window's local maximum may be < 1.0, and cubic interpolation
   between discrete nodes can introduce ~1e-3 overshoot outside [0, 1]. A single
   post-crop Min-Max [0, 1] pass guarantees exact `[0.0000, 1.0000]` bounds on every
   compiled 1,000-point tensor.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from .cleaning import minmax_normalize_batch

N_READY_POINTS: int = 1000

# Canonical 1,000-point grids for 1D-ViT (SMAE), 1D-ResNet-34 (Barlow Twins), and Downstream tasks
GRID_FP_1000: np.ndarray = np.linspace(650.0, 1750.0, N_READY_POINTS, dtype=np.float32)
GRID_HW_1000: np.ndarray = np.linspace(2765.0, 3200.0, N_READY_POINTS, dtype=np.float32)
GRID_CH_1000: np.ndarray = np.linspace(2800.0, 3045.0, N_READY_POINTS, dtype=np.float32)
GRID_FR_JENA_1000: np.ndarray = np.linspace(650.0, 3200.0, N_READY_POINTS, dtype=np.float32)
GRID_FR_RENZO_1000: np.ndarray = np.linspace(380.0, 3049.0, N_READY_POINTS, dtype=np.float32)

__all__ = [
    "N_READY_POINTS",
    "GRID_FP_1000",
    "GRID_HW_1000",
    "GRID_CH_1000",
    "GRID_FR_JENA_1000",
    "GRID_FR_RENZO_1000",
    "resample_cubic_spline_batch",
]


def resample_cubic_spline_batch(
    wn_src: np.ndarray,
    spectra: np.ndarray,
    wn_target: np.ndarray,
    renormalize_minmax: bool = True,
    batch_size: int = 25000,
) -> np.ndarray:
    """
    Resample a 2D batch of spectra `(N, C_src)` from `wn_src` onto `wn_target` `(C_tgt,)`
    using a vectorized Natural Cubic Spline with boundary clamping and optional
    post-crop `[0, 1]` Min-Max re-anchoring.
    """
    wn_s = np.asarray(wn_src, dtype=np.float64).flatten()
    wn_t = np.asarray(wn_target, dtype=np.float64).flatten()
    X = np.asarray(spectra, dtype=np.float32)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    # Ensure strictly increasing source wavenumber axis
    if wn_s[0] > wn_s[-1]:
        wn_s = wn_s[::-1]
        X = X[:, ::-1]

    # Guard against any duplicate wavenumbers if present
    diffs = np.diff(wn_s)
    if np.any(diffs <= 0):
        uniq_mask = np.concatenate(([True], diffs > 1e-7))
        wn_s = wn_s[uniq_mask]
        X = X[:, uniq_mask]

    # Clamp target evaluation points to [wn_s[0], wn_s[-1]] so out-of-bounds edges
    # take the exact boundary value instead of cubic polynomial blow-up
    wn_t_clamped = np.clip(wn_t, wn_s[0], wn_s[-1])

    n_total = X.shape[0]
    out = np.empty((n_total, wn_t.size), dtype=np.float32)

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk = X[start:end].astype(np.float64, copy=False)
        cs = CubicSpline(wn_s, chunk, axis=1, bc_type="natural", extrapolate=True)
        res = cs(wn_t_clamped).astype(np.float32)
        if renormalize_minmax:
            res = minmax_normalize_batch(res)
        out[start:end] = res

    return out
