"""
Preprocessing algorithms for Raman spectroscopy in SPECTRA-BREAST:
- Vectorized Whitaker-Hayes cosmic-ray despiking (`despike_whitaker_hayes_batch`)
- Vectorized pentadiagonal LDL^T Asymmetric Least Squares baseline subtraction (`als_baseline_batch`)
- Per-spectrum Min-Max [0, 1] normalization (`minmax_normalize_batch`)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

__all__ = [
    "despike_whitaker_hayes_batch",
    "als_baseline_batch",
    "minmax_normalize_batch",
    "clean_raw_spectra_batch",
]


def despike_whitaker_hayes_batch(
    spectra: np.ndarray,
    threshold: float = 7.0,
    window: int = 5,
) -> np.ndarray:
    """
    Remove narrow cosmic-ray spikes from a 2D batch of Raman spectra (N, L)
    using the Whitaker-Hayes modified Z-score on first differences.

    Only channels exceeding the Z-score threshold are replaced by a local
    1D median filter of width ``window``; all other channels remain untouched.
    """
    Y = np.asarray(spectra, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    diff = np.diff(Y, axis=1)
    med = np.median(diff, axis=1, keepdims=True)
    mad = np.median(np.abs(diff - med), axis=1, keepdims=True)
    mod_z = 0.6745 * np.abs(diff - med) / (mad + 1e-9)

    spike_mask = np.zeros_like(Y, dtype=bool)
    spike_mask[:, 1:] |= mod_z > threshold
    spike_mask[:, :-1] |= mod_z > threshold

    if not np.any(spike_mask):
        return Y.copy()

    Y_out = Y.copy()
    rows_with_spikes = np.where(np.any(spike_mask, axis=1))[0]
    if rows_with_spikes.size > 0:
        med_rows = median_filter(Y[rows_with_spikes], size=(1, window), mode="nearest")
        mask_sub = spike_mask[rows_with_spikes]
        Y_sub = Y_out[rows_with_spikes]
        Y_sub[mask_sub] = med_rows[mask_sub]
        Y_out[rows_with_spikes] = Y_sub
    return Y_out


def als_baseline_batch(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 8,
    chunk_size: int = 16384,
) -> np.ndarray:
    """
    Fast vectorized Asymmetric Least Squares (ALS, Eilers & Boelens 2005)
    baseline subtraction using an exact symmetric pentadiagonal LDL^T solver
    vectorized across the batch dimension.

    Parameters
    ----------
    spectra : np.ndarray
        2D array of shape (N, L) containing input Raman spectra.
    lam : float, default=1e5
        Smoothness penalty parameter lambda on second differences.
    p : float, default=0.01
        Asymmetry parameter in (0, 1), typically 0.005 to 0.01 for Raman peaks.
    niter : int, default=8
        Number of iteratively reweighted least squares iterations.
    chunk_size : int, default=16384
        Maximum number of spectra processed per vectorized batch to bound RAM.

    Returns
    -------
    np.ndarray
        Baseline-subtracted spectra (spectra - baseline) of shape (N, L).
    """
    Y = np.asarray(spectra, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    N, L = Y.shape
    out = np.empty((N, L), dtype=np.float64)

    d0_base = lam * np.array([1.0, 5.0] + [6.0] * (L - 4) + [5.0, 1.0], dtype=np.float64)
    d1_base = lam * np.array([-2.0] + [-4.0] * (L - 3) + [-2.0], dtype=np.float64)
    d2_base = np.full(L - 2, lam * 1.0, dtype=np.float64)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B = end - start
        YT = np.ascontiguousarray(Y[start:end].T, dtype=np.float64)
        WT = np.ones((L, B), dtype=np.float64)
        d = np.empty((L, B), dtype=np.float64)
        l1 = np.empty((L - 1, B), dtype=np.float64)
        l2 = np.empty((L - 2, B), dtype=np.float64)
        z = np.empty((L, B), dtype=np.float64)

        for _ in range(niter):
            d[0] = d0_base[0] + WT[0]
            l1[0] = d1_base[0] / d[0]
            l2[0] = d2_base[0] / d[0]

            d[1] = (d0_base[1] + WT[1]) - l1[0] * l1[0] * d[0]
            l1[1] = (d1_base[1] - l2[0] * l1[0] * d[0]) / d[1]
            l2[1] = d2_base[1] / d[1]

            for j in range(2, L):
                dj = (
                    (d0_base[j] + WT[j])
                    - l1[j - 1] * l1[j - 1] * d[j - 1]
                    - l2[j - 2] * l2[j - 2] * d[j - 2]
                )
                d[j] = dj
                if j < L - 1:
                    l1[j] = (d1_base[j] - l2[j - 1] * l1[j - 1] * d[j - 1]) / dj
                if j < L - 2:
                    l2[j] = d2_base[j] / dj

            z[0] = WT[0] * YT[0]
            z[1] = WT[1] * YT[1] - l1[0] * z[0]
            for j in range(2, L):
                z[j] = WT[j] * YT[j] - l1[j - 1] * z[j - 1] - l2[j - 2] * z[j - 2]

            z /= d
            z[L - 2] -= l1[L - 2] * z[L - 1]
            for j in range(L - 3, -1, -1):
                z[j] -= l1[j] * z[j + 1] + l2[j] * z[j + 2]

            WT = np.where(YT > z, p, 1.0 - p)

        out[start:end] = (YT - z).T

    return out


def minmax_normalize_batch(spectra: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Apply strict per-spectrum Min-Max [0, 1] normalization along the wavenumber axis:
        x_norm = (x - min(x)) / (max(x) - min(x) + eps)
    """
    Y = np.asarray(spectra, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[None, :]
    mn = np.nanmin(Y, axis=1, keepdims=True)
    mx = np.nanmax(Y, axis=1, keepdims=True)
    span = mx - mn
    normed = np.where(span > eps, (Y - mn) / np.maximum(span, eps), 0.0)
    return np.clip(normed, 0.0, 1.0).astype(np.float32)


def clean_raw_spectra_batch(
    spectra: np.ndarray,
    lam: float = 1e5,
    p: float = 0.01,
    niter: int = 8,
    despike_threshold: float = 7.0,
) -> np.ndarray:
    """
    Complete native-grid cleaning pipeline for unprocessed RAW Raman spectra:
    1. Whitaker-Hayes cosmic-ray despiking
    2. Vectorized ALS fluorescence baseline subtraction
    3. Single per-spectrum Min-Max [0, 1] normalization
    """
    despiked = despike_whitaker_hayes_batch(spectra, threshold=despike_threshold)
    baselined = als_baseline_batch(despiked, lam=lam, p=p, niter=niter)
    return minmax_normalize_batch(baselined)
