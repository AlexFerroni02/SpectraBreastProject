"""
Dataset wrappers and temperature-balanced batch iterators for Self-Supervised
Pre-Training (`1D-ViT SMAE` and `1D-ResNet-34 Barlow Twins`) and Leakage-Free
Downstream Fine-Tuning (`ibd`, `bacteria_id`, `riverd_oral_hw`, `dataset_renzo`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .augmentations import RamanPhysicalAugmentor, SpectralBlockMaskGenerator
from .splits import compute_temperature_sampling_weights


class SSLPretrainDataset:
    """
    Dataset & Temperature-Balanced Batch Loader for `data/02_ready/pretrain/` archives:
    - `pretrain_vit_fp_1000.npz` (`384,241` spectra, `650.0-1750.0 cm^-1`, `1,000` pts)
    - `pretrain_vit_hw_1000.npz` (`448,358` spectra, `2765.0-3200.0 cm^-1`, `1,000` pts)
    - `pretrain_vit_ch_1000.npz` (`448,358` spectra, `2800.0-3045.0 cm^-1`, `1,000` pts)
    - `pretrain_cnn_all_1000.npz` (`716,686` spectra, multi-regime `1,000` pts)

    Supports two SSL paradigms:
    1. `mode='smae'` (`1D-ViT` Masked Autoencoder):
       Returns `{"spectra": (B, 1000), "patch_mask": (B, 50), "point_mask": (B, 1000), "source_id": (B,)}`.
    2. `mode='barlow_twins'` (`1D-ResNet-34` Self-Supervised Contrastive/Redundancy Reduction):
       Returns `{"view_1": (B, 1000), "view_2": (B, 1000), "clean": (B, 1000), "source_id": (B,)}`.
    """

    def __init__(
        self,
        npz_path: Path,
        mode: str = "smae",
        sampling_alpha: float = 0.5,
        mask_generator: SpectralBlockMaskGenerator | None = None,
        augmentor: RamanPhysicalAugmentor | None = None,
    ) -> None:
        self.npz_path = Path(npz_path)
        self.mode = mode
        self.sampling_alpha = sampling_alpha
        self.mask_generator = mask_generator or SpectralBlockMaskGenerator()
        self.augmentor = augmentor or RamanPhysicalAugmentor()

        archive = np.load(self.npz_path, allow_pickle=True)
        self.spectra: np.ndarray = archive["spectra"]
        self.source_id: np.ndarray = archive["source_id"]
        self.source_names: list[str] = [str(x) for x in archive["source_names"]]
        self.wavenumbers: np.ndarray | None = archive["wavenumbers"] if "wavenumbers" in archive else None

        self.sample_weights, self.weight_info = compute_temperature_sampling_weights(
            self.source_id, alpha=self.sampling_alpha
        )

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def sample_batch(
        self,
        batch_size: int = 256,
        balanced: bool = True,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Draw a single mini-batch of size `batch_size` using temperature-balanced sampling."""
        rng = np.random.default_rng(seed)
        if balanced:
            # Fast hierarchical sampling: first pick source according to P_alpha(c), then uniform within source
            p_cohort = self.weight_info["p_sqrt_alpha05"]
            chosen_cohorts = rng.choice(len(self.source_names), size=batch_size, p=p_cohort)
            indices = np.empty(batch_size, dtype=np.int64)
            for c_id in np.unique(chosen_cohorts):
                c_mask = chosen_cohorts == c_id
                n_draw = int(np.sum(c_mask))
                pool = np.where(self.source_id == c_id)[0]
                indices[c_mask] = rng.choice(pool, size=n_draw, replace=(n_draw > pool.size))
        else:
            indices = rng.choice(len(self), size=batch_size, replace=False)

        x = self.spectra[indices].astype(np.float32, copy=True)
        sid = self.source_id[indices]

        if self.mode == "smae":
            patch_mask, point_mask = self.mask_generator.generate_masks(
                batch_size, mode="block", rng=rng
            )
            return {
                "spectra": x,
                "patch_mask": patch_mask,
                "point_mask": point_mask,
                "source_id": sid,
                "indices": indices,
            }
        elif self.mode == "barlow_twins":
            v1, v2 = self.augmentor.generate_barlow_twins_views(x, rng=rng)
            return {
                "view_1": v1,
                "view_2": v2,
                "clean": x,
                "source_id": sid,
                "indices": indices,
            }
        else:
            raise ValueError(f"Unknown SSL mode: {self.mode!r}")

    def iter_batches(
        self,
        batch_size: int = 256,
        num_batches: int = 100,
        balanced: bool = True,
        seed: int = 42,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield `num_batches` mini-batches deterministically from `seed`."""
        rng = np.random.default_rng(seed)
        for _ in range(num_batches):
            sub_seed = int(rng.integers(0, 2**31 - 1))
            yield self.sample_batch(batch_size=batch_size, balanced=balanced, seed=sub_seed)


class DownstreamFoldLoader:
    """
    Leakage-free train/test fold loader for any of the 4 downstream benchmarks:
    - `ibd` (`FP`: 228 patients, 5-fold stratified patient CV)
    - `bacteria_id` (`FP`: 3,000 finetune with 5-fold val + 3,000 clinical test)
    - `riverd_oral_hw` / `riverd_oral_ch` (`HW`/`CH`: 8,422 spectra, 186 patients, 5-fold StratifiedGroupKFold)
    - `dataset_renzo` (`FR`/`FP`/`CH`: 3,200 spectra, 47 mice, 5-fold StratifiedGroupKFold)
    """

    def __init__(self, ready_dir: Path) -> None:
        self.ready_dir = Path(ready_dir)
        self.splits_dir = self.ready_dir / "downstream" / "splits"

    def get_fold(
        self,
        benchmark: str,
        fold: int = 0,
        regime: str = "default",
    ) -> dict[str, Any]:
        """Return `(X_train, y_train, X_test, y_test, wavenumbers, metadata)` for `fold`."""
        if benchmark == "ibd":
            d = np.load(self.ready_dir / "downstream" / "FP" / "ibd_fp_1000.npz", allow_pickle=True)
            s = np.load(self.splits_dir / "ibd_5fold_splits.npz", allow_pickle=True)
            X = d["spectra_native_1000"] if regime == "native" else d["spectra"]
            wn = d["wavenumbers_native_1000"] if regime == "native" else d["wavenumbers"]
            y = s["labels"]
            te_m = s["fold_id"] == fold
            tr_m = ~te_m
            return {
                "X_train": X[tr_m], "y_train": y[tr_m],
                "X_test": X[te_m], "y_test": y[te_m],
                "wavenumbers": wn,
                "train_groups": s["patient_ids"][tr_m],
                "test_groups": s["patient_ids"][te_m],
            }
        elif benchmark == "bacteria_id":
            d = np.load(self.ready_dir / "downstream" / "FP" / "bacteria_id_fp_1000.npz", allow_pickle=True)
            s = np.load(self.splits_dir / "bacteria_id_splits.npz", allow_pickle=True)
            X_ft = d["X_finetune_native_1000"] if regime == "native" else d["X_finetune"]
            X_te = d["X_test_native_1000"] if regime == "native" else d["X_test"]
            wn = d["wavenumbers_native_1000"] if regime == "native" else d["wavenumbers"]
            val_m = s["finetune_val_fold"] == fold
            tr_m = ~val_m
            return {
                "X_train": X_ft[tr_m], "y_train": d["y_finetune"][tr_m],
                "X_val": X_ft[val_m], "y_val": d["y_finetune"][val_m],
                "X_test": X_te, "y_test": d["y_test"],
                "wavenumbers": wn,
            }
        elif benchmark in ("riverd_oral_hw", "riverd_oral_ch"):
            s = np.load(self.splits_dir / "riverd_oral_5fold_grouped_splits.npz", allow_pickle=True)
            if benchmark == "riverd_oral_ch" or regime == "CH":
                d = np.load(self.ready_dir / "downstream" / "CH" / "riverd_oral_ch_1000.npz", allow_pickle=True)
                X, wn = d["spectra"], d["wavenumbers"]
            else:
                d = np.load(self.ready_dir / "downstream" / "HW" / "riverd_oral_hw_1000.npz", allow_pickle=True)
                X = d["spectra_broad_1000"] if regime == "broad" else d["spectra"]
                wn = d["wavenumbers_broad_1000"] if regime == "broad" else d["wavenumbers"]
            y = s["labels"]
            te_m = s["fold_id"] == fold
            tr_m = ~te_m
            return {
                "X_train": X[tr_m], "y_train": y[tr_m],
                "X_test": X[te_m], "y_test": y[te_m],
                "wavenumbers": wn,
                "train_groups": s["sample_ids"][tr_m],
                "test_groups": s["sample_ids"][te_m],
            }
        elif benchmark == "dataset_renzo":
            d = np.load(self.ready_dir / "downstream" / "FR" / "dataset_renzo_fr_1000.npz", allow_pickle=True)
            s = np.load(self.splits_dir / "dataset_renzo_5fold_grouped_splits.npz", allow_pickle=True)
            if regime == "FP":
                X, wn = d["spectra_fp"], d["wavenumbers_fp"]
            elif regime == "CH":
                X, wn = d["spectra_ch"], d["wavenumbers_ch"]
            else:
                X, wn = d["spectra"], d["wavenumbers"]
            y = s["labels"]
            te_m = s["fold_id"] == fold
            tr_m = ~te_m
            return {
                "X_train": X[tr_m], "y_train": y[tr_m],
                "X_test": X[te_m], "y_test": y[te_m],
                "wavenumbers": wn,
                "class_names": s["class_names"],
                "train_groups": s["sample_id"][tr_m],
                "test_groups": s["sample_id"][te_m],
            }
        else:
            raise ValueError(f"Unknown downstream benchmark: {benchmark!r}")
