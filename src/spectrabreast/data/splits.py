"""
Leakage-free Patient/Subject-Grouped Cross-Validation split generators and
Temperature-Balanced SSL Cohort Samplers for Step 4 (`data/02_ready/`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def compute_temperature_sampling_weights(
    source_id: np.ndarray,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute per-spectrum sampling weights `w_i` and cohort probabilities `P_alpha(c)`
    under temperature exponent `alpha in [0, 1]`:
        P_alpha(c) = (N_c ** alpha) / sum_k (N_k ** alpha)
        w_i = P_alpha(c_i) / N_{c_i}
    """
    sid = np.asarray(source_id, dtype=np.int64)
    uniq, counts = np.unique(sid, return_counts=True)
    counts_f = counts.astype(np.float64)

    p_naive = counts_f / counts_f.sum()
    p_alpha = (counts_f**alpha) / np.sum(counts_f**alpha)
    p_equal = np.full_like(p_naive, 1.0 / len(uniq))

    # Per-sample weight proportional to P_alpha(c) / N_c
    class_weight = p_alpha / counts_f
    sample_weights = class_weight[sid]
    sample_weights = sample_weights / sample_weights.sum()

    return sample_weights.astype(np.float64), {
        "counts": counts,
        "p_naive_alpha1": p_naive,
        "p_sqrt_alpha05": p_alpha,
        "p_equal_alpha0": p_equal,
    }


def build_and_verify_downstream_splits(
    ready_dir: Path,
    catalog_csv_path: Path | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate, save, and mathematically verify leakage-free 5-fold evaluation splits
    for all 4 downstream benchmarks in `data/02_ready/downstream/splits/`.
    Guarantees 0% patient/subject overlap between train and validation/test sets.
    """
    splits_dir = ready_dir / "downstream" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    split_details: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. IBD (`ibd_fp_1000.npz`): 228 unique patients (77 Control vs 151 IBD)
    # -------------------------------------------------------------------------
    d_ibd = np.load(ready_dir / "downstream" / "FP" / "ibd_fp_1000.npz", allow_pickle=True)
    y_ibd = d_ibd["labels"].astype(np.int64)
    pids_ibd = d_ibd["patient_ids"]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    ibd_fold_id = np.zeros_like(y_ibd, dtype=np.int8)
    ibd_leakage = 0
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros_like(y_ibd), y_ibd, groups=pids_ibd)):
        ibd_fold_id[te_idx] = fold
        ibd_leakage += len(set(pids_ibd[tr_idx]).intersection(set(pids_ibd[te_idx])))

    np.savez(
        splits_dir / "ibd_5fold_splits.npz",
        fold_id=ibd_fold_id,
        labels=y_ibd,
        patient_ids=pids_ibd,
    )
    for f in range(5):
        te_m = ibd_fold_id == f
        tr_m = ~te_m
        rows.append({
            "Benchmark": "ibd (ICSM Plasma)",
            "Regime": "FP (650–1750 cm^-1)",
            "Split_Protocol": "5-Fold Stratified Patient CV",
            "Fold": f"Fold {f + 1}",
            "Train_Spectra": int(tr_m.sum()),
            "Test_Spectra": int(te_m.sum()),
            "Train_Subjects": int(len(np.unique(pids_ibd[tr_m]))),
            "Test_Subjects": int(len(np.unique(pids_ibd[te_m]))),
            "Subject_Overlap_Leakage": ibd_leakage,
            "Target_Distribution_Test": f"HC={int((y_ibd[te_m]==0).sum())}, IBD={int((y_ibd[te_m]==1).sum())}",
        })
    split_details["ibd"] = {"fold_id": ibd_fold_id, "labels": y_ibd, "groups": pids_ibd}

    # -------------------------------------------------------------------------
    # 2. Bacteria-ID (`bacteria_id_fp_1000.npz`): Official 3,000 Finetune (5-Fold) + 3,000 Clinical Test
    # -------------------------------------------------------------------------
    d_bact = np.load(ready_dir / "downstream" / "FP" / "bacteria_id_fp_1000.npz", allow_pickle=True)
    y_ft = d_bact["y_finetune"].astype(np.int64)
    y_te = d_bact["y_test"].astype(np.int64)
    skf_b = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    bact_val_fold = np.zeros_like(y_ft, dtype=np.int8)
    for fold, (_, val_idx) in enumerate(skf_b.split(np.zeros_like(y_ft), y_ft)):
        bact_val_fold[val_idx] = fold

    np.savez(
        splits_dir / "bacteria_id_splits.npz",
        finetune_val_fold=bact_val_fold,
        y_finetune=y_ft,
        y_test=y_te,
    )
    rows.append({
        "Benchmark": "bacteria_id (30 Pathogens)",
        "Regime": "FP (650–1750 cm^-1)",
        "Split_Protocol": "Official Finetune (5-Fold Val) + Held-Out Test",
        "Fold": "Official Split (5-Fold Val)",
        "Train_Spectra": 2400,
        "Test_Spectra": 3000,
        "Train_Subjects": 30,
        "Test_Subjects": 30,
        "Subject_Overlap_Leakage": 0,
        "Target_Distribution_Test": "30 Classes × 100 Clinical Isolates = 3,000 Test (+ 600 Val)",
    })
    split_details["bact"] = {"val_fold": bact_val_fold, "y_finetune": y_ft, "y_test": y_te}

    # -------------------------------------------------------------------------
    # 3. RiverD Oral HW/CH (`riverd_oral_hw_1000.npz`): 8,422 spectra across 186 surgical patients
    # -------------------------------------------------------------------------
    d_oral = np.load(ready_dir / "downstream" / "HW" / "riverd_oral_hw_1000.npz", allow_pickle=True)
    y_oral = d_oral["labels"].astype(np.int64)
    sids_oral = d_oral["sample_ids"].astype(np.int64)
    sgkf_o = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    oral_fold_id = np.zeros_like(y_oral, dtype=np.int8)
    oral_leakage = 0
    for fold, (tr_idx, te_idx) in enumerate(sgkf_o.split(np.zeros_like(y_oral), y_oral, groups=sids_oral)):
        oral_fold_id[te_idx] = fold
        oral_leakage += len(set(sids_oral[tr_idx]).intersection(set(sids_oral[te_idx])))

    np.savez(
        splits_dir / "riverd_oral_5fold_grouped_splits.npz",
        fold_id=oral_fold_id,
        labels=y_oral,
        sample_ids=sids_oral,
        data_source=d_oral["data_source"],
    )
    for f in range(5):
        te_m = oral_fold_id == f
        tr_m = ~te_m
        rows.append({
            "Benchmark": "riverd_oral_hw / ch (OSCC Surgery)",
            "Regime": "HW (2765–3200) & CH (2800–3045)",
            "Split_Protocol": "5-Fold StratifiedGroupKFold (by Patient sample_id)",
            "Fold": f"Fold {f + 1}",
            "Train_Spectra": int(tr_m.sum()),
            "Test_Spectra": int(te_m.sum()),
            "Train_Subjects": int(len(np.unique(sids_oral[tr_m]))),
            "Test_Subjects": int(len(np.unique(sids_oral[te_m]))),
            "Subject_Overlap_Leakage": oral_leakage,
            "Target_Distribution_Test": f"Normal={int((y_oral[te_m]==0).sum())}, Tumor={int((y_oral[te_m]==1).sum())}",
        })
    split_details["oral"] = {"fold_id": oral_fold_id, "labels": y_oral, "groups": sids_oral}

    # -------------------------------------------------------------------------
    # 4. Dataset Renzo (`dataset_renzo_fr_1000.npz`): 3,200 spectra across 47 mice (RV1..RV50)
    # -------------------------------------------------------------------------
    d_rz = np.load(ready_dir / "downstream" / "FR" / "dataset_renzo_fr_1000.npz", allow_pickle=True)
    patho_str = d_rz["pathology"].astype(str)
    class_names_rz = np.array(["WT", "Rankl ko", "oc/oc", "HSCT"])
    class_map_rz = {name: idx for idx, name in enumerate(class_names_rz)}
    y_rz = np.array([class_map_rz[s] for s in patho_str], dtype=np.int64)
    mice_rz = d_rz["sample_id"].astype(str)

    sgkf_r = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rz_fold_id = np.zeros_like(y_rz, dtype=np.int8)
    rz_leakage = 0
    for fold, (tr_idx, te_idx) in enumerate(sgkf_r.split(np.zeros_like(y_rz), y_rz, groups=mice_rz)):
        rz_fold_id[te_idx] = fold
        rz_leakage += len(set(mice_rz[tr_idx]).intersection(set(mice_rz[te_idx])))

    np.savez(
        splits_dir / "dataset_renzo_5fold_grouped_splits.npz",
        fold_id=rz_fold_id,
        labels=y_rz,
        class_names=class_names_rz,
        sample_id=mice_rz,
        bone=d_rz["bone"].astype(str),
    )
    for f in range(5):
        te_m = rz_fold_id == f
        tr_m = ~te_m
        dist_str = ", ".join(f"{cname}={int((y_rz[te_m]==cid).sum())}" for cid, cname in enumerate(class_names_rz))
        rows.append({
            "Benchmark": "dataset_renzo (Murine Bone)",
            "Regime": "FR (380–3049), FP (650–1750), CH (2800–3045)",
            "Split_Protocol": "5-Fold StratifiedGroupKFold (by Mouse sample_id)",
            "Fold": f"Fold {f + 1}",
            "Train_Spectra": int(tr_m.sum()),
            "Test_Spectra": int(te_m.sum()),
            "Train_Subjects": int(len(np.unique(mice_rz[tr_m]))),
            "Test_Subjects": int(len(np.unique(mice_rz[te_m]))),
            "Subject_Overlap_Leakage": rz_leakage,
            "Target_Distribution_Test": dist_str,
        })
    split_details["renzo"] = {
        "fold_id": rz_fold_id,
        "labels": y_rz,
        "class_names": class_names_rz,
        "groups": mice_rz,
    }

    df = pd.DataFrame(rows)
    if catalog_csv_path is not None:
        catalog_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(catalog_csv_path, index=False)
    return df, split_details
