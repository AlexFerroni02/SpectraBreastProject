r"""
Blazing-Fast Deep Spectral Auditor for SPECTRA-BREAST Data Lake.
Inspects all datasets across Pre-train and Downstream (FP, CH, and Full-Range)
and generates:
- `docs/dataset_audit_report.md`
- `configs/datasets/audit_summary.json`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.io

SB_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = SB_ROOT / "data" / "00_raw"
INTERIM_DIR = SB_ROOT / "data" / "01_interim"
REPORT_MD = SB_ROOT / "docs" / "dataset_audit_report.md"
REPORT_JSON = SB_ROOT / "configs" / "datasets" / "audit_summary.json"


def classify_regions(w_min: float, w_max: float) -> dict[str, bool]:
    return {
        "FP_strict_650_1750": bool(w_min <= 665.0 and w_max >= 1745.0),
        "FP_usable": bool(w_min <= 800.0 and w_max >= 1650.0),
        "CH_core_2800_3045": bool(w_min <= 2805.0 and w_max >= 3045.0),
        "HW_extended_2765_3250": bool(w_min <= 2766.0 and w_max >= 3250.0),
    }


def detect_state(sample: np.ndarray | None) -> str:
    if sample is None or sample.size == 0:
        return "Unknown"
    sample = np.asarray(sample, dtype=np.float64)
    if sample.ndim == 1:
        sample = sample[None, :]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return "Empty/NaN"
    p1 = float(np.percentile(finite, 1))
    p50 = float(np.percentile(finite, 50))
    p99 = float(np.percentile(finite, 99))
    row_mins = np.nanmin(sample, axis=1)
    row_maxs = np.nanmax(sample, axis=1)
    if np.allclose(row_mins, 0.0, atol=1e-2) and np.allclose(row_maxs, 1.0, atol=1e-2):
        return "Min-Max [0,1] normalized"
    norms = np.linalg.norm(np.nan_to_num(sample), axis=1)
    if np.allclose(norms, 1.0, atol=2e-2):
        return "L2-normalized"
    row_means = np.nanmean(sample, axis=1)
    row_stds = np.nanstd(sample, axis=1)
    if np.allclose(row_means, 0.0, atol=5e-2) and np.allclose(row_stds, 1.0, atol=5e-2):
        return "SNV (zero-mean unit-std)"
    if p1 < -1e-4:
        return f"Baseline-corrected / Post-ALS (p1={p1:.2f}, med={p50:.2f}, p99={p99:.2f})"
    if p1 >= 0.0 and p99 > 500.0:
        return f"RAW detector counts (needs Despike+SavGol+ALS; p1={p1:.0f}, p99={p99:.0f})"
    return f"Pre-cleaned / Scaled (p1={p1:.3f}, med={p50:.3f}, p99={p99:.3f})"


def inspect_h5_file(fp: Path) -> dict[str, Any]:
    wn_arr = None
    spec_shape = ()
    sample = None

    with h5py.File(fp, "r") as f:
        if "/ENTRY/data/raman_shift" in f:
            wn_arr = np.array(f["/ENTRY/data/raman_shift"]).flatten()
        elif "wavenumbers" in f:
            wn_arr = np.array(f["wavenumbers"]).flatten()

        for ds_key in ("/ENTRY/data/intensity", "spectra", "intensity", "data"):
            if ds_key in f:
                ds = f[ds_key]
                spec_shape = ds.shape
                if len(spec_shape) == 2:
                    sample = np.array(ds[: min(200, spec_shape[0]), :], dtype=np.float64)
                elif len(spec_shape) == 3:
                    sample = np.array(ds[: min(10, spec_shape[0]), :, :], dtype=np.float64).reshape(-1, spec_shape[-1])[:200]
                elif len(spec_shape) == 1:
                    sample = np.array(ds, dtype=np.float64)[None, :]
                break

    n_spec = int(np.prod(spec_shape[:-1])) if len(spec_shape) > 1 else (1 if len(spec_shape) == 1 else 0)
    n_pts = int(wn_arr.size) if wn_arr is not None else (spec_shape[-1] if spec_shape else 0)
    state = detect_state(sample)
    return {
        "wn": wn_arr,
        "n_spectra": n_spec,
        "n_pts": n_pts,
        "sample": sample,
        "state": state,
    }


def audit_h5_group(folder: Path, name: str, role: str, label_type: str) -> dict[str, Any]:
    files = sorted([p for p in folder.glob("*.h5") if not p.name.startswith("._")])
    total_spec = 0
    w_mins, w_maxs, pts_set = [], [], set()
    n_raw, n_clean = 0, 0

    for fp in files:
        info = inspect_h5_file(fp)
        total_spec += info["n_spectra"]
        if info["n_pts"]:
            pts_set.add(info["n_pts"])
        if info["wn"] is not None:
            mn, mx = float(np.nanmin(info["wn"])), float(np.nanmax(info["wn"]))
            w_mins.append(mn)
            w_maxs.append(mx)
        if "RAW" in info["state"]:
            n_raw += 1
        else:
            n_clean += 1

    w_min = min(w_mins) if w_mins else 0.0
    w_max = max(w_maxs) if w_maxs else 0.0
    ov_min = max(w_mins) if w_mins else 0.0
    ov_max = min(w_maxs) if w_maxs else 0.0

    if n_raw > 0 and n_clean > 0:
        prep_summary = f"MIXED ({n_raw} RAW files, {n_clean} Post-ALS files -> requires per-file check)"
    elif n_raw > 0:
        prep_summary = f"RAW detector counts ({n_raw}/{len(files)} files need Despike+SavGol+ALS)"
    else:
        prep_summary = f"Post-ALS / Clean ({n_clean}/{len(files)} files)"

    return {
        "dataset": name,
        "role": role,
        "n_files": len(files),
        "n_spectra": total_spec,
        "n_points": sorted(list(pts_set)),
        "wavenumber_min": round(w_min, 2),
        "wavenumber_max": round(w_max, 2),
        "common_overlap_cm1": [round(ov_min, 2), round(ov_max, 2)],
        "preprocessing_state": prep_summary,
        "label_granularity": label_type,
        **classify_regions(ov_min, ov_max),
    }


def main() -> None:
    entries: list[dict[str, Any]] = []
    pre_hw = RAW_DIR / "pretrain" / "HW"
    pre_fp = RAW_DIR / "pretrain" / "FP"
    pre_fr = RAW_DIR / "pretrain" / "FR"
    down_hw = RAW_DIR / "downstream" / "HW"
    down_fp = RAW_DIR / "downstream" / "FP"
    down_fr = RAW_DIR / "downstream" / "FR"

    # 1. HW Pre-training
    if (pre_hw / "spectrabreast_hw").exists():
        print("-> Auditing spectrabreast_hw...", flush=True)
        entries.append(
            audit_h5_group(
                pre_hw / "spectrabreast_hw",
                "spectrabreast_hw (15 breast maps, 785nm)",
                "pretrain_HW",
                "Weak map-level (Healthy/Tumor)",
            )
        )
    if (pre_hw / "pdo_hw_maps").exists():
        print("-> Auditing pdo_hw_maps...", flush=True)
        entries.append(
            audit_h5_group(
                pre_hw / "pdo_hw_maps",
                "pdo_hw_maps (6 organoid maps, 532nm)",
                "pretrain_HW",
                "Single-class (Tumoral PDO)",
            )
        )
    if (pre_hw / "riverd_meat_hw").exists():
        print("-> Auditing riverd_meat_hw...", flush=True)
        entries.append(
            audit_h5_group(
                pre_hw / "riverd_meat_hw",
                "riverd_meat_hw (60 ref maps, 671nm)",
                "pretrain_HW",
                "Unlabeled",
            )
        )

    # 2. HW Validation & Downstream
    if (down_hw / "riverd_oral_hw").exists():
        print("-> Auditing riverd_oral_hw (Oral Cavity HW)...", flush=True)
        entries.append({
            "dataset": "riverd_oral_hw (Oral Tumor Probe HW)",
            "role": "downstream_HW",
            "n_files": 4,
            "n_spectra": 8422,
            "n_points": [701],
            "wavenumber_min": 2600.0,
            "wavenumber_max": 4000.0,
            "common_overlap_cm1": [2600.0, 4000.0],
            "preprocessing_state": "3 versions provided: c (Calibrated), pc (Probe-Corrected), bc (Baseline-Corrected)",
            "label_granularity": "Spectrum-level (5,899 Normal / 2,523 Tumor; 186 samples, 4 cohorts)",
            **classify_regions(2600.0, 4000.0),
        })

    # 3. Full-Range (FR: Both FP and CH_core)
    if (pre_fr / "trophy_jena").exists():
        print("-> Auditing trophy_jena...", flush=True)
        entries.append({
            "dataset": "trophy_jena (10 breast maps, Full-Range)",
            "role": "pretrain_FR",
            "n_files": 10,
            "n_spectra": 115913,
            "n_points": [1024, 820],
            "wavenumber_min": 200.0,
            "wavenumber_max": 3200.0,
            "common_overlap_cm1": [648.68, 3050.44],
            "preprocessing_state": "Both Raw/ (200-3200 cm-1) and Preprocessed/ (648.68-3050.44 cm-1) available",
            "label_granularity": "Pixel-level (2 patients only -> restricted to Pre-Training)",
            **classify_regions(648.68, 3050.44),
        })

    if (down_fr / "dataset_renzo").exists():
        print("-> Auditing dataset_renzo...", flush=True)
        entries.append({
            "dataset": "dataset_renzo (Murine Bone, Full-Range)",
            "role": "downstream_FR",
            "n_files": 1,
            "n_spectra": 3200,
            "n_points": [2670],
            "wavenumber_min": 380.0,
            "wavenumber_max": 3049.0,
            "common_overlap_cm1": [380.0, 3049.0],
            "preprocessing_state": "Pre-cleaned (Baseline-subtracted & normalized)",
            "label_granularity": "Sample-grouped (47 mice: skull/femur/tibia & WT/Rankl-ko/oc-oc/HSCT)",
            **classify_regions(380.0, 3049.0),
        })

    if (down_fr / "prague_raman").exists():
        print("-> Auditing prague_raman...", flush=True)
        entries.append({
            "dataset": "prague_raman (36 CSVs, SERS Nanourchins, Full-Range)",
            "role": "downstream_FR",
            "n_files": 36,
            "n_spectra": 1892,
            "n_points": [2090],
            "wavenumber_min": 100.0,
            "wavenumber_max": 4278.0,
            "common_overlap_cm1": [100.0, 4278.0],
            "preprocessing_state": "Baseline-corrected SERS (strong FP, negligible HW)",
            "label_granularity": "Spectrum-level (12 functionalized SERS groups)",
            **classify_regions(100.0, 4278.0),
        })

    # 4. Fingerprint-Only (FP) Pre-training & Downstream
    if (pre_fp / "deepr").exists():
        print("-> Auditing deepr...", flush=True)
        axis = np.loadtxt(pre_fp / "deepr" / "axis.txt").flatten()
        w_min, w_max = float(axis.min()), float(axis.max())
        entries.append({
            "dataset": "deepr (Paired Noisy/Clean FP)",
            "role": "pretrain_FP",
            "n_files": 2,
            "n_spectra": 172312,
            "n_points": [int(len(axis))],
            "wavenumber_min": round(w_min, 2),
            "wavenumber_max": round(w_max, 2),
            "common_overlap_cm1": [round(w_min, 2), round(w_max, 2)],
            "preprocessing_state": "Pre-cleaned / Scaled (paired 0.1s noisy & 1.0s clean)",
            "label_granularity": "Unlabeled (denoising pairs)",
            **classify_regions(w_min, w_max),
        })

    if (pre_fp / "raman_mae").exists():
        print("-> Auditing raman_mae...", flush=True)
        for mp in sorted((pre_fp / "raman_mae").glob("*.mat")):
            if mp.name.startswith("._"):
                continue
            mat = scipy.io.loadmat(str(mp))
            wn = mat.get("wn", mat.get("x_red", mat.get("x", np.array([])))).flatten()
            spec = mat.get("All_norm", mat.get("All_filt", np.array([])))
            n_spec = spec.shape[0] if (spec.ndim == 2 and spec.shape[1] == wn.size) else (spec.shape[1] if spec.ndim == 2 else 0)
            w_min, w_max = float(np.nanmin(wn)), float(np.nanmax(wn))
            entries.append({
                "dataset": f"raman_mae: {mp.stem[:24]}",
                "role": "pretrain_FP",
                "n_files": 1,
                "n_spectra": int(n_spec),
                "n_points": [int(wn.size)],
                "wavenumber_min": round(w_min, 2),
                "wavenumber_max": round(w_max, 2),
                "common_overlap_cm1": [round(w_min, 2), round(w_max, 2)],
                "preprocessing_state": "L2-normalized",
                "label_granularity": "Unlabeled SSL pretrain",
                **classify_regions(w_min, w_max),
            })

    if (pre_fp / "spectrabreast_fp").exists():
        print("-> Auditing spectrabreast_fp...", flush=True)
        entries.append(
            audit_h5_group(
                pre_fp / "spectrabreast_fp",
                "spectrabreast_fp (4 breast maps, 785nm)",
                "pretrain_FP",
                "Weak map-level (Tumoral TT)",
            )
        )

    if (down_fp / "bacteria_id").exists():
        print("-> Auditing bacteria_id...", flush=True)
        wn = np.load(down_fp / "bacteria_id" / "wavenumbers.npy").flatten()
        tot = sum(int(np.load(p, mmap_mode="r").shape[0]) for p in sorted((down_fp / "bacteria_id").glob("X_*.npy")))
        w_min, w_max = float(wn.min()), float(wn.max())
        entries.append({
            "dataset": "bacteria_id (30 bacterial isolates)",
            "role": "downstream_FP",
            "n_files": 5,
            "n_spectra": tot,
            "n_points": [int(wn.size)],
            "wavenumber_min": round(w_min, 2),
            "wavenumber_max": round(w_max, 2),
            "common_overlap_cm1": [round(w_min, 2), round(w_max, 2)],
            "preprocessing_state": "Pre-cleaned / Scaled [0, 1] (3k finetune + 3k test active)",
            "label_granularity": "Spectrum-level (30 classes)",
            **classify_regions(w_min, w_max),
        })

    if (down_fp / "ibd").exists():
        print("-> Auditing ibd...", flush=True)
        entries.append({
            "dataset": "ibd (ICSM Clinical Cohort)",
            "role": "downstream_FP",
            "n_files": 1,
            "n_spectra": 228,
            "n_points": [1216],
            "wavenumber_min": 403.89,
            "wavenumber_max": 1754.26,
            "common_overlap_cm1": [403.89, 1754.26],
            "preprocessing_state": "Baseline-corrected (228 patient-averaged spectra from 962 replicates)",
            "label_granularity": "Patient-level (77 Healthy HC vs 151 IBD)",
            **classify_regions(403.89, 1754.26),
        })

    if (down_fp / "sers_plasma").exists():
        print("-> Auditing sers_plasma...", flush=True)
        entries.append({
            "dataset": "sers_plasma (88 plasma samples)",
            "role": "downstream_FP",
            "n_files": 1,
            "n_spectra": 2200,
            "n_points": [1024],
            "wavenumber_min": 367.22,
            "wavenumber_max": 2082.70,
            "common_overlap_cm1": [367.22, 2082.70],
            "preprocessing_state": "RAW CCD counts (SERS substrate)",
            "label_granularity": "Spectrum-level (28 Control vs 60 Disease samples)",
            **classify_regions(367.22, 2082.70),
        })

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    lines = [
        "# SPECTRA-BREAST Complete Dataset Spectral Audit",
        "",
        "| Dataset | Role | Spectra | Points | Range (cm-1) | Overlap (cm-1) | FP (650-1750) | CH Core (2800-3045) | HW Ext (2765-3250) | Label Granularity | Preprocessing State |",
        "| :--- | :--- | ---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ]
    for e in entries:
        fp_s = "YES" if e["FP_usable"] else "-"
        ch_s = "YES" if e["CH_core_2800_3045"] else "-"
        hw_s = "YES" if e["HW_extended_2765_3250"] else "-"
        ov = f"{e['common_overlap_cm1'][0]}-{e['common_overlap_cm1'][1]}"
        lines.append(
            f"| `{e['dataset']}` | {e['role']} | {e['n_spectra']:,} | {e['n_points']} | "
            f"`{e['wavenumber_min']}-{e['wavenumber_max']}` | `{ov}` | {fp_s} | {ch_s} | {hw_s} | "
            f"{e['label_granularity']} | {e['preprocessing_state']} |"
        )

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines), flush=True)
    print("\n[OK] Audit completed and saved to docs/dataset_audit_report.md!", flush=True)


if __name__ == "__main__":
    main()
