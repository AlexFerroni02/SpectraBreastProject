"""
High-performance parser, reader, and visualizer for High-Wavenumber (HW / CH) Raman maps.
Supports FAIRaman standard HDF5 format used across SPECTRA-BREAST pre-training datasets:
- `spectrabreast_hw` (15 maps, 785nm, breast tissue)
- `pdo_hw_maps` (6 maps, 532nm, patient-derived organoids)
- `riverd_meat_hw` (60 maps, 671nm, reference tissue)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CH_DIR = Path(__file__).resolve().parents[3] / "data" / "00_raw" / "pretrain" / "HW"
DEFAULT_HW_DIR = DEFAULT_CH_DIR


def _clean_str(val: Any) -> str:
    """Normalize HDF5 dataset/attribute string value."""
    if val is None:
        return ""
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="ignore")
    val_str = str(val).strip()
    if val_str.lower() in ("nan", "none", "null", ""):
        return ""
    return val_str


def _read_h5_scalar(h5: h5py.File, path: str, default: Any = "") -> Any:
    """Read scalar dataset or group attribute safely."""
    if path in h5:
        obj = h5[path]
        if isinstance(obj, h5py.Dataset):
            try:
                v = obj[()]
                if isinstance(v, bytes):
                    return v.decode("utf-8", errors="ignore").strip()
                if isinstance(v, np.ndarray) and v.size == 1:
                    v = v.item()
                return v
            except Exception:
                return default
    return default


@dataclass
class CHMapMetadata:
    """Structured FAIRaman metadata extracted from an HDF5 Raman map."""

    dataset: str
    filename: str
    filepath: str
    file_size_mb: float

    # Instrument & Acquisition
    instrument: str
    laser_nm: float | None
    laser_power_mw: str
    lens: str
    exposure_time_s: float | None
    accumulation_count: int | None
    substrate: str
    start_time: str
    title: str

    # Sample & Clinical
    sample_id: str
    donor_id: str
    diagnosis_code: str
    sample_type: str
    sample_notes: str

    # Spatial & Grid
    ny: int
    nx: int
    n_spectra: int
    n_wavenumbers: int
    coordinate_units: str
    coordinate_mode: str
    x_min_um: float | None
    x_max_um: float | None
    y_min_um: float | None
    y_max_um: float | None
    step_x_um: float | None
    step_y_um: float | None

    # Spectral
    wavenumber_min: float
    wavenumber_max: float
    wavenumber_step_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CHMap:
    """In-memory or lazy representation of a 2D/3D Raman hyperspectral map."""

    def __init__(
        self,
        metadata: CHMapMetadata,
        wavenumbers: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        intensity: np.ndarray | None = None,
        h5_filepath: str | Path | None = None,
    ):
        self.metadata = metadata
        self.wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
        self.x_coords = np.asarray(x_coords, dtype=np.float64)
        self.y_coords = np.asarray(y_coords, dtype=np.float64)
        self._intensity = intensity
        self._h5_filepath = Path(h5_filepath) if h5_filepath else None

    @property
    def intensity(self) -> np.ndarray:
        """Hyperspectral intensity array with shape (ny, nx, n_wavenumbers)."""
        if self._intensity is None:
            if self._h5_filepath and self._h5_filepath.exists():
                with h5py.File(self._h5_filepath, "r") as h5:
                    ds = h5["ENTRY/data/intensity"]
                    self._intensity = _reshape_intensity(
                        ds[:], self.metadata.ny, self.metadata.nx, self.metadata.n_wavenumbers
                    )
            else:
                raise ValueError("Intensity array not loaded and H5 file path unavailable.")
        return self._intensity

    def mean_spectrum(self) -> np.ndarray:
        """Compute spatial mean spectrum across all pixels."""
        flat = self.intensity.reshape(-1, self.metadata.n_wavenumbers)
        return np.nanmean(flat, axis=0)

    def std_spectrum(self) -> np.ndarray:
        """Compute standard deviation spectrum across all pixels."""
        flat = self.intensity.reshape(-1, self.metadata.n_wavenumbers)
        return np.nanstd(flat, axis=0)

    def pixel_spectrum(self, y_idx: int, x_idx: int) -> np.ndarray:
        """Retrieve spectrum at specific grid coordinate (y, x)."""
        y_c = max(0, min(y_idx, self.metadata.ny - 1))
        x_c = max(0, min(x_idx, self.metadata.nx - 1))
        return self.intensity[y_c, x_c, :]

    def integrated_map(self, wn_min: float | None = None, wn_max: float | None = None) -> np.ndarray:
        """Compute 2D spatial heatmap integrated across a given wavenumber window (default: 2800-3050 cm-1)."""
        if wn_min is None:
            wn_min = 2800.0
        if wn_max is None:
            wn_max = 3050.0

        mask = (self.wavenumbers >= wn_min) & (self.wavenumbers <= wn_max)
        if not np.any(mask):
            mask = np.ones_like(self.wavenumbers, dtype=bool)

        return np.nansum(self.intensity[:, :, mask], axis=-1)

    def peak_map(self, peak_wn: float = 2935.0, window_cm: float = 15.0) -> np.ndarray:
        """Compute 2D spatial heatmap of peak intensity near specified wavenumber (e.g. 2935 cm-1)."""
        mask = (self.wavenumbers >= peak_wn - window_cm) & (self.wavenumbers <= peak_wn + window_cm)
        if not np.any(mask):
            idx = np.argmin(np.abs(self.wavenumbers - peak_wn))
            return self.intensity[:, :, idx]
        return np.nanmax(self.intensity[:, :, mask], axis=-1)

    def summary_table(self) -> pd.DataFrame:
        """Return a formatted single-row DataFrame summary."""
        return pd.DataFrame([self.metadata.to_dict()])

    def __repr__(self) -> str:
        m = self.metadata
        return (
            f"CHMap(dataset='{m.dataset}', file='{m.filename}', "
            f"grid=({m.ny}x{m.nx}), spectra={m.n_spectra}, "
            f"wn=[{m.wavenumber_min:.1f}, {m.wavenumber_max:.1f}] cm-1, laser={m.laser_nm}nm)"
        )


def _reshape_intensity(arr: np.ndarray, ny: int, nx: int, n_wn: int) -> np.ndarray:
    """Ensure intensity array conforms to (ny, nx, n_wn)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape == (ny, nx, n_wn):
            return arr
        if arr.shape == (nx, ny, n_wn):
            return np.transpose(arr, (1, 0, 2))
    elif arr.ndim == 2:
        if arr.shape[0] == ny * nx and arr.shape[1] == n_wn:
            return arr.reshape(ny, nx, n_wn)
        if arr.shape[1] == ny * nx and arr.shape[0] == n_wn:
            return arr.T.reshape(ny, nx, n_wn)
    elif arr.ndim == 1 and arr.size == ny * nx * n_wn:
        return arr.reshape(ny, nx, n_wn)

    # Fallback reshape
    if arr.shape[-1] == n_wn:
        n_sp = int(np.prod(arr.shape[:-1]))
        if ny * nx == n_sp:
            return arr.reshape(ny, nx, n_wn)
        return arr.reshape(1, n_sp, n_wn)

    return arr.reshape(ny, nx, -1)


def read_ch_map(file_path: str | Path, load_intensity: bool = True) -> CHMap:
    """
    Read an HDF5 Raman map from the FAIRaman dataset.

    Parameters
    ----------
    file_path : str | Path
        Path to the .h5 file.
    load_intensity : bool, default=True
        Whether to load full intensity data into memory immediately.
        If False, intensity is loaded lazily on first access.

    Returns
    -------
    CHMap
        High-level CH map object containing metadata, coordinates, and hyperspectral data.
    """
    fp = Path(file_path).resolve()
    if not fp.exists():
        raise FileNotFoundError(f"HDF5 file does not exist: {fp}")

    dataset_name = fp.parent.name
    file_size_mb = round(fp.stat().st_size / (1024 * 1024), 2)

    with h5py.File(fp, "r") as h5:
        data_grp = h5.get("ENTRY/data")
        if data_grp is None:
            raise ValueError(f"File {fp.name} is missing 'ENTRY/data' group.")

        d_attrs = dict(data_grp.attrs)

        # Wavenumbers
        if "raman_shift" in data_grp:
            wn = np.array(data_grp["raman_shift"]).flatten()
        else:
            raise ValueError(f"File {fp.name} is missing 'ENTRY/data/raman_shift'.")

        n_wn = int(wn.size)
        wn_min = float(np.nanmin(wn))
        wn_max = float(np.nanmax(wn))
        wn_step_mean = float(np.nanmean(np.diff(wn))) if n_wn > 1 else 0.0

        # Grid dimensions
        ny = int(d_attrs.get("ny", 1))
        nx = int(d_attrs.get("nx", 1))
        n_spectra = int(d_attrs.get("spectral_count", ny * nx))

        # Check intensity shape
        if "intensity" in data_grp:
            raw_sh = data_grp["intensity"].shape
            if len(raw_sh) >= 2 and ny * nx != np.prod(raw_sh[:-1]):
                if len(raw_sh) == 3:
                    ny, nx = int(raw_sh[0]), int(raw_sh[1])
                elif len(raw_sh) == 2:
                    ny, nx = 1, int(raw_sh[0])
                n_spectra = ny * nx
        else:
            raw_sh = (ny, nx, n_wn)

        # Spatial coordinates
        coord_units = _clean_str(d_attrs.get("coordinate_units", "micrometers"))
        coord_mode = _clean_str(d_attrs.get("coordinate_mode", "regular_grid"))

        if "x" in data_grp:
            x_arr = np.array(data_grp["x"]).flatten()
        else:
            x_arr = np.arange(nx, dtype=np.float64)

        if "y" in data_grp:
            y_arr = np.array(data_grp["y"]).flatten()
        else:
            y_arr = np.arange(ny, dtype=np.float64)

        x_min = float(np.nanmin(x_arr)) if x_arr.size > 0 else None
        x_max = float(np.nanmax(x_arr)) if x_arr.size > 0 else None
        y_min = float(np.nanmin(y_arr)) if y_arr.size > 0 else None
        y_max = float(np.nanmax(y_arr)) if y_arr.size > 0 else None

        step_x = float(np.abs(np.diff(x_arr)).mean()) if x_arr.size > 1 else None
        step_y = float(np.abs(np.diff(y_arr)).mean()) if y_arr.size > 1 else None

        # Metadata fields
        laser_raw = _read_h5_scalar(h5, "ENTRY/instrument/laser/wavelength")
        try:
            laser_nm = float(laser_raw) if laser_raw not in ("", None) else None
        except Exception:
            laser_nm = None

        exp_raw = _read_h5_scalar(h5, "ENTRY/measurement/exposure_time")
        try:
            exp_time = float(exp_raw) if exp_raw not in ("", None) else None
        except Exception:
            exp_time = None

        acc_raw = _read_h5_scalar(h5, "ENTRY/measurement/accumulation_count")
        try:
            acc_count = int(acc_raw) if acc_raw not in ("", None) else None
        except Exception:
            acc_count = None

        meta = CHMapMetadata(
            dataset=dataset_name,
            filename=fp.name,
            filepath=str(fp),
            file_size_mb=file_size_mb,
            instrument=_clean_str(_read_h5_scalar(h5, "ENTRY/instrument/name")),
            laser_nm=laser_nm,
            laser_power_mw=_clean_str(_read_h5_scalar(h5, "ENTRY/instrument/laser/power")),
            lens=_clean_str(_read_h5_scalar(h5, "ENTRY/instrument/optical_system/lens")),
            exposure_time_s=exp_time,
            accumulation_count=acc_count,
            substrate=_clean_str(_read_h5_scalar(h5, "ENTRY/measurement/substrate")),
            start_time=_clean_str(_read_h5_scalar(h5, "ENTRY/start_time")),
            title=_clean_str(_read_h5_scalar(h5, "ENTRY/title", fp.name)),
            sample_id=_clean_str(_read_h5_scalar(h5, "SAMPLE/sample_id")),
            donor_id=_clean_str(_read_h5_scalar(h5, "SAMPLE/SAMPLE_DONOR/donor_id")),
            diagnosis_code=_clean_str(_read_h5_scalar(h5, "SAMPLE/SAMPLE_DONOR/diagnosis_code")),
            sample_type=_clean_str(_read_h5_scalar(h5, "SAMPLE/SAMPLE_INFO/sample_type")),
            sample_notes=_clean_str(_read_h5_scalar(h5, "SAMPLE/SAMPLE_INFO/sample_notes")),
            ny=ny,
            nx=nx,
            n_spectra=n_spectra,
            n_wavenumbers=n_wn,
            coordinate_units=coord_units,
            coordinate_mode=coord_mode,
            x_min_um=x_min,
            x_max_um=x_max,
            y_min_um=y_min,
            y_max_um=y_max,
            step_x_um=step_x,
            step_y_um=step_y,
            wavenumber_min=round(wn_min, 2),
            wavenumber_max=round(wn_max, 2),
            wavenumber_step_mean=round(wn_step_mean, 3),
        )

        intensity_arr = None
        if load_intensity and "intensity" in data_grp:
            raw_intens = data_grp["intensity"][:]
            intensity_arr = _reshape_intensity(raw_intens, ny, nx, n_wn)

    return CHMap(
        metadata=meta,
        wavenumbers=wn,
        x_coords=x_arr,
        y_coords=y_arr,
        intensity=intensity_arr,
        h5_filepath=fp,
    )


def scan_ch_maps(base_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Rapidly scan all CH maps across pretrain datasets without loading full matrices into RAM.

    Returns a pandas DataFrame indexed by file with all FAIRaman metadata.
    """
    root_dir = Path(base_dir).resolve() if base_dir else DEFAULT_CH_DIR
    if not root_dir.exists():
        raise FileNotFoundError(f"CH pretrain folder not found at: {root_dir}")

    h5_files = sorted(root_dir.rglob("*.h5"))
    records = []

    for fp in h5_files:
        try:
            ch_map = read_ch_map(fp, load_intensity=False)
            records.append(ch_map.metadata.to_dict())
        except Exception as e:
            records.append({
                "dataset": fp.parent.name,
                "filename": fp.name,
                "filepath": str(fp),
                "error": str(e),
            })

    df = pd.DataFrame(records)
    return df


def plot_ch_map(
    ch_map: CHMap,
    wn_band: Tuple[float, float] = (2800.0, 3050.0),
    peak_wn: float = 2935.0,
    cmap: str = "magma",
    figsize: Tuple[int, int] = (15, 6),
) -> plt.Figure:
    """
    Generate an elegant multi-panel diagnostic plot for a CH Raman map:
    - Panel 1: 2D Spatial Hyperspectral Map (integrated intensity in CH region).
    - Panel 2: Raman Spectra (spatial mean ± 1 std dev band + key peak markers).
    """
    fig, (ax_map, ax_spec) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.25]})
    m = ch_map.metadata

    # 1. Spatial 2D Map
    img = ch_map.integrated_map(wn_band[0], wn_band[1])

    extent = None
    if m.x_min_um is not None and m.x_max_um is not None and m.y_min_um is not None and m.y_max_um is not None:
        if m.x_min_um != m.x_max_um and m.y_min_um != m.y_max_um:
            extent = [m.x_min_um, m.x_max_um, m.y_max_um, m.y_min_um]

    im = ax_map.imshow(img, cmap=cmap, aspect="auto", extent=extent)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Integrated Intensity (Counts)", fontsize=10)

    units_lbl = f"({m.coordinate_units})" if m.coordinate_units else "(pixels)"
    ax_map.set_xlabel(f"X {units_lbl}", fontsize=11)
    ax_map.set_ylabel(f"Y {units_lbl}", fontsize=11)
    ax_map.set_title(
        f"Spatial Map: {m.filename}\n"
        f"Grid: {m.ny} x {m.nx} ({m.n_spectra:,} spectra) | Band: {wn_band[0]:.0f}-{wn_band[1]:.0f} cm$^{{-1}}$",
        fontsize=11,
        fontweight="bold",
    )

    # 2. Spectral profile
    wn = ch_map.wavenumbers
    mean_s = ch_map.mean_spectrum()
    std_s = ch_map.std_spectrum()

    ax_spec.plot(wn, mean_s, color="#0066cc", lw=2, label="Mean Spectrum")
    ax_spec.fill_between(wn, mean_s - std_s, mean_s + std_s, color="#66b2ff", alpha=0.3, label="±1 Std Dev")

    # High-intensity pixel for contrast
    flat_img = img.flatten()
    max_idx = int(np.nanargmax(flat_img))
    y_max, x_max = divmod(max_idx, m.nx)
    if y_max < m.ny and x_max < m.nx:
        max_spec = ch_map.pixel_spectrum(y_max, x_max)
        ax_spec.plot(wn, max_spec, color="#e65100", lw=1.2, alpha=0.8, linestyle="--", label=f"Max Pixel ({y_max},{x_max})")

    # Known CH Raman vibration bands
    ch_markers = [
        (2850, "CH$_2$ sym (Lipids)"),
        (2880, "CH$_2$ asym"),
        (2935, "CH$_3$ sym (Proteins/Lipids)"),
        (2960, "CH$_3$ asym"),
    ]
    for p_wn, label in ch_markers:
        if wn.min() <= p_wn <= wn.max():
            ax_spec.axvline(p_wn, color="gray", linestyle=":", alpha=0.6)
            ax_spec.text(p_wn, ax_spec.get_ylim()[1] if ax_spec.get_ylim()[1] > 0 else 1, f" {p_wn}",
                         rotation=90, va="top", ha="left", fontsize=8, color="#555555")

    # Shaded band for pretrain target region
    if wn.min() <= wn_band[0] and wn.max() >= wn_band[1]:
        ax_spec.axvspan(wn_band[0], wn_band[1], color="#ffeb3b", alpha=0.15, label=f"Target SSL Region ({wn_band[0]:.0f}-{wn_band[1]:.0f})")

    ax_spec.set_xlabel("Raman Shift (cm$^{-1}$)", fontsize=11)
    ax_spec.set_ylabel("Intensity (Counts)", fontsize=11)
    ax_spec.set_title(
        f"Raman Spectra Profile (Laser: {m.laser_nm} nm, Exp: {m.exposure_time_s}s)\n"
        f"Sample: {m.sample_id or 'N/A'} | Donor: {m.donor_id or 'N/A'} | Code: {m.diagnosis_code or 'N/A'}",
        fontsize=11,
        fontweight="bold",
    )
    ax_spec.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax_spec.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig
