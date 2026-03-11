"""I/O helpers for the cross-generator dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .constants import DATASET_PATHS
from .latents import final_labels_frame


def require_parquet_support() -> None:
    """Ensure the runtime can read and write parquet files."""

    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("pyarrow is required for parquet support in this dataset pipeline.")


def write_latents_parquet(output_root: Path, latents: pd.DataFrame) -> Path:
    """Persist the intermediate latent table."""

    require_parquet_support()
    path = output_root / DATASET_PATHS.latents_parquet
    latents.to_parquet(path, index=False)
    return path


def read_latents_parquet(output_root: Path) -> pd.DataFrame:
    """Load the intermediate latent table."""

    require_parquet_support()
    return pd.read_parquet(output_root / DATASET_PATHS.latents_parquet)


def write_labels_parquet(output_root: Path, latents: pd.DataFrame) -> Path:
    """Persist the public labels table."""

    require_parquet_support()
    path = output_root / DATASET_PATHS.labels_parquet
    final_labels_frame(latents).to_parquet(path, index=False)
    return path


def read_labels_parquet(output_root: Path) -> pd.DataFrame:
    """Load the assembled public labels table."""

    require_parquet_support()
    return pd.read_parquet(output_root / DATASET_PATHS.labels_parquet)


def write_spectra_h5(
    output_root: Path,
    labels: pd.DataFrame,
    wavelength_um: np.ndarray,
    transit_depth_noiseless: np.ndarray,
    transit_depth_noisy: np.ndarray,
    sigma_ppm: np.ndarray,
) -> Path:
    """Write the assembled HDF5 spectra bundle."""

    path = output_root / DATASET_PATHS.spectra_h5
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    noiseless = np.asarray(transit_depth_noiseless, dtype=np.float32)
    noisy = np.asarray(transit_depth_noisy, dtype=np.float32)
    sigma = np.asarray(sigma_ppm, dtype=np.float32)

    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample_id", data=np.asarray(labels["sample_id"].tolist(), dtype="S32"))
        handle.create_dataset("generator", data=np.asarray(labels["generator"].tolist(), dtype="S16"))
        handle.create_dataset("split", data=np.asarray(labels["split"].tolist(), dtype="S16"))
        handle.create_dataset("wavelength_um", data=wavelength)
        handle.create_dataset("transit_depth_noiseless", data=noiseless, compression="gzip")
        handle.create_dataset("transit_depth_noisy", data=noisy, compression="gzip")
        handle.create_dataset("sigma_ppm", data=sigma)
    return path


def read_spectra_h5(output_root: Path) -> dict[str, Any]:
    """Load the assembled spectra arrays."""

    path = output_root / DATASET_PATHS.spectra_h5
    with h5py.File(path, "r") as handle:
        return {
            "sample_id": handle["sample_id"][:].astype("U32"),
            "generator": handle["generator"][:].astype("U16"),
            "split": handle["split"][:].astype("U16"),
            "wavelength_um": np.asarray(handle["wavelength_um"][:], dtype=np.float64),
            "transit_depth_noiseless": np.asarray(handle["transit_depth_noiseless"][:], dtype=np.float64),
            "transit_depth_noisy": np.asarray(handle["transit_depth_noisy"][:], dtype=np.float64),
            "sigma_ppm": np.asarray(handle["sigma_ppm"][:], dtype=np.float64),
        }
