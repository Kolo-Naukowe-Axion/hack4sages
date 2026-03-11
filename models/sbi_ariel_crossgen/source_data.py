"""Loading helpers for the published cross-generator dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from .constants import NOISY_SPECTRA_DATASET, SAFE_AUX_FEATURE_COLS, TARGET_COLS


def load_crossgen_source(
    source_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    labels_path = source_dir / "labels.parquet"
    spectra_path = source_dir / "spectra.h5"

    labels = pd.read_parquet(labels_path).reset_index(drop=True)
    with h5py.File(spectra_path, "r") as handle:
        h5_sample_id = handle["sample_id"][:].astype("U64")
        h5_generator = handle["generator"][:].astype("U16")
        h5_split = handle["split"][:].astype("U16")
        wavelength_um = np.asarray(handle["wavelength_um"][:], dtype=np.float32)
        noisy_spectra = np.asarray(handle[NOISY_SPECTRA_DATASET][:], dtype=np.float32)
        sigma_ppm = np.asarray(handle["sigma_ppm"][:], dtype=np.float32)

    index_frame = pd.DataFrame(
        {
            "sample_id": h5_sample_id,
            "_row_index": np.arange(len(h5_sample_id), dtype=np.int64),
            "_generator_h5": h5_generator,
            "_split_h5": h5_split,
        }
    )
    merged = labels.merge(index_frame, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        raise AssertionError("labels.parquet and spectra.h5 have inconsistent sample coverage.")
    if not np.array_equal(merged["generator"].to_numpy(dtype="U16"), merged["_generator_h5"].to_numpy(dtype="U16")):
        raise AssertionError("Generator mismatch between labels.parquet and spectra.h5.")
    if not np.array_equal(merged["split"].to_numpy(dtype="U16"), merged["_split_h5"].to_numpy(dtype="U16")):
        raise AssertionError("Split mismatch between labels.parquet and spectra.h5.")

    row_index = merged["_row_index"].to_numpy(dtype=np.int64)
    aligned = merged.drop(columns=["_row_index", "_generator_h5", "_split_h5"]).reset_index(drop=True)
    return aligned, noisy_spectra[row_index], sigma_ppm[row_index], wavelength_um


def build_feature_arrays(
    labels: pd.DataFrame,
    noisy_spectra: np.ndarray,
    sigma_ppm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aux = labels[SAFE_AUX_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    noise_scalar = np.log10(np.clip(sigma_ppm.astype(np.float32, copy=False), 1.0, None)).reshape(-1, 1)
    targets = labels[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)
    spectra = noisy_spectra.astype(np.float32, copy=True)
    return spectra, noise_scalar.astype(np.float32), aux, targets


def limit_indices(indices: np.ndarray, limit: Optional[int]) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return indices
    return indices[:limit]
