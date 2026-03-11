"""Raw ADC2023 loading and stratification helpers for the five-gas FMPE workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from .constants import (
    AUX_FEATURE_COLS,
    COARSE_ABUNDANCE_QUANTILES,
    COARSE_STRATIFY_MIN_COUNT,
    HDF5_GROUP_PREFIX,
    LOG10_AUX_FEATURE_COLS,
    PRESENCE_THRESHOLD_LOG10_VMR,
    PRIMARY_STRATIFY_MIN_COUNT,
    RAW_SPECTRAL_CHANNELS,
    TARGET_COLS,
    WAVELENGTH_DATASET,
)


def _drop_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    unnamed = [column for column in frame.columns if column.startswith("Unnamed:")]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def _load_spectra(hdf5_path: Path, planet_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectra = np.empty((len(planet_ids), 52, len(RAW_SPECTRAL_CHANNELS)), dtype=np.float32)
    wavelength_um: Optional[np.ndarray] = None

    with h5py.File(hdf5_path, "r") as handle:
        for row_index, planet_id in enumerate(planet_ids.tolist()):
            group_name = f"{HDF5_GROUP_PREFIX}{planet_id}"
            if group_name not in handle:
                raise AssertionError(f"Missing HDF5 group {group_name} in {hdf5_path}.")
            group = handle[group_name]

            group_wavelength = np.asarray(group[WAVELENGTH_DATASET][:], dtype=np.float32)
            if wavelength_um is None:
                wavelength_um = group_wavelength
            elif not np.allclose(group_wavelength, wavelength_um, atol=1.0e-8):
                raise AssertionError(f"Wavelength grid mismatch detected for {planet_id}.")

            for channel_index, channel_name in enumerate(RAW_SPECTRAL_CHANNELS):
                channel_values = np.asarray(group[channel_name][:], dtype=np.float32)
                if channel_values.shape != (52,):
                    raise AssertionError(f"{group_name}/{channel_name} has shape {channel_values.shape}, expected (52,).")
                spectra[row_index, :, channel_index] = channel_values

    if wavelength_um is None:
        raise AssertionError(f"No spectra found in {hdf5_path}.")
    return spectra, wavelength_um


def load_training_dataset(data_root: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = Path(data_root).expanduser().resolve()
    aux_path = root / "TrainingData" / "AuxillaryTable.csv"
    target_path = root / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv"
    spectral_path = root / "TrainingData" / "SpectralData.hdf5"

    aux = _drop_unnamed_columns(pd.read_csv(aux_path))
    targets = _drop_unnamed_columns(pd.read_csv(target_path))
    merged = aux.merge(targets[["planet_ID", *TARGET_COLS]], on="planet_ID", how="inner", validate="one_to_one")
    if len(merged) != len(aux) or len(merged) != len(targets):
        raise AssertionError("Auxiliary features and target table do not align one-to-one on planet_ID.")
    if list(aux.columns) != ["planet_ID", *AUX_FEATURE_COLS]:
        raise AssertionError(f"Unexpected auxiliary columns: {list(aux.columns)}")

    spectra, wavelength_um = _load_spectra(spectral_path, merged["planet_ID"].to_numpy(dtype="U32"))
    return merged.reset_index(drop=True), spectra, wavelength_um


def load_test_dataset(data_root: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = Path(data_root).expanduser().resolve()
    aux_path = root / "TestData" / "AuxillaryTable.csv"
    spectral_path = root / "TestData" / "SpectralData.hdf5"

    aux = _drop_unnamed_columns(pd.read_csv(aux_path))
    if list(aux.columns) != ["planet_ID", *AUX_FEATURE_COLS]:
        raise AssertionError(f"Unexpected test auxiliary columns: {list(aux.columns)}")

    spectra, wavelength_um = _load_spectra(spectral_path, aux["planet_ID"].to_numpy(dtype="U32"))
    return aux.reset_index(drop=True), spectra, wavelength_um


def transform_aux_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[AUX_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    for column_index, column_name in enumerate(AUX_FEATURE_COLS):
        if column_name in LOG10_AUX_FEATURE_COLS:
            values[:, column_index] = np.log10(np.clip(values[:, column_index], 1.0e-12, None))
    return values.astype(np.float32)


def _presence_signature(targets: np.ndarray) -> np.ndarray:
    presence = (targets >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)
    bit_weights = (1 << np.arange(presence.shape[1], dtype=np.int64)).reshape(1, -1)
    return (presence * bit_weights).sum(axis=1).astype(np.int64)


def _coarse_abundance_labels(targets: np.ndarray) -> np.ndarray:
    mean_abundance = targets.mean(axis=1)
    quantiles = np.quantile(mean_abundance, COARSE_ABUNDANCE_QUANTILES)
    edges = np.unique(np.asarray(quantiles, dtype=np.float64))
    abundance_bin = np.digitize(mean_abundance, edges, right=False)
    presence_count = (targets >= PRESENCE_THRESHOLD_LOG10_VMR).sum(axis=1)
    return np.asarray([f"{count}_{bin_index}" for count, bin_index in zip(presence_count, abundance_bin)], dtype="U16")


def _can_stratify(labels: np.ndarray, min_count: int) -> bool:
    counts = pd.Series(labels).value_counts()
    return len(counts) > 1 and int(counts.min()) >= int(min_count)


def build_stratify_labels(targets: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    primary = _presence_signature(targets)
    if _can_stratify(primary, PRIMARY_STRATIFY_MIN_COUNT):
        return primary, "presence_signature"

    coarse = _coarse_abundance_labels(targets)
    if _can_stratify(coarse, COARSE_STRATIFY_MIN_COUNT):
        return coarse, "coarse_abundance"

    return None, "unstratified"
