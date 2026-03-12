"""Raw TauREx loading helpers for the five-gas FMPE workflow."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .constants import (
    AU_M,
    AUX_FEATURE_COLS,
    FIXED_PLANET_DISTANCE_AU,
    FIXED_STAR_DISTANCE_PC,
    FIXED_STAR_MASS_KG,
    FIXED_STAR_TEMPERATURE_K,
    G_NEWTON,
    LOG10_AUX_FEATURE_COLS,
    NOISE_PPM_TO_TRANSIT_DEPTH,
    REQUIRED_LABEL_COLUMNS,
    REQUIRED_SPECTRA_KEYS,
    RJUP_M,
    SECONDS_PER_DAY,
    SOLAR_RADIUS_M,
    SPECTRAL_LENGTH,
)


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return values.astype("U64")
    return values.astype(str)


def load_labels_table(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root).expanduser().resolve()
    labels = pd.read_parquet(root / "labels.parquet").copy()
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise KeyError(f"TauREx labels.parquet is missing required columns: {missing}")
    return labels.reset_index(drop=True)


def load_spectral_bundle(data_root: str | Path) -> dict[str, np.ndarray]:
    root = Path(data_root).expanduser().resolve()
    with h5py.File(root / "spectra.h5", "r") as handle:
        missing = [key for key in REQUIRED_SPECTRA_KEYS if key not in handle]
        if missing:
            raise KeyError(f"TauREx spectra.h5 is missing required datasets: {missing}")
        payload = {
            "sample_id": _decode_string_array(handle["sample_id"][:]),
            "generator": _decode_string_array(handle["generator"][:]),
            "split": _decode_string_array(handle["split"][:]),
            "wavelength_um": np.asarray(handle["wavelength_um"][:], dtype=np.float32),
            "spectra": np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32),
            "sigma_ppm": np.asarray(handle["sigma_ppm"][:], dtype=np.float32),
        }
    if payload["spectra"].ndim != 2 or payload["spectra"].shape[1] != SPECTRAL_LENGTH:
        raise RuntimeError(
            f"Unexpected TauREx spectra shape {payload['spectra'].shape}; expected (*, {SPECTRAL_LENGTH})."
        )
    if payload["sigma_ppm"].shape[0] != payload["spectra"].shape[0]:
        raise RuntimeError("TauREx sigma_ppm row count does not match the spectra row count.")
    return payload


def validate_alignment(labels: pd.DataFrame, spectra: dict[str, np.ndarray]) -> None:
    row_count = len(labels)
    if row_count != int(spectra["spectra"].shape[0]):
        raise RuntimeError("labels.parquet and spectra.h5 have different row counts.")
    if not np.array_equal(labels["sample_id"].to_numpy(dtype=str), spectra["sample_id"]):
        raise RuntimeError("sample_id ordering mismatch between labels.parquet and spectra.h5.")
    if not np.array_equal(labels["generator"].to_numpy(dtype=str), spectra["generator"]):
        raise RuntimeError("generator ordering mismatch between labels.parquet and spectra.h5.")
    if not np.array_equal(labels["split"].to_numpy(dtype=str), spectra["split"]):
        raise RuntimeError("split ordering mismatch between labels.parquet and spectra.h5.")


def orbital_period_days(planet_distance_au: np.ndarray, star_mass_kg: np.ndarray) -> np.ndarray:
    semi_major_axis_m = planet_distance_au.astype(np.float64) * AU_M
    period_seconds = 2.0 * math.pi * np.sqrt(np.power(semi_major_axis_m, 3) / (G_NEWTON * star_mass_kg.astype(np.float64)))
    return (period_seconds / SECONDS_PER_DAY).astype(np.float32)


def build_auxiliary_frame(labels: pd.DataFrame) -> pd.DataFrame:
    row_count = len(labels)
    planet_radius_m = labels["planet_radius_rjup"].to_numpy(dtype=np.float64) * RJUP_M
    planet_surface_gravity = (10.0 ** labels["log_g_cgs"].to_numpy(dtype=np.float64)) / 100.0
    planet_mass_kg = planet_surface_gravity * np.square(planet_radius_m) / G_NEWTON
    star_mass_kg = np.full(row_count, FIXED_STAR_MASS_KG, dtype=np.float32)
    planet_distance = np.full(row_count, FIXED_PLANET_DISTANCE_AU, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "sample_id": labels["sample_id"].astype(str).to_numpy(dtype="U64"),
            "generator": labels["generator"].astype(str).to_numpy(dtype="U16"),
            "split": labels["split"].astype(str).to_numpy(dtype="U16"),
            "star_distance": np.full(row_count, FIXED_STAR_DISTANCE_PC, dtype=np.float32),
            "star_mass_kg": star_mass_kg,
            "star_radius_m": labels["star_radius_rsun"].to_numpy(dtype=np.float32) * SOLAR_RADIUS_M,
            "star_temperature": np.full(row_count, FIXED_STAR_TEMPERATURE_K, dtype=np.float32),
            "planet_mass_kg": planet_mass_kg.astype(np.float32),
            "planet_orbital_period": orbital_period_days(planet_distance, star_mass_kg),
            "planet_distance": planet_distance,
            "planet_surface_gravity": planet_surface_gravity.astype(np.float32),
        }
    )
    return frame.loc[:, ["sample_id", "generator", "split", *AUX_FEATURE_COLS]].reset_index(drop=True)


def build_noise_matrix(sigma_ppm: np.ndarray, spectral_length: int = SPECTRAL_LENGTH) -> np.ndarray:
    sigma_depth = sigma_ppm.astype(np.float32).reshape(-1, 1) * NOISE_PPM_TO_TRANSIT_DEPTH
    return np.repeat(sigma_depth, spectral_length, axis=1).astype(np.float32)


def transform_aux_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[AUX_FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
    for column_index, column_name in enumerate(AUX_FEATURE_COLS):
        if column_name in LOG10_AUX_FEATURE_COLS:
            values[:, column_index] = np.log10(np.clip(values[:, column_index], 1.0e-12, None))
    return values.astype(np.float32)


def selector_indices(labels: pd.DataFrame, *, generator: str, split: str) -> np.ndarray:
    mask = (labels["generator"].astype(str) == generator) & (labels["split"].astype(str) == split)
    return np.flatnonzero(mask.to_numpy())
