"""Build a compact ADC2023 reference bundle for remote generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pandas as pd

from .constants import (
    ADC_AUX_COLUMNS,
    ADC_OUTPUT_INSTRUMENT_WIDTH_ASC,
    EMPIRICAL_AUX_COLUMNS,
    EMPIRICAL_FEATURE_COLUMNS,
    OFFICIAL_BASELINE_WLGRID_ASC,
)


def _load_canonical_arrays(dataset_root: Path) -> Dict[str, np.ndarray]:
    spectral_path = dataset_root / "TrainingData" / "SpectralData.hdf5"

    with h5py.File(spectral_path, "r") as handle:
        first_key = sorted(handle.keys())[0]
        group = handle[first_key]
        wlgrid = group["instrument_wlgrid"][:]
        instrument_width = group["instrument_width"][:]

    return {"wlgrid": wlgrid, "instrument_width": instrument_width}


def build_reference_bundle(dataset_root: Path, output_path: Path) -> Path:
    """Create the compact empirical prior bundle used by remote generation."""

    aux_path = dataset_root / "TrainingData" / "AuxillaryTable.csv"
    fm_path = dataset_root / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv"
    arrays = _load_canonical_arrays(dataset_root)

    if not np.allclose(arrays["wlgrid"], OFFICIAL_BASELINE_WLGRID_ASC, atol=1.0e-8):
        raise ValueError("Local ADC wlgrid does not match the official ADC2023 baseline wlgrid.")

    if not np.allclose(arrays["instrument_width"], ADC_OUTPUT_INSTRUMENT_WIDTH_ASC, atol=1.0e-8):
        raise ValueError("Local ADC instrument_width does not match the recorded canonical metadata field.")

    aux = pd.read_csv(aux_path)
    fm = pd.read_csv(fm_path)
    merged = aux.merge(
        fm[["planet_ID", "planet_radius", "planet_temp"]],
        on="planet_ID",
        how="inner",
        validate="one_to_one",
    )

    if list(aux.columns) != ADC_AUX_COLUMNS:
        raise ValueError(f"Unexpected AuxillaryTable columns: {list(aux.columns)}")

    feature_matrix = merged[EMPIRICAL_FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    feature_log = np.log10(feature_matrix)
    feature_center = feature_log.mean(axis=0)
    feature_scale = feature_log.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    feature_z = (feature_log - feature_center) / feature_scale

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        planet_id=merged["planet_ID"].to_numpy(dtype="U32"),
        feature_matrix_z=feature_z,
        feature_center=feature_center,
        feature_scale=feature_scale,
        feature_min=feature_log.min(axis=0),
        feature_max=feature_log.max(axis=0),
        empirical_feature_columns=np.array(EMPIRICAL_FEATURE_COLUMNS, dtype="U64"),
        empirical_aux_columns=np.array(EMPIRICAL_AUX_COLUMNS, dtype="U64"),
        aux_values=merged[EMPIRICAL_AUX_COLUMNS].to_numpy(dtype=np.float64, copy=True),
        aux_planet_mass_kg=merged["planet_mass_kg"].to_numpy(dtype=np.float64, copy=True),
        aux_planet_surface_gravity=merged["planet_surface_gravity"].to_numpy(dtype=np.float64, copy=True),
        aux_planet_radius=merged["planet_radius"].to_numpy(dtype=np.float64, copy=True),
        aux_planet_temp=merged["planet_temp"].to_numpy(dtype=np.float64, copy=True),
        canonical_wlgrid=arrays["wlgrid"],
        canonical_instrument_width=arrays["instrument_width"],
    )

    return output_path

