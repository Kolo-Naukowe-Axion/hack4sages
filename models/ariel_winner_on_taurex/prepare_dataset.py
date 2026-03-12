"""Prepare the TauREx cross-generator bundle for the winner-style five-gas independent NSF model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .constants import (
    AU_M,
    AUX_COLUMNS,
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_ROOT,
    FIXED_PLANET_DISTANCE_AU,
    FIXED_STAR_DISTANCE_PC,
    FIXED_STAR_MASS_KG,
    FIXED_STAR_TEMPERATURE_K,
    G_NEWTON,
    GENERATOR_ID_OFFSETS,
    HOLDOUT_GENERATOR,
    HOLDOUT_SPLIT,
    NOISE_PPM_TO_TRANSIT_DEPTH,
    RJUP_M,
    SECONDS_PER_DAY,
    SOLAR_RADIUS_M,
    SPECTRAL_LENGTH,
    TARGET_COLUMNS,
    TRAIN_GENERATOR,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from .preprocessing import fit_scalers


REQUIRED_LABEL_COLUMNS = (
    "sample_id",
    "generator",
    "split",
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    *TARGET_COLUMNS,
)

REQUIRED_SPECTRA_KEYS = (
    "sample_id",
    "generator",
    "split",
    "wavelength_um",
    "transit_depth_noisy",
    "sigma_ppm",
)


def _decode_string_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return values.astype("U32")
    return values.astype(str)


def load_labels_table(data_root: Path) -> pd.DataFrame:
    labels = pd.read_parquet(data_root / "labels.parquet").copy()
    missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in labels.columns]
    if missing:
        raise KeyError(f"TauREx labels.parquet is missing required columns: {missing}")
    return labels


def load_spectral_bundle(data_root: Path) -> dict[str, np.ndarray]:
    with h5py.File(data_root / "spectra.h5", "r") as handle:
        missing = [key for key in REQUIRED_SPECTRA_KEYS if key not in handle]
        if missing:
            raise KeyError(f"TauREx spectra.h5 is missing required datasets: {missing}")
        payload = {
            "sample_id": _decode_string_array(handle["sample_id"][:]),
            "generator": _decode_string_array(handle["generator"][:]),
            "split": _decode_string_array(handle["split"][:]),
            "wavelength_um": np.asarray(handle["wavelength_um"][:], dtype=np.float64),
            "spectra": np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32),
            "sigma_ppm": np.asarray(handle["sigma_ppm"][:], dtype=np.float32),
        }
    if payload["spectra"].ndim != 2 or payload["spectra"].shape[1] != SPECTRAL_LENGTH:
        raise RuntimeError(f"Unexpected TauREx spectra shape {payload['spectra'].shape}; expected (*, {SPECTRAL_LENGTH}).")
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


def build_auxiliary_matrix(labels: pd.DataFrame) -> np.ndarray:
    row_count = len(labels)
    planet_radius_m = labels["planet_radius_rjup"].to_numpy(dtype=np.float64) * RJUP_M
    planet_surface_gravity = (10.0 ** labels["log_g_cgs"].to_numpy(dtype=np.float64)) / 100.0
    planet_mass_kg = planet_surface_gravity * np.square(planet_radius_m) / G_NEWTON
    star_mass_kg = np.full(row_count, FIXED_STAR_MASS_KG, dtype=np.float32)
    planet_distance = np.full(row_count, FIXED_PLANET_DISTANCE_AU, dtype=np.float32)
    aux = np.column_stack(
        [
            np.full(row_count, FIXED_STAR_DISTANCE_PC, dtype=np.float32),
            star_mass_kg,
            labels["star_radius_rsun"].to_numpy(dtype=np.float32) * SOLAR_RADIUS_M,
            np.full(row_count, FIXED_STAR_TEMPERATURE_K, dtype=np.float32),
            planet_mass_kg.astype(np.float32),
            orbital_period_days(planet_distance, star_mass_kg),
            planet_distance,
            planet_surface_gravity.astype(np.float32),
        ]
    )
    if aux.shape[1] != len(AUX_COLUMNS):
        raise RuntimeError(f"Unexpected TauREx auxiliary shape {aux.shape}; expected {len(AUX_COLUMNS)} columns.")
    return aux.astype(np.float32)


def build_noise_matrix(sigma_ppm: np.ndarray) -> np.ndarray:
    sigma_depth = sigma_ppm.astype(np.float32).reshape(-1, 1) * NOISE_PPM_TO_TRANSIT_DEPTH
    return np.repeat(sigma_depth, SPECTRAL_LENGTH, axis=1).astype(np.float32)


def encode_sample_ids(labels: pd.DataFrame) -> np.ndarray:
    sample_ids = labels["sample_id"].astype(str).to_numpy()
    generators = labels["generator"].astype(str).to_numpy()
    numeric_suffix = labels["sample_id"].astype(str).str.extract(r"(\d+)$", expand=False).fillna("-1").astype(np.int64).to_numpy()
    offsets = np.array(
        [GENERATOR_ID_OFFSETS.get(generator, 10_000_000 + idx * 1_000_000) for idx, generator in enumerate(generators)],
        dtype=np.int64,
    )
    encoded = offsets + numeric_suffix
    if np.unique(encoded).size != encoded.size:
        # Fall back to a unique stable row-based encoding if a future generator overlaps the reserved ranges.
        encoded = np.arange(1, len(sample_ids) + 1, dtype=np.int64)
    return encoded


def save_split(path: Path, *, ids: np.ndarray, aux: np.ndarray, spectra: np.ndarray, noise: np.ndarray, targets: np.ndarray | None) -> None:
    payload = {
        "planet_id": ids.astype(np.int64),
        "aux_raw": aux.astype(np.float32),
        "spectra_raw": spectra.astype(np.float32),
        "noise_raw": noise.astype(np.float32),
    }
    if targets is not None:
        payload["targets_raw"] = targets.astype(np.float32)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.expanduser().resolve()
    labels = load_labels_table(data_root)
    spectra_bundle = load_spectral_bundle(data_root)
    validate_alignment(labels, spectra_bundle)

    ids = encode_sample_ids(labels)
    aux = build_auxiliary_matrix(labels)
    targets = labels.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    spectra = spectra_bundle["spectra"].astype(np.float32)
    noise = build_noise_matrix(spectra_bundle["sigma_ppm"])

    train_mask = (labels["generator"] == TRAIN_GENERATOR) & (labels["split"] == TRAIN_SPLIT)
    validation_mask = (labels["generator"] == TRAIN_GENERATOR) & (labels["split"] == VALIDATION_SPLIT)
    holdout_mask = (labels["generator"] == HOLDOUT_GENERATOR) & (labels["split"] == HOLDOUT_SPLIT)

    if not bool(train_mask.any()):
        raise RuntimeError("No TauREx training rows found in labels.parquet.")
    if not bool(validation_mask.any()):
        raise RuntimeError("No TauREx validation rows found in labels.parquet.")
    if not bool(holdout_mask.any()):
        raise RuntimeError("No POSEIDON holdout rows found in labels.parquet.")

    scalers = fit_scalers(aux[train_mask.to_numpy()], spectra[train_mask.to_numpy()], noise[train_mask.to_numpy()], targets[train_mask.to_numpy()])
    scalers.save(output / "scalers.npz")
    np.save(output / "wavelength_um.npy", spectra_bundle["wavelength_um"].astype(np.float64))

    save_split(
        output / "train.npz",
        ids=ids[train_mask.to_numpy()],
        aux=aux[train_mask.to_numpy()],
        spectra=spectra[train_mask.to_numpy()],
        noise=noise[train_mask.to_numpy()],
        targets=targets[train_mask.to_numpy()],
    )
    save_split(
        output / "validation.npz",
        ids=ids[validation_mask.to_numpy()],
        aux=aux[validation_mask.to_numpy()],
        spectra=spectra[validation_mask.to_numpy()],
        noise=noise[validation_mask.to_numpy()],
        targets=targets[validation_mask.to_numpy()],
    )
    save_split(
        output / "holdout.npz",
        ids=ids[holdout_mask.to_numpy()],
        aux=aux[holdout_mask.to_numpy()],
        spectra=spectra[holdout_mask.to_numpy()],
        noise=noise[holdout_mask.to_numpy()],
        targets=targets[holdout_mask.to_numpy()],
    )
    save_split(
        output / "testdata.npz",
        ids=ids[holdout_mask.to_numpy()],
        aux=aux[holdout_mask.to_numpy()],
        spectra=spectra[holdout_mask.to_numpy()],
        noise=noise[holdout_mask.to_numpy()],
        targets=None,
    )

    manifest = {
        "data_root": str(data_root),
        "dataset": "crossgen_biosignatures",
        "train_generator": TRAIN_GENERATOR,
        "validation_split": VALIDATION_SPLIT,
        "holdout_generator": HOLDOUT_GENERATOR,
        "holdout_split": HOLDOUT_SPLIT,
        "testdata_mirrors_holdout": True,
        "context_layout": {
            "aux_engineered": 12,
            "spectrum_stats": 2,
            "spectrum_normalized": SPECTRAL_LENGTH,
            "noise_scaled": SPECTRAL_LENGTH,
            "total": 12 + 2 + SPECTRAL_LENGTH + SPECTRAL_LENGTH,
        },
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "holdout": int(holdout_mask.sum()),
            "testdata": int(holdout_mask.sum()),
        },
        "target_columns": TARGET_COLUMNS,
        "wavelength_bins": int(len(spectra_bundle["wavelength_um"])),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
