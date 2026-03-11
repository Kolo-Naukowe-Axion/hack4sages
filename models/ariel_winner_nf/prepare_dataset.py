"""Prepare ADC2023 data for the winner-style five-gas independent NSF model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import AUX_COLUMNS, DEFAULT_DATA_ROOT, DEFAULT_PREPARED_ROOT, SPECTRAL_LENGTH, TARGET_COLUMNS
from .preprocessing import PreparedScalers, fit_scalers, transform_targets


def load_spectral_table(hdf5_path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        keys = sorted(handle.keys(), key=lambda name: int("".join(ch for ch in name if ch.isdigit())))
        spectra = np.stack([handle[key]["instrument_spectrum"][:] for key in keys], axis=0).astype(np.float32)
        noise = np.stack([handle[key]["instrument_noise"][:] for key in keys], axis=0).astype(np.float32)
        widths = np.stack([handle[key]["instrument_width"][:] for key in keys], axis=0).astype(np.float32)
    if spectra.shape[1] != SPECTRAL_LENGTH or noise.shape[1] != SPECTRAL_LENGTH or widths.shape[1] != SPECTRAL_LENGTH:
        raise RuntimeError("Unexpected spectral shape in ADC2023 HDF5 data.")
    return spectra, noise, widths


def load_training_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    aux = pd.read_csv(data_root / "TrainingData" / "AuxillaryTable.csv")
    targets = pd.read_csv(data_root / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv")
    merged = aux.merge(targets[["planet_ID", *TARGET_COLUMNS]], on="planet_ID", how="inner", validate="one_to_one")
    return merged, aux


def load_test_table(data_root: Path) -> pd.DataFrame:
    return pd.read_csv(data_root / "TestData" / "AuxillaryTable.csv")


def numeric_planet_ids(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.extract(r"(\d+)$", expand=False).astype(np.int64).to_numpy()


def build_stratify_labels(targets: np.ndarray) -> np.ndarray:
    presence = (targets > -8.0).astype(np.int32)
    weights = (1 << np.arange(targets.shape[1], dtype=np.int32)).reshape(1, -1)
    labels = (presence * weights).sum(axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    rare = set(unique[counts < 3].tolist())
    if rare:
        labels = np.array([label if label not in rare else -1 for label in labels], dtype=np.int32)
    return labels


def split_indices(num_rows: int, targets: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = build_stratify_labels(targets)
    all_indices = np.arange(num_rows, dtype=np.int64)
    val_plus_holdout = 4142 + 4143
    stratify_first = labels if np.unique(labels).size > 1 else None
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=val_plus_holdout,
        random_state=seed,
        shuffle=True,
        stratify=stratify_first,
    )
    temp_labels = labels[temp_idx]
    stratify_second = temp_labels if np.unique(temp_labels).size > 1 else None
    val_idx, holdout_idx = train_test_split(
        temp_idx,
        test_size=4143,
        random_state=seed,
        shuffle=True,
        stratify=stratify_second,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(holdout_idx)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root.expanduser().resolve()
    merged, train_aux_table = load_training_tables(data_root)
    test_aux_table = load_test_table(data_root)
    train_spectra, train_noise, _ = load_spectral_table(data_root / "TrainingData" / "SpectralData.hdf5", prefix="Planet_train")
    test_spectra, test_noise, _ = load_spectral_table(data_root / "TestData" / "SpectralData.hdf5", prefix="Planet_test")

    aux_train = merged[AUX_COLUMNS].to_numpy(dtype=np.float32)
    targets_train = merged[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    ids_train = numeric_planet_ids(merged["planet_ID"])
    aux_test = test_aux_table[AUX_COLUMNS].to_numpy(dtype=np.float32)
    ids_test = numeric_planet_ids(test_aux_table["planet_ID"])

    train_idx, val_idx, holdout_idx = split_indices(len(merged), targets_train, seed=args.seed)

    scalers = fit_scalers(aux_train[train_idx], train_spectra[train_idx], train_noise[train_idx], targets_train[train_idx])
    scalers.save(output / "scalers.npz")

    save_split(
        output / "train.npz",
        ids=ids_train[train_idx],
        aux=aux_train[train_idx],
        spectra=train_spectra[train_idx],
        noise=train_noise[train_idx],
        targets=targets_train[train_idx],
    )
    save_split(
        output / "validation.npz",
        ids=ids_train[val_idx],
        aux=aux_train[val_idx],
        spectra=train_spectra[val_idx],
        noise=train_noise[val_idx],
        targets=targets_train[val_idx],
    )
    save_split(
        output / "holdout.npz",
        ids=ids_train[holdout_idx],
        aux=aux_train[holdout_idx],
        spectra=train_spectra[holdout_idx],
        noise=train_noise[holdout_idx],
        targets=targets_train[holdout_idx],
    )
    save_split(
        output / "testdata.npz",
        ids=ids_test,
        aux=aux_test,
        spectra=test_spectra,
        noise=test_noise,
        targets=None,
    )

    manifest = {
        "data_root": str(data_root),
        "seed": int(args.seed),
        "context_layout": {
            "aux_engineered": 12,
            "spectrum_stats": 2,
            "spectrum_normalized": 52,
            "noise_scaled": 52,
            "total": 118,
        },
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "holdout": int(len(holdout_idx)),
            "testdata": int(len(ids_test)),
        },
        "target_columns": TARGET_COLUMNS,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
