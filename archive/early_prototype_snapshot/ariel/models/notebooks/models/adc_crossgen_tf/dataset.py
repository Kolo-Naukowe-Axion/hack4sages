"""Dataset loading and preprocessing utilities for the TensorFlow ADC adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from .constants import AUX_COLUMNS, PRESENCE_COLUMNS, TARGET_COLUMNS


@dataclass(frozen=True)
class ScalarStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "ScalarStandardizer":
        mean = float(np.mean(values, dtype=np.float64))
        scale = float(np.std(values, dtype=np.float64))
        if scale == 0.0:
            scale = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "scale": self.scale}


@dataclass(frozen=True)
class ArrayStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayStandardizer":
        values64 = values.astype(np.float64, copy=False)
        mean = values64.mean(axis=0)
        scale = values64.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.scale + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


@dataclass(frozen=True)
class SplitArrays:
    sample_ids: np.ndarray
    generator: np.ndarray
    split: np.ndarray
    spectra: np.ndarray
    aux: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    presence: np.ndarray
    sigma_ppm: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.spectra.shape[0])


@dataclass(frozen=True)
class PreparedData:
    train: SplitArrays
    tau_val: SplitArrays
    poseidon: SplitArrays
    spectral_scaler: ScalarStandardizer
    aux_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer
    wavelength_um: np.ndarray
    metadata: dict[str, Any]


def _decode_text_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind == "S":
        return np.char.decode(array, "utf-8")
    if array.dtype.kind == "O":
        decoded = []
        for value in array.tolist():
            if isinstance(value, (bytes, bytearray)):
                decoded.append(value.decode("utf-8"))
            else:
                decoded.append(str(value))
        return np.asarray(decoded, dtype=str)
    return array.astype(str)


def load_aligned_dataset(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    labels_path = data_root / "labels.parquet"
    spectra_path = data_root / "spectra.h5"

    labels = pd.read_parquet(labels_path).reset_index(drop=True)
    with h5py.File(spectra_path, "r") as handle:
        sample_ids = _decode_text_array(handle["sample_id"][:])
        generators = _decode_text_array(handle["generator"][:])
        splits = _decode_text_array(handle["split"][:])
        spectra = np.asarray(handle["transit_depth_noisy"][:], dtype=np.float32)
        sigma_ppm = np.asarray(handle["sigma_ppm"][:], dtype=np.float32)
        wavelength_um = np.asarray(handle["wavelength_um"][:], dtype=np.float32)

    index_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "_row_index": np.arange(len(sample_ids), dtype=np.int64),
            "_generator_h5": generators,
            "_split_h5": splits,
        }
    )
    merged = labels.merge(index_frame, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        raise AssertionError("labels.parquet and spectra.h5 do not align by sample_id.")
    if not np.array_equal(merged["generator"].to_numpy(dtype=str), merged["_generator_h5"].to_numpy(dtype=str)):
        raise AssertionError("Generator mismatch between labels.parquet and spectra.h5.")
    if not np.array_equal(merged["split"].to_numpy(dtype=str), merged["_split_h5"].to_numpy(dtype=str)):
        raise AssertionError("Split mismatch between labels.parquet and spectra.h5.")

    row_index = merged["_row_index"].to_numpy(dtype=np.int64)
    labels = merged.drop(columns=["_row_index", "_generator_h5", "_split_h5"]).reset_index(drop=True)
    return labels, spectra[row_index], sigma_ppm[row_index], wavelength_um


def build_aux_targets(
    labels: pd.DataFrame,
    sigma_ppm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_aux = labels[["planet_radius_rjup", "log_g_cgs", "temperature_k", "star_radius_rsun"]].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    log10_sigma = np.log10(np.clip(sigma_ppm.astype(np.float32, copy=False), 1.0, None)).reshape(-1, 1)
    aux = np.concatenate([safe_aux, log10_sigma], axis=1).astype(np.float32)
    targets = labels[TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    presence = labels[PRESENCE_COLUMNS].to_numpy(dtype=np.int64, copy=True)
    return aux, targets, presence


def _slice_rows(indices: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None:
        return indices
    return indices[: int(limit)]


def _make_split(
    labels: pd.DataFrame,
    spectra: np.ndarray,
    aux: np.ndarray,
    targets: np.ndarray,
    raw_targets: np.ndarray,
    presence: np.ndarray,
    sigma_ppm: np.ndarray,
) -> SplitArrays:
    return SplitArrays(
        sample_ids=labels["sample_id"].to_numpy(dtype=str),
        generator=labels["generator"].to_numpy(dtype=str),
        split=labels["split"].to_numpy(dtype=str),
        spectra=spectra.astype(np.float32, copy=False),
        aux=aux.astype(np.float32, copy=False),
        targets=targets.astype(np.float32, copy=False),
        raw_targets=raw_targets.astype(np.float32, copy=True),
        presence=presence.astype(np.int64, copy=True),
        sigma_ppm=sigma_ppm.astype(np.float32, copy=True),
    )


def _augment_training_spectra(train_spectra: np.ndarray, sigma_ppm: np.ndarray, repeat: int, seed: int) -> np.ndarray:
    repeat = max(1, int(repeat))
    rng = np.random.default_rng(seed)
    sigma = (sigma_ppm.astype(np.float32) * 1.0e-6).reshape(-1, 1)
    noise = rng.normal(
        loc=0.0,
        scale=np.broadcast_to(sigma, (repeat, train_spectra.shape[0], train_spectra.shape[1])),
    ).astype(np.float32)
    return (train_spectra[None, :, :] + noise).reshape(-1, train_spectra.shape[1]).astype(np.float32)


def prepare_data(
    data_root: Path,
    seed: int = 42,
    augment_repeat: int = 5,
    train_limit: int | None = None,
    val_limit: int | None = None,
    poseidon_limit: int | None = None,
) -> PreparedData:
    labels, spectra, sigma_ppm, wavelength_um = load_aligned_dataset(Path(data_root))
    aux_raw, targets_raw, presence = build_aux_targets(labels, sigma_ppm)

    train_idx = np.flatnonzero((labels["generator"] == "tau") & (labels["split"] == "train"))
    val_idx = np.flatnonzero((labels["generator"] == "tau") & (labels["split"] == "val"))
    poseidon_idx = np.flatnonzero(labels["generator"] == "poseidon")

    train_idx = _slice_rows(train_idx, train_limit)
    val_idx = _slice_rows(val_idx, val_limit)
    poseidon_idx = _slice_rows(poseidon_idx, poseidon_limit)

    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    val_labels = labels.iloc[val_idx].reset_index(drop=True)
    poseidon_labels = labels.iloc[poseidon_idx].reset_index(drop=True)

    train_spectra_raw = spectra[train_idx]
    val_spectra_raw = spectra[val_idx]
    poseidon_spectra_raw = spectra[poseidon_idx]

    train_aux_raw = aux_raw[train_idx]
    val_aux_raw = aux_raw[val_idx]
    poseidon_aux_raw = aux_raw[poseidon_idx]

    train_targets_raw = targets_raw[train_idx]
    val_targets_raw = targets_raw[val_idx]
    poseidon_targets_raw = targets_raw[poseidon_idx]

    train_presence = presence[train_idx]
    val_presence = presence[val_idx]
    poseidon_presence = presence[poseidon_idx]

    train_sigma = sigma_ppm[train_idx]
    val_sigma = sigma_ppm[val_idx]
    poseidon_sigma = sigma_ppm[poseidon_idx]

    spectral_scaler = ScalarStandardizer.fit(train_spectra_raw)
    aux_scaler = ArrayStandardizer.fit(train_aux_raw)
    target_scaler = ArrayStandardizer.fit(train_targets_raw)

    train_spectra_augmented = _augment_training_spectra(train_spectra_raw, train_sigma, augment_repeat, seed)
    train_aux_augmented = np.repeat(train_aux_raw, repeats=max(1, int(augment_repeat)), axis=0)
    train_targets_augmented = np.repeat(train_targets_raw, repeats=max(1, int(augment_repeat)), axis=0)
    train_ids_augmented = np.repeat(train_labels["sample_id"].to_numpy(dtype=str), repeats=max(1, int(augment_repeat)))
    train_sigma_augmented = np.repeat(train_sigma.astype(np.float32), repeats=max(1, int(augment_repeat)))
    train_presence_augmented = np.repeat(train_presence, repeats=max(1, int(augment_repeat)), axis=0)
    train_generator_augmented = np.repeat(train_labels["generator"].to_numpy(dtype=str), repeats=max(1, int(augment_repeat)))
    train_split_augmented = np.repeat(train_labels["split"].to_numpy(dtype=str), repeats=max(1, int(augment_repeat)))

    train_split = SplitArrays(
        sample_ids=train_ids_augmented,
        generator=train_generator_augmented,
        split=train_split_augmented,
        spectra=spectral_scaler.transform(train_spectra_augmented),
        aux=aux_scaler.transform(train_aux_augmented),
        targets=target_scaler.transform(train_targets_augmented),
        raw_targets=train_targets_augmented.astype(np.float32, copy=True),
        presence=train_presence_augmented.astype(np.int64, copy=True),
        sigma_ppm=train_sigma_augmented.astype(np.float32, copy=True),
    )
    tau_val_split = _make_split(
        val_labels,
        spectral_scaler.transform(val_spectra_raw),
        aux_scaler.transform(val_aux_raw),
        target_scaler.transform(val_targets_raw),
        val_targets_raw,
        val_presence,
        val_sigma,
    )
    poseidon_split = _make_split(
        poseidon_labels,
        spectral_scaler.transform(poseidon_spectra_raw),
        aux_scaler.transform(poseidon_aux_raw),
        target_scaler.transform(poseidon_targets_raw),
        poseidon_targets_raw,
        poseidon_presence,
        poseidon_sigma,
    )

    metadata = {
        "rows_total": int(len(labels)),
        "train_rows_raw": int(len(train_idx)),
        "train_rows_augmented": int(train_split.rows),
        "tau_val_rows": int(len(val_idx)),
        "poseidon_rows": int(len(poseidon_idx)),
        "spectrum_length": int(train_split.spectra.shape[1]),
        "aux_columns": AUX_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "presence_columns": PRESENCE_COLUMNS,
    }

    return PreparedData(
        train=train_split,
        tau_val=tau_val_split,
        poseidon=poseidon_split,
        spectral_scaler=spectral_scaler,
        aux_scaler=aux_scaler,
        target_scaler=target_scaler,
        wavelength_um=wavelength_um,
        metadata=metadata,
    )
