"""ADC-format dataset loading and cached preparation for Ariel regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from .constants import (
    AUX_COLUMNS,
    COARSE_ABUNDANCE_QUANTILES,
    COARSE_STRATIFY_MIN_COUNT,
    DEFAULT_PREPARED_CACHE_SUBDIR,
    FIXED_SPECTRAL_CHANNELS,
    HDF5_GROUP_PREFIX,
    LOG10_AUX_COLUMNS,
    MODEL_SPECTRAL_CHANNELS,
    PRESENCE_THRESHOLD_LOG10_VMR,
    PRIMARY_STRATIFY_MIN_COUNT,
    RAW_SPECTRAL_CHANNELS,
    SAMPLE_SPECTRAL_CHANNELS,
    TARGET_COLUMNS,
    WAVELENGTH_DATASET,
)


@dataclass
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

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ArrayStandardizer":
        return cls(mean=np.asarray(state["mean"], dtype=np.float32), scale=np.asarray(state["scale"], dtype=np.float32))


@dataclass
class SpectralStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    fixed_channels: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, fixed_channels: np.ndarray) -> "SpectralStandardizer":
        values64 = values.astype(np.float64, copy=False)
        mean = values64.mean(axis=0)
        scale = values64.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(
            mean=mean.astype(np.float32),
            scale=scale.astype(np.float32),
            fixed_channels=fixed_channels.astype(np.float32, copy=True),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = ((values - self.mean[None, :, :]) / self.scale[None, :, :]).astype(np.float32)
        fixed = np.broadcast_to(self.fixed_channels[None, :, :], (values.shape[0], *self.fixed_channels.shape))
        return np.concatenate([scaled, fixed.astype(np.float32)], axis=1)

    def state_dict(self) -> dict[str, list[list[float]]]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "fixed_channels": self.fixed_channels.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SpectralStandardizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
            fixed_channels=np.asarray(state["fixed_channels"], dtype=np.float32),
        )


@dataclass
class LabeledSplit:
    planet_ids: np.ndarray
    aux: torch.Tensor
    spectra: torch.Tensor
    targets: torch.Tensor
    raw_targets: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.aux.shape[0])


@dataclass
class InferenceSplit:
    planet_ids: np.ndarray
    aux: torch.Tensor
    spectra: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.aux.shape[0])


@dataclass
class PreparedData:
    train: LabeledSplit
    val: LabeledSplit
    holdout: LabeledSplit
    testdata: InferenceSplit
    aux_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer
    spectral_scaler: SpectralStandardizer
    wavelength_um: np.ndarray
    split_manifest: dict[str, Any]
    prepared_manifest: dict[str, Any]


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


def load_training_dataset(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = Path(data_root).expanduser().resolve()
    aux_path = root / "TrainingData" / "AuxillaryTable.csv"
    target_path = root / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv"
    spectral_path = root / "TrainingData" / "SpectralData.hdf5"

    aux = _drop_unnamed_columns(pd.read_csv(aux_path))
    targets = _drop_unnamed_columns(pd.read_csv(target_path))
    merged = aux.merge(targets[["planet_ID", *TARGET_COLUMNS]], on="planet_ID", how="inner", validate="one_to_one")
    if len(merged) != len(aux) or len(merged) != len(targets):
        raise AssertionError("Auxiliary features and target table do not align one-to-one on planet_ID.")
    if list(aux.columns) != ["planet_ID", *AUX_COLUMNS]:
        raise AssertionError(f"Unexpected auxiliary columns: {list(aux.columns)}")

    spectra, wavelength_um = _load_spectra(spectral_path, merged["planet_ID"].to_numpy(dtype="U32"))
    return merged.reset_index(drop=True), spectra, wavelength_um


def load_test_dataset(data_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    root = Path(data_root).expanduser().resolve()
    aux_path = root / "TestData" / "AuxillaryTable.csv"
    spectral_path = root / "TestData" / "SpectralData.hdf5"

    aux = _drop_unnamed_columns(pd.read_csv(aux_path))
    if list(aux.columns) != ["planet_ID", *AUX_COLUMNS]:
        raise AssertionError(f"Unexpected test auxiliary columns: {list(aux.columns)}")

    spectra, wavelength_um = _load_spectra(spectral_path, aux["planet_ID"].to_numpy(dtype="U32"))
    return aux.reset_index(drop=True), spectra, wavelength_um


def transform_aux_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[AUX_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    for column_index, column_name in enumerate(AUX_COLUMNS):
        if column_name in LOG10_AUX_COLUMNS:
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


def _limit_indices(indices: np.ndarray, limit: Optional[int]) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return np.sort(indices.astype(np.int64))
    return np.sort(indices[: int(limit)].astype(np.int64))


def _make_labeled_split(
    planet_ids: np.ndarray,
    aux_values: np.ndarray,
    spectra_values: np.ndarray,
    targets_scaled: np.ndarray,
    raw_targets: np.ndarray,
) -> LabeledSplit:
    return LabeledSplit(
        planet_ids=planet_ids.astype("U32"),
        aux=torch.from_numpy(aux_values.astype(np.float32, copy=False)),
        spectra=torch.from_numpy(spectra_values.astype(np.float32, copy=False)),
        targets=torch.from_numpy(targets_scaled.astype(np.float32, copy=False)),
        raw_targets=raw_targets.astype(np.float32, copy=True),
    )


def _make_inference_split(planet_ids: np.ndarray, aux_values: np.ndarray, spectra_values: np.ndarray) -> InferenceSplit:
    return InferenceSplit(
        planet_ids=planet_ids.astype("U32"),
        aux=torch.from_numpy(aux_values.astype(np.float32, copy=False)),
        spectra=torch.from_numpy(spectra_values.astype(np.float32, copy=False)),
    )


def _expected_manifest(
    data_root: Path,
    seed: int,
    train_limit: Optional[int],
    val_limit: Optional[int],
    holdout_limit: Optional[int],
    test_limit: Optional[int],
) -> dict[str, Any]:
    return {
        "version": 3,
        "data_root": str(Path(data_root).expanduser().resolve()),
        "seed": int(seed),
        "split_fractions": {"train": 0.8, "val": 0.1, "holdout": 0.1},
        "limits": {
            "train": train_limit,
            "val": val_limit,
            "holdout": holdout_limit,
            "testdata": test_limit,
        },
        "aux_columns": AUX_COLUMNS,
        "log10_aux_columns": LOG10_AUX_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "raw_spectral_channels": RAW_SPECTRAL_CHANNELS,
        "sample_spectral_channels": SAMPLE_SPECTRAL_CHANNELS,
        "fixed_spectral_channels": FIXED_SPECTRAL_CHANNELS,
        "model_spectral_channels": MODEL_SPECTRAL_CHANNELS,
        "sample_spectral_normalization": {
            "mode": "divide_by_sample_mean",
            "reference_channel": SAMPLE_SPECTRAL_CHANNELS[0],
            "applied_channels": SAMPLE_SPECTRAL_CHANNELS,
        },
        "presence_threshold_log10_vmr": PRESENCE_THRESHOLD_LOG10_VMR,
    }


def _load_cache_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _save_prepared_cache(
    cache_dir: Path,
    manifest: dict[str, Any],
    prepared: PreparedData,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_json(cache_dir / "manifest.json", manifest)
    _save_json(
        cache_dir / "scalers.json",
        {
            "aux_scaler": prepared.aux_scaler.state_dict(),
            "target_scaler": prepared.target_scaler.state_dict(),
            "spectral_scaler": prepared.spectral_scaler.state_dict(),
        },
    )
    _save_json(cache_dir / "split_manifest.json", prepared.split_manifest)
    np.save(cache_dir / "wavelength_um.npy", prepared.wavelength_um.astype(np.float32))

    for split_name, split in (("train", prepared.train), ("val", prepared.val), ("holdout", prepared.holdout)):
        np.save(cache_dir / f"{split_name}_planet_ids.npy", split.planet_ids)
        np.save(cache_dir / f"{split_name}_aux.npy", split.aux.cpu().numpy())
        np.save(cache_dir / f"{split_name}_spectra.npy", split.spectra.cpu().numpy())
        np.save(cache_dir / f"{split_name}_targets.npy", split.targets.cpu().numpy())
        np.save(cache_dir / f"{split_name}_raw_targets.npy", split.raw_targets)

    np.save(cache_dir / "testdata_planet_ids.npy", prepared.testdata.planet_ids)
    np.save(cache_dir / "testdata_aux.npy", prepared.testdata.aux.cpu().numpy())
    np.save(cache_dir / "testdata_spectra.npy", prepared.testdata.spectra.cpu().numpy())


def _load_prepared_cache(cache_dir: Path, expected_manifest: dict[str, Any]) -> Optional[PreparedData]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    cached_manifest = _load_cache_json(manifest_path)
    if cached_manifest != expected_manifest:
        return None

    scalers = _load_cache_json(cache_dir / "scalers.json")
    split_manifest = _load_cache_json(cache_dir / "split_manifest.json")
    wavelength_um = np.load(cache_dir / "wavelength_um.npy").astype(np.float32)

    def load_labeled_split(split_name: str) -> LabeledSplit:
        return _make_labeled_split(
            planet_ids=np.load(cache_dir / f"{split_name}_planet_ids.npy"),
            aux_values=np.load(cache_dir / f"{split_name}_aux.npy").astype(np.float32),
            spectra_values=np.load(cache_dir / f"{split_name}_spectra.npy").astype(np.float32),
            targets_scaled=np.load(cache_dir / f"{split_name}_targets.npy").astype(np.float32),
            raw_targets=np.load(cache_dir / f"{split_name}_raw_targets.npy").astype(np.float32),
        )

    prepared_manifest = dict(cached_manifest)
    prepared_manifest["cache_dir"] = str(cache_dir)
    prepared_manifest["cache_hit"] = True

    return PreparedData(
        train=load_labeled_split("train"),
        val=load_labeled_split("val"),
        holdout=load_labeled_split("holdout"),
        testdata=_make_inference_split(
            planet_ids=np.load(cache_dir / "testdata_planet_ids.npy"),
            aux_values=np.load(cache_dir / "testdata_aux.npy").astype(np.float32),
            spectra_values=np.load(cache_dir / "testdata_spectra.npy").astype(np.float32),
        ),
        aux_scaler=ArrayStandardizer.from_state_dict(scalers["aux_scaler"]),
        target_scaler=ArrayStandardizer.from_state_dict(scalers["target_scaler"]),
        spectral_scaler=SpectralStandardizer.from_state_dict(scalers["spectral_scaler"]),
        wavelength_um=wavelength_um,
        split_manifest=split_manifest,
        prepared_manifest=prepared_manifest,
    )


def resolve_prepared_cache_dir(output_dir: Path, prepared_cache_dir: Optional[str | Path]) -> Path:
    if prepared_cache_dir is None:
        return output_dir / DEFAULT_PREPARED_CACHE_SUBDIR
    cache_path = Path(prepared_cache_dir).expanduser()
    if cache_path.is_absolute():
        return cache_path
    return output_dir / cache_path


def _normalize_fixed_channel(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = float(values.mean())
    scale = float(values.std())
    if scale == 0.0:
        scale = 1.0
    return ((values - mean) / scale).astype(np.float32)


def _normalize_sample_spectra(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32, copy=True)
    if values.ndim != 3:
        raise AssertionError(f"Expected sample spectra with shape (N, C, L), got {values.shape}.")
    reference = values[:, 0, :]
    sample_mean = reference.mean(axis=1, keepdims=True)
    sample_mean = np.clip(sample_mean, 1.0e-12, None)
    return (values / sample_mean[:, None, :]).astype(np.float32)


def prepare_data(
    data_root: Path,
    output_dir: Path,
    prepared_cache_dir: Optional[str | Path] = None,
    seed: int = 42,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    holdout_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> PreparedData:
    root = Path(data_root).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    cache_dir = resolve_prepared_cache_dir(output_path, prepared_cache_dir)
    expected_manifest = _expected_manifest(root, seed, train_limit, val_limit, holdout_limit, test_limit)

    cached = _load_prepared_cache(cache_dir, expected_manifest)
    if cached is not None:
        return cached

    labeled_frame, labeled_spectra_raw, wavelength_um = load_training_dataset(root)
    test_frame, test_spectra_raw, test_wavelength_um = load_test_dataset(root)
    if not np.allclose(test_wavelength_um, wavelength_um, atol=1.0e-8):
        raise AssertionError("Training and test wavelength grids do not match.")

    labeled_aux_raw = transform_aux_features(labeled_frame)
    test_aux_raw = transform_aux_features(test_frame)
    labeled_targets_raw = labeled_frame[TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)

    sample_channel_indices = [RAW_SPECTRAL_CHANNELS.index(name) for name in SAMPLE_SPECTRAL_CHANNELS]
    width_channel_index = RAW_SPECTRAL_CHANNELS.index("instrument_width")

    labeled_sample_spectra = np.transpose(labeled_spectra_raw[:, :, sample_channel_indices], (0, 2, 1)).astype(np.float32)
    test_sample_spectra = np.transpose(test_spectra_raw[:, :, sample_channel_indices], (0, 2, 1)).astype(np.float32)
    labeled_sample_spectra = _normalize_sample_spectra(labeled_sample_spectra)
    test_sample_spectra = _normalize_sample_spectra(test_sample_spectra)
    fixed_channels = np.stack(
        [
            _normalize_fixed_channel(labeled_spectra_raw[0, :, width_channel_index]),
            _normalize_fixed_channel(wavelength_um),
        ],
        axis=0,
    ).astype(np.float32)

    stratify_all, stratify_mode_all = build_stratify_labels(labeled_targets_raw)
    all_indices = np.arange(len(labeled_frame), dtype=np.int64)
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=stratify_all if stratify_all is not None else None,
    )

    temp_targets_raw = labeled_targets_raw[temp_indices]
    stratify_temp, stratify_mode_temp = build_stratify_labels(temp_targets_raw)
    temp_positions = np.arange(len(temp_indices), dtype=np.int64)
    val_positions, holdout_positions = train_test_split(
        temp_positions,
        test_size=0.5,
        random_state=seed + 1,
        shuffle=True,
        stratify=stratify_temp if stratify_temp is not None else None,
    )

    train_indices = _limit_indices(train_indices, train_limit)
    val_indices = _limit_indices(temp_indices[val_positions], val_limit)
    holdout_indices = _limit_indices(temp_indices[holdout_positions], holdout_limit)
    test_indices = _limit_indices(np.arange(len(test_frame), dtype=np.int64), test_limit)

    aux_scaler = ArrayStandardizer.fit(labeled_aux_raw[train_indices])
    target_scaler = ArrayStandardizer.fit(labeled_targets_raw[train_indices])
    spectral_scaler = SpectralStandardizer.fit(labeled_sample_spectra[train_indices], fixed_channels=fixed_channels)

    def transform_labeled(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        aux_scaled = aux_scaler.transform(labeled_aux_raw[indices])
        spectra_scaled = spectral_scaler.transform(labeled_sample_spectra[indices])
        raw_targets = labeled_targets_raw[indices]
        targets_scaled = target_scaler.transform(raw_targets)
        return aux_scaled, spectra_scaled, targets_scaled, raw_targets

    train_aux, train_spectra_scaled, train_targets_scaled, train_targets_raw = transform_labeled(train_indices)
    val_aux, val_spectra_scaled, val_targets_scaled, val_targets_raw = transform_labeled(val_indices)
    holdout_aux, holdout_spectra_scaled, holdout_targets_scaled, holdout_targets_raw = transform_labeled(holdout_indices)
    test_aux_scaled = aux_scaler.transform(test_aux_raw[test_indices])
    test_spectra_scaled = spectral_scaler.transform(test_sample_spectra[test_indices])

    split_manifest = {
        "rows_total": int(len(labeled_frame)),
        "testdata_rows_total": int(len(test_frame)),
        "train_rows": int(len(train_indices)),
        "val_rows": int(len(val_indices)),
        "holdout_rows": int(len(holdout_indices)),
        "testdata_rows": int(len(test_indices)),
        "wavelength_bins": int(len(wavelength_um)),
        "wavelength_min_um": float(wavelength_um.min()),
        "wavelength_max_um": float(wavelength_um.max()),
        "raw_spectrum_shape": [52, len(RAW_SPECTRAL_CHANNELS)],
        "model_spectrum_shape": [len(MODEL_SPECTRAL_CHANNELS), 52],
        "aux_columns": AUX_COLUMNS,
        "log10_aux_columns": LOG10_AUX_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "raw_spectral_channels": RAW_SPECTRAL_CHANNELS,
        "sample_spectral_channels": SAMPLE_SPECTRAL_CHANNELS,
        "fixed_spectral_channels": FIXED_SPECTRAL_CHANNELS,
        "model_spectral_channels": MODEL_SPECTRAL_CHANNELS,
        "sample_spectral_normalization": {
            "mode": "divide_by_sample_mean",
            "reference_channel": SAMPLE_SPECTRAL_CHANNELS[0],
            "applied_channels": SAMPLE_SPECTRAL_CHANNELS,
        },
        "presence_threshold_log10_vmr": PRESENCE_THRESHOLD_LOG10_VMR,
        "split_seed": int(seed),
        "split_fractions": {"train": 0.8, "val": 0.1, "holdout": 0.1},
        "primary_stratify_mode": stratify_mode_all,
        "secondary_stratify_mode": stratify_mode_temp,
    }

    prepared_manifest = dict(expected_manifest)
    prepared_manifest["cache_dir"] = str(cache_dir)
    prepared_manifest["cache_hit"] = False

    prepared = PreparedData(
        train=_make_labeled_split(
            planet_ids=labeled_frame.iloc[train_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=train_aux,
            spectra_values=train_spectra_scaled,
            targets_scaled=train_targets_scaled,
            raw_targets=train_targets_raw,
        ),
        val=_make_labeled_split(
            planet_ids=labeled_frame.iloc[val_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=val_aux,
            spectra_values=val_spectra_scaled,
            targets_scaled=val_targets_scaled,
            raw_targets=val_targets_raw,
        ),
        holdout=_make_labeled_split(
            planet_ids=labeled_frame.iloc[holdout_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=holdout_aux,
            spectra_values=holdout_spectra_scaled,
            targets_scaled=holdout_targets_scaled,
            raw_targets=holdout_targets_raw,
        ),
        testdata=_make_inference_split(
            planet_ids=test_frame.iloc[test_indices]["planet_ID"].to_numpy(dtype="U32"),
            aux_values=test_aux_scaled,
            spectra_values=test_spectra_scaled,
        ),
        aux_scaler=aux_scaler,
        target_scaler=target_scaler,
        spectral_scaler=spectral_scaler,
        wavelength_um=wavelength_um.astype(np.float32),
        split_manifest=split_manifest,
        prepared_manifest=prepared_manifest,
    )
    _save_prepared_cache(cache_dir, expected_manifest, prepared)
    return prepared
