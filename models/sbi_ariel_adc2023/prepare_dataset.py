"""Prepare ADC2023 train/validation/holdout/test arrays for five-gas FMPE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import (
    AUX_FEATURE_COLS,
    CONTEXT_DIM,
    CONTEXT_FILENAME_TEMPLATE,
    DATASET_TYPE,
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_DIR,
    HOLDOUT_SPLIT,
    LOG10_AUX_FEATURE_COLS,
    MANIFEST_FILENAME,
    METADATA_FILENAME_TEMPLATE,
    NOISE_FIELD,
    NORMALIZATION_FILENAME,
    NORMALIZATION_MODE,
    RAW_SPECTRAL_CHANNELS,
    RAW_TARGET_FILENAME_TEMPLATE,
    SPECTRUM_FIELD,
    TARGET_COLS,
    TARGET_FILENAME_TEMPLATE,
    TESTDATA_SPLIT,
    THETA_DIM,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    WAVELENGTH_FILENAME,
)


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = values.std(axis=0, ddof=0).astype(np.float32)
    return np.where(scale == 0.0, 1.0, scale).astype(np.float32)


def _normalize_spectrum(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32, copy=True)
    sample_mean = values.mean(axis=1, keepdims=True)
    sample_mean = np.clip(sample_mean, 1.0e-12, None)
    return (values / sample_mean).astype(np.float32)


def _log_noise(values: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(values.astype(np.float32, copy=False), 1.0e-12, None)).astype(np.float32)


def _compute_zscore_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = _safe_scale(values)
    return mean, scale


def _apply_zscore(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def _limit_indices(indices: np.ndarray, limit: Optional[int]) -> np.ndarray:
    if limit is None or limit >= len(indices):
        return np.sort(indices.astype(np.int64))
    return np.sort(indices[: int(limit)].astype(np.int64))


def _ensure_dataset_complete(data_root: Path) -> None:
    required = [
        data_root / "TrainingData" / "AuxillaryTable.csv",
        data_root / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv",
        data_root / "TrainingData" / "SpectralData.hdf5",
        data_root / "TestData" / "AuxillaryTable.csv",
        data_root / "TestData" / "SpectralData.hdf5",
    ]
    missing = [path for path in required if not path.exists()]
    if not missing:
        return
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "ADC2023 dataset is incomplete. Expected these files under "
        f"{data_root}:\n{missing_lines}"
    )


def _metadata_frame(planet_ids: np.ndarray, source_indices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "planet_ID": planet_ids.astype("U32"),
            "source_row_index": source_indices.astype(np.int64),
        }
    )


def prepare_dataset(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_dir: str | Path = DEFAULT_PREPARED_DIR,
    *,
    overwrite: bool = False,
    seed: int = 42,
    train_limit: Optional[int] = None,
    validation_limit: Optional[int] = None,
    holdout_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
) -> dict[str, Any]:
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    _ensure_dataset_complete(data_root)

    from .raw_dataset import build_stratify_labels, load_test_dataset, load_training_dataset, transform_aux_features

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_frame, labeled_spectra_raw, wavelength_um = load_training_dataset(data_root)
    test_frame, test_spectra_raw, test_wavelength_um = load_test_dataset(data_root)
    if not np.allclose(wavelength_um, test_wavelength_um, atol=1.0e-8):
        raise AssertionError("Training and test wavelength grids do not match.")

    spectrum_index = RAW_SPECTRAL_CHANNELS.index(SPECTRUM_FIELD)
    noise_index = RAW_SPECTRAL_CHANNELS.index(NOISE_FIELD)

    labeled_spectrum = _normalize_spectrum(labeled_spectra_raw[:, :, spectrum_index])
    test_spectrum = _normalize_spectrum(test_spectra_raw[:, :, spectrum_index])
    labeled_noise = _log_noise(labeled_spectra_raw[:, :, noise_index])
    test_noise = _log_noise(test_spectra_raw[:, :, noise_index])
    labeled_aux = transform_aux_features(labeled_frame)
    test_aux = transform_aux_features(test_frame)
    labeled_targets = labeled_frame[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)

    all_indices = np.arange(len(labeled_frame), dtype=np.int64)
    stratify_all, stratify_mode_all = build_stratify_labels(labeled_targets)
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=stratify_all if stratify_all is not None else None,
    )

    temp_targets = labeled_targets[temp_indices]
    stratify_temp, stratify_mode_temp = build_stratify_labels(temp_targets)
    temp_positions = np.arange(len(temp_indices), dtype=np.int64)
    validation_positions, holdout_positions = train_test_split(
        temp_positions,
        test_size=0.5,
        random_state=seed + 1,
        shuffle=True,
        stratify=stratify_temp if stratify_temp is not None else None,
    )

    train_indices = _limit_indices(train_indices, train_limit)
    validation_indices = _limit_indices(temp_indices[validation_positions], validation_limit)
    holdout_indices = _limit_indices(temp_indices[holdout_positions], holdout_limit)
    test_indices = _limit_indices(np.arange(len(test_frame), dtype=np.int64), test_limit)

    spectrum_mean, spectrum_scale = _compute_zscore_stats(labeled_spectrum[train_indices])
    noise_mean, noise_scale = _compute_zscore_stats(labeled_noise[train_indices])
    aux_mean, aux_scale = _compute_zscore_stats(labeled_aux[train_indices])
    target_mean, target_scale = _compute_zscore_stats(labeled_targets[train_indices])

    def transform_context(indices: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                _apply_zscore(labeled_spectrum[indices], spectrum_mean, spectrum_scale),
                _apply_zscore(labeled_noise[indices], noise_mean, noise_scale),
                _apply_zscore(labeled_aux[indices], aux_mean, aux_scale),
            ],
            axis=1,
        ).astype(np.float32)

    train_context = transform_context(train_indices)
    validation_context = transform_context(validation_indices)
    holdout_context = transform_context(holdout_indices)
    test_context = np.concatenate(
        [
            _apply_zscore(test_spectrum[test_indices], spectrum_mean, spectrum_scale),
            _apply_zscore(test_noise[test_indices], noise_mean, noise_scale),
            _apply_zscore(test_aux[test_indices], aux_mean, aux_scale),
        ],
        axis=1,
    ).astype(np.float32)

    train_targets = _apply_zscore(labeled_targets[train_indices], target_mean, target_scale)
    validation_targets = _apply_zscore(labeled_targets[validation_indices], target_mean, target_scale)
    holdout_targets = _apply_zscore(labeled_targets[holdout_indices], target_mean, target_scale)

    split_payloads = [
        (TRAIN_SPLIT, train_indices, train_context, train_targets, labeled_targets[train_indices]),
        (VALIDATION_SPLIT, validation_indices, validation_context, validation_targets, labeled_targets[validation_indices]),
        (HOLDOUT_SPLIT, holdout_indices, holdout_context, holdout_targets, labeled_targets[holdout_indices]),
    ]
    for split_name, indices, context, targets, raw_targets in split_payloads:
        np.save(output_dir / CONTEXT_FILENAME_TEMPLATE.format(split_name=split_name), context.astype(np.float32))
        np.save(output_dir / TARGET_FILENAME_TEMPLATE.format(split_name=split_name), targets.astype(np.float32))
        np.save(output_dir / RAW_TARGET_FILENAME_TEMPLATE.format(split_name=split_name), raw_targets.astype(np.float32))
        _metadata_frame(
            labeled_frame.iloc[indices]["planet_ID"].to_numpy(dtype="U32"),
            indices,
        ).to_csv(output_dir / METADATA_FILENAME_TEMPLATE.format(split_name=split_name), index=False)

    np.save(output_dir / CONTEXT_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT), test_context.astype(np.float32))
    _metadata_frame(
        test_frame.iloc[test_indices]["planet_ID"].to_numpy(dtype="U32"),
        test_indices,
    ).to_csv(output_dir / METADATA_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT), index=False)

    np.savez(
        output_dir / NORMALIZATION_FILENAME,
        spectrum_mean=spectrum_mean,
        spectrum_scale=spectrum_scale,
        noise_mean=noise_mean,
        noise_scale=noise_scale,
        aux_mean=aux_mean,
        aux_scale=aux_scale,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    np.save(output_dir / WAVELENGTH_FILENAME, wavelength_um.astype(np.float32))

    manifest = {
        "data_root": str(data_root),
        "prepared_dir": str(output_dir),
        "dataset_type": DATASET_TYPE,
        "normalization_mode": NORMALIZATION_MODE,
        "seed": int(seed),
        "split_fractions": {"train": 0.8, "validation": 0.1, "holdout": 0.1},
        "split_counts": {
            TRAIN_SPLIT: int(len(train_indices)),
            VALIDATION_SPLIT: int(len(validation_indices)),
            HOLDOUT_SPLIT: int(len(holdout_indices)),
            TESTDATA_SPLIT: int(len(test_indices)),
        },
        "context_dim": CONTEXT_DIM,
        "theta_dim": THETA_DIM,
        "target_columns": TARGET_COLS,
        "aux_feature_cols": AUX_FEATURE_COLS,
        "log10_aux_feature_cols": LOG10_AUX_FEATURE_COLS,
        "wavelength_bins": int(len(wavelength_um)),
        "wavelength_min_um": float(wavelength_um.min()),
        "wavelength_max_um": float(wavelength_um.max()),
        "feature_order": [
            "normalized_instrument_spectrum[52]",
            "log10_instrument_noise[52]",
            *AUX_FEATURE_COLS,
        ],
        "spectrum_normalization": {
            "mode": "divide_by_sample_mean_then_train_zscore",
            "field": SPECTRUM_FIELD,
        },
        "noise_normalization": {
            "mode": "log10_then_train_zscore",
            "field": NOISE_FIELD,
        },
        "aux_normalization": {
            "mode": "log10_selected_then_train_zscore",
        },
        "target_normalization": {
            "mode": "train_zscore",
        },
        "stratify_mode_train_split": stratify_mode_all,
        "stratify_mode_validation_split": stratify_mode_temp,
    }
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ADC2023 five-gas arrays for FMPE training.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to the full ADC2023 dataset root.")
    parser.add_argument("--output", default=str(DEFAULT_PREPARED_DIR), help="Prepared output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split generation.")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap for train rows.")
    parser.add_argument("--validation-limit", type=int, default=None, help="Optional cap for validation rows.")
    parser.add_argument("--holdout-limit", type=int, default=None, help="Optional cap for holdout rows.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional cap for test rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(
        data_root=args.data_root,
        output_dir=args.output,
        overwrite=args.overwrite,
        seed=args.seed,
        train_limit=args.train_limit,
        validation_limit=args.validation_limit,
        holdout_limit=args.holdout_limit,
        test_limit=args.test_limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
