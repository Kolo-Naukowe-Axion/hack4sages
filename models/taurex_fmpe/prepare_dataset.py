"""Prepare TauREx-only train/validation/holdout/test arrays for five-gas FMPE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import (
    AUX_FEATURE_COLS,
    CONTEXT_DIM,
    CONTEXT_FILENAME_TEMPLATE,
    DATASET_TYPE,
    DEFAULT_DATA_ROOT,
    DEFAULT_PREPARED_DIR,
    EXCLUDED_GENERATORS,
    HOLDOUT_SPLIT,
    LOG10_AUX_FEATURE_COLS,
    MANIFEST_FILENAME,
    METADATA_FILENAME_TEMPLATE,
    NORMALIZATION_FILENAME,
    NORMALIZATION_MODE,
    RAW_TARGET_FILENAME_TEMPLATE,
    SOURCE_HOLDOUT_GENERATOR,
    SOURCE_HOLDOUT_SPLIT,
    SOURCE_TRAIN_GENERATOR,
    SOURCE_TRAIN_SPLIT,
    SOURCE_VALIDATION_GENERATOR,
    SOURCE_VALIDATION_SPLIT,
    SPECTRAL_LENGTH,
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
        data_root / "labels.parquet",
        data_root / "spectra.h5",
    ]
    missing = [path for path in required if not path.exists()]
    if not missing:
        return
    missing_lines = "\n".join(f"- {path}" for path in missing)
    raise FileNotFoundError(
        "TauREx dataset is incomplete. Expected these files under "
        f"{data_root}:\n{missing_lines}"
    )


def _metadata_frame(metadata_rows: pd.DataFrame, source_indices: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": metadata_rows["sample_id"].to_numpy(dtype="U64"),
            "generator": metadata_rows["generator"].to_numpy(dtype="U16"),
            "split": metadata_rows["split"].to_numpy(dtype="U16"),
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

    from .raw_dataset import (
        build_auxiliary_frame,
        build_noise_matrix,
        load_labels_table,
        load_spectral_bundle,
        selector_indices,
        transform_aux_features,
        validate_alignment,
    )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Pass --overwrite to replace outputs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels_table(data_root)
    spectra_bundle = load_spectral_bundle(data_root)
    validate_alignment(labels, spectra_bundle)

    aux_frame = build_auxiliary_frame(labels)
    raw_spectrum = spectra_bundle["spectra"].astype(np.float32)
    raw_noise = build_noise_matrix(spectra_bundle["sigma_ppm"], spectral_length=SPECTRAL_LENGTH)
    wavelength_um = spectra_bundle["wavelength_um"].astype(np.float32)
    if raw_spectrum.shape[1] != SPECTRAL_LENGTH:
        raise AssertionError(f"Expected {SPECTRAL_LENGTH} wavelength bins, got {raw_spectrum.shape[1]}.")

    labeled_spectrum = _normalize_spectrum(raw_spectrum)
    labeled_noise = _log_noise(raw_noise)
    labeled_aux = transform_aux_features(aux_frame)
    labeled_targets = labels[TARGET_COLS].to_numpy(dtype=np.float32, copy=True)

    train_source_indices = selector_indices(labels, generator=SOURCE_TRAIN_GENERATOR, split=SOURCE_TRAIN_SPLIT)
    validation_source_indices = selector_indices(labels, generator=SOURCE_VALIDATION_GENERATOR, split=SOURCE_VALIDATION_SPLIT)
    holdout_source_indices = selector_indices(labels, generator=SOURCE_HOLDOUT_GENERATOR, split=SOURCE_HOLDOUT_SPLIT)

    if len(train_source_indices) == 0:
        raise RuntimeError("No TauREx training rows found in labels.parquet.")
    if len(validation_source_indices) == 0:
        raise RuntimeError("No TauREx validation rows found in labels.parquet.")
    if len(holdout_source_indices) == 0:
        raise RuntimeError("No TauREx holdout rows found in labels.parquet.")

    train_indices = _limit_indices(train_source_indices, train_limit)
    validation_indices = _limit_indices(validation_source_indices, validation_limit)
    holdout_indices = _limit_indices(holdout_source_indices, holdout_limit)
    if test_limit is None and holdout_limit is not None:
        test_indices = holdout_indices.copy()
    else:
        test_indices = _limit_indices(holdout_source_indices, test_limit)

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
    test_context = transform_context(test_indices)

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
            aux_frame.iloc[indices].reset_index(drop=True),
            indices,
        ).to_csv(output_dir / METADATA_FILENAME_TEMPLATE.format(split_name=split_name), index=False)

    np.save(output_dir / CONTEXT_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT), test_context.astype(np.float32))
    _metadata_frame(
        aux_frame.iloc[test_indices].reset_index(drop=True),
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
        "split_strategy": "embedded_generator_splits",
        "excluded_generators": list(EXCLUDED_GENERATORS),
        "holdout_mirrors_validation": True,
        "split_selectors": {
            TRAIN_SPLIT: {"generator": SOURCE_TRAIN_GENERATOR, "split": SOURCE_TRAIN_SPLIT},
            VALIDATION_SPLIT: {"generator": SOURCE_VALIDATION_GENERATOR, "split": SOURCE_VALIDATION_SPLIT},
            HOLDOUT_SPLIT: {"generator": SOURCE_HOLDOUT_GENERATOR, "split": SOURCE_HOLDOUT_SPLIT},
            TESTDATA_SPLIT: {"generator": SOURCE_HOLDOUT_GENERATOR, "split": SOURCE_HOLDOUT_SPLIT},
        },
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
            f"normalized_transit_depth[{SPECTRAL_LENGTH}]",
            f"log10_white_noise[{SPECTRAL_LENGTH}]",
            *AUX_FEATURE_COLS,
        ],
        "spectrum_normalization": {
            "mode": "divide_by_sample_mean_then_train_zscore",
            "field": "transit_depth_noisy",
        },
        "noise_normalization": {
            "mode": "log10_then_train_zscore",
            "field": "sigma_ppm",
        },
        "aux_normalization": {
            "mode": "log10_selected_then_train_zscore",
        },
        "target_normalization": {
            "mode": "train_zscore",
        },
    }
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare TauREx-only five-gas arrays for FMPE training.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Path to the TauREx dataset root.")
    parser.add_argument("--output", default=str(DEFAULT_PREPARED_DIR), help="Prepared output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty directory.")
    parser.add_argument("--seed", type=int, default=42, help="Recorded seed value for reproducibility metadata.")
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
