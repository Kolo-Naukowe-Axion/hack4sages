"""Prepare explicit train/validation/holdout splits for FMPE training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .constants import (
    AUX_FILENAME_TEMPLATE,
    CONTEXT_DIM,
    DATASET_TYPE,
    MANIFEST_FILENAME,
    METADATA_FILENAME_TEMPLATE,
    NOISE_FIELD_NAME,
    NOISE_FILENAME_TEMPLATE,
    NORMALIZATION_MODE,
    NORMALIZATION_FILENAME,
    SAFE_AUX_FEATURE_COLS,
    SPLIT_SPECS,
    SPECTRA_FILENAME_TEMPLATE,
    TARGET_COLS,
    TARGET_FILENAME_TEMPLATE,
    THETA_DIM,
    WAVELENGTH_FILENAME,
)
from .source_data import build_feature_arrays, limit_indices, load_crossgen_source


def _compute_min_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = values.min(axis=0).astype(np.float32)
    maximum = values.max(axis=0).astype(np.float32)
    maximum = np.where(maximum == minimum, minimum + 1.0, maximum).astype(np.float32)
    return minimum, maximum


def _compute_min_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    minimum = values.min(axis=0).astype(np.float32)
    ddof = 1 if values.shape[0] > 1 else 0
    scale = values.std(axis=0, ddof=ddof).astype(np.float32)
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32)
    return minimum, scale


def _save_split_arrays(
    output_dir: Path,
    split_name: str,
    metadata_frame,
    spectra: np.ndarray,
    noise_scalar: np.ndarray,
    aux: np.ndarray,
    targets: np.ndarray,
) -> None:
    np.save(output_dir / SPECTRA_FILENAME_TEMPLATE.format(split_name=split_name), spectra.astype(np.float32))
    np.save(output_dir / NOISE_FILENAME_TEMPLATE.format(split_name=split_name), noise_scalar.astype(np.float32))
    np.save(output_dir / AUX_FILENAME_TEMPLATE.format(split_name=split_name), aux.astype(np.float32))
    np.save(output_dir / TARGET_FILENAME_TEMPLATE.format(split_name=split_name), targets.astype(np.float32))
    metadata_frame.to_csv(output_dir / METADATA_FILENAME_TEMPLATE.format(split_name=split_name), index=False)


def prepare_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    limit_tau_train: Optional[int] = None,
    limit_tau_val: Optional[int] = None,
    limit_poseidon: Optional[int] = None,
) -> dict:
    source_path = Path(source_dir).expanduser().resolve()
    target_path = Path(output_dir).expanduser().resolve()
    if target_path.exists() and any(target_path.iterdir()) and not overwrite:
        raise FileExistsError(f"{target_path} already exists and is not empty. Pass --overwrite to replace outputs.")
    target_path.mkdir(parents=True, exist_ok=True)

    labels, noisy_spectra, sigma_ppm, wavelength_um = load_crossgen_source(source_path)
    spectra, noise_scalar, aux, targets = build_feature_arrays(labels, noisy_spectra, sigma_ppm)
    row_index = np.arange(len(labels), dtype=np.int64)

    limits = {
        "tau_train": limit_tau_train,
        "tau_val": limit_tau_val,
        "poseidon_holdout": limit_poseidon,
    }

    split_counts: dict[str, int] = {}
    selected_indices: dict[str, np.ndarray] = {}
    for split_name, spec in SPLIT_SPECS.items():
        mask = labels["generator"].eq(spec["generator"])
        mask &= labels["split"].eq(spec["split"])
        indices = np.flatnonzero(mask.to_numpy())
        indices = limit_indices(indices, limits.get(split_name))
        selected_indices[split_name] = indices
        split_counts[split_name] = int(len(indices))

        metadata = labels.iloc[indices][["sample_id", "generator", "split"]].copy()
        metadata.insert(0, "source_row_index", row_index[indices])
        _save_split_arrays(
            target_path,
            split_name,
            metadata,
            spectra[indices],
            noise_scalar[indices],
            aux[indices],
            targets[indices],
        )

    train_indices = selected_indices["tau_train"]
    spectra_min, spectra_scale = _compute_min_std(spectra[train_indices])
    noise_min, noise_max = _compute_min_max(noise_scalar[train_indices])
    aux_min, aux_max = _compute_min_max(aux[train_indices])
    targets_min, targets_max = _compute_min_max(targets[train_indices])
    np.savez(
        target_path / NORMALIZATION_FILENAME,
        spectra_min=spectra_min,
        spectra_scale=spectra_scale,
        noise_min=noise_min,
        noise_max=noise_max,
        aux_min=aux_min,
        aux_max=aux_max,
        targets_min=targets_min,
        targets_max=targets_max,
    )
    np.save(target_path / WAVELENGTH_FILENAME, wavelength_um.astype(np.float32))

    manifest = {
        "source_dir": str(source_path),
        "prepared_dir": str(target_path),
        "dataset_type": DATASET_TYPE,
        "normalization_mode": NORMALIZATION_MODE,
        "split_counts": split_counts,
        "safe_aux_feature_cols": SAFE_AUX_FEATURE_COLS,
        "target_columns": TARGET_COLS,
        "noise_field": NOISE_FIELD_NAME,
        "feature_order": ["spectra_bins", NOISE_FIELD_NAME, *SAFE_AUX_FEATURE_COLS],
        "context_dim": CONTEXT_DIM,
        "theta_dim": THETA_DIM,
        "wavelength_bins": int(len(wavelength_um)),
        "wavelength_min_um": float(wavelength_um.min()),
        "wavelength_max_um": float(wavelength_um.max()),
    }
    (target_path / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare explicit TauREx/POSEIDON splits for FMPE training.")
    parser.add_argument("--source", required=True, help="Path to the published cross-generator dataset.")
    parser.add_argument("--output", required=True, help="Path where the prepared split arrays will be written.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty directory.")
    parser.add_argument("--limit-tau-train", type=int, default=None, help="Optional deterministic cap for tau/train rows.")
    parser.add_argument("--limit-tau-val", type=int, default=None, help="Optional deterministic cap for tau/val rows.")
    parser.add_argument("--limit-poseidon", type=int, default=None, help="Optional deterministic cap for POSEIDON holdout rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        overwrite=args.overwrite,
        limit_tau_train=args.limit_tau_train,
        limit_tau_val=args.limit_tau_val,
        limit_poseidon=args.limit_poseidon,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
