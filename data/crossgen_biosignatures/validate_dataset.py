"""Validation helpers for the cross-generator biosignature dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    DATASET_PATHS,
    LOG_G_RANGE_CGS,
    NOISE_SIGMA_PPM_RANGE,
    PLANET_RADIUS_RANGE_RJUP,
    POSEIDON_GENERATOR_KEY,
    STAR_RADIUS_RANGE_RSUN,
    TARGET_RESOLUTION,
    TARGET_WAVELENGTH_MAX_UM,
    TARGET_WAVELENGTH_MIN_UM,
    TAUREX_GENERATOR_KEY,
    TEMPERATURE_RANGE_K,
    TRACE_LOG10_VMR_RANGE,
    TRACE_VMR_MAX_TOTAL,
)
from .dataset_io import read_labels_parquet, read_spectra_h5
from .grid import make_constant_resolution_grid
from .latents import FINAL_LABEL_COLUMNS


def validate_dataset(
    output_root: Path,
    expected_counts: dict[str, int] | None = None,
    enforce_prevalence_bounds: bool = True,
    required_generators: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate the assembled dataset bundle."""

    labels_path = output_root / DATASET_PATHS.labels_parquet
    spectra_path = output_root / DATASET_PATHS.spectra_h5
    manifest_path = output_root / DATASET_PATHS.manifest_json
    if not labels_path.exists():
        raise FileNotFoundError(str(labels_path))
    if not spectra_path.exists():
        raise FileNotFoundError(str(spectra_path))
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))

    labels = read_labels_parquet(output_root).sort_values(["generator", "sample_id"]).reset_index(drop=True)
    spectra = read_spectra_h5(output_root)
    available_generators = tuple(sorted(labels["generator"].astype(str).unique().tolist()))

    if required_generators is not None and not set(required_generators).issubset(set(available_generators)):
        raise AssertionError(
            f"Dataset is missing required generators. Required={required_generators}, observed={available_generators}"
        )

    if labels.columns.tolist() != list(FINAL_LABEL_COLUMNS):
        raise AssertionError("labels.parquet columns do not match the public schema.")
    if len(labels) != len(spectra["sample_id"]):
        raise AssertionError("labels.parquet and spectra.h5 differ in row count.")

    if not np.array_equal(labels["sample_id"].astype(str).to_numpy(dtype="U32"), spectra["sample_id"]):
        raise AssertionError("sample_id order mismatch between labels.parquet and spectra.h5.")
    if not np.array_equal(labels["generator"].astype(str).to_numpy(dtype="U16"), spectra["generator"]):
        raise AssertionError("generator order mismatch between labels.parquet and spectra.h5.")
    if not np.array_equal(labels["split"].astype(str).to_numpy(dtype="U16"), spectra["split"]):
        raise AssertionError("split order mismatch between labels.parquet and spectra.h5.")

    expected_centers, _ = make_constant_resolution_grid(
        TARGET_WAVELENGTH_MIN_UM,
        TARGET_WAVELENGTH_MAX_UM,
        TARGET_RESOLUTION,
    )
    if not np.allclose(spectra["wavelength_um"], expected_centers, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Stored wavelength grid does not match the configured shared grid.")
    if spectra["transit_depth_noisy"].shape != spectra["transit_depth_noiseless"].shape:
        raise AssertionError("Noisy and noiseless spectra arrays differ in shape.")
    if spectra["transit_depth_noisy"].shape[1] != len(expected_centers):
        raise AssertionError("Feature tensor shape does not match the stored wavelength grid.")
    if not np.isfinite(spectra["transit_depth_noisy"]).all() or not np.isfinite(spectra["transit_depth_noiseless"]).all():
        raise AssertionError("Spectra contain NaN or Inf values.")
    if np.any(spectra["transit_depth_noisy"] <= 0.0) or np.any(spectra["transit_depth_noiseless"] <= 0.0):
        raise AssertionError("Transit depths must remain strictly positive.")

    if expected_counts is not None:
        counts = labels["generator"].value_counts().to_dict()
        for generator, expected_count in expected_counts.items():
            if int(counts.get(generator, 0)) != int(expected_count):
                raise AssertionError(f"Generator {generator} has {counts.get(generator, 0)} rows instead of {expected_count}.")

    numeric_bounds = {
        "planet_radius_rjup": PLANET_RADIUS_RANGE_RJUP,
        "log_g_cgs": LOG_G_RANGE_CGS,
        "temperature_k": TEMPERATURE_RANGE_K,
        "star_radius_rsun": STAR_RADIUS_RANGE_RSUN,
    }
    for column_name, (lower, upper) in numeric_bounds.items():
        values = labels[column_name].to_numpy(dtype=np.float64)
        if np.any(values < lower) or np.any(values > upper):
            raise AssertionError(f"{column_name} falls outside the configured prior.")

    if np.any(labels["trace_vmr_total"].to_numpy(dtype=np.float64) >= TRACE_VMR_MAX_TOTAL):
        raise AssertionError("trace_vmr_total exceeds the configured maximum.")
    if np.any(spectra["sigma_ppm"] < NOISE_SIGMA_PPM_RANGE[0]) or np.any(spectra["sigma_ppm"] > NOISE_SIGMA_PPM_RANGE[1]):
        raise AssertionError("sigma_ppm falls outside the configured noise prior.")

    target_names = [column for column in labels.columns if column.startswith("log10_vmr_")]
    binary_names = [column for column in labels.columns if column.startswith("present_")]
    for column_name in target_names:
        values = labels[column_name].to_numpy(dtype=np.float64)
        if np.any(values < TRACE_LOG10_VMR_RANGE[0]) or np.any(values > TRACE_LOG10_VMR_RANGE[1]):
            raise AssertionError(f"{column_name} falls outside the configured prior.")
    for column_name in binary_names:
        species = column_name.removeprefix("present_")
        expected_binary = (labels[f"log10_vmr_{species}"].to_numpy(dtype=np.float64) >= -8.0).astype(np.int64)
        actual_binary = labels[column_name].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_binary, actual_binary):
            raise AssertionError(f"{column_name} does not match the configured presence threshold.")

    if enforce_prevalence_bounds:
        for generator in available_generators:
            subset = labels.loc[labels["generator"] == generator].reset_index(drop=True)
            for column_name in binary_names:
                prevalence = float(subset[column_name].mean())
                if prevalence <= 0.15 or prevalence >= 0.85:
                    raise AssertionError(f"{column_name} prevalence for {generator} is out of bounds: {prevalence:.3f}")

    manifest = json.loads(manifest_path.read_text())
    if "generator_summary" not in manifest:
        raise AssertionError("manifest.json is missing generator summaries.")
    if {TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY}.issubset(set(available_generators)) and "comparison_summary" not in manifest:
        raise AssertionError("manifest.json is missing generator comparison summaries.")

    return {
        "sample_count": int(len(labels)),
        "feature_shape": tuple(int(value) for value in spectra["transit_depth_noisy"].shape),
        "target_names": target_names,
        "binary_names": binary_names,
        "generator_counts": {str(key): int(value) for key, value in labels["generator"].value_counts().to_dict().items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_dataset(args.output_root)
    print("Validated cross-generator biosignature dataset:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
