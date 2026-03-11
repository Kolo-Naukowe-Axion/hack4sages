"""Validate a generated pRT transmission benchmark dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd

from .constants import (
    BENCHMARK_RESOLUTION,
    CLOUD_LOG10_P_RANGE,
    DatasetPaths,
    DETECTION_THRESHOLD_LOG_X,
    HAZE_GAMMA_RANGE,
    HAZE_LOG10_KAPPA0_RANGE,
    HEAVY_SPECIES_MAX_FRACTION,
    LOG_G_RANGE_CGS,
    PRESSURE_LEVELS,
    PRIMARY_TARGET_SPECIES,
    SPLIT_COUNTS,
    WAVELENGTH_MAX_UM,
    WAVELENGTH_MIN_UM,
)
from .generate_dataset import assign_splits
from .grid import build_rebin_matrix, make_log_resolving_power_grid


BENCHMARK_PATHS = DatasetPaths()


def _require_parquet_support() -> None:
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("Reading parquet requires pyarrow. Install it in the validation environment first.")


def _check_required_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AssertionError("%s is missing required columns: %s" % (name, ", ".join(missing)))


def validate_dataset(output_root: Path) -> Dict[str, Any]:
    _require_parquet_support()
    spectra_path = output_root / BENCHMARK_PATHS.spectra_h5
    labels_path = output_root / BENCHMARK_PATHS.labels_parquet
    provenance_path = output_root / BENCHMARK_PATHS.provenance_parquet
    if not spectra_path.exists():
        raise FileNotFoundError(str(spectra_path))
    if not labels_path.exists():
        raise FileNotFoundError(str(labels_path))
    if not provenance_path.exists():
        raise FileNotFoundError(str(provenance_path))

    labels = pd.read_parquet(labels_path).sort_values("sample_id").reset_index(drop=True)
    provenance = pd.read_parquet(provenance_path).sort_values("sample_id").reset_index(drop=True)
    if len(labels) != len(provenance):
        raise AssertionError("labels.parquet and provenance.parquet differ in row count.")

    expected_label_columns = [
        "sample_id",
        "split",
        "planet_radius_ref_m",
        "terminator_temperature_k",
        "log_g_cgs",
        "log10_p_cloud_bar",
        "log10_kappa0",
        "gamma_scat",
        "heavy_species_mass_fraction",
        "temperature_regime",
    ]
    for species in PRIMARY_TARGET_SPECIES:
        expected_label_columns.append("log_X_%s" % species)
        expected_label_columns.append("present_%s" % species)
    _check_required_columns(labels, expected_label_columns, "labels.parquet")
    _check_required_columns(provenance, ["sample_id", "split", "ood_candidate", "generation_attempt"], "provenance.parquet")

    benchmark_wavelength, benchmark_edges = make_log_resolving_power_grid(
        WAVELENGTH_MIN_UM,
        WAVELENGTH_MAX_UM,
        BENCHMARK_RESOLUTION,
    )

    with h5py.File(spectra_path, "r") as handle:
        required_paths = [
            "sample_id",
            "split",
            "benchmark/wavelength_um",
            "benchmark/transit_depth_noisy",
            "benchmark/transit_depth_noiseless",
            "benchmark/sigma_1sigma",
            "native/wavelength_um",
            "native/transit_depth_noiseless",
            "native/transit_radius_m",
        ]
        for path in required_paths:
            if path not in handle:
                raise AssertionError("spectra.h5 is missing dataset %s" % path)

        sample_ids_h5 = handle["sample_id"][:].astype("U16")
        splits_h5 = handle["split"][:].astype("U16")
        benchmark_wavelength_h5 = np.asarray(handle["benchmark/wavelength_um"][:], dtype=np.float64)
        native_wavelength_h5 = np.asarray(handle["native/wavelength_um"][:], dtype=np.float64)
        benchmark_noisy = np.asarray(handle["benchmark/transit_depth_noisy"][:], dtype=np.float64)
        benchmark_noiseless = np.asarray(handle["benchmark/transit_depth_noiseless"][:], dtype=np.float64)
        benchmark_sigma = np.asarray(handle["benchmark/sigma_1sigma"][:], dtype=np.float64)
        native_noiseless = np.asarray(handle["native/transit_depth_noiseless"][:], dtype=np.float64)
        native_radius_m = np.asarray(handle["native/transit_radius_m"][:], dtype=np.float64)

    sample_ids_labels = labels["sample_id"].astype(str).to_numpy(dtype="U16")
    if not np.array_equal(sample_ids_h5, sample_ids_labels):
        raise AssertionError("sample_id ordering mismatch between spectra.h5 and labels.parquet.")
    if not np.array_equal(sample_ids_h5, provenance["sample_id"].astype(str).to_numpy(dtype="U16")):
        raise AssertionError("sample_id ordering mismatch between spectra.h5 and provenance.parquet.")

    if benchmark_noisy.shape != benchmark_noiseless.shape or benchmark_noisy.shape != benchmark_sigma.shape:
        raise AssertionError("Benchmark arrays do not share the same shape.")
    if native_noiseless.shape != native_radius_m.shape:
        raise AssertionError("Native arrays do not share the same shape.")
    if benchmark_noisy.shape[0] != len(labels):
        raise AssertionError("spectra row count does not match labels row count.")

    if not np.all(np.diff(benchmark_wavelength_h5) > 0.0):
        raise AssertionError("Benchmark wavelength grid is not strictly increasing.")
    if not np.all(np.diff(native_wavelength_h5) > 0.0):
        raise AssertionError("Native wavelength grid is not strictly increasing.")
    if not np.allclose(benchmark_wavelength_h5, benchmark_wavelength, rtol=0.0, atol=1.0e-12):
        raise AssertionError("Stored benchmark wavelength grid does not match the configured fixed-R grid.")

    if not np.isfinite(benchmark_noisy).all() or not np.isfinite(benchmark_noiseless).all() or not np.isfinite(native_noiseless).all():
        raise AssertionError("Non-finite values detected in spectra.")
    if not np.isfinite(native_radius_m).all() or not np.isfinite(benchmark_sigma).all():
        raise AssertionError("Non-finite values detected in radii or sigma.")

    if np.any(native_noiseless <= 0.0) or np.any(native_noiseless >= 0.1):
        raise AssertionError("Native transit depth is outside the expected physical range.")
    if np.any(benchmark_noiseless <= 0.0) or np.any(benchmark_noiseless >= 0.1):
        raise AssertionError("Benchmark noiseless transit depth is outside the expected physical range.")
    if np.any(benchmark_noisy <= 0.0) or np.any(benchmark_noisy >= 0.1):
        raise AssertionError("Benchmark noisy transit depth is outside the expected physical range.")
    if np.any(benchmark_sigma <= 0.0):
        raise AssertionError("Benchmark sigma values must be strictly positive.")

    if np.any(labels["heavy_species_mass_fraction"].to_numpy(dtype=np.float64) >= HEAVY_SPECIES_MAX_FRACTION):
        raise AssertionError("Heavy-species mass fraction exceeds the configured maximum.")
    if np.any(labels["log_g_cgs"].to_numpy(dtype=np.float64) < LOG_G_RANGE_CGS[0]) or np.any(
        labels["log_g_cgs"].to_numpy(dtype=np.float64) > LOG_G_RANGE_CGS[1]
    ):
        raise AssertionError("log_g_cgs falls outside the configured prior.")
    if np.any(labels["log10_p_cloud_bar"].to_numpy(dtype=np.float64) < CLOUD_LOG10_P_RANGE[0]) or np.any(
        labels["log10_p_cloud_bar"].to_numpy(dtype=np.float64) > CLOUD_LOG10_P_RANGE[1]
    ):
        raise AssertionError("log10_p_cloud_bar falls outside the configured prior.")
    if np.any(labels["log10_kappa0"].to_numpy(dtype=np.float64) < HAZE_LOG10_KAPPA0_RANGE[0]) or np.any(
        labels["log10_kappa0"].to_numpy(dtype=np.float64) > HAZE_LOG10_KAPPA0_RANGE[1]
    ):
        raise AssertionError("log10_kappa0 falls outside the configured prior.")
    if np.any(labels["gamma_scat"].to_numpy(dtype=np.float64) < HAZE_GAMMA_RANGE[0]) or np.any(
        labels["gamma_scat"].to_numpy(dtype=np.float64) > HAZE_GAMMA_RANGE[1]
    ):
        raise AssertionError("gamma_scat falls outside the configured prior.")

    for species in PRIMARY_TARGET_SPECIES:
        values = labels["log_X_%s" % species].to_numpy(dtype=np.float64)
        expected_presence = (values >= DETECTION_THRESHOLD_LOG_X).astype(np.int64)
        actual_presence = labels["present_%s" % species].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_presence, actual_presence):
            raise AssertionError("Presence labels for %s do not match the detection threshold." % species)

    rebin_matrix = build_rebin_matrix(native_wavelength_h5, benchmark_edges)
    reconstructed = native_noiseless.dot(rebin_matrix.T)
    max_rebin_error = float(np.max(np.abs(reconstructed - benchmark_noiseless)))
    if max_rebin_error > 5.0e-7:
        raise AssertionError("Benchmark noiseless spectra do not match native rebinning within tolerance.")

    recomputed_split = assign_splits(labels, provenance, int(_load_seed(output_root)))
    if not np.array_equal(recomputed_split.to_numpy(dtype="U16"), labels["split"].to_numpy(dtype="U16")):
        raise AssertionError("Split assignment is not reproducible from labels and provenance.")
    if not np.array_equal(labels["split"].to_numpy(dtype="U16"), provenance["split"].to_numpy(dtype="U16")):
        raise AssertionError("Split columns disagree between labels and provenance.")
    if not np.array_equal(labels["split"].to_numpy(dtype="U16"), splits_h5):
        raise AssertionError("Split ordering mismatch between spectra.h5 and labels.parquet.")

    if len(labels) >= sum(SPLIT_COUNTS.values()):
        split_counts = labels["split"].value_counts().to_dict()
        for split_name, expected_count in SPLIT_COUNTS.items():
            if int(split_counts.get(split_name, 0)) != int(expected_count):
                raise AssertionError("Canonical split %s has %d rows instead of %d." % (split_name, split_counts.get(split_name, 0), expected_count))

    if len(labels) >= 1000:
        for species in PRIMARY_TARGET_SPECIES:
            prevalence = float(labels["present_%s" % species].mean())
            if prevalence <= 0.0 or prevalence >= 1.0:
                raise AssertionError("Species %s has collapsed prevalence %.3f." % (species, prevalence))
        for split_name in ("train", "val", "test_id"):
            subset = labels.loc[labels["split"] == split_name, "temperature_regime"]
            if len(subset) > 0 and len(set(subset.tolist())) < 3:
                raise AssertionError("Split %s does not cover all three temperature regimes." % split_name)
        if int(provenance.loc[labels["split"] == "test_ood", "ood_candidate"].min()) != 1:
            raise AssertionError("test_ood contains rows that are not tagged as OOD candidates.")

    return {
        "sample_count": int(len(labels)),
        "benchmark_bins": int(benchmark_noisy.shape[1]),
        "native_bins": int(native_noiseless.shape[1]),
        "pressure_levels": PRESSURE_LEVELS,
        "max_rebin_error": max_rebin_error,
    }


def _load_seed(output_root: Path) -> int:
    manifest_path = output_root / BENCHMARK_PATHS.manifest_json
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        if "seed" in payload:
            return int(payload["seed"])
    return 10032026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_dataset(args.output_root)
    print("Validated transmission benchmark dataset:")
    for key, value in summary.items():
        print("  %s: %s" % (key, value))


if __name__ == "__main__":
    main()
