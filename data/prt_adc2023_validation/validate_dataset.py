"""Validation checks for the generated pRT-based ADC2023 validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

from .constants import ADC_AUX_COLUMNS, PAPER_PARAMETER_COLUMNS, canonical_output_instrument_width, canonical_output_wlgrid
from .physics import (
    compute_pt_summary,
    distance_pc_to_cm,
    formula_tolerances,
    orbital_period_days,
    planet_mass_kg_from_logg_and_radius,
    surface_gravity_m_s2,
)


def _read_group_keys(path: Path) -> List[str]:
    with h5py.File(path, "r") as handle:
        return sorted(handle.keys())


def _assert_close(name: str, lhs: np.ndarray, rhs: np.ndarray, atol: float) -> None:
    if not np.allclose(lhs, rhs, atol=atol):
        raise AssertionError(f"{name} mismatch: max abs diff = {np.max(np.abs(lhs - rhs))}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(output_root: Path) -> Dict[str, int]:
    validation_root = output_root / "ValidationData"
    truth_root = validation_root / "Ground Truth Package"

    spectral_path = validation_root / "SpectralData.hdf5"
    aux_path = validation_root / "AuxillaryTable.csv"
    noiseless_path = truth_root / "NoiselessSpectralData.hdf5"
    native_path = truth_root / "NativeSpectra_R400.hdf5"
    fm_path = truth_root / "FM_Parameter_Table.csv"
    manifest_path = output_root / "manifest.json"

    tolerances = formula_tolerances()
    aux = pd.read_csv(aux_path)
    fm = pd.read_csv(fm_path)
    manifest = json.loads(manifest_path.read_text())

    if list(aux.columns) != ADC_AUX_COLUMNS:
        raise AssertionError(f"Unexpected auxiliary columns: {list(aux.columns)}")

    merged = aux.merge(fm, on="planet_ID", how="inner", validate="one_to_one", suffixes=("_aux", "_truth"))
    spectral_keys = _read_group_keys(spectral_path)
    noiseless_keys = _read_group_keys(noiseless_path)
    native_keys = _read_group_keys(native_path)

    expected_keys = [f"Planet_{planet_id}" for planet_id in merged["planet_ID"].tolist()]
    if spectral_keys != expected_keys:
        raise AssertionError("SpectralData.hdf5 keys do not match FM_Parameter_Table.csv planet IDs.")
    if noiseless_keys != expected_keys or native_keys != expected_keys:
        raise AssertionError("Truth HDF5 keys do not match FM_Parameter_Table.csv planet IDs.")

    output_wlgrid = canonical_output_wlgrid()
    output_width = canonical_output_instrument_width()
    normalized_residuals: List[np.ndarray] = []
    with h5py.File(spectral_path, "r") as spectral_handle, h5py.File(noiseless_path, "r") as noiseless_handle:
        for group_name, row in merged.iterrows():
            planet_id = row["planet_ID"]
            spectral_group = spectral_handle[f"Planet_{planet_id}"]
            noiseless_group = noiseless_handle[f"Planet_{planet_id}"]

            _assert_close(
                f"{planet_id} instrument_wlgrid",
                spectral_group["instrument_wlgrid"][:],
                output_wlgrid,
                tolerances["wl_abs"],
            )
            _assert_close(
                f"{planet_id} instrument_width",
                spectral_group["instrument_width"][:],
                output_width,
                tolerances["wl_abs"],
            )
            _assert_close(
                f"{planet_id} noiseless instrument_wlgrid",
                noiseless_group["instrument_wlgrid"][:],
                output_wlgrid,
                tolerances["wl_abs"],
            )
            if spectral_group["instrument_spectrum"][:].shape != (52,):
                raise AssertionError(f"{planet_id} noisy spectrum is not length 52.")
            if spectral_group["instrument_noise"][:].shape != (52,):
                raise AssertionError(f"{planet_id} noise vector is not length 52.")
            if noiseless_group["instrument_spectrum"][:].shape != (52,):
                raise AssertionError(f"{planet_id} noiseless spectrum is not length 52.")

            noisy = spectral_group["instrument_spectrum"][:]
            noiseless = noiseless_group["instrument_spectrum"][:]
            noise = spectral_group["instrument_noise"][:]
            if not np.isfinite(noisy).all() or not np.isfinite(noiseless).all() or not np.isfinite(noise).all():
                raise AssertionError(f"{planet_id} contains NaN/Inf values.")
            if np.allclose(noiseless, 0.0):
                raise AssertionError(f"{planet_id} noiseless ADC spectrum is identically zero.")
            if not np.allclose(noise, noise[0], atol=tolerances["wl_abs"]):
                raise AssertionError(f"{planet_id} instrument_noise is not constant across ADC bins.")
            if not np.isclose(noise[0], row["sampled_sigma_scaled"], rtol=tolerances["physics_rel"]):
                raise AssertionError(f"{planet_id} stored sigma does not match instrument_noise.")
            normalized_residuals.append((noisy - noiseless) / noise)

    with h5py.File(native_path, "r") as native_handle:
        for planet_id in merged["planet_ID"]:
            group = native_handle[f"Planet_{planet_id}"]
            if group["native_wlgrid"][:].shape != (379,):
                raise AssertionError(f"{planet_id} native wlgrid is not length 379.")
            if group["native_noiseless_spectrum"][:].shape != (379,):
                raise AssertionError(f"{planet_id} native spectrum is not length 379.")

    for row in merged.itertuples(index=False):
        expected_mass = planet_mass_kg_from_logg_and_radius(row.log_g, row.r_p_r_jup)
        expected_gravity = surface_gravity_m_s2(row.log_g)
        expected_period = orbital_period_days(row.star_mass_kg, row.planet_distance)
        pt_summary = compute_pt_summary({name: getattr(row, name) for name in PAPER_PARAMETER_COLUMNS})

        if not np.isclose(row.planet_mass_kg_aux, expected_mass, rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} auxiliary planet_mass_kg does not match log_g and radius.")
        if not np.isclose(row.planet_mass_kg_truth, expected_mass, rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} truth planet_mass_kg does not match log_g and radius.")
        if not np.isclose(row.planet_surface_gravity, expected_gravity, rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} surface gravity does not match log_g.")
        if not np.isclose(row.planet_orbital_period, expected_period, rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} orbital period violates Kepler consistency.")
        if not np.isclose(row.star_distance, row.star_distance_pc, rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} star_distance and truth star_distance_pc disagree.")
        if not np.isclose(row.d_pl_cm, distance_pc_to_cm(row.star_distance), rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} d_pl_cm does not match star_distance.")
        if not np.isclose(row.t_connect, pt_summary["t_connect"], rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} stored t_connect is inconsistent.")
        if not np.isclose(row.t3, pt_summary["t3"], rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} stored t3 is inconsistent.")
        if not np.isclose(row.t2, pt_summary["t2"], rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} stored t2 is inconsistent.")
        if not np.isclose(row.t1, pt_summary["t1"], rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} stored t1 is inconsistent.")
        if not np.isclose(row.delta, pt_summary["delta"], rtol=tolerances["physics_rel"]):
            raise AssertionError(f"{row.planet_ID} stored delta is inconsistent.")
        if not (0.05 <= row.sampled_sigma_scaled <= 0.50):
            raise AssertionError(f"{row.planet_ID} sampled_sigma_scaled is outside the configured range.")
        if hasattr(row, "generation_attempt") and row.generation_attempt < 1:
            raise AssertionError(f"{row.planet_ID} generation_attempt must be >= 1.")

    expected_manifest_files = {
        "ValidationData/SpectralData.hdf5",
        "ValidationData/AuxillaryTable.csv",
        "ValidationData/Ground Truth Package/NoiselessSpectralData.hdf5",
        "ValidationData/Ground Truth Package/NativeSpectra_R400.hdf5",
        "ValidationData/Ground Truth Package/FM_Parameter_Table.csv",
    }
    if set(manifest["files"].keys()) != expected_manifest_files:
        raise AssertionError("Manifest file list is incomplete.")
    for relative_path, expected_digest in manifest["files"].items():
        actual_digest = _sha256(output_root / relative_path)
        if actual_digest != expected_digest:
            raise AssertionError(f"Manifest checksum mismatch for {relative_path}.")

    residual_z = np.concatenate(normalized_residuals)
    if abs(float(residual_z.mean())) > 0.15:
        raise AssertionError("Noise residual mean is inconsistent with zero-mean Gaussian noise.")
    if not 0.8 <= float(residual_z.std(ddof=0)) <= 1.2:
        raise AssertionError("Noise residual scatter is inconsistent with the stored sigma.")

    return {
        "sample_count": int(len(merged)),
        "spectral_groups": int(len(spectral_keys)),
        "native_groups": int(len(native_keys)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_dataset(args.output_root)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
