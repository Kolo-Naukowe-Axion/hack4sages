"""Remote-capable generator for the pRT-based ADC2023 validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import numpy as np
import pandas as pd

from .constants import (
    ADC_AUX_COLUMNS,
    CANONICAL_SAMPLE_COUNT,
    CLOUD_SPECIES,
    CONTINUUM_OPACITIES,
    DEFAULT_NEIGHBOR_COUNT,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SHARD_SIZE,
    DEFAULT_WORKERS,
    HDF5_GROUP_PREFIX,
    LINE_SPECIES_R400,
    PAPER_PARAMETER_COLUMNS,
    PAPER_PRIORS,
    PLANET_ID_PREFIX,
    PRESSURE_SCALING,
    PRESSURE_SIMPLE,
    PRESSURE_WIDTH,
    RAYLEIGH_SPECIES,
    SCALE_FACTOR,
    SIGMA_MAX_SCALED,
    SIGMA_MIN_SCALED,
    SIGMA_UNIT_W_M2_UM,
    canonical_output_instrument_width,
    canonical_output_wlgrid,
    official_binning_wlgrid_desc,
    official_binning_wlwidth_desc,
)
from .empirical_prior import EmpiricalConditionalPrior
from .physics import (
    distance_pc_to_cm,
    orbital_period_days,
    planet_mass_kg_from_logg_and_radius,
    planet_radius_m,
    surface_gravity_m_s2,
    compute_pt_summary,
)


WORKER_STATE: Dict[str, Any] = {}
MAX_GENERATION_ATTEMPTS = 64
PROGRESS_FILENAME = "progress.json"
DEFAULT_PROGRESS_UPDATE_SECONDS = 10.0


@dataclass(frozen=True)
class GeneratorConfig:
    """All runtime configuration needed by workers and assembly."""

    output_root: Path
    bundle_path: Path
    p_rt_input_data_path: Path
    sample_count: int
    shard_size: int
    seed: int
    workers: int
    neighbor_count: int
    pressure_scaling: int
    pressure_simple: int
    pressure_width: int
    skip_existing: bool
    run_started_utc: str
    progress_update_seconds: float


def _import_taurex_binner() -> Tuple[Any, np.ndarray, np.ndarray]:
    from taurex.binning import FluxBinner
    from taurex.util.util import wnwidth_to_wlwidth

    wlgrid_desc = official_binning_wlgrid_desc()
    wlwidth_desc = official_binning_wlwidth_desc()
    wngrid = 10000.0 / wlgrid_desc
    wnwidth = wnwidth_to_wlwidth(wlgrid_desc, wlwidth_desc)
    return FluxBinner(wngrid, wnwidth), wngrid, wnwidth


def _progress_path(output_root: Path) -> Path:
    return output_root / "work" / PROGRESS_FILENAME


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _elapsed_seconds(started_utc: str) -> float:
    started = datetime.fromisoformat(started_utc)
    return max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _base_progress_payload(config: GeneratorConfig) -> Dict[str, Any]:
    return {
        "output_root": str(config.output_root),
        "bundle_path": str(config.bundle_path),
        "p_rt_input_data_path": str(config.p_rt_input_data_path),
        "sample_count": int(config.sample_count),
        "shard_size": int(config.shard_size),
        "workers": int(config.workers),
        "neighbor_count": int(config.neighbor_count),
        "skip_existing": bool(config.skip_existing),
        "run_started_utc": config.run_started_utc,
        "pid": os.getpid(),
        "log_path": os.environ.get("PRT_ADC_LOG_PATH"),
    }


def _write_progress(
    config: GeneratorConfig,
    *,
    stage: str,
    generated_samples: int,
    persisted_samples: int,
    shards_completed: int,
    shards_total: int,
    current_shard_index: int | None = None,
    current_shard_ordinal: int | None = None,
    current_shard_start: int | None = None,
    current_shard_stop: int | None = None,
    current_shard_generated: int | None = None,
    message: str | None = None,
    error: str | None = None,
) -> None:
    elapsed = _elapsed_seconds(config.run_started_utc)
    rate_per_minute = (generated_samples / elapsed) * 60.0 if elapsed > 0.0 and generated_samples > 0 else None
    eta_seconds = (
        (config.sample_count - generated_samples) / (generated_samples / elapsed)
        if elapsed > 0.0 and generated_samples > 0 and generated_samples < config.sample_count
        else 0.0
        if generated_samples >= config.sample_count
        else None
    )
    payload = _base_progress_payload(config)
    payload.update(
        {
            "stage": stage,
            "last_update_utc": _utc_now_iso(),
            "elapsed_seconds": elapsed,
            "generated_samples": int(generated_samples),
            "persisted_samples": int(persisted_samples),
            "remaining_samples": int(max(config.sample_count - generated_samples, 0)),
            "generation_rate_samples_per_minute": rate_per_minute,
            "estimated_remaining_seconds": eta_seconds,
            "shards_completed": int(shards_completed),
            "shards_total": int(shards_total),
            "current_shard_index": int(current_shard_index) if current_shard_index is not None else None,
            "current_shard_ordinal": int(current_shard_ordinal) if current_shard_ordinal is not None else None,
            "current_shard_start": int(current_shard_start) if current_shard_start is not None else None,
            "current_shard_stop": int(current_shard_stop) if current_shard_stop is not None else None,
            "current_shard_generated": int(current_shard_generated) if current_shard_generated is not None else None,
            "message": message,
            "error": error,
        }
    )
    _atomic_write_json(_progress_path(config.output_root), payload)


def _mark_progress_failed(config: GeneratorConfig, exc: BaseException) -> None:
    progress_path = _progress_path(config.output_root)
    payload = _base_progress_payload(config)
    if progress_path.exists():
        payload.update(json.loads(progress_path.read_text()))
    payload.update(
        {
            "stage": "failed",
            "last_update_utc": _utc_now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    _atomic_write_json(progress_path, payload)


def _resolve_cloud_parameter_names(
    model_module: Any,
    atmosphere: Any,
    distance_cm: float,
    pressure_scaling: int,
    pressure_simple: int,
    pressure_width: int,
) -> Tuple[str, str]:
    """Resolve the exact eq-scaling parameter names accepted by pRT 2.6.7."""

    import petitRADTRANS.retrieval.parameter as prm

    base_parameters = {
        "C/O": 0.55,
        "Fe/H": 0.0,
        "log_pquench": -5.0,
        "fsed": 3.0,
        "log_kzz": 8.5,
        "sigma_lnorm": 2.0,
        "log_g": 3.75,
        "R_pl": planet_radius_m(1.0),
        "T_int": 1063.6,
        "T3": 0.26,
        "T2": 0.29,
        "T1": 0.32,
        "alpha": 1.39,
        "log_delta": 0.48,
        "D_pl": distance_cm,
        "pressure_scaling": pressure_scaling,
        "pressure_simple": pressure_simple,
        "pressure_width": pressure_width,
    }
    candidates = [
        ("eq_scaling_Fe(c)", "eq_scaling_MgSiO3(c)"),
        ("eq_scaling_Fe", "eq_scaling_MgSiO3"),
        ("log_X_cb_Fe(c)", "log_X_cb_MgSiO3(c)"),
    ]

    for fe_name, mg_name in candidates:
        trial = dict(base_parameters)
        trial[fe_name] = -0.86
        trial[mg_name] = -0.65
        parameters = {
            key: prm.Parameter(name=key, value=value, is_free_parameter=False)
            for key, value in trial.items()
        }
        try:
            model_module.emission_model_diseq(atmosphere, parameters, AMR=True)
            return fe_name, mg_name
        except Exception:
            continue

    raise RuntimeError("Unable to resolve the cloud scaling parameter names for emission_model_diseq().")


def _init_worker(
    bundle_path: str,
    p_rt_input_data_path: str,
    neighbor_count: int,
    pressure_scaling: int,
    pressure_simple: int,
    pressure_width: int,
) -> None:
    os.environ["pRT_input_data_path"] = p_rt_input_data_path
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prtval")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import petitRADTRANS as prt
    import petitRADTRANS.retrieval.models as models
    import petitRADTRANS.retrieval.parameter as prm

    prior = EmpiricalConditionalPrior(Path(bundle_path))
    binner, wngrid, wnwidth = _import_taurex_binner()

    atmosphere = prt.Radtrans(
        line_species=LINE_SPECIES_R400,
        cloud_species=CLOUD_SPECIES,
        rayleigh_species=RAYLEIGH_SPECIES,
        continuum_opacities=CONTINUUM_OPACITIES,
        wlen_bords_micron=[0.95, 2.45],
        do_scat_emis=True,
    )
    levels = pressure_simple + len(atmosphere.cloud_species) * (pressure_scaling - 1) * pressure_width
    atmosphere.setup_opa_structure(np.logspace(-6, 3, levels))

    fe_name, mg_name = _resolve_cloud_parameter_names(
        models,
        atmosphere,
        distance_pc_to_cm(10.0),
        pressure_scaling,
        pressure_simple,
        pressure_width,
    )

    WORKER_STATE.clear()
    WORKER_STATE.update(
        {
            "prior": prior,
            "atmosphere": atmosphere,
            "models": models,
            "parameter_cls": prm.Parameter,
            "binner": binner,
            "wngrid": wngrid,
            "wnwidth": wnwidth,
            "fe_cloud_name": fe_name,
            "mg_cloud_name": mg_name,
            "output_wlgrid": canonical_output_wlgrid(),
            "output_instrument_width": canonical_output_instrument_width(),
            "neighbor_count": int(neighbor_count),
            "pressure_scaling": int(pressure_scaling),
            "pressure_simple": int(pressure_simple),
            "pressure_width": int(pressure_width),
        }
    )


def _sample_theta(rng: np.random.Generator) -> Dict[str, float]:
    return {prior.name: float(rng.uniform(prior.lower, prior.upper)) for prior in PAPER_PRIORS}


def _planet_id(sample_index: int) -> str:
    return f"{PLANET_ID_PREFIX}{sample_index + 1:05d}"


def _build_prt_parameter_dict(theta: Dict[str, float], distance_cm: float) -> Dict[str, float]:
    params = {
        "C/O": theta["c_o"],
        "Fe/H": theta["fe_h"],
        "log_pquench": theta["log_p_quench"],
        WORKER_STATE["fe_cloud_name"]: theta["s_eq_fe"],
        WORKER_STATE["mg_cloud_name"]: theta["s_eq_mgsio3"],
        "fsed": theta["f_sed"],
        "log_kzz": theta["log_kzz"],
        "sigma_lnorm": theta["sigma_g"],
        "log_g": theta["log_g"],
        "R_pl": planet_radius_m(theta["r_p_r_jup"]),
        "T_int": theta["t_int"],
        "T3": theta["t3_unit"],
        "T2": theta["t2_unit"],
        "T1": theta["t1_unit"],
        "alpha": theta["alpha"],
        "log_delta": theta["log_delta_unit"],
        "D_pl": distance_cm,
        "pressure_scaling": WORKER_STATE["pressure_scaling"],
        "pressure_simple": WORKER_STATE["pressure_simple"],
        "pressure_width": WORKER_STATE["pressure_width"],
    }
    return params


def _run_prt_spectrum(theta: Dict[str, float], distance_pc: float) -> Tuple[np.ndarray, np.ndarray]:
    parameter_cls = WORKER_STATE["parameter_cls"]
    models = WORKER_STATE["models"]
    atmosphere = WORKER_STATE["atmosphere"]

    distance_cm = distance_pc_to_cm(distance_pc)
    raw_parameters = _build_prt_parameter_dict(theta, distance_cm)
    parameters = {
        name: parameter_cls(name=name, value=value, is_free_parameter=False)
        for name, value in raw_parameters.items()
    }
    native_wavelengths, native_flux = models.emission_model_diseq(atmosphere, parameters, AMR=True)
    native_wavelengths = np.asarray(native_wavelengths, dtype=np.float64)
    native_flux = np.asarray(native_flux, dtype=np.float64) * SCALE_FACTOR
    return native_wavelengths, native_flux


def _bin_native_spectrum(native_wavelengths: np.ndarray, native_flux: np.ndarray) -> np.ndarray:
    native_wavenumbers = 10000.0 / np.asarray(native_wavelengths, dtype=np.float64)
    binner_output = WORKER_STATE["binner"].bindown(native_wavenumbers, native_flux)
    binned_flux = np.asarray(binner_output[1], dtype=np.float64)
    if binned_flux.shape != (52,):
        raise RuntimeError(f"Expected 52 ADC bins, got {binned_flux.shape}")
    return binned_flux[::-1].copy()


def _generate_one(task: Tuple[int, int]) -> Dict[str, Any]:
    sample_index, base_seed = task
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, sample_index]))
    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        theta = _sample_theta(rng)
        pt_summary = compute_pt_summary(theta)
        planet_mass_kg = planet_mass_kg_from_logg_and_radius(theta["log_g"], theta["r_p_r_jup"])
        planet_surface_gravity = surface_gravity_m_s2(theta["log_g"])

        empirical = WORKER_STATE["prior"].sample(
            planet_mass_kg=planet_mass_kg,
            planet_surface_gravity=planet_surface_gravity,
            planet_radius_r_jup=theta["r_p_r_jup"],
            t_connect_k=pt_summary["t_connect"],
            neighbor_count=WORKER_STATE["neighbor_count"],
            rng=rng,
        )

        star_distance_pc = empirical.aux_values["star_distance"]
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                native_wl, native_flux = _run_prt_spectrum(theta, star_distance_pc)
                binned_noiseless = _bin_native_spectrum(native_wl, native_flux)
        except Exception:
            continue

        warning_messages = [str(warning.message) for warning in caught_warnings]
        if any("Cloud rescaling lead to nan opacities" in message for message in warning_messages):
            continue
        if (
            not np.isfinite(native_flux).all()
            or not np.isfinite(binned_noiseless).all()
            or np.allclose(binned_noiseless, 0.0)
        ):
            continue

        sigma_scaled = float(rng.uniform(SIGMA_MIN_SCALED, SIGMA_MAX_SCALED))
        sigma_unscaled = sigma_scaled * SIGMA_UNIT_W_M2_UM
        instrument_noise = np.full(52, sigma_scaled, dtype=np.float64)
        noisy_binned = binned_noiseless + rng.normal(0.0, sigma_scaled, size=52)

        aux_row = {
            "planet_ID": _planet_id(sample_index),
            "star_distance": float(star_distance_pc),
            "star_mass_kg": float(empirical.aux_values["star_mass_kg"]),
            "star_radius_m": float(empirical.aux_values["star_radius_m"]),
            "star_temperature": float(empirical.aux_values["star_temperature"]),
            "planet_mass_kg": float(planet_mass_kg),
            "planet_distance": float(empirical.aux_values["planet_distance"]),
            "planet_orbital_period": float(
                orbital_period_days(
                    empirical.aux_values["star_mass_kg"],
                    empirical.aux_values["planet_distance"],
                )
            ),
            "planet_surface_gravity": float(planet_surface_gravity),
        }

        truth_row = {
            "planet_ID": aux_row["planet_ID"],
            **{name: theta[name] for name in PAPER_PARAMETER_COLUMNS},
            "t_connect": pt_summary["t_connect"],
            "t3": pt_summary["t3"],
            "t2": pt_summary["t2"],
            "t1": pt_summary["t1"],
            "delta": pt_summary["delta"],
            "sampled_sigma_scaled": sigma_scaled,
            "sampled_sigma_w_m2_um": sigma_unscaled,
            "star_distance_pc": aux_row["star_distance"],
            "d_pl_cm": distance_pc_to_cm(aux_row["star_distance"]),
            "planet_mass_kg": aux_row["planet_mass_kg"],
            "planet_surface_gravity_m_s2": aux_row["planet_surface_gravity"],
            "planet_radius_r_jup": theta["r_p_r_jup"],
            "planet_radius_m": planet_radius_m(theta["r_p_r_jup"]),
            "generation_attempt": attempt,
            "selected_empirical_row_index": empirical.row_index,
            "selected_empirical_planet_id": empirical.planet_id,
            "selected_empirical_rank": empirical.rank,
            "selected_empirical_distance": empirical.distance,
        }

        return {
            "planet_id": aux_row["planet_ID"],
            "native_wavelengths": native_wl,
            "native_noiseless_flux": native_flux,
            "binned_noiseless_flux": binned_noiseless,
            "binned_noisy_flux": noisy_binned,
            "instrument_noise": instrument_noise,
            "aux_row": aux_row,
            "truth_row": truth_row,
        }

    raise RuntimeError(
        f"Unable to generate a finite spectrum for sample_index={sample_index} after "
        f"{MAX_GENERATION_ATTEMPTS} attempts."
    )


def _shard_ranges(sample_count: int, shard_size: int) -> List[Tuple[int, int, int]]:
    shard_count = math.ceil(sample_count / shard_size)
    ranges: List[Tuple[int, int, int]] = []
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        stop = min(sample_count, start + shard_size)
        ranges.append((shard_index, start, stop))
    return ranges


def _write_shard(shard_path: Path, records: List[Dict[str, Any]]) -> None:
    payload: Dict[str, Any] = {
        "planet_id": np.array([record["planet_id"] for record in records], dtype="U16"),
        "native_wavelengths": np.stack([record["native_wavelengths"] for record in records], axis=0),
        "native_noiseless_flux": np.stack([record["native_noiseless_flux"] for record in records], axis=0),
        "binned_noiseless_flux": np.stack([record["binned_noiseless_flux"] for record in records], axis=0),
        "binned_noisy_flux": np.stack([record["binned_noisy_flux"] for record in records], axis=0),
        "instrument_noise": np.stack([record["instrument_noise"] for record in records], axis=0),
        "aux_rows_json": np.array(
            [json.dumps(record["aux_row"], sort_keys=True) for record in records],
            dtype="U4096",
        ),
        "truth_rows_json": np.array(
            [json.dumps(record["truth_row"], sort_keys=True) for record in records],
            dtype="U8192",
        ),
    }
    np.savez_compressed(shard_path, **payload)


def generate_shards(config: GeneratorConfig) -> List[Path]:
    """Generate all shard files, skipping existing ones when requested."""

    work_root = config.output_root / "work"
    shard_root = work_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_paths: List[Path] = []
    shard_ranges = _shard_ranges(config.sample_count, config.shard_size)
    shards_total = len(shard_ranges)
    persisted_samples = 0
    shards_completed = 0
    for shard_index, start, stop in shard_ranges:
        shard_path = shard_root / f"shard_{shard_index:05d}.npz"
        if config.skip_existing and shard_path.exists():
            persisted_samples += stop - start
            shards_completed += 1

    _write_progress(
        config,
        stage="generating",
        generated_samples=persisted_samples,
        persisted_samples=persisted_samples,
        shards_completed=shards_completed,
        shards_total=shards_total,
        message="generation initialized",
    )

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=config.workers,
        initializer=_init_worker,
        initargs=(
            str(config.bundle_path),
            str(config.p_rt_input_data_path),
            config.neighbor_count,
            config.pressure_scaling,
            config.pressure_simple,
            config.pressure_width,
        ),
    ) as pool:
        persisted_samples = 0
        shards_completed = 0
        for ordinal, (shard_index, start, stop) in enumerate(shard_ranges, start=1):
            shard_path = shard_root / f"shard_{shard_index:05d}.npz"
            shard_paths.append(shard_path)
            if config.skip_existing and shard_path.exists():
                print(
                    f"[generate] skipping existing shard {ordinal}/{len(shard_ranges)}: "
                    f"{shard_path.name} ({start}:{stop})",
                    flush=True,
                )
                persisted_samples += stop - start
                shards_completed += 1
                _write_progress(
                    config,
                    stage="generating",
                    generated_samples=persisted_samples,
                    persisted_samples=persisted_samples,
                    shards_completed=shards_completed,
                    shards_total=shards_total,
                    current_shard_index=shard_index,
                    current_shard_ordinal=ordinal,
                    current_shard_start=start,
                    current_shard_stop=stop,
                    current_shard_generated=stop - start,
                    message=f"skipped existing {shard_path.name}",
                )
                continue

            print(
                f"[generate] shard {ordinal}/{len(shard_ranges)} -> {shard_path.name} ({start}:{stop})",
                flush=True,
            )
            tasks = [(sample_index, config.seed) for sample_index in range(start, stop)]
            records: List[Dict[str, Any]] = []
            last_progress_update = 0.0
            _write_progress(
                config,
                stage="generating",
                generated_samples=persisted_samples,
                persisted_samples=persisted_samples,
                shards_completed=shards_completed,
                shards_total=shards_total,
                current_shard_index=shard_index,
                current_shard_ordinal=ordinal,
                current_shard_start=start,
                current_shard_stop=stop,
                current_shard_generated=0,
                message=f"running {shard_path.name}",
            )
            for completed_in_shard, record in enumerate(pool.imap(_generate_one, tasks, chunksize=4), start=1):
                records.append(record)
                now = time.monotonic()
                if (
                    completed_in_shard == 1
                    or completed_in_shard == (stop - start)
                    or now - last_progress_update >= config.progress_update_seconds
                ):
                    _write_progress(
                        config,
                        stage="generating",
                        generated_samples=persisted_samples + completed_in_shard,
                        persisted_samples=persisted_samples,
                        shards_completed=shards_completed,
                        shards_total=shards_total,
                        current_shard_index=shard_index,
                        current_shard_ordinal=ordinal,
                        current_shard_start=start,
                        current_shard_stop=stop,
                        current_shard_generated=completed_in_shard,
                        message=f"running {shard_path.name}",
                    )
                    last_progress_update = now
            _write_shard(shard_path, records)
            persisted_samples += len(records)
            shards_completed += 1
            _write_progress(
                config,
                stage="generating",
                generated_samples=persisted_samples,
                persisted_samples=persisted_samples,
                shards_completed=shards_completed,
                shards_total=shards_total,
                current_shard_index=shard_index,
                current_shard_ordinal=ordinal,
                current_shard_start=start,
                current_shard_stop=stop,
                current_shard_generated=len(records),
                message=f"wrote {shard_path.name}",
            )

    return shard_paths


def _load_shard_records(shard_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for shard_path in sorted(shard_paths):
        payload = np.load(shard_path, allow_pickle=False)
        for index, planet_id in enumerate(payload["planet_id"].tolist()):
            records.append(
                {
                    "planet_id": str(planet_id),
                    "native_wavelengths": payload["native_wavelengths"][index],
                    "native_noiseless_flux": payload["native_noiseless_flux"][index],
                    "binned_noiseless_flux": payload["binned_noiseless_flux"][index],
                    "binned_noisy_flux": payload["binned_noisy_flux"][index],
                    "instrument_noise": payload["instrument_noise"][index],
                    "aux_row": json.loads(str(payload["aux_rows_json"][index])),
                    "truth_row": json.loads(str(payload["truth_rows_json"][index])),
                }
            )
    return records


def _write_final_hdf5_spectral(path: Path, records: List[Dict[str, Any]], key: str) -> None:
    wlgrid = canonical_output_wlgrid()
    instrument_width = canonical_output_instrument_width()
    with h5py.File(path, "w") as handle:
        for record in records:
            group = handle.create_group(f"{HDF5_GROUP_PREFIX}{record['planet_id']}")
            group.create_dataset("instrument_wlgrid", data=wlgrid)
            group.create_dataset("instrument_width", data=instrument_width)
            group.create_dataset("instrument_noise", data=record["instrument_noise"])
            group.create_dataset("instrument_spectrum", data=record[key])


def _write_native_hdf5(path: Path, records: List[Dict[str, Any]]) -> None:
    with h5py.File(path, "w") as handle:
        for record in records:
            group = handle.create_group(f"{HDF5_GROUP_PREFIX}{record['planet_id']}")
            group.create_dataset("native_wlgrid", data=record["native_wavelengths"])
            group.create_dataset("native_noiseless_spectrum", data=record["native_noiseless_flux"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_versions(p_rt_input_data_path: Path) -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "h5py": h5py.__version__,
    }

    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        pass

    try:
        os.environ.setdefault("pRT_input_data_path", str(p_rt_input_data_path))
        import petitRADTRANS

        versions["petitRADTRANS"] = petitRADTRANS.__version__
    except Exception:
        pass

    try:
        import taurex

        versions["taurex"] = taurex.__version__
    except Exception:
        pass

    return versions


def assemble_dataset(config: GeneratorConfig, shard_paths: Iterable[Path]) -> Dict[str, Path]:
    """Assemble shard outputs into the final ValidationData directory."""

    validation_root = config.output_root / "ValidationData"
    truth_root = validation_root / "Ground Truth Package"
    validation_root.mkdir(parents=True, exist_ok=True)
    truth_root.mkdir(parents=True, exist_ok=True)

    records = _load_shard_records(shard_paths)
    records.sort(key=lambda record: record["planet_id"])
    if len(records) != config.sample_count:
        raise RuntimeError(f"Expected {config.sample_count} records, found {len(records)}")

    print(f"[assemble] writing final dataset with {len(records)} records", flush=True)
    _write_progress(
        config,
        stage="assembling",
        generated_samples=len(records),
        persisted_samples=len(records),
        shards_completed=math.ceil(config.sample_count / config.shard_size),
        shards_total=math.ceil(config.sample_count / config.shard_size),
        message="assembling final ValidationData outputs",
    )

    spectral_path = validation_root / "SpectralData.hdf5"
    aux_path = validation_root / "AuxillaryTable.csv"
    noiseless_path = truth_root / "NoiselessSpectralData.hdf5"
    native_path = truth_root / "NativeSpectra_R400.hdf5"
    fm_path = truth_root / "FM_Parameter_Table.csv"
    manifest_path = config.output_root / "manifest.json"

    _write_final_hdf5_spectral(spectral_path, records, "binned_noisy_flux")
    _write_final_hdf5_spectral(noiseless_path, records, "binned_noiseless_flux")
    _write_native_hdf5(native_path, records)

    aux_rows = pd.DataFrame([record["aux_row"] for record in records], columns=ADC_AUX_COLUMNS)
    aux_rows.to_csv(aux_path, index=False)

    truth_rows = pd.DataFrame([record["truth_row"] for record in records])
    truth_rows.to_csv(fm_path, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": config.sample_count,
        "scale_factor": SCALE_FACTOR,
        "sigma_scaled_min": SIGMA_MIN_SCALED,
        "sigma_scaled_max": SIGMA_MAX_SCALED,
        "sigma_unit_w_m2_um": SIGMA_UNIT_W_M2_UM,
        "official_sources": {
            "paper_2025": "https://arxiv.org/html/2410.21477v1",
            "paper_2023": "https://arxiv.org/abs/2301.06575",
            "adc2023_baseline": "https://github.com/ucl-exoplanets/ADC2023-baseline",
            "petit_radtrans_docs": "https://petitradtrans.readthedocs.io/en/latest/content/installation.html",
            "sbi_ear_repo": "https://github.com/francois-rozet/sbi-ear",
            "sbi_ear_input_data_bundle": "https://keeper.mpdl.mpg.de/f/78b3c66857924b5aacdd/?dl=1",
        },
        "software_versions": _runtime_versions(config.p_rt_input_data_path),
        "input_data": {
            "path": str(config.p_rt_input_data_path),
            "opacity_resolution": 400,
            "opacity_rebin_pattern": "authors_rebin_py",
        },
        "reference_bundle_sha256": _sha256(config.bundle_path),
        "line_species_r400": list(LINE_SPECIES_R400),
        "cloud_species": list(CLOUD_SPECIES),
        "canonical_output_wlgrid": canonical_output_wlgrid().tolist(),
        "canonical_output_instrument_width": canonical_output_instrument_width().tolist(),
        "official_binning_wlgrid_desc": official_binning_wlgrid_desc().tolist(),
        "official_binning_wlwidth_desc": official_binning_wlwidth_desc().tolist(),
        "parameter_priors": [{k: getattr(prior, k) for k in ("name", "lower", "upper")} for prior in PAPER_PRIORS],
        "assumptions": {
            "pRT_truth_authoritative": True,
            "aux_star_orbit_fields_empirical_covariates": True,
            "star_distance_flux_coupled": True,
            "output_instrument_width_from_local_adc_hdf5": True,
            "official_binning_from_adc2023_baseline_taurex_fluxbinner": True,
        },
        "files": {
            "ValidationData/SpectralData.hdf5": _sha256(spectral_path),
            "ValidationData/AuxillaryTable.csv": _sha256(aux_path),
            "ValidationData/Ground Truth Package/NoiselessSpectralData.hdf5": _sha256(noiseless_path),
            "ValidationData/Ground Truth Package/NativeSpectra_R400.hdf5": _sha256(native_path),
            "ValidationData/Ground Truth Package/FM_Parameter_Table.csv": _sha256(fm_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    _write_progress(
        config,
        stage="completed",
        generated_samples=len(records),
        persisted_samples=len(records),
        shards_completed=math.ceil(config.sample_count / config.shard_size),
        shards_total=math.ceil(config.sample_count / config.shard_size),
        message="generation and assembly completed",
    )

    return {
        "spectral_path": spectral_path,
        "aux_path": aux_path,
        "noiseless_path": noiseless_path,
        "native_path": native_path,
        "fm_path": fm_path,
        "manifest_path": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--p-rt-input-data-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=CANONICAL_SAMPLE_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--neighbor-count", type=int, default=DEFAULT_NEIGHBOR_COUNT)
    parser.add_argument("--pressure-scaling", type=int, default=PRESSURE_SCALING)
    parser.add_argument("--pressure-simple", type=int, default=PRESSURE_SIMPLE)
    parser.add_argument("--pressure-width", type=int, default=PRESSURE_WIDTH)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--progress-update-seconds", type=float, default=DEFAULT_PROGRESS_UPDATE_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prtval")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    config = GeneratorConfig(
        output_root=args.output_root,
        bundle_path=args.bundle_path,
        p_rt_input_data_path=args.p_rt_input_data_path,
        sample_count=args.sample_count,
        shard_size=args.shard_size,
        seed=args.seed,
        workers=args.workers,
        neighbor_count=args.neighbor_count,
        pressure_scaling=args.pressure_scaling,
        pressure_simple=args.pressure_simple,
        pressure_width=args.pressure_width,
        skip_existing=args.skip_existing,
        run_started_utc=_utc_now_iso(),
        progress_update_seconds=args.progress_update_seconds,
    )

    try:
        if args.assemble_only:
            shard_root = config.output_root / "work" / "shards"
            shard_paths = sorted(shard_root.glob("shard_*.npz"))
        else:
            shard_paths = generate_shards(config)

        assemble_dataset(config, shard_paths)
    except Exception as exc:
        _mark_progress_failed(config, exc)
        raise


if __name__ == "__main__":
    main()
