"""Generate a transmission benchmark dataset with petitRADTRANS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from .constants import (
    BENCHMARK_RESOLUTION,
    CLOUD_LOG10_P_RANGE,
    CONTINUUM_OPACITIES,
    DatasetPaths,
    DEFAULT_PROGRESS_UPDATE_SECONDS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SHARD_SIZE,
    DEFAULT_WORKER_CANDIDATES,
    DETECTION_THRESHOLD_LOG_X,
    HAZE_GAMMA_RANGE,
    HAZE_LOG10_KAPPA0_RANGE,
    HDF5_CHUNK_ROWS,
    LINE_SPECIES_R400,
    LOG_G_RANGE_CGS,
    MAX_GENERATION_ATTEMPTS,
    MIN_SEMIMAJOR_AXIS_TO_STELLAR_RADIUS,
    NATIVE_RESOLUTION,
    NOISE_SLOPE_FRACTION_RANGE,
    NOISE_WHITE_PPM_RANGE,
    PLANET_ID_PREFIX,
    PLANET_RADIUS_RANGE_RJUP,
    PRESSURE_LEVELS,
    PRESSURE_MAX_BAR,
    PRESSURE_MIN_BAR,
    PRIMARY_TARGET_SPECIES,
    RAYLEIGH_SPECIES,
    REFERENCE_PRESSURE_BAR,
    SOURCE_SPECIES,
    SPLIT_COUNTS,
    SYSTEMATIC_AMPLITUDE_PPM_RANGE,
    SYSTEMATIC_PERIOD_OCTAVES_RANGE,
    WAVELENGTH_MAX_UM,
    WAVELENGTH_MIN_UM,
)
from .grid import build_rebin_matrix, make_log_resolving_power_grid
from .physics import (
    LIGHT_SPEED_CGS,
    build_mass_fraction_profile,
    is_ood_candidate,
    make_species_presence_labels,
    planet_mass_kg_from_logg_and_radius,
    planet_radius_m,
    ppm_to_transit_depth,
    sample_log_abundances,
    sample_star_and_orbit,
    sample_temperature_regime,
    stable_hash_u64,
    star_radius_m,
    surface_gravity_cgs,
    surface_gravity_m_s2,
)


WORKER_STATE: Dict[str, Any] = {}
PROGRESS_FILENAME = "progress.json"
BENCHMARK_PATHS = DatasetPaths()


@dataclass(frozen=True)
class GeneratorConfig:
    output_root: Path
    p_rt_input_data_path: Path
    sample_count: int
    shard_size: int
    seed: int
    workers: int
    skip_existing: bool
    run_started_utc: str
    progress_update_seconds: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress_path(output_root: Path) -> Path:
    return output_root / "work" / PROGRESS_FILENAME


def _base_progress_payload(config: GeneratorConfig) -> Dict[str, Any]:
    return {
        "output_root": str(config.output_root),
        "p_rt_input_data_path": str(config.p_rt_input_data_path),
        "sample_count": int(config.sample_count),
        "shard_size": int(config.shard_size),
        "workers": int(config.workers),
        "skip_existing": bool(config.skip_existing),
        "run_started_utc": config.run_started_utc,
        "pid": os.getpid(),
        "log_path": os.environ.get("PRT_BENCH_LOG_PATH"),
    }


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / ("." + path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _elapsed_seconds(started_utc: str) -> float:
    started = datetime.fromisoformat(started_utc)
    return max((datetime.now(timezone.utc) - started).total_seconds(), 0.0)


def _write_progress(
    config: GeneratorConfig,
    *,
    stage: str,
    generated_samples: int,
    persisted_samples: int,
    shards_completed: int,
    shards_total: int,
    current_shard_index: Optional[int] = None,
    current_shard_ordinal: Optional[int] = None,
    current_shard_start: Optional[int] = None,
    current_shard_stop: Optional[int] = None,
    current_shard_generated: Optional[int] = None,
    resample_reason_counts: Optional[Dict[str, int]] = None,
    message: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    elapsed = _elapsed_seconds(config.run_started_utc)
    rate_per_minute = None
    eta_seconds = None
    if elapsed > 0.0 and generated_samples > 0:
        rate_per_minute = (float(generated_samples) / elapsed) * 60.0
        if generated_samples < config.sample_count:
            eta_seconds = (config.sample_count - generated_samples) / (generated_samples / elapsed)
        else:
            eta_seconds = 0.0

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
            "current_shard_index": current_shard_index,
            "current_shard_ordinal": current_shard_ordinal,
            "current_shard_start": current_shard_start,
            "current_shard_stop": current_shard_stop,
            "current_shard_generated": current_shard_generated,
            "resample_reason_counts": dict(sorted((resample_reason_counts or {}).items())),
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
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    )
    _atomic_write_json(progress_path, payload)


def _planet_id(sample_index: int) -> str:
    return "%s%06d" % (PLANET_ID_PREFIX, sample_index + 1)


def _resolve_default_workers() -> int:
    cpu_count = os.cpu_count() or min(DEFAULT_WORKER_CANDIDATES)
    return int(min(max(cpu_count - 2, 1), max(DEFAULT_WORKER_CANDIDATES)))


def _native_wavelengths_um(atmosphere: Any) -> Tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(LIGHT_SPEED_CGS / atmosphere.freq / 1.0e-4, dtype=np.float64)
    order = np.argsort(wavelengths)
    return wavelengths[order], order


def _build_prt_abundances(sampled_abundances: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    prt_abundances = {
        LINE_SPECIES_R400[species]: values for species, values in sampled_abundances.items() if species in LINE_SPECIES_R400
    }
    prt_abundances["H2"] = sampled_abundances["H2"]
    prt_abundances["He"] = sampled_abundances["He"]
    return prt_abundances


def _build_noise_model(
    benchmark_wavelength_um: np.ndarray,
    noiseless_depth: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    white_noise_ppm = float(rng.uniform(*NOISE_WHITE_PPM_RANGE))
    slope_fraction = float(rng.uniform(*NOISE_SLOPE_FRACTION_RANGE))
    slope_sign = float(rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64)))
    systematic_amp_ppm = float(rng.uniform(*SYSTEMATIC_AMPLITUDE_PPM_RANGE))
    systematic_phase_rad = float(rng.uniform(0.0, 2.0 * math.pi))
    systematic_period_octaves = float(rng.uniform(*SYSTEMATIC_PERIOD_OCTAVES_RANGE))

    x = np.linspace(-1.0, 1.0, len(benchmark_wavelength_um), dtype=np.float64)
    sigma_ppm = white_noise_ppm * (1.0 + slope_sign * slope_fraction * x)
    sigma_ppm = np.clip(sigma_ppm, white_noise_ppm * 0.15, None)

    log_wave = np.log2(benchmark_wavelength_um / benchmark_wavelength_um[0])
    systematic_ppm = systematic_amp_ppm * np.sin((2.0 * math.pi * log_wave / systematic_period_octaves) + systematic_phase_rad)
    noise_depth = rng.normal(loc=0.0, scale=ppm_to_transit_depth(sigma_ppm), size=len(benchmark_wavelength_um))
    noisy_depth = noiseless_depth + ppm_to_transit_depth(systematic_ppm) + noise_depth

    metadata = {
        "white_noise_ppm": white_noise_ppm,
        "noise_slope_fraction": slope_fraction,
        "noise_slope_sign": slope_sign,
        "systematic_amplitude_ppm": systematic_amp_ppm,
        "systematic_phase_rad": systematic_phase_rad,
        "systematic_period_octaves": systematic_period_octaves,
    }
    return noisy_depth.astype(np.float64), ppm_to_transit_depth(sigma_ppm).astype(np.float64), metadata


def _record_resample(counter: Counter, reason: str) -> None:
    counter[reason] += 1


def _generate_one(task: Tuple[int, int]) -> Dict[str, Any]:
    sample_index, base_seed = task
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, sample_index]))
    pressures_bar = WORKER_STATE["pressures_bar"]
    temperature_profile = np.empty(len(pressures_bar), dtype=np.float64)
    resample_counts = Counter()

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        temperature_regime, terminator_temperature_k = sample_temperature_regime(rng)
        planet_radius_r_jup = float(rng.uniform(*PLANET_RADIUS_RANGE_RJUP))
        log_g_cgs = float(rng.uniform(*LOG_G_RANGE_CGS))
        log_abundances = sample_log_abundances(rng, temperature_regime)

        try:
            abundances_short, mmw, heavy_fraction = build_mass_fraction_profile(log_abundances, len(pressures_bar))
        except ValueError:
            _record_resample(resample_counts, "heavy_fraction")
            continue

        try:
            star_system = sample_star_and_orbit(rng, terminator_temperature_k, MIN_SEMIMAJOR_AXIS_TO_STELLAR_RADIUS)
        except RuntimeError:
            _record_resample(resample_counts, "star_orbit")
            continue

        temperature_profile.fill(terminator_temperature_k)
        atmosphere = WORKER_STATE["atmosphere"]
        cloud_log10_p = float(rng.uniform(*CLOUD_LOG10_P_RANGE))
        haze_log10_kappa0 = float(rng.uniform(*HAZE_LOG10_KAPPA0_RANGE))
        haze_gamma = float(rng.uniform(*HAZE_GAMMA_RANGE))
        prt_abundances = _build_prt_abundances(abundances_short)

        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                atmosphere.calc_transm(
                    temp=temperature_profile,
                    abunds=prt_abundances,
                    gravity=surface_gravity_cgs(log_g_cgs),
                    mmw=mmw,
                    P0_bar=REFERENCE_PRESSURE_BAR,
                    R_pl=planet_radius_m(planet_radius_r_jup) * 100.0,
                    Pcloud=10.0 ** cloud_log10_p,
                    kappa_zero=10.0 ** haze_log10_kappa0,
                    gamma_scat=haze_gamma,
                )
        except Exception:
            _record_resample(resample_counts, "prt_exception")
            continue

        if caught_warnings:
            warning_messages = [str(w.message) for w in caught_warnings]
            if any("nan" in message.lower() or "invalid" in message.lower() for message in warning_messages):
                _record_resample(resample_counts, "prt_warning")
                continue

        native_radius_cm = np.asarray(atmosphere.transm_rad, dtype=np.float64)[WORKER_STATE["native_order"]]
        native_radius_m = native_radius_cm / 100.0
        star_radius_value_m = float(star_system["star_radius_m"])
        native_depth = (native_radius_m / star_radius_value_m) ** 2
        benchmark_noiseless = WORKER_STATE["rebin_matrix"].dot(native_depth)
        benchmark_noisy, sigma_1sigma, noise_metadata = _build_noise_model(
            WORKER_STATE["benchmark_wavelength_um"],
            benchmark_noiseless,
            rng,
        )

        if (
            not np.isfinite(native_radius_m).all()
            or not np.isfinite(native_depth).all()
            or not np.isfinite(benchmark_noiseless).all()
            or not np.isfinite(benchmark_noisy).all()
            or not np.isfinite(sigma_1sigma).all()
        ):
            _record_resample(resample_counts, "nonfinite")
            continue

        if (
            np.any(native_depth <= 0.0)
            or np.any(native_depth >= 0.1)
            or np.any(benchmark_noiseless <= 0.0)
            or np.any(benchmark_noiseless >= 0.1)
            or np.any(benchmark_noisy <= 0.0)
            or np.any(benchmark_noisy >= 0.1)
        ):
            _record_resample(resample_counts, "depth_bounds")
            continue

        sample_id = _planet_id(sample_index)
        planet_mass_kg = planet_mass_kg_from_logg_and_radius(log_g_cgs, planet_radius_r_jup)
        label_row = {
            "sample_id": sample_id,
            "planet_radius_ref_m": planet_radius_m(planet_radius_r_jup),
            "planet_radius_ref_r_jup": planet_radius_r_jup,
            "terminator_temperature_k": terminator_temperature_k,
            "log_g_cgs": log_g_cgs,
            "planet_surface_gravity_m_s2": surface_gravity_m_s2(log_g_cgs),
            "planet_mass_kg": planet_mass_kg,
            "planet_mass_mjup": planet_mass_kg / 1.89813e27,
            "reference_pressure_bar": REFERENCE_PRESSURE_BAR,
            "log10_p_cloud_bar": cloud_log10_p,
            "log10_kappa0": haze_log10_kappa0,
            "gamma_scat": haze_gamma,
            "heavy_species_mass_fraction": heavy_fraction,
            "h2_mass_fraction": float(abundances_short["H2"][0]),
            "he_mass_fraction": float(abundances_short["He"][0]),
            "star_class": star_system["star_class"],
            "star_temperature_k": star_system["star_temperature_k"],
            "star_radius_m": star_system["star_radius_m"],
            "star_mass_kg": star_system["star_mass_kg"],
            "star_distance_pc": star_system["star_distance_pc"],
            "semi_major_axis_m": star_system["semi_major_axis_m"],
            "semi_major_axis_au": star_system["semi_major_axis_au"],
            "orbital_period_days": star_system["orbital_period_days"],
            "insolation_flux_w_m2": star_system["insolation_flux_w_m2"],
            "temperature_regime": temperature_regime,
        }
        for species, log_x in log_abundances.items():
            label_row["log_X_%s" % species] = float(log_x)
        label_row.update(make_species_presence_labels(log_abundances, DETECTION_THRESHOLD_LOG_X))

        provenance_row = {
            "sample_id": sample_id,
            "generation_attempt": attempt,
            "ood_candidate": int(is_ood_candidate(label_row)),
        }
        provenance_row.update(noise_metadata)
        provenance_row["resample_counts_json"] = json.dumps(dict(sorted(resample_counts.items())), sort_keys=True)

        return {
            "sample_id": sample_id,
            "native_transit_radius_m": native_radius_m.astype(np.float64),
            "native_transit_depth_noiseless": native_depth.astype(np.float64),
            "benchmark_transit_depth_noiseless": benchmark_noiseless.astype(np.float64),
            "benchmark_transit_depth_noisy": benchmark_noisy.astype(np.float64),
            "benchmark_sigma_1sigma": sigma_1sigma.astype(np.float64),
            "label_row": label_row,
            "provenance_row": provenance_row,
            "resample_counts": dict(resample_counts),
        }

    raise RuntimeError(
        "Unable to generate a finite transmission spectrum for sample_index=%d after %d attempts."
        % (sample_index, MAX_GENERATION_ATTEMPTS)
    )


def _init_worker(p_rt_input_data_path: str) -> None:
    os.environ["pRT_input_data_path"] = p_rt_input_data_path
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-prt-benchmark")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import petitRADTRANS as prt

    pressures_bar = np.logspace(np.log10(PRESSURE_MIN_BAR), np.log10(PRESSURE_MAX_BAR), PRESSURE_LEVELS)
    atmosphere = prt.Radtrans(
        line_species=list(LINE_SPECIES_R400.values()),
        rayleigh_species=list(RAYLEIGH_SPECIES),
        continuum_opacities=list(CONTINUUM_OPACITIES),
        wlen_bords_micron=[WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM],
    )
    atmosphere.setup_opa_structure(pressures_bar)

    benchmark_wavelength_um, benchmark_edges_um = make_log_resolving_power_grid(
        WAVELENGTH_MIN_UM,
        WAVELENGTH_MAX_UM,
        BENCHMARK_RESOLUTION,
    )
    native_wavelength_um, native_order = _native_wavelengths_um(atmosphere)
    rebin_matrix = build_rebin_matrix(native_wavelength_um, benchmark_edges_um)

    WORKER_STATE.clear()
    WORKER_STATE.update(
        {
            "atmosphere": atmosphere,
            "pressures_bar": pressures_bar,
            "native_wavelength_um": native_wavelength_um,
            "native_order": native_order,
            "benchmark_wavelength_um": benchmark_wavelength_um,
            "benchmark_edges_um": benchmark_edges_um,
            "rebin_matrix": rebin_matrix,
        }
    )


def _shard_ranges(sample_count: int, shard_size: int) -> List[Tuple[int, int, int]]:
    ranges = []
    shard_count = int(math.ceil(float(sample_count) / float(shard_size)))
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        stop = min(sample_count, start + shard_size)
        ranges.append((shard_index, start, stop))
    return ranges


def _records_to_shard_payload(records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    return {
        "sample_id": np.asarray([record["sample_id"] for record in records], dtype="U16"),
        "native_transit_radius_m": np.stack([record["native_transit_radius_m"] for record in records], axis=0),
        "native_transit_depth_noiseless": np.stack(
            [record["native_transit_depth_noiseless"] for record in records], axis=0
        ),
        "benchmark_transit_depth_noiseless": np.stack(
            [record["benchmark_transit_depth_noiseless"] for record in records], axis=0
        ),
        "benchmark_transit_depth_noisy": np.stack(
            [record["benchmark_transit_depth_noisy"] for record in records], axis=0
        ),
        "benchmark_sigma_1sigma": np.stack([record["benchmark_sigma_1sigma"] for record in records], axis=0),
        "label_rows_json": np.asarray(
            [json.dumps(record["label_row"], sort_keys=True) for record in records],
            dtype="U16384",
        ),
        "provenance_rows_json": np.asarray(
            [json.dumps(record["provenance_row"], sort_keys=True) for record in records],
            dtype="U16384",
        ),
    }


def _write_shard_payload(shard_path: Path, payload: Dict[str, np.ndarray]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = shard_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, **payload)
    tmp_path.replace(shard_path)


def _writer_loop(write_queue: Any, ack_queue: Any) -> None:
    while True:
        item = write_queue.get()
        if item is None:
            ack_queue.put({"type": "done"})
            return
        shard_path = Path(item["shard_path"])
        payload = item["payload"]
        _write_shard_payload(shard_path, payload)
        ack_queue.put(
            {
                "type": "written",
                "shard_path": str(shard_path),
                "count": int(payload["sample_id"].shape[0]),
            }
        )


def _drain_writer_acks(ack_queue: Any) -> Tuple[int, int]:
    persisted = 0
    shards = 0
    while True:
        try:
            ack = ack_queue.get_nowait()
        except Empty:
            return persisted, shards
        if ack.get("type") == "written":
            persisted += int(ack["count"])
            shards += 1


def _load_shard_payload(shard_path: Path) -> Dict[str, np.ndarray]:
    loaded = np.load(shard_path, allow_pickle=False)
    return {key: loaded[key] for key in loaded.files}


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
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_parquet_support() -> None:
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("Writing parquet requires pyarrow. Install it in the generation environment first.")


def _resolve_split_targets(sample_count: int) -> Dict[str, int]:
    canonical_count = sum(SPLIT_COUNTS.values())
    if sample_count == canonical_count:
        return dict(SPLIT_COUNTS)

    ratios = {name: float(count) / canonical_count for name, count in SPLIT_COUNTS.items()}
    scaled = {name: int(math.floor(sample_count * ratio)) for name, ratio in ratios.items()}
    assigned = sum(scaled.values())
    scaled["train"] += sample_count - assigned
    return scaled


def assign_splits(labels: pd.DataFrame, provenance: pd.DataFrame, seed: int) -> pd.Series:
    targets = _resolve_split_targets(len(labels))
    sample_ids = labels["sample_id"].astype(str).to_numpy()
    ood_flags = provenance["ood_candidate"].to_numpy(dtype=np.int64)
    ordering = np.argsort([stable_hash_u64("split", seed, sample_id) for sample_id in sample_ids])
    ood_order = [idx for idx in ordering if int(ood_flags[idx]) == 1]

    target_ood = targets["test_ood"]
    if len(ood_order) < target_ood and len(labels) >= sum(SPLIT_COUNTS.values()):
        raise RuntimeError(
            "Expected at least %d OOD candidates for the canonical split, found %d." % (target_ood, len(ood_order))
        )
    target_ood = min(target_ood, len(ood_order))

    split = np.full(len(labels), "train", dtype=object)
    chosen_ood = set(ood_order[:target_ood])
    for idx in chosen_ood:
        split[idx] = "test_ood"

    remaining = [idx for idx in ordering if idx not in chosen_ood]
    val_count = min(targets["val"], len(remaining))
    for idx in remaining[:val_count]:
        split[idx] = "val"
    remaining = remaining[val_count:]

    test_id_count = min(targets["test_id"], len(remaining))
    for idx in remaining[:test_id_count]:
        split[idx] = "test_id"

    return pd.Series(split, index=labels.index, dtype="string")


def generate_shards(config: GeneratorConfig) -> List[Path]:
    work_root = config.output_root / "work"
    shard_root = work_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    shard_ranges = _shard_ranges(config.sample_count, config.shard_size)
    shards_total = len(shard_ranges)
    persisted_samples = 0
    shards_completed = 0
    shard_paths: List[Path] = []
    resample_reason_counts = Counter()

    for shard_index, start, stop in shard_ranges:
        shard_path = shard_root / ("shard_%05d.npz" % shard_index)
        if config.skip_existing and shard_path.exists():
            persisted_samples += stop - start
            shards_completed += 1
        shard_paths.append(shard_path)

    _write_progress(
        config,
        stage="generating",
        generated_samples=persisted_samples,
        persisted_samples=persisted_samples,
        shards_completed=shards_completed,
        shards_total=shards_total,
        resample_reason_counts=dict(resample_reason_counts),
        message="generation initialized",
    )

    ctx = get_context("spawn")
    write_queue = ctx.Queue(maxsize=2)
    ack_queue = ctx.Queue()
    writer = ctx.Process(target=_writer_loop, args=(write_queue, ack_queue), daemon=True)
    writer.start()

    try:
        with ctx.Pool(
            processes=config.workers,
            initializer=_init_worker,
            initargs=(str(config.p_rt_input_data_path),),
        ) as pool:
            generated_samples = persisted_samples
            last_progress_update = 0.0
            for ordinal, (shard_index, start, stop) in enumerate(shard_ranges, start=1):
                shard_path = shard_root / ("shard_%05d.npz" % shard_index)
                if config.skip_existing and shard_path.exists():
                    persisted_delta, shard_delta = _drain_writer_acks(ack_queue)
                    persisted_samples += persisted_delta
                    shards_completed += shard_delta
                    continue

                print("[generate] shard %d/%d -> %s (%d:%d)" % (ordinal, shards_total, shard_path.name, start, stop), flush=True)
                tasks = [(sample_index, config.seed) for sample_index in range(start, stop)]
                records: List[Dict[str, Any]] = []
                _write_progress(
                    config,
                    stage="generating",
                    generated_samples=generated_samples,
                    persisted_samples=persisted_samples,
                    shards_completed=shards_completed,
                    shards_total=shards_total,
                    current_shard_index=shard_index,
                    current_shard_ordinal=ordinal,
                    current_shard_start=start,
                    current_shard_stop=stop,
                    current_shard_generated=0,
                    resample_reason_counts=dict(resample_reason_counts),
                    message="running %s" % shard_path.name,
                )
                for completed_in_shard, record in enumerate(pool.imap(_generate_one, tasks, chunksize=4), start=1):
                    records.append(record)
                    generated_samples += 1
                    resample_reason_counts.update(record["resample_counts"])
                    persisted_delta, shard_delta = _drain_writer_acks(ack_queue)
                    persisted_samples += persisted_delta
                    shards_completed += shard_delta
                    now = time.monotonic()
                    if (
                        completed_in_shard == 1
                        or completed_in_shard == (stop - start)
                        or now - last_progress_update >= config.progress_update_seconds
                    ):
                        _write_progress(
                            config,
                            stage="generating",
                            generated_samples=generated_samples,
                            persisted_samples=persisted_samples,
                            shards_completed=shards_completed,
                            shards_total=shards_total,
                            current_shard_index=shard_index,
                            current_shard_ordinal=ordinal,
                            current_shard_start=start,
                            current_shard_stop=stop,
                            current_shard_generated=completed_in_shard,
                            resample_reason_counts=dict(resample_reason_counts),
                            message="running %s" % shard_path.name,
                        )
                        last_progress_update = now

                write_queue.put({"shard_path": str(shard_path), "payload": _records_to_shard_payload(records)})

            write_queue.put(None)
            while True:
                ack = ack_queue.get()
                if ack.get("type") == "done":
                    break
                if ack.get("type") == "written":
                    persisted_samples += int(ack["count"])
                    shards_completed += 1
                    _write_progress(
                        config,
                        stage="generating",
                        generated_samples=generated_samples,
                        persisted_samples=persisted_samples,
                        shards_completed=shards_completed,
                        shards_total=shards_total,
                        resample_reason_counts=dict(resample_reason_counts),
                        message="wrote %s" % Path(ack["shard_path"]).name,
                    )
    finally:
        writer.join(timeout=5.0)
        if writer.is_alive():
            writer.terminate()
            writer.join(timeout=1.0)

    return shard_paths


def _load_all_records(shard_paths: Iterable[Path]) -> Dict[str, Any]:
    sample_ids: List[np.ndarray] = []
    native_radius_m: List[np.ndarray] = []
    native_depth: List[np.ndarray] = []
    benchmark_noisy: List[np.ndarray] = []
    benchmark_noiseless: List[np.ndarray] = []
    benchmark_sigma: List[np.ndarray] = []
    label_rows: List[Dict[str, Any]] = []
    provenance_rows: List[Dict[str, Any]] = []

    for shard_path in sorted(shard_paths):
        payload = _load_shard_payload(shard_path)
        sample_ids.append(payload["sample_id"])
        native_radius_m.append(payload["native_transit_radius_m"])
        native_depth.append(payload["native_transit_depth_noiseless"])
        benchmark_noiseless.append(payload["benchmark_transit_depth_noiseless"])
        benchmark_noisy.append(payload["benchmark_transit_depth_noisy"])
        benchmark_sigma.append(payload["benchmark_sigma_1sigma"])
        label_rows.extend(json.loads(str(item)) for item in payload["label_rows_json"])
        provenance_rows.extend(json.loads(str(item)) for item in payload["provenance_rows_json"])

    return {
        "sample_id": np.concatenate(sample_ids, axis=0),
        "native_transit_radius_m": np.concatenate(native_radius_m, axis=0),
        "native_transit_depth_noiseless": np.concatenate(native_depth, axis=0),
        "benchmark_transit_depth_noiseless": np.concatenate(benchmark_noiseless, axis=0),
        "benchmark_transit_depth_noisy": np.concatenate(benchmark_noisy, axis=0),
        "benchmark_sigma_1sigma": np.concatenate(benchmark_sigma, axis=0),
        "labels": pd.DataFrame(label_rows),
        "provenance": pd.DataFrame(provenance_rows),
    }


def _write_spectra_h5(
    path: Path,
    sample_ids: np.ndarray,
    splits: np.ndarray,
    benchmark_wavelength_um: np.ndarray,
    native_wavelength_um: np.ndarray,
    benchmark_noisy: np.ndarray,
    benchmark_noiseless: np.ndarray,
    benchmark_sigma: np.ndarray,
    native_depth_noiseless: np.ndarray,
    native_radius_m: np.ndarray,
) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    chunk_rows = min(HDF5_CHUNK_ROWS, len(sample_ids))
    sample_ids_str = np.asarray(sample_ids, dtype=object)
    splits_str = np.asarray(splits, dtype=object)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("sample_id", data=sample_ids_str, dtype=string_dtype)
        handle.create_dataset("split", data=splits_str, dtype=string_dtype)
        benchmark_group = handle.create_group("benchmark")
        native_group = handle.create_group("native")
        benchmark_group.create_dataset("wavelength_um", data=benchmark_wavelength_um)
        native_group.create_dataset("wavelength_um", data=native_wavelength_um)
        for name, values in (
            ("transit_depth_noisy", benchmark_noisy),
            ("transit_depth_noiseless", benchmark_noiseless),
            ("sigma_1sigma", benchmark_sigma),
        ):
            benchmark_group.create_dataset(
                name,
                data=values,
                chunks=(chunk_rows, values.shape[1]),
                compression="lzf",
                shuffle=True,
            )
        for name, values in (
            ("transit_depth_noiseless", native_depth_noiseless),
            ("transit_radius_m", native_radius_m),
        ):
            native_group.create_dataset(
                name,
                data=values,
                chunks=(chunk_rows, values.shape[1]),
                compression="lzf",
                shuffle=True,
            )


def assemble_dataset(config: GeneratorConfig, shard_paths: Iterable[Path]) -> Dict[str, Path]:
    _require_parquet_support()
    records = _load_all_records(shard_paths)
    if len(records["sample_id"]) != config.sample_count:
        raise RuntimeError("Expected %d records, found %d." % (config.sample_count, len(records["sample_id"])))

    labels = records["labels"].sort_values("sample_id").reset_index(drop=True)
    provenance = records["provenance"].sort_values("sample_id").reset_index(drop=True)
    order = np.argsort(records["sample_id"].astype(str))
    sample_ids = np.asarray(records["sample_id"][order], dtype="U16")

    benchmark_wavelength_um, _ = make_log_resolving_power_grid(
        WAVELENGTH_MIN_UM,
        WAVELENGTH_MAX_UM,
        BENCHMARK_RESOLUTION,
    )
    native_wavelength_um = None
    if shard_paths:
        payload = _load_shard_payload(sorted(shard_paths)[0])
        native_length = payload["native_transit_radius_m"].shape[1]
        native_wavelength_um = np.empty(native_length, dtype=np.float64)
    if native_wavelength_um is None:
        raise RuntimeError("No shard payloads were found for assembly.")

    _init_worker(str(config.p_rt_input_data_path))
    native_wavelength_um = WORKER_STATE["native_wavelength_um"]

    native_radius_m = records["native_transit_radius_m"][order]
    native_depth = records["native_transit_depth_noiseless"][order]
    benchmark_noiseless = records["benchmark_transit_depth_noiseless"][order]
    benchmark_noisy = records["benchmark_transit_depth_noisy"][order]
    benchmark_sigma = records["benchmark_sigma_1sigma"][order]

    split_series = assign_splits(labels, provenance, config.seed)
    labels["split"] = split_series
    provenance["split"] = split_series

    spectra_path = config.output_root / BENCHMARK_PATHS.spectra_h5
    labels_path = config.output_root / BENCHMARK_PATHS.labels_parquet
    provenance_path = config.output_root / BENCHMARK_PATHS.provenance_parquet
    manifest_path = config.output_root / BENCHMARK_PATHS.manifest_json

    _write_progress(
        config,
        stage="assembling",
        generated_samples=config.sample_count,
        persisted_samples=config.sample_count,
        shards_completed=int(math.ceil(float(config.sample_count) / float(config.shard_size))),
        shards_total=int(math.ceil(float(config.sample_count) / float(config.shard_size))),
        message="assembling spectra and tables",
    )

    _write_spectra_h5(
        spectra_path,
        sample_ids=sample_ids,
        splits=labels["split"].to_numpy(dtype="U16"),
        benchmark_wavelength_um=benchmark_wavelength_um,
        native_wavelength_um=native_wavelength_um,
        benchmark_noisy=benchmark_noisy,
        benchmark_noiseless=benchmark_noiseless,
        benchmark_sigma=benchmark_sigma,
        native_depth_noiseless=native_depth,
        native_radius_m=native_radius_m,
    )
    labels.to_parquet(labels_path, index=False)
    provenance.to_parquet(provenance_path, index=False)

    split_counts = labels["split"].value_counts().sort_index().to_dict()
    resample_reason_counts = Counter()
    for raw_json in provenance["resample_counts_json"].astype(str):
        resample_reason_counts.update(json.loads(raw_json))

    manifest = {
        "created_utc": _utc_now_iso(),
        "sample_count": int(config.sample_count),
        "seed": int(config.seed),
        "worker_count": int(config.workers),
        "pressure_grid_bar": {
            "min": PRESSURE_MIN_BAR,
            "max": PRESSURE_MAX_BAR,
            "levels": PRESSURE_LEVELS,
            "reference_pressure_bar": REFERENCE_PRESSURE_BAR,
        },
        "wavelength_grid_um": {
            "min": WAVELENGTH_MIN_UM,
            "max": WAVELENGTH_MAX_UM,
            "native_resolution": NATIVE_RESOLUTION,
            "benchmark_resolution": BENCHMARK_RESOLUTION,
            "benchmark_bins": int(len(benchmark_wavelength_um)),
            "native_bins": int(len(native_wavelength_um)),
        },
        "input_data": {
            "path": str(config.p_rt_input_data_path),
            "source_species": list(SOURCE_SPECIES),
            "line_species_r400": dict(LINE_SPECIES_R400),
        },
        "software_versions": _runtime_versions(config.p_rt_input_data_path),
        "split_counts": split_counts,
        "resample_reason_counts": dict(sorted(resample_reason_counts.items())),
        "files": {
            BENCHMARK_PATHS.spectra_h5: _sha256(spectra_path),
            BENCHMARK_PATHS.labels_parquet: _sha256(labels_path),
            BENCHMARK_PATHS.provenance_parquet: _sha256(provenance_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    _write_progress(
        config,
        stage="completed",
        generated_samples=config.sample_count,
        persisted_samples=config.sample_count,
        shards_completed=int(math.ceil(float(config.sample_count) / float(config.shard_size))),
        shards_total=int(math.ceil(float(config.sample_count) / float(config.shard_size))),
        resample_reason_counts=dict(sorted(resample_reason_counts.items())),
        message="generation and assembly completed",
    )

    return {
        "spectra_path": spectra_path,
        "labels_path": labels_path,
        "provenance_path": provenance_path,
        "manifest_path": manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--p-rt-input-data-path", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=_resolve_default_workers())
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--progress-update-seconds", type=float, default=DEFAULT_PROGRESS_UPDATE_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeneratorConfig(
        output_root=args.output_root,
        p_rt_input_data_path=args.p_rt_input_data_path,
        sample_count=args.sample_count,
        shard_size=args.shard_size,
        seed=args.seed,
        workers=args.workers,
        skip_existing=args.skip_existing,
        run_started_utc=_utc_now_iso(),
        progress_update_seconds=args.progress_update_seconds,
    )

    try:
        if args.assemble_only:
            shard_paths = sorted((config.output_root / "work" / "shards").glob("shard_*.npz"))
        else:
            shard_paths = generate_shards(config)
        if args.generate_only:
            _write_progress(
                config,
                stage="generated_shards",
                generated_samples=config.sample_count,
                persisted_samples=config.sample_count,
                shards_completed=int(math.ceil(float(config.sample_count) / float(config.shard_size))),
                shards_total=int(math.ceil(float(config.sample_count) / float(config.shard_size))),
                message="shard generation completed",
            )
        else:
            assemble_dataset(config, shard_paths)
    except Exception as exc:
        _mark_progress_failed(config, exc)
        raise


if __name__ == "__main__":
    main()
