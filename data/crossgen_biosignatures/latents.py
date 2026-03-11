"""Latent sampling and schema helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .constants import (
    BACKGROUND_H2_FRACTION,
    BACKGROUND_HE_FRACTION,
    GENERATOR_KEYS,
    LOG_G_RANGE_CGS,
    MASTER_SEED,
    NOISE_SIGMA_PPM_RANGE,
    PLANET_RADIUS_RANGE_RJUP,
    POSEIDON_GENERATOR_KEY,
    POSEIDON_TEST_COUNT,
    PRESENCE_THRESHOLD_LOG10_VMR,
    STAR_RADIUS_RANGE_RSUN,
    TAUREX_GENERATOR_KEY,
    TAUREX_SAMPLE_COUNT,
    TAUREX_TRAIN_COUNT,
    TAUREX_VAL_COUNT,
    TEMPERATURE_RANGE_K,
    TRACE_LOG10_VMR_RANGE,
    TRACE_SPECIES_KEYS,
    TRACE_VMR_MAX_TOTAL,
)
from .utils import derived_seed


TRACE_COLUMN_NAMES = tuple(f"log10_vmr_{species}" for species in TRACE_SPECIES_KEYS)
PRESENT_COLUMN_NAMES = tuple(f"present_{species}" for species in TRACE_SPECIES_KEYS)
FINAL_LABEL_COLUMNS = (
    "sample_id",
    "generator",
    "split",
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    "trace_vmr_total",
    "vmr_h2",
    "vmr_he",
    *TRACE_COLUMN_NAMES,
    *PRESENT_COLUMN_NAMES,
)


def sample_id_for(generator: str, ordinal: int) -> str:
    """Return the canonical sample identifier for a generator-local ordinal."""

    if generator == TAUREX_GENERATOR_KEY:
        return f"tau_{ordinal:06d}"
    if generator == POSEIDON_GENERATOR_KEY:
        return f"poseidon_{ordinal:06d}"
    raise ValueError(f"Unsupported generator {generator!r}.")


def expected_split_counts() -> dict[str, dict[str, int]]:
    """Return the canonical split counts for the dataset."""

    return {
        TAUREX_GENERATOR_KEY: {"train": TAUREX_TRAIN_COUNT, "val": TAUREX_VAL_COUNT},
        POSEIDON_GENERATOR_KEY: {"test": POSEIDON_TEST_COUNT},
    }


def generator_counts() -> dict[str, int]:
    """Return the expected row count per generator."""

    return {
        TAUREX_GENERATOR_KEY: TAUREX_SAMPLE_COUNT,
        POSEIDON_GENERATOR_KEY: POSEIDON_TEST_COUNT,
    }


def _sample_valid_batch(rng: np.random.Generator, batch_size: int) -> pd.DataFrame:
    trace_logs = rng.uniform(TRACE_LOG10_VMR_RANGE[0], TRACE_LOG10_VMR_RANGE[1], size=(batch_size, len(TRACE_COLUMN_NAMES)))
    trace_vmr = np.power(10.0, trace_logs)
    trace_totals = trace_vmr.sum(axis=1)
    valid_mask = trace_totals < TRACE_VMR_MAX_TOTAL
    if not np.any(valid_mask):
        return pd.DataFrame(columns=FINAL_LABEL_COLUMNS)

    valid_logs = trace_logs[valid_mask]
    valid_totals = trace_totals[valid_mask]
    vmr_h2 = (1.0 - valid_totals) * BACKGROUND_H2_FRACTION
    vmr_he = (1.0 - valid_totals) * BACKGROUND_HE_FRACTION
    data: dict[str, np.ndarray] = {
        "planet_radius_rjup": rng.uniform(PLANET_RADIUS_RANGE_RJUP[0], PLANET_RADIUS_RANGE_RJUP[1], size=batch_size)[valid_mask],
        "log_g_cgs": rng.uniform(LOG_G_RANGE_CGS[0], LOG_G_RANGE_CGS[1], size=batch_size)[valid_mask],
        "temperature_k": rng.uniform(TEMPERATURE_RANGE_K[0], TEMPERATURE_RANGE_K[1], size=batch_size)[valid_mask],
        "star_radius_rsun": rng.uniform(STAR_RADIUS_RANGE_RSUN[0], STAR_RADIUS_RANGE_RSUN[1], size=batch_size)[valid_mask],
        "trace_vmr_total": valid_totals,
        "vmr_h2": vmr_h2,
        "vmr_he": vmr_he,
        "sigma_ppm": rng.uniform(NOISE_SIGMA_PPM_RANGE[0], NOISE_SIGMA_PPM_RANGE[1], size=batch_size)[valid_mask],
    }
    for column_idx, column_name in enumerate(TRACE_COLUMN_NAMES):
        data[column_name] = valid_logs[:, column_idx]
    for column_name in TRACE_COLUMN_NAMES:
        present_column = "present_" + column_name.removeprefix("log10_vmr_")
        data[present_column] = (data[column_name] >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)
    return pd.DataFrame(data)


def _sample_latents_for_generator(
    generator: str,
    count: int,
    splits: Iterable[str],
    seed: int,
    start_ordinal: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    remaining = int(count)
    frames: list[pd.DataFrame] = []
    while remaining > 0:
        batch_size = max(remaining * 2, 1_024)
        batch = _sample_valid_batch(rng, batch_size=batch_size)
        if len(batch) == 0:
            continue
        frames.append(batch.iloc[:remaining].reset_index(drop=True))
        remaining -= len(frames[-1])

    frame = pd.concat(frames, ignore_index=True)
    frame.insert(0, "split", list(splits))
    frame.insert(0, "generator", generator)
    frame.insert(0, "sample_id", [sample_id_for(generator, start_ordinal + ordinal + 1) for ordinal in range(count)])
    frame.insert(3, "sample_index", np.arange(start_ordinal + 1, start_ordinal + count + 1, dtype=np.int64))
    return frame


def build_latents(master_seed: int = MASTER_SEED) -> pd.DataFrame:
    """Construct the full canonical latent table for the dataset."""

    split_counts = expected_split_counts()
    tau_splits = ["train"] * split_counts[TAUREX_GENERATOR_KEY]["train"] + ["val"] * split_counts[TAUREX_GENERATOR_KEY]["val"]
    poseidon_splits = ["test"] * split_counts[POSEIDON_GENERATOR_KEY]["test"]

    tau_frame = _sample_latents_for_generator(
        generator=TAUREX_GENERATOR_KEY,
        count=TAUREX_SAMPLE_COUNT,
        splits=tau_splits,
        seed=derived_seed(master_seed, TAUREX_GENERATOR_KEY, "latents"),
    )
    poseidon_frame = _sample_latents_for_generator(
        generator=POSEIDON_GENERATOR_KEY,
        count=POSEIDON_TEST_COUNT,
        splits=poseidon_splits,
        seed=derived_seed(master_seed, POSEIDON_GENERATOR_KEY, "latents"),
    )
    combined = pd.concat([tau_frame, poseidon_frame], ignore_index=True)
    combined.insert(0, "row_index", np.arange(len(combined), dtype=np.int64))
    return combined


def build_tau_extension_latents(
    count: int,
    *,
    master_seed: int = MASTER_SEED,
    start_ordinal: int = TAUREX_SAMPLE_COUNT,
    val_fraction: float | None = None,
) -> pd.DataFrame:
    """Construct an append-compatible TauREx-only extension latent table."""

    if count <= 0:
        raise ValueError("TauREx extension count must be positive.")
    if start_ordinal < 0:
        raise ValueError("TauREx extension start_ordinal must be non-negative.")

    if val_fraction is None:
        val_fraction = TAUREX_VAL_COUNT / TAUREX_SAMPLE_COUNT

    raw_val_count = int(round(float(count) * float(val_fraction)))
    if count == 1:
        val_count = 0
    else:
        val_count = min(max(raw_val_count, 1), count - 1)
    train_count = count - val_count
    splits = ["train"] * train_count + ["val"] * val_count

    tau_frame = _sample_latents_for_generator(
        generator=TAUREX_GENERATOR_KEY,
        count=count,
        splits=splits,
        seed=derived_seed(master_seed, TAUREX_GENERATOR_KEY, "extension_latents", start_ordinal, count),
        start_ordinal=start_ordinal,
    )
    tau_frame.insert(0, "row_index", np.arange(len(tau_frame), dtype=np.int64))
    return tau_frame


def final_labels_frame(latents: pd.DataFrame) -> pd.DataFrame:
    """Project the intermediate latent table to the public label schema."""

    return latents.loc[:, FINAL_LABEL_COLUMNS].copy()


def validate_latent_frame(
    latents: pd.DataFrame,
    *,
    expected_counts: dict[str, int] | None = None,
    required_generators: Iterable[str] | None = None,
) -> None:
    """Validate latent priors and dataset counts before assembly."""

    observed_generators = set(latents["generator"].astype(str).unique().tolist())
    required = set(GENERATOR_KEYS if required_generators is None else required_generators)
    if not required.issubset(observed_generators):
        raise AssertionError("Latent table is missing at least one generator.")
    if latents["sample_id"].duplicated().any():
        raise AssertionError("Latent table contains duplicate sample_id values.")
    counts = latents["generator"].value_counts().to_dict()
    expected = generator_counts() if expected_counts is None else {str(key): int(value) for key, value in expected_counts.items()}
    for generator, expected_count in expected.items():
        if int(counts.get(generator, 0)) != int(expected_count):
            raise AssertionError(f"Generator {generator} has {counts.get(generator, 0)} rows instead of {expected_count}.")
    if np.any(latents["trace_vmr_total"].to_numpy(dtype=np.float64) >= TRACE_VMR_MAX_TOTAL):
        raise AssertionError("trace_vmr_total exceeds the configured maximum.")
    for column_name in TRACE_COLUMN_NAMES:
        presence_column = "present_" + column_name.removeprefix("log10_vmr_")
        expected_presence = (latents[column_name].to_numpy(dtype=np.float64) >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)
        actual_presence = latents[presence_column].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_presence, actual_presence):
            raise AssertionError(f"Presence labels are inconsistent for {presence_column}.")
