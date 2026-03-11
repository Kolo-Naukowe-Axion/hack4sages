"""Constants for the cross-generator biosignature dataset."""

from __future__ import annotations

from dataclasses import dataclass

DATASET_NAME = "crossgen_biosignatures"

MASTER_SEED = 20260310

TAUREX_SAMPLE_COUNT = 41_423
TAUREX_TRAIN_COUNT = 37_281
TAUREX_VAL_COUNT = 4_142
POSEIDON_TEST_COUNT = 685

TRACE_SPECIES = (
    ("h2o", "H2O"),
    ("co2", "CO2"),
    ("co", "CO"),
    ("ch4", "CH4"),
    ("nh3", "NH3"),
)

TRACE_SPECIES_KEYS = tuple(key for key, _ in TRACE_SPECIES)
TRACE_SPECIES_NAMES = tuple(name for _, name in TRACE_SPECIES)

PLANET_RADIUS_RANGE_RJUP = (0.7, 1.5)
LOG_G_RANGE_CGS = (2.8, 3.7)
TEMPERATURE_RANGE_K = (500.0, 1800.0)
STAR_RADIUS_RANGE_RSUN = (0.2, 1.3)
TRACE_LOG10_VMR_RANGE = (-12.0, -2.0)
TRACE_VMR_MAX_TOTAL = 0.10
PRESENCE_THRESHOLD_LOG10_VMR = -8.0

BACKGROUND_H2_FRACTION = 0.85
BACKGROUND_HE_FRACTION = 0.15
BACKGROUND_HE_TO_H2_RATIO = BACKGROUND_HE_FRACTION / BACKGROUND_H2_FRACTION

PRESSURE_LEVELS = 100
PRESSURE_MIN_BAR = 1.0e-6
PRESSURE_MAX_BAR = 1.0e2
REFERENCE_PRESSURE_BAR = 10.0

TARGET_WAVELENGTH_MIN_UM = 0.6
TARGET_WAVELENGTH_MAX_UM = 5.2
TARGET_RESOLUTION = 100.0
POSEIDON_NATIVE_RESOLUTION = 1_000.0

NOISE_SIGMA_PPM_RANGE = (20.0, 100.0)

FIXED_STAR_TEMPERATURE_K = 5_500.0
FIXED_STAR_LOG_G_CGS = 4.5
FIXED_STAR_METALLICITY = 0.0
FIXED_STAR_DISTANCE_PC = 10.0
FIXED_STAR_MASS_MSUN = 1.0

FIXED_PLANET_SEMIMAJOR_AXIS_AU = 0.05
FIXED_SYSTEM_DISTANCE_PC = 10.0

POSEIDON_OPACITY_TREATMENT = "opacity_sampling"
POSEIDON_OPACITY_DATABASE = "High-T"
POSEIDON_DATABASE_VERSION = "1.3"
POSEIDON_FINE_TEMPERATURE_GRID_K = tuple(float(value) for value in range(400, 1901, 100))
POSEIDON_FINE_LOG10_PRESSURE_BAR = tuple(-6.0 + 0.25 * idx for idx in range(33))

JUPITER_RADIUS_M = 7.1492e7
SOLAR_RADIUS_M = 6.957e8
JUPITER_MASS_KG = 1.89813e27
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
AU_M = 1.495978707e11
PARSEC_M = 3.085677581491367e16

TAUREX_GENERATOR_KEY = "tau"
POSEIDON_GENERATOR_KEY = "poseidon"
GENERATOR_KEYS = (TAUREX_GENERATOR_KEY, POSEIDON_GENERATOR_KEY)


@dataclass(frozen=True)
class DatasetPaths:
    """File layout for assembled and intermediate artifacts."""

    latents_parquet: str = "latents.parquet"
    labels_parquet: str = "labels.parquet"
    spectra_h5: str = "spectra.h5"
    manifest_json: str = "manifest.json"
    shards_dir: str = "shards"
    metadata_dir: str = "meta"
    baseline_json: str = "baseline_smoke.json"
    baseline_predictions_csv: str = "baseline_poseidon_predictions.csv"


DATASET_PATHS = DatasetPaths()

