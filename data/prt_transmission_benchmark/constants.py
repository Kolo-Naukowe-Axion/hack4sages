"""Constants for the pRT transmission benchmark generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

PRESSURE_MIN_BAR = 1.0e-6
PRESSURE_MAX_BAR = 1.0e2
PRESSURE_LEVELS = 100
REFERENCE_PRESSURE_BAR = 10.0
WAVELENGTH_MIN_UM = 0.5
WAVELENGTH_MAX_UM = 5.0
NATIVE_RESOLUTION = 400
BENCHMARK_RESOLUTION = 100

DEFAULT_SAMPLE_COUNT = 150_000
DEFAULT_SHARD_SIZE = 512
DEFAULT_RANDOM_SEED = 10032026
DEFAULT_PROGRESS_UPDATE_SECONDS = 10.0
DEFAULT_TUNING_SAMPLE_COUNT = 4_096
DEFAULT_WORKER_CANDIDATES = (16, 18, 20, 22)

MAX_GENERATION_ATTEMPTS = 64
PLANET_ID_PREFIX = "tx"

STAR_CLASS_WEIGHTS = (
    ("M", 0.25),
    ("K", 0.35),
    ("G", 0.25),
    ("F", 0.15),
)

STAR_CLASS_RANGES = {
    "M": {
        "mass_solar": (0.20, 0.60),
        "radius_solar": (0.25, 0.65),
        "temperature_k": (3000.0, 3900.0),
        "distance_pc": (8.0, 120.0),
    },
    "K": {
        "mass_solar": (0.60, 0.90),
        "radius_solar": (0.65, 0.95),
        "temperature_k": (3900.0, 5200.0),
        "distance_pc": (10.0, 180.0),
    },
    "G": {
        "mass_solar": (0.90, 1.10),
        "radius_solar": (0.90, 1.15),
        "temperature_k": (5200.0, 6000.0),
        "distance_pc": (20.0, 240.0),
    },
    "F": {
        "mass_solar": (1.10, 1.45),
        "radius_solar": (1.15, 1.60),
        "temperature_k": (6000.0, 7200.0),
        "distance_pc": (30.0, 320.0),
    },
}

TEMPERATURE_REGIMES = (
    ("cool", 0.30, (500.0, 900.0)),
    ("warm", 0.45, (900.0, 1600.0)),
    ("hot", 0.25, (1600.0, 2500.0)),
)

PRIMARY_TARGET_SPECIES = (
    "H2O",
    "CO2",
    "CO",
    "CH4",
    "NH3",
    "H2S",
    "PH3",
    "Na",
    "K",
    "TiO",
    "VO",
)

LINE_SPECIES_R400 = {
    "H2O": "H2O_HITEMP_R_400",
    "CO2": "CO2_R_400",
    "CO": "CO_all_iso_HITEMP_R_400",
    "CH4": "CH4_R_400",
    "NH3": "NH3_R_400",
    "H2S": "H2S_R_400",
    "PH3": "PH3_R_400",
    "Na": "Na_allard_R_400",
    "K": "K_allard_R_400",
    "TiO": "TiO_all_Exomol_R_400",
    "VO": "VO_R_400",
}

SOURCE_SPECIES = (
    "H2O_HITEMP",
    "CO2",
    "CO_all_iso_HITEMP",
    "CH4",
    "NH3",
    "H2S",
    "PH3",
    "Na_allard",
    "K_allard",
    "TiO_all_Exomol",
    "VO",
)

RAYLEIGH_SPECIES = ("H2", "He")
CONTINUUM_OPACITIES = ("H2-H2", "H2-He")

MOLECULAR_WEIGHTS_AMU = {
    "H2": 2.01588,
    "He": 4.002602,
    "H2O": 18.01528,
    "CO2": 44.0095,
    "CO": 28.0101,
    "CH4": 16.04246,
    "NH3": 17.03052,
    "H2S": 34.08088,
    "PH3": 33.99758,
    "Na": 22.989769,
    "K": 39.0983,
    "TiO": 63.866,
    "VO": 66.9409,
}

SPECIES_LOG_X_BOUNDS = {
    "H2O": (-12.0, -2.0),
    "CO2": (-12.0, -2.0),
    "CO": (-12.0, -2.0),
    "CH4": (-12.0, -2.0),
    "NH3": (-12.0, -4.0),
    "H2S": (-12.0, -4.0),
    "PH3": (-12.0, -5.0),
    "Na": (-12.0, -5.0),
    "K": (-12.0, -5.0),
    "TiO": (-14.0, -5.0),
    "VO": (-14.0, -5.0),
}

COOL_MAXIMA = {
    "CO": -8.0,
    "TiO": -8.0,
    "VO": -8.0,
}

HOT_MAXIMA = {
    "CH4": -8.0,
    "NH3": -8.0,
    "H2S": -8.0,
}

HEAVY_SPECIES_MAX_FRACTION = 0.15
H2_HE_REMAINDER_RATIO = (0.74, 0.26)
DETECTION_THRESHOLD_LOG_X = -8.0

PLANET_RADIUS_RANGE_RJUP = (0.6, 1.8)
LOG_G_RANGE_CGS = (2.6, 3.7)
NOISE_WHITE_PPM_RANGE = (15.0, 120.0)
NOISE_SLOPE_FRACTION_RANGE = (0.00, 0.80)
SYSTEMATIC_AMPLITUDE_PPM_RANGE = (0.0, 80.0)
SYSTEMATIC_PERIOD_OCTAVES_RANGE = (0.8, 2.8)
CLOUD_LOG10_P_RANGE = (-6.0, 1.0)
HAZE_LOG10_KAPPA0_RANGE = (-10.0, -2.0)
HAZE_GAMMA_RANGE = (-8.0, 2.0)
MIN_SEMIMAJOR_AXIS_TO_STELLAR_RADIUS = 3.0

SPLIT_COUNTS = {
    "train": 120_000,
    "val": 10_000,
    "test_id": 10_000,
    "test_ood": 10_000,
}

HDF5_CHUNK_ROWS = 512


@dataclass(frozen=True)
class DatasetPaths:
    """Final output paths for the assembled benchmark dataset."""

    spectra_h5: str = "spectra.h5"
    labels_parquet: str = "labels.parquet"
    provenance_parquet: str = "provenance.parquet"
    manifest_json: str = "manifest.json"
