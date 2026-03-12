"""Constants for the TauREx-only five-gas FMPE workflow."""

from __future__ import annotations

from pathlib import Path


DATASET_TYPE = "TauRExFiveGasFMPEPreparedDataset"
NORMALIZATION_MODE = "taurex_fivegas_context_zscore_v1"

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
HOLDOUT_SPLIT = "holdout"
TESTDATA_SPLIT = "testdata"

SOURCE_TRAIN_GENERATOR = "tau"
SOURCE_TRAIN_SPLIT = "train"
SOURCE_VALIDATION_GENERATOR = "tau"
SOURCE_VALIDATION_SPLIT = "val"
SOURCE_HOLDOUT_GENERATOR = "tau"
SOURCE_HOLDOUT_SPLIT = "val"
EXCLUDED_GENERATORS = ("poseidon",)

AUX_FEATURE_COLS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
]

LOG10_AUX_FEATURE_COLS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
]

TARGET_COLS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]

REQUIRED_LABEL_COLUMNS = (
    "sample_id",
    "generator",
    "split",
    "planet_radius_rjup",
    "log_g_cgs",
    "star_radius_rsun",
    *TARGET_COLS,
)

REQUIRED_SPECTRA_KEYS = (
    "sample_id",
    "generator",
    "split",
    "wavelength_um",
    "transit_depth_noisy",
    "sigma_ppm",
)

SPECTRAL_LENGTH = 218
NOISE_PPM_TO_TRANSIT_DEPTH = 1.0e-6
POINT_ESTIMATE_CHOICES = ("auto", "mean", "median")

RJUP_M = 69_911_000.0
G_NEWTON = 6.674e-11
SOLAR_RADIUS_M = 6.957e8
SOLAR_MASS_KG = 1.98847e30
AU_M = 1.495978707e11
SECONDS_PER_DAY = 86_400.0

FIXED_STAR_DISTANCE_PC = 10.0
FIXED_STAR_MASS_KG = SOLAR_MASS_KG
FIXED_STAR_TEMPERATURE_K = 5_500.0
FIXED_PLANET_DISTANCE_AU = 0.05

CONTEXT_FILENAME_TEMPLATE = "{split_name}_context.npy"
TARGET_FILENAME_TEMPLATE = "{split_name}_targets.npy"
RAW_TARGET_FILENAME_TEMPLATE = "{split_name}_raw_targets.npy"
METADATA_FILENAME_TEMPLATE = "{split_name}_metadata.csv"

MANIFEST_FILENAME = "manifest.json"
NORMALIZATION_FILENAME = "normalization_stats.npz"
WAVELENGTH_FILENAME = "wavelength_um.npy"

CONTEXT_DIM = SPECTRAL_LENGTH + SPECTRAL_LENGTH + len(AUX_FEATURE_COLS)
THETA_DIM = len(TARGET_COLS)

DEFAULT_DATA_ROOT = Path("data/TauREx set")
DEFAULT_PREPARED_DIR = Path("data/generated-data/taurex_fmpe_prepared")
DEFAULT_RUN_ROOT = Path("local_runs")
