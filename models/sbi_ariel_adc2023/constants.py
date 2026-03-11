"""Constants for the ADC2023 five-gas FMPE workflow."""

from __future__ import annotations

from pathlib import Path


DATASET_TYPE = "ADC2023FiveGasFMPEPreparedDataset"
NORMALIZATION_MODE = "adc2023_fivegas_context_zscore_v1"

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
HOLDOUT_SPLIT = "holdout"
TESTDATA_SPLIT = "testdata"

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
    "log_H2O",
    "log_CO2",
    "log_CO",
    "log_CH4",
    "log_NH3",
]

RAW_SPECTRAL_CHANNELS = [
    "instrument_spectrum",
    "instrument_noise",
    "instrument_width",
    "instrument_wlgrid",
]

SPECTRUM_FIELD = "instrument_spectrum"
NOISE_FIELD = "instrument_noise"
WAVELENGTH_DATASET = "instrument_wlgrid"
HDF5_GROUP_PREFIX = "Planet_"
PRESENCE_THRESHOLD_LOG10_VMR = -8.0
PRIMARY_STRATIFY_MIN_COUNT = 10
COARSE_STRATIFY_MIN_COUNT = 4
COARSE_ABUNDANCE_QUANTILES = (0.25, 0.5, 0.75)
POINT_ESTIMATE_CHOICES = ("auto", "mean", "median")

CONTEXT_FILENAME_TEMPLATE = "{split_name}_context.npy"
TARGET_FILENAME_TEMPLATE = "{split_name}_targets.npy"
RAW_TARGET_FILENAME_TEMPLATE = "{split_name}_raw_targets.npy"
METADATA_FILENAME_TEMPLATE = "{split_name}_metadata.csv"

MANIFEST_FILENAME = "manifest.json"
NORMALIZATION_FILENAME = "normalization_stats.npz"
WAVELENGTH_FILENAME = "wavelength_um.npy"

CONTEXT_DIM = 52 + 52 + len(AUX_FEATURE_COLS)
THETA_DIM = len(TARGET_COLS)

DEFAULT_DATA_ROOT = Path("data/full-ariel")
DEFAULT_PREPARED_DIR = Path("data/generated-data/adc2023_fivegas_fmpe_prepared")
DEFAULT_RUN_ROOT = Path("local_runs")
