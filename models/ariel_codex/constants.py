"""Constants for the Ariel five-gas quantum regressor."""

from __future__ import annotations

from pathlib import Path


AUX_COLUMNS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
]

LOG10_AUX_COLUMNS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
]

TARGET_COLUMNS = [
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

SAMPLE_SPECTRAL_CHANNELS = [
    "instrument_spectrum",
    "instrument_noise",
]

FIXED_SPECTRAL_CHANNELS = [
    "instrument_width",
    "instrument_wlgrid",
]

MODEL_SPECTRAL_CHANNELS = [
    "instrument_spectrum",
    "instrument_noise",
    "instrument_width_template",
    "wavelength_um",
]

WAVELENGTH_DATASET = "instrument_wlgrid"
HDF5_GROUP_PREFIX = "Planet_"
PRESENCE_THRESHOLD_LOG10_VMR = -8.0
PRIMARY_STRATIFY_MIN_COUNT = 10
COARSE_STRATIFY_MIN_COUNT = 4
COARSE_ABUNDANCE_QUANTILES = (0.25, 0.5, 0.75)

DEFAULT_DATA_ROOT = Path("data/ariel-ml-dataset")
DEFAULT_OUTPUT_DIR = Path("outputs/ariel_quantum_regression")
DEFAULT_PREPARED_CACHE_SUBDIR = "prepared_cache"
