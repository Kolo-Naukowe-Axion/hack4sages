"""Constants for the TauREx non-quantum five-gas regressor."""

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

ENGINEERED_AUX_COLUMNS = [
    "raw_spectrum_mean",
    "raw_spectrum_std",
    "raw_spectrum_gradient_std",
    "raw_spectrum_curvature_std",
    "raw_noise_mean",
    "radius_from_spectrum_rjup",
    "radius_from_gravity_rjup",
    "radius_delta_rjup",
    "radius_ratio",
    "mean_signal_to_noise",
]

ENHANCED_AUX_COLUMNS = [*AUX_COLUMNS, *ENGINEERED_AUX_COLUMNS]

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

TAUREX_TARGET_COLUMNS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
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

DERIVED_SPECTRAL_CHANNELS = [
    "spectrum_gradient",
    "spectrum_curvature",
    "signal_to_noise",
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

ENHANCED_MODEL_SPECTRAL_CHANNELS = [
    *SAMPLE_SPECTRAL_CHANNELS,
    *DERIVED_SPECTRAL_CHANNELS,
    "instrument_width_template",
    "wavelength_um",
]

WAVELENGTH_DATASET = "instrument_wlgrid"
HDF5_GROUP_PREFIX = "Planet_"
PRESENCE_THRESHOLD_LOG10_VMR = -8.0
PRIMARY_STRATIFY_MIN_COUNT = 10
COARSE_STRATIFY_MIN_COUNT = 4
COARSE_ABUNDANCE_QUANTILES = (0.25, 0.5, 0.75)

SUPPORTED_FEATURE_RECIPES = ("legacy", "spectral_plus")

DEFAULT_DATA_ROOT = Path("data/ariel-ml-dataset")
DEFAULT_OUTPUT_DIR = Path("outputs/taurex_codex_without_quant")
DEFAULT_PREPARED_CACHE_SUBDIR = "prepared_cache"
