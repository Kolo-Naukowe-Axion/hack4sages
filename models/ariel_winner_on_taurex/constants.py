"""Constants for the TauREx adaptation of the winner-style five-gas independent NSF model."""

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

TARGET_COLUMNS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]

SPECTRAL_LENGTH = 218
ENGINEERED_AUX_COLUMNS = [
    *AUX_COLUMNS,
    "radius_from_spectrum_jupiter",
    "radius_from_gravity_jupiter",
    "gravity_radius_vertical_line",
    "radius_combo_jupiter",
]

RJUP_M = 69_911_000.0
G_NEWTON = 6.674e-11
SOLAR_RADIUS_M = 6.957e8
SOLAR_MASS_KG = 1.98847e30
AU_M = 1.495978707e11
SECONDS_PER_DAY = 86_400.0
NOISE_PPM_TO_TRANSIT_DEPTH = 1.0e-6

FIXED_STAR_DISTANCE_PC = 10.0
FIXED_STAR_MASS_KG = SOLAR_MASS_KG
FIXED_STAR_TEMPERATURE_K = 5_500.0
FIXED_PLANET_DISTANCE_AU = 0.05

TRAIN_GENERATOR = "tau"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "val"
HOLDOUT_GENERATOR = "poseidon"
HOLDOUT_SPLIT = "test"

GENERATOR_ID_OFFSETS = {
    "tau": 0,
    "poseidon": 1_000_000,
}

VERTICAL_LINE_LOW = 0.71488e8 / RJUP_M
VERTICAL_LINE_HIGH = 0.714881e8 / RJUP_M

DEFAULT_DATA_ROOT = Path("data/TauREx set")
DEFAULT_PREPARED_ROOT = Path("data/generated-data/ariel_winner_on_taurex_prepared")
DEFAULT_RUNS_ROOT = Path("local_runs")
