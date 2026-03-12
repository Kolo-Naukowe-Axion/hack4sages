"""Constants for the ADC2023 winner-style five-gas independent NSF model."""

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
    "log_H2O",
    "log_CO2",
    "log_CO",
    "log_CH4",
    "log_NH3",
]

SPECTRAL_LENGTH = 52
ENGINEERED_AUX_COLUMNS = [
    *AUX_COLUMNS,
    "radius_from_spectrum_jupiter",
    "radius_from_gravity_jupiter",
    "gravity_radius_vertical_line",
    "radius_combo_jupiter",
]

RJUP_M = 69_911_000.0
G_NEWTON = 6.674e-11
VERTICAL_LINE_LOW = 0.71488e8 / RJUP_M
VERTICAL_LINE_HIGH = 0.714881e8 / RJUP_M

DEFAULT_DATA_ROOT = Path("data/full-ariel")
DEFAULT_SPLIT_ROOT = Path("data/val_dataset")
DEFAULT_PREPARED_ROOT = Path("data/generated-data/ariel_winner_nf_prepared")
DEFAULT_RUNS_ROOT = Path("local_runs")
