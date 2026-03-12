"""Constants for the Ariel winner-family tracedata rerun package."""

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
    "planet_radius",
    "planet_temp",
    "log_H2O",
    "log_CO2",
    "log_CO",
    "log_CH4",
    "log_NH3",
]

FIVE_GAS_TARGET_COLUMNS = TARGET_COLUMNS[2:]

QUARTILE_COLUMN_GROUPS = {
    "planet_radius": ("planet_radius_q1", "planet_radius_q2", "planet_radius_q3"),
    "planet_temp": ("T_q1", "T_q2", "T_q3"),
    "log_H2O": ("log_H2O_q1", "log_H2O_q2", "log_H2O_q3"),
    "log_CO2": ("log_CO2_q1", "log_CO2_q2", "log_CO2_q3"),
    "log_CO": ("log_CO_q1", "log_CO_q2", "log_CO_q3"),
    "log_CH4": ("log_CH4_q1", "log_CH4_q2", "log_CH4_q3"),
    "log_NH3": ("log_NH3_q1", "log_NH3_q2", "log_NH3_q3"),
}

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
DEFAULT_PREPARED_ROOT = Path("data/generated-data/ariel_winner_trace_nf_prepared")
DEFAULT_RUNS_ROOT = Path("local_runs")

TRACE_FILE = Path("TrainingData/Ground Truth Package/Tracedata.hdf5")
QUARTILES_FILE = Path("TrainingData/Ground Truth Package/QuartilesTable.csv")
FM_TARGETS_FILE = Path("TrainingData/Ground Truth Package/FM_Parameter_Table.csv")
TRAIN_AUX_FILE = Path("TrainingData/AuxillaryTable.csv")
TRAIN_SPECTRA_FILE = Path("TrainingData/SpectralData.hdf5")
TEST_AUX_FILE = Path("TestData/AuxillaryTable.csv")
TEST_SPECTRA_FILE = Path("TestData/SpectralData.hdf5")
