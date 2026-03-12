"""Constants for the isolated TensorFlow ADC crossgen baseline."""

from __future__ import annotations

from pathlib import Path

AUX_COLUMNS = [
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
    "log10_sigma_ppm",
]

TARGET_COLUMNS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]

PRESENCE_COLUMNS = [
    "present_h2o",
    "present_co2",
    "present_co",
    "present_ch4",
    "present_nh3",
]

PRESENCE_THRESHOLD_LOG10_VMR = -8.0

DEFAULT_DATA_ROOT = Path("data/generated-data/crossgen_biosignatures_20260311")
DEFAULT_OUTPUT_DIR = Path("outputs/adc_crossgen_tf")

DEFAULT_FILTERS = (32, 64, 64)
