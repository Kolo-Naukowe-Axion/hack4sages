"""Constants shared by the cross-generator FMPE adapter."""

from __future__ import annotations

DATASET_TYPE = "CrossGenRealFullNormalizedArielDataset"
NORMALIZATION_MODE = "upstream_real_full_normalized_v1"

SAFE_AUX_FEATURE_COLS = [
    "planet_radius_rjup",
    "log_g_cgs",
    "temperature_k",
    "star_radius_rsun",
]

LEAKY_FEATURE_COLS = [
    "trace_vmr_total",
    "vmr_h2",
    "vmr_he",
    "transit_depth_noiseless",
]

TARGET_COLS = [
    "log10_vmr_h2o",
    "log10_vmr_co2",
    "log10_vmr_co",
    "log10_vmr_ch4",
    "log10_vmr_nh3",
]

SPLIT_SPECS = {
    "tau_train": {"generator": "tau", "split": "train"},
    "tau_val": {"generator": "tau", "split": "val"},
    "poseidon_holdout": {"generator": "poseidon", "split": "test"},
}

NOISY_SPECTRA_DATASET = "transit_depth_noisy"
NOISE_FIELD_NAME = "log10_sigma_ppm"

SPECTRA_FILENAME_TEMPLATE = "{split_name}_spectra.npy"
NOISE_FILENAME_TEMPLATE = "{split_name}_noise_scalar.npy"
AUX_FILENAME_TEMPLATE = "{split_name}_aux.npy"
TARGET_FILENAME_TEMPLATE = "{split_name}_targets.npy"
METADATA_FILENAME_TEMPLATE = "{split_name}_metadata.csv"

NORMALIZATION_FILENAME = "normalization_stats.npz"
MANIFEST_FILENAME = "manifest.json"
WAVELENGTH_FILENAME = "wavelength_um.npy"

CONTEXT_DIM = 218 + 1 + len(SAFE_AUX_FEATURE_COLS)
THETA_DIM = len(TARGET_COLS)
