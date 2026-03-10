"""Canonical constants for the pRT-based ADC2023 validation-set generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SCALE_FACTOR = 1.0e16
SIGMA_MIN_SCALED = 0.05
SIGMA_MAX_SCALED = 0.50
SIGMA_UNIT_W_M2_UM = 1.0e-16
PRESSURE_SCALING = 10
PRESSURE_SIMPLE = 100
PRESSURE_WIDTH = 3
CANONICAL_SAMPLE_COUNT = 20_000
DEFAULT_SHARD_SIZE = 250
DEFAULT_RANDOM_SEED = 24032026
DEFAULT_NEIGHBOR_COUNT = 64
DEFAULT_WORKERS = 4
PLANET_ID_PREFIX = "val"
HDF5_GROUP_PREFIX = "Planet_"

ADC_AUX_COLUMNS = [
    "planet_ID",
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
]

EMPIRICAL_AUX_COLUMNS = [
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_distance",
    "planet_orbital_period",
]

EMPIRICAL_FEATURE_COLUMNS = [
    "planet_mass_kg",
    "planet_surface_gravity",
    "planet_radius",
    "planet_temp",
]

PAPER_PARAMETER_COLUMNS = [
    "c_o",
    "fe_h",
    "log_p_quench",
    "s_eq_fe",
    "s_eq_mgsio3",
    "f_sed",
    "log_kzz",
    "sigma_g",
    "log_g",
    "r_p_r_jup",
    "t_int",
    "t3_unit",
    "t2_unit",
    "t1_unit",
    "alpha",
    "log_delta_unit",
]

LINE_SPECIES_R400 = [
    "H2O_HITEMP_R_400",
    "CO_all_iso_HITEMP_R_400",
    "CH4_R_400",
    "NH3_R_400",
    "CO2_R_400",
    "H2S_R_400",
    "VO_R_400",
    "TiO_all_Exomol_R_400",
    "PH3_R_400",
    "Na_allard_R_400",
    "K_allard_R_400",
]

CLOUD_SPECIES = ["MgSiO3(c)_cd", "Fe(c)_cd"]
RAYLEIGH_SPECIES = ["H2", "He"]
CONTINUUM_OPACITIES = ["H2-H2", "H2-He"]

OFFICIAL_BASELINE_WLGRID_ASC = np.array(
    [
        0.55,
        0.7,
        0.95,
        1.156375,
        1.27490344,
        1.40558104,
        1.5496531,
        1.70849254,
        1.88361302,
        1.9695975,
        2.00918641,
        2.04957106,
        2.09076743,
        2.13279186,
        2.17566098,
        2.21939176,
        2.26400154,
        2.30950797,
        2.35592908,
        2.40328325,
        2.45158925,
        2.50086619,
        2.5511336,
        2.60241139,
        2.65471985,
        2.70807972,
        2.76251213,
        2.81803862,
        2.8746812,
        2.93246229,
        2.99140478,
        3.05153202,
        3.11286781,
        3.17543645,
        3.23926272,
        3.30437191,
        3.37078978,
        3.43854266,
        3.50765736,
        3.57816128,
        3.65008232,
        3.72344897,
        4.03216667,
        4.30545796,
        4.59727234,
        4.90886524,
        5.24157722,
        5.59683967,
        5.97618103,
        6.3812333,
        6.81373911,
        7.2755592,
    ],
    dtype=np.float64,
)

OFFICIAL_BASELINE_WLWIDTH_ASC = np.array(
    [
        0.10083333,
        0.20416667,
        0.30767045,
        0.11301861,
        0.12460302,
        0.13737483,
        0.15145575,
        0.16697996,
        0.18409541,
        0.03919888,
        0.03998678,
        0.04079051,
        0.0416104,
        0.04244677,
        0.04329995,
        0.04417028,
        0.0450581,
        0.04596377,
        0.04688764,
        0.04783008,
        0.04879147,
        0.04977218,
        0.0507726,
        0.05179313,
        0.05283417,
        0.05389614,
        0.05497945,
        0.05608453,
        0.05721183,
        0.05836179,
        0.05953486,
        0.06073151,
        0.06195222,
        0.06319746,
        0.06446773,
        0.06576353,
        0.06708537,
        0.06843379,
        0.06980931,
        0.07121248,
        0.07264385,
        0.07410399,
        0.26461764,
        0.28255283,
        0.30170364,
        0.32215244,
        0.34398722,
        0.36730191,
        0.39219681,
        0.41877904,
        0.44716295,
        0.47747067,
    ],
    dtype=np.float64,
)

# This comes from the local ADC dataset HDF5 and is the authoritative output field.
ADC_OUTPUT_INSTRUMENT_WIDTH_ASC = np.array(
    [
        1.0083333333333342,
        2.041666666666666,
        3.0767045454545454,
        1.130186121739875,
        1.246030199218751,
        1.3737482946385539,
        1.514557494839177,
        1.6697996380602298,
        1.8409541009613477,
        3.9198880789132777,
        3.9986778292990516,
        4.079051253668244,
        4.161040183866361,
        4.244677091561767,
        4.329995101101734,
        4.417028002634619,
        4.505810265488258,
        4.596377051823551,
        4.6887642305662255,
        4.783008391600796,
        4.879146860270672,
        4.977217712162857,
        5.077259788177438,
        5.179312709919472,
        5.283416895388932,
        5.389613574986215,
        5.497944807843348,
        5.608453498480933,
        5.721183413800252,
        5.836179200417958,
        5.953486402345644,
        6.073151479032627,
        6.195221823761604,
        6.319745782419926,
        6.446772672645854,
        6.576352803366364,
        6.708537494713971,
        6.843379098358031,
        6.980931018234507,
        7.121247731700565,
        7.2643848111077845,
        7.41039894581171,
        2.646176391297556,
        2.825528346708357,
        3.017036379095571,
        3.2215244003457224,
        3.4398721652579624,
        3.6730190564589424,
        3.921968125840975,
        4.187790409925792,
        4.471629537709747,
        4.774706650820687,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class ParameterPrior:
    """Uniform prior for one of the 16 paper parameters."""

    name: str
    lower: float
    upper: float


PAPER_PRIORS = [
    ParameterPrior("c_o", 0.1, 1.6),
    ParameterPrior("fe_h", -1.5, 1.5),
    ParameterPrior("log_p_quench", -6.0, 3.0),
    ParameterPrior("s_eq_fe", -2.3, 1.0),
    ParameterPrior("s_eq_mgsio3", -2.3, 1.0),
    ParameterPrior("f_sed", 0.0, 10.0),
    ParameterPrior("log_kzz", 5.0, 13.0),
    ParameterPrior("sigma_g", 1.05, 3.0),
    ParameterPrior("log_g", 2.0, 5.5),
    ParameterPrior("r_p_r_jup", 0.9, 2.0),
    ParameterPrior("t_int", 300.0, 2300.0),
    ParameterPrior("t3_unit", 0.0, 1.0),
    ParameterPrior("t2_unit", 0.0, 1.0),
    ParameterPrior("t1_unit", 0.0, 1.0),
    ParameterPrior("alpha", 1.0, 2.0),
    ParameterPrior("log_delta_unit", 0.0, 1.0),
]


def canonical_output_wlgrid() -> np.ndarray:
    """Return the local ADC output wavelength grid in ascending micron order."""

    return OFFICIAL_BASELINE_WLGRID_ASC.copy()


def canonical_output_instrument_width() -> np.ndarray:
    """Return the local ADC output metadata width/resolution field."""

    return ADC_OUTPUT_INSTRUMENT_WIDTH_ASC.copy()


def official_binning_wlgrid_desc() -> np.ndarray:
    """Return the official baseline wl grid used to derive the TauREx binning object."""

    return OFFICIAL_BASELINE_WLGRID_ASC[::-1].copy()


def official_binning_wlwidth_desc() -> np.ndarray:
    """Return the official baseline wavelength-bin widths in micron."""

    return OFFICIAL_BASELINE_WLWIDTH_ASC[::-1].copy()
