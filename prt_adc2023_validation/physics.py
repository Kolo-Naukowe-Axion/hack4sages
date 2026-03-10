"""Physics helpers for the validation-set generator."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


G_SI = 6.67430e-11
JUPITER_RADIUS_M = 7.1492e7
PARSEC_M = 3.085677581491367e16
PARSEC_CM = PARSEC_M * 100.0
AU_M = 1.495978707e11
DAY_S = 86400.0


def compute_pt_summary(theta: Dict[str, float]) -> Dict[str, float]:
    """Return the transformed PT parameters used internally by pRT."""

    t_int = float(theta["t_int"])
    t_connect = ((3.0 / 4.0) * (t_int ** 4) * (0.1 + 2.0 / 3.0)) ** 0.25
    t3 = t_connect * (1.0 - float(theta["t3_unit"]))
    t2 = t3 * (1.0 - float(theta["t2_unit"]))
    t1 = t2 * (1.0 - float(theta["t1_unit"]))
    alpha = float(theta["alpha"])
    log_delta_unit = float(theta["log_delta_unit"])
    delta = (1.0e6 * (10.0 ** (-3.0 + 5.0 * log_delta_unit))) ** (-alpha)

    return {
        "t_connect": t_connect,
        "t3": t3,
        "t2": t2,
        "t1": t1,
        "delta": delta,
    }


def planet_radius_m(r_p_r_jup: float) -> float:
    """Convert Jupiter radii to meters."""

    return float(r_p_r_jup) * JUPITER_RADIUS_M


def surface_gravity_m_s2(log_g_cgs: float) -> float:
    """Convert log10(surface gravity [cm/s^2]) to m/s^2."""

    return (10.0 ** float(log_g_cgs)) / 100.0


def planet_mass_kg_from_logg_and_radius(log_g_cgs: float, r_p_r_jup: float) -> float:
    """Derive mass from surface gravity and radius."""

    gravity = surface_gravity_m_s2(log_g_cgs)
    radius = planet_radius_m(r_p_r_jup)
    return gravity * (radius ** 2) / G_SI


def orbital_period_days(star_mass_kg: float, planet_distance_au: float) -> float:
    """Return Keplerian orbital period in days for M_p << M_star."""

    semi_major_axis_m = float(planet_distance_au) * AU_M
    period_s = 2.0 * math.pi * math.sqrt((semi_major_axis_m ** 3) / (G_SI * float(star_mass_kg)))
    return period_s / DAY_S


def distance_pc_to_cm(distance_pc: float) -> float:
    """Convert parsecs to centimeters for pRT's D_pl parameter."""

    return float(distance_pc) * PARSEC_CM


def formula_tolerances() -> Dict[str, float]:
    """Numeric tolerances used by the validator."""

    return {
        "wl_abs": 1.0e-10,
        "physics_rel": 1.0e-10,
        "noise_rel": 5.0e-2,
    }

