"""Physics and sampling helpers for the transmission benchmark."""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Tuple, Union

import numpy as np

from .constants import (
    COOL_MAXIMA,
    H2_HE_REMAINDER_RATIO,
    HEAVY_SPECIES_MAX_FRACTION,
    HOT_MAXIMA,
    MOLECULAR_WEIGHTS_AMU,
    PRIMARY_TARGET_SPECIES,
    SPECIES_LOG_X_BOUNDS,
    STAR_CLASS_RANGES,
    STAR_CLASS_WEIGHTS,
    TEMPERATURE_REGIMES,
)

G_SI = 6.67430e-11
AU_M = 1.495978707e11
DAY_S = 86400.0
SOLAR_MASS_KG = 1.98847e30
SOLAR_RADIUS_M = 6.957e8
JUPITER_RADIUS_M = 7.1492e7
PARSEC_M = 3.085677581491367e16
LIGHT_SPEED_CGS = 2.99792458e10
SIGMA_SB = 5.670374419e-8


def stable_hash_u64(*parts: object) -> int:
    """Return a stable unsigned 64-bit integer from arbitrary parts."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def choose_weighted(options: Iterable[Tuple[str, float]], rng: np.random.Generator) -> str:
    names = []
    weights = []
    for name, weight in options:
        names.append(name)
        weights.append(float(weight))
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    return str(rng.choice(np.asarray(names, dtype=object), p=probabilities))


def sample_temperature_regime(rng: np.random.Generator) -> Tuple[str, float]:
    regime = choose_weighted(((name, weight) for name, weight, _ in TEMPERATURE_REGIMES), rng)
    bounds = {name: value_range for name, _, value_range in TEMPERATURE_REGIMES}[regime]
    return regime, float(rng.uniform(bounds[0], bounds[1]))


def sample_star_metadata(rng: np.random.Generator) -> Dict[str, Union[float, str]]:
    star_class = choose_weighted(STAR_CLASS_WEIGHTS, rng)
    ranges = STAR_CLASS_RANGES[star_class]
    return {
        "star_class": star_class,
        "star_mass_solar": float(rng.uniform(*ranges["mass_solar"])),
        "star_radius_solar": float(rng.uniform(*ranges["radius_solar"])),
        "star_temperature_k": float(rng.uniform(*ranges["temperature_k"])),
        "star_distance_pc": float(rng.uniform(*ranges["distance_pc"])),
    }


def planet_radius_m(radius_r_jup: float) -> float:
    return float(radius_r_jup) * JUPITER_RADIUS_M


def star_radius_m(radius_solar: float) -> float:
    return float(radius_solar) * SOLAR_RADIUS_M


def star_mass_kg(mass_solar: float) -> float:
    return float(mass_solar) * SOLAR_MASS_KG


def surface_gravity_m_s2(log_g_cgs: float) -> float:
    return (10.0 ** float(log_g_cgs)) / 100.0


def surface_gravity_cgs(log_g_cgs: float) -> float:
    return 10.0 ** float(log_g_cgs)


def planet_mass_kg_from_logg_and_radius(log_g_cgs: float, radius_r_jup: float) -> float:
    gravity = surface_gravity_m_s2(log_g_cgs)
    radius = planet_radius_m(radius_r_jup)
    return gravity * radius * radius / G_SI


def semi_major_axis_m(star_radius_m_value: float, star_temperature_k: float, equilibrium_temperature_k: float) -> float:
    return 0.5 * float(star_radius_m_value) * (float(star_temperature_k) / float(equilibrium_temperature_k)) ** 2


def orbital_period_days(star_mass_kg_value: float, semi_major_axis_m_value: float) -> float:
    period_s = 2.0 * math.pi * math.sqrt((semi_major_axis_m_value ** 3) / (G_SI * float(star_mass_kg_value)))
    return period_s / DAY_S


def insolation_flux_w_m2(star_temperature_k: float, star_radius_m_value: float, semi_major_axis_m_value: float) -> float:
    return SIGMA_SB * (float(star_temperature_k) ** 4) * (float(star_radius_m_value) / float(semi_major_axis_m_value)) ** 2


def ppm_to_transit_depth(ppm: Union[np.ndarray, float]) -> np.ndarray:
    return np.asarray(ppm, dtype=np.float64) * 1.0e-6


def sample_log_abundances(rng: np.random.Generator, temperature_regime: str) -> Dict[str, float]:
    log_x: Dict[str, float] = {}
    for species in PRIMARY_TARGET_SPECIES:
        lower, upper = SPECIES_LOG_X_BOUNDS[species]
        if temperature_regime == "cool" and species in COOL_MAXIMA:
            upper = min(upper, COOL_MAXIMA[species])
        if temperature_regime == "hot" and species in HOT_MAXIMA:
            upper = min(upper, HOT_MAXIMA[species])
        log_x[species] = float(rng.uniform(lower, upper))
    return log_x


def build_mass_fraction_profile(log_x: Dict[str, float], pressure_levels: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, float]:
    heavy_fraction = 0.0
    abundances: Dict[str, np.ndarray] = {}
    for species, value in log_x.items():
        mass_fraction = 10.0 ** float(value)
        heavy_fraction += mass_fraction
        abundances[species] = np.full(pressure_levels, mass_fraction, dtype=np.float64)

    if heavy_fraction >= HEAVY_SPECIES_MAX_FRACTION:
        raise ValueError("Heavy-species fraction exceeds the configured maximum.")

    remaining = 1.0 - heavy_fraction
    h2_fraction = remaining * H2_HE_REMAINDER_RATIO[0]
    he_fraction = remaining * H2_HE_REMAINDER_RATIO[1]
    abundances["H2"] = np.full(pressure_levels, h2_fraction, dtype=np.float64)
    abundances["He"] = np.full(pressure_levels, he_fraction, dtype=np.float64)

    inverse_mmw = 0.0
    for species, profile in abundances.items():
        inverse_mmw += float(profile[0]) / MOLECULAR_WEIGHTS_AMU[species]
    mmw = np.full(pressure_levels, 1.0 / inverse_mmw, dtype=np.float64)
    return abundances, mmw, heavy_fraction


def sample_star_and_orbit(
    rng: np.random.Generator,
    terminator_temperature_k: float,
    min_axis_to_stellar_radius: float,
) -> Dict[str, Union[float, str]]:
    for _ in range(32):
        star = sample_star_metadata(rng)
        radius_m = star_radius_m(float(star["star_radius_solar"]))
        mass_kg = star_mass_kg(float(star["star_mass_solar"]))
        semi_major_m = semi_major_axis_m(radius_m, float(star["star_temperature_k"]), terminator_temperature_k)
        if semi_major_m / radius_m < min_axis_to_stellar_radius:
            continue
        return {
            **star,
            "star_radius_m": radius_m,
            "star_mass_kg": mass_kg,
            "semi_major_axis_m": semi_major_m,
            "semi_major_axis_au": semi_major_m / AU_M,
            "orbital_period_days": orbital_period_days(mass_kg, semi_major_m),
            "insolation_flux_w_m2": insolation_flux_w_m2(float(star["star_temperature_k"]), radius_m, semi_major_m),
        }
    raise RuntimeError("Unable to sample a physically consistent star/orbit configuration.")


def make_species_presence_labels(log_x: Dict[str, float], threshold: float) -> Dict[str, int]:
    return {f"present_{species}": int(value >= threshold) for species, value in log_x.items()}


def is_ood_candidate(record: Dict[str, Union[float, str, int]]) -> bool:
    hot_hazy_optical = (
        record["temperature_regime"] == "hot"
        and (int(record["present_TiO"]) == 1 or int(record["present_VO"]) == 1)
        and float(record["gamma_scat"]) > -6.5
        and float(record["log10_p_cloud_bar"]) < -2.0
    )
    return bool(hot_hazy_optical)
