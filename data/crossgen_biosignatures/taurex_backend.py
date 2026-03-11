"""TauREx backend wrapper for the cross-generator dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    BACKGROUND_HE_TO_H2_RATIO,
    FIXED_PLANET_SEMIMAJOR_AXIS_AU,
    FIXED_STAR_DISTANCE_PC,
    FIXED_STAR_MASS_MSUN,
    FIXED_STAR_METALLICITY,
    FIXED_STAR_TEMPERATURE_K,
    GRAVITATIONAL_CONSTANT_SI,
    JUPITER_MASS_KG,
    JUPITER_RADIUS_M,
    PRESSURE_LEVELS,
    PRESSURE_MAX_BAR,
    PRESSURE_MIN_BAR,
    TRACE_SPECIES,
)


def _candidate_ktable_dirs() -> list[Path]:
    env_path = os.environ.get("CROSSGEN_TAUREX_KTABLE_DIR")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    home = Path.home()
    candidates.extend(
        [
            home / ".cache" / "crossgen-taurex-ktables",
            home / "crossgen-taurex-ktables",
        ]
    )
    return candidates


def resolve_taurex_ktable_dir() -> Path:
    """Locate the staged flat k-table directory used by TauREx."""

    required_files = tuple(f"{species_name}.h5" for _, species_name in TRACE_SPECIES)
    for candidate in _candidate_ktable_dirs():
        if candidate.is_dir() and all((candidate / filename).exists() for filename in required_files):
            return candidate

    expected = ", ".join(required_files)
    searched = ", ".join(str(path) for path in _candidate_ktable_dirs())
    raise FileNotFoundError(
        f"TauREx k-table directory not found. Expected {expected} in one of: {searched}"
    )


class TauRExBackend:
    """Render native transmission spectra with TauREx."""

    generator_key = "tau"

    def __init__(self) -> None:
        self._taurex_module: Any | None = None

    def _load_imports(self) -> dict[str, Any]:
        if self._taurex_module is None:
            import h5py
            import taurex
            from taurex.cache import GlobalCache
            from taurex.cache.ktablecache import KTableCache
            from taurex.chemistry import ConstantGas, TaurexChemistry
            from taurex.contributions import AbsorptionContribution, RayleighContribution
            from taurex.model import TransmissionModel
            from taurex.planet import Planet
            from taurex.pressure import SimplePressureProfile
            from taurex.stellar import BlackbodyStar
            from taurex.temperature import Isothermal

            if not hasattr(h5py.Dataset, "copy"):
                h5py.Dataset.copy = lambda self: self[()]  # type: ignore[attr-defined]

            self._taurex_module = {
                "module": taurex,
                "GlobalCache": GlobalCache,
                "KTableCache": KTableCache,
                "ConstantGas": ConstantGas,
                "TaurexChemistry": TaurexChemistry,
                "AbsorptionContribution": AbsorptionContribution,
                "RayleighContribution": RayleighContribution,
                "TransmissionModel": TransmissionModel,
                "Planet": Planet,
                "SimplePressureProfile": SimplePressureProfile,
                "BlackbodyStar": BlackbodyStar,
                "Isothermal": Isothermal,
            }
        return self._taurex_module

    def software_versions(self) -> dict[str, str]:
        imports = self._load_imports()
        module = imports["module"]
        return {"taurex": getattr(module, "__version__", "unknown")}

    def render_native(self, sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        imports = self._load_imports()
        ktable_dir = resolve_taurex_ktable_dir()

        global_cache = imports["GlobalCache"]()
        global_cache["opacity_method"] = "ktables"
        global_cache["ktable_path"] = str(ktable_dir)
        global_cache["xsec_interpolation"] = "linear"
        global_cache["deactive_molecules"] = None

        ktable_cache = imports["KTableCache"]()
        ktable_cache.set_ktable_path(str(ktable_dir))
        ktable_cache.clear_cache()

        chemistry = imports["TaurexChemistry"](fill_gases=["H2", "He"], ratio=BACKGROUND_HE_TO_H2_RATIO)
        for species_key, species_name in TRACE_SPECIES:
            chemistry.addGas(imports["ConstantGas"](species_name, mix_ratio=10.0 ** float(sample[f"log10_vmr_{species_key}"])))

        pressure_profile = imports["SimplePressureProfile"](
            nlayers=PRESSURE_LEVELS,
            atm_min_pressure=PRESSURE_MIN_BAR * 1.0e5,
            atm_max_pressure=PRESSURE_MAX_BAR * 1.0e5,
        )
        temperature_profile = imports["Isothermal"](T=float(sample["temperature_k"]))

        planet_radius_rjup = float(sample["planet_radius_rjup"])
        planet_radius_m = planet_radius_rjup * JUPITER_RADIUS_M
        gravity_m_s2 = (10.0 ** float(sample["log_g_cgs"])) / 100.0
        planet_mass_kg = gravity_m_s2 * (planet_radius_m ** 2) / GRAVITATIONAL_CONSTANT_SI
        planet_mass_mjup = planet_mass_kg / JUPITER_MASS_KG

        planet = imports["Planet"](
            planet_mass=planet_mass_mjup,
            planet_radius=planet_radius_rjup,
            planet_sma=FIXED_PLANET_SEMIMAJOR_AXIS_AU,
        )
        star = imports["BlackbodyStar"](
            temperature=FIXED_STAR_TEMPERATURE_K,
            radius=float(sample["star_radius_rsun"]),
            distance=FIXED_STAR_DISTANCE_PC,
            mass=FIXED_STAR_MASS_MSUN,
            metallicity=FIXED_STAR_METALLICITY,
        )
        model = imports["TransmissionModel"](
            planet=planet,
            star=star,
            pressure_profile=pressure_profile,
            temperature_profile=temperature_profile,
            chemistry=chemistry,
            contributions=[
                imports["AbsorptionContribution"](),
                imports["RayleighContribution"](),
            ],
            nlayers=PRESSURE_LEVELS,
            atm_min_pressure=PRESSURE_MIN_BAR * 1.0e5,
            atm_max_pressure=PRESSURE_MAX_BAR * 1.0e5,
        )
        model.build()
        native_wavenumber, native_depth, _, _ = model.model()
        wavelength_um = 10000.0 / np.asarray(native_wavenumber, dtype=np.float64)
        depth = np.asarray(native_depth, dtype=np.float64)
        order = np.argsort(wavelength_um)
        return wavelength_um[order], depth[order]
