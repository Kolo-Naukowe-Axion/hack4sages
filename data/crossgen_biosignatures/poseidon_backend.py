"""POSEIDON backend wrapper for the cross-generator dataset."""

from __future__ import annotations

import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    BACKGROUND_HE_TO_H2_RATIO,
    FIXED_PLANET_SEMIMAJOR_AXIS_AU,
    FIXED_STAR_LOG_G_CGS,
    FIXED_STAR_METALLICITY,
    FIXED_STAR_TEMPERATURE_K,
    FIXED_SYSTEM_DISTANCE_PC,
    JUPITER_RADIUS_M,
    PARSEC_M,
    POSEIDON_DATABASE_VERSION,
    POSEIDON_FINE_LOG10_PRESSURE_BAR,
    POSEIDON_FINE_TEMPERATURE_GRID_K,
    POSEIDON_NATIVE_RESOLUTION,
    POSEIDON_OPACITY_DATABASE,
    POSEIDON_OPACITY_TREATMENT,
    PRESSURE_LEVELS,
    PRESSURE_MAX_BAR,
    PRESSURE_MIN_BAR,
    REFERENCE_PRESSURE_BAR,
    SOLAR_RADIUS_M,
    TARGET_WAVELENGTH_MAX_UM,
    TARGET_WAVELENGTH_MIN_UM,
    TRACE_SPECIES,
)
from .utils import ensure_directory


class PoseidonBackend:
    """Render native transmission spectra with POSEIDON."""

    generator_key = "poseidon"

    def __init__(self) -> None:
        self._imports: dict[str, Any] | None = None
        self._model: Any | None = None
        self._opacities: Any | None = None
        self._native_wavelength_um: np.ndarray | None = None
        self._pressure_grid_bar = np.geomspace(PRESSURE_MIN_BAR, PRESSURE_MAX_BAR, PRESSURE_LEVELS).astype(np.float64)

    @property
    def native_wavelength_um(self) -> np.ndarray:
        self._initialise()
        assert self._native_wavelength_um is not None
        return self._native_wavelength_um

    def _configure_environment(self) -> None:
        poseidon_input_root = os.environ.get("POSEIDON_input_data")
        if poseidon_input_root is None:
            candidates = (
                Path.home() / "poseidon-inputs",
                Path.home() / "hack4sages" / "input_data",
            )
            poseidon_input_root = str(next((path for path in candidates if path.exists()), candidates[0]))
            os.environ["POSEIDON_input_data"] = poseidon_input_root

        mpl_config_dir = os.environ.get("MPLCONFIGDIR")
        if mpl_config_dir is None:
            mpl_config_dir = str(Path(tempfile.gettempdir()) / "crossgen-matplotlib")
            os.environ["MPLCONFIGDIR"] = mpl_config_dir
        ensure_directory(Path(mpl_config_dir))

    def _initialise(self) -> None:
        if self._imports is not None:
            return

        self._configure_environment()

        from POSEIDON.core import (
            compute_spectrum,
            create_planet,
            create_star,
            define_model,
            make_atmosphere,
            read_opacities,
            wl_grid_constant_R,
        )

        self._imports = {
            "compute_spectrum": compute_spectrum,
            "create_planet": create_planet,
            "create_star": create_star,
            "define_model": define_model,
            "make_atmosphere": make_atmosphere,
            "read_opacities": read_opacities,
            "wl_grid_constant_R": wl_grid_constant_R,
        }
        self._native_wavelength_um = np.asarray(
            wl_grid_constant_R(TARGET_WAVELENGTH_MIN_UM, TARGET_WAVELENGTH_MAX_UM, POSEIDON_NATIVE_RESOLUTION),
            dtype=np.float64,
        )
        self._model = define_model(
            model_name="crossgen_biosignatures",
            bulk_species=["H2", "He"],
            param_species=[species_name for _, species_name in TRACE_SPECIES],
            object_type="transiting",
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="cloud-free",
            gravity_setting="fixed",
            mass_setting="fixed",
            stellar_contam=None,
        )
        self._opacities = read_opacities(
            self._model,
            self._native_wavelength_um,
            opacity_treatment=POSEIDON_OPACITY_TREATMENT,
            T_fine=np.asarray(POSEIDON_FINE_TEMPERATURE_GRID_K, dtype=np.float64),
            log_P_fine=np.asarray(POSEIDON_FINE_LOG10_PRESSURE_BAR, dtype=np.float64),
            opacity_database=POSEIDON_OPACITY_DATABASE,
            device="cpu",
            database_version=POSEIDON_DATABASE_VERSION,
        )

    def software_versions(self) -> dict[str, str]:
        version = "unknown"
        for distribution_name in ("POSEIDON", "poseidon"):
            try:
                version = importlib.metadata.version(distribution_name)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
        return {"poseidon": version}

    def render_native(self, sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        self._initialise()
        assert self._imports is not None
        assert self._model is not None
        assert self._opacities is not None
        assert self._native_wavelength_um is not None

        star = self._imports["create_star"](
            R_s=float(sample["star_radius_rsun"]) * SOLAR_RADIUS_M,
            T_eff=FIXED_STAR_TEMPERATURE_K,
            log_g=FIXED_STAR_LOG_G_CGS,
            Met=FIXED_STAR_METALLICITY,
            stellar_grid="blackbody",
            wl=self._native_wavelength_um,
        )
        planet_radius_m = float(sample["planet_radius_rjup"]) * JUPITER_RADIUS_M
        planet = self._imports["create_planet"](
            planet_name=str(sample["sample_id"]),
            R_p=planet_radius_m,
            log_g=float(sample["log_g_cgs"]),
            T_eq=float(sample["temperature_k"]),
            d=FIXED_SYSTEM_DISTANCE_PC * PARSEC_M,
            a_p=FIXED_PLANET_SEMIMAJOR_AXIS_AU * 1.495978707e11,
        )
        atmosphere = self._imports["make_atmosphere"](
            planet=planet,
            model=self._model,
            P=self._pressure_grid_bar,
            P_ref=REFERENCE_PRESSURE_BAR,
            R_p_ref=planet_radius_m,
            PT_params=np.asarray([float(sample["temperature_k"])], dtype=np.float64),
            log_X_params=np.asarray(
                [float(sample[f"log10_vmr_{species_key}"]) for species_key, _ in TRACE_SPECIES],
                dtype=np.float64,
            ),
            He_fraction=BACKGROUND_HE_TO_H2_RATIO,
        )
        spectrum = self._imports["compute_spectrum"](
            planet=planet,
            star=star,
            model=self._model,
            atmosphere=atmosphere,
            opac=self._opacities,
            wl=self._native_wavelength_um,
            spectrum_type="transmission",
            suppress_print=True,
        )
        return self._native_wavelength_um, np.asarray(spectrum, dtype=np.float64)
