"""Immutable data contracts for the ExoBiome data layer.

Paradigm: immutability. Every record below is a frozen dataclass — once built
it cannot be mutated, which lets every downstream (pure) transformation reason
about its input. Names and column order mirror ``models/sbi_ariel_adc2023/constants.py`` so the
records map 1:1 onto what the trained models expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np


def _freeze_array(arr: np.ndarray) -> None:
    """Make an ndarray read-only in place"""
    arr.flags.writeable = False


def _frozen_mapping(mapping: dict) -> MappingProxyType:
    """Return a read-only view over a copy of mapping"""
    return MappingProxyType(dict(mapping))

# Target gases, in the exact order the models emit them
GASES: tuple[str, ...] = ("log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3")

#: Auxiliary stellar/planetary features
AUX_COLS: tuple[str, ...] = (
    "star_distance",
    "star_mass_kg",
    "star_radius_m",
    "star_temperature",
    "planet_mass_kg",
    "planet_orbital_period",
    "planet_distance",
    "planet_surface_gravity",
)

# Aux columns that are log10-scaled before standardization, everything except``star_temperature``
LOG10_AUX_COLS: frozenset[str] = frozenset(AUX_COLS) - {"star_temperature"}

# Number of wavelength bins in an ADC2023 spectrum
N_BINS: int = 52


class DataError(ValueError):
    """Raised when input data is malformed. A dedicated domain error so the UI layer can show a friendly message."""

@dataclass(frozen=True, slots=True)
class RawSpectrum:
    """One ADC2023 spectrum exactly as stored on disk (4 channels x 52 bins)"""

    planet_id: str
    flux: np.ndarray # instrument_spectrum
    noise: np.ndarray # instrument_noise
    width: np.ndarray # instrument_width
    wavelength: np.ndarray # instrument_wlgrid in micrometres

    def __post_init__(self) -> None:
        for name in ("flux", "noise", "width", "wavelength"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.shape != (N_BINS,):
                raise DataError(
                    f"RawSpectrum.{name} must be an ndarray of shape ({N_BINS},), "
                    f"got {getattr(arr, 'shape', type(arr))}."
                )
            # flux/noise drive the model — reject NaN/inf early instead of letting
            # them silently propagate into a NaN prediction.
            if name in ("flux", "noise") and not np.isfinite(arr).all():
                raise DataError(f"RawSpectrum.{name} contains non-finite values (NaN/inf).")
            _freeze_array(arr)


@dataclass(frozen=True, slots=True)
class AuxFeatures:
    """The 8 auxiliary features for one planet as AUX_FEATURES"""

    planet_id: str
    values: np.ndarray  # shape (8,)

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.shape != (len(AUX_COLS),):
            raise DataError(
                f"AuxFeatures.values must be an ndarray of shape ({len(AUX_COLS)},), "
                f"got {getattr(self.values, 'shape', type(self.values))}."
            )
        if not np.isfinite(self.values).all():
            raise DataError("AuxFeatures.values contains non-finite values (NaN/inf).")
        _freeze_array(self.values)


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """The true atmospheric abundances for one planet, if known"""
    log_vmr: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_vmr", _frozen_mapping(self.log_vmr))


@dataclass(frozen=True, slots=True)
class PlanetRecord:
    """Everything the data layer hands to a model: spectrum + aux + optional truth"""

    spectrum: RawSpectrum
    aux: AuxFeatures
    truth: GroundTruth | None = None

    @property
    def planet_id(self) -> str:
        return self.spectrum.planet_id


@dataclass(frozen=True, slots=True)
class Prediction:
    """A model's prediction for one planet — physical (de-normalized) log VMR.
    The shared output contract: both the quantum model and adc_winner return
    this, so the comparison layer treats them uniformly.
    """

    model_name: str
    log_vmr: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_vmr", _frozen_mapping(self.log_vmr))


@dataclass(frozen=True, slots=True)
class CuratedPlanet:
    """A planet offered in the app's dropdown."""
    planet_id: str
    dominant_gas: str
    label: str


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """One gas, compared across models (and against truth when available)."""
    gas: str
    true: float | None
    preds: dict[str, float] # {model_name: predicted value}
    errors: dict[str, float] = field(default_factory=dict) # {model_name: |pred - true|}

    def __post_init__(self) -> None:
        object.__setattr__(self, "preds", _frozen_mapping(self.preds))
        object.__setattr__(self, "errors", _frozen_mapping(self.errors))
