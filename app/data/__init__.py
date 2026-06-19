"""ExoBiome data layer (Osoba 2): immutable contracts + pure data pipeline.

Layout:
  types         — frozen dataclasses (the shared contract)
  loading       — I/O boundary: load by id, parse/export upload, lazy iteration
  preprocessing — pure transforms producing the quantum model's input
  pipeline      — functional composition (pipe/compose, decorators, closures)
  comparison    — pure comparison of model predictions vs ground truth
"""

from .types import (
    AUX_COLS,
    GASES,
    AuxFeatures,
    ComparisonRow,
    CuratedPlanet,
    DataError,
    GroundTruth,
    PlanetRecord,
    Prediction,
    RawSpectrum,
)

__all__ = [
    "AUX_COLS",
    "GASES",
    "AuxFeatures",
    "ComparisonRow",
    "CuratedPlanet",
    "DataError",
    "GroundTruth",
    "PlanetRecord",
    "Prediction",
    "RawSpectrum",
]
