"""Pure spectrum perturbations for the interactive what-if panel (Osoba 4).

Each function returns a NEW immutable ``RawSpectrum`` / ``PlanetRecord`` - the
input is never mutated - so the UI rebuilds the model input from scratch on every
slider move. This is the functional paradigm feeding the OOP model: pure
functions + immutable data in, a fresh prediction out.
"""

from __future__ import annotations

import numpy as np

from app.data.types import PlanetRecord, RawSpectrum


def _respawn(spectrum: RawSpectrum, flux: np.ndarray) -> RawSpectrum:
    """Build a fresh RawSpectrum with a new flux, copying the other channels."""
    return RawSpectrum(
        spectrum.planet_id,
        flux=np.asarray(flux, dtype=np.float32),
        noise=np.asarray(spectrum.noise, dtype=np.float32).copy(),
        width=np.asarray(spectrum.width, dtype=np.float32).copy(),
        wavelength=np.asarray(spectrum.wavelength, dtype=np.float32).copy(),
    )


def scale_features(spectrum: RawSpectrum, depth: float) -> RawSpectrum:
    """Amplify (depth>1) or flatten (depth<1) absorption features by scaling the
    flux deviations from its mean. depth=1 leaves the spectrum unchanged."""
    flux = np.asarray(spectrum.flux, dtype=np.float64)
    mean = float(flux.mean())
    return _respawn(spectrum, mean + (flux - mean) * float(depth))


def add_noise(spectrum: RawSpectrum, level: float, seed: int = 0) -> RawSpectrum:
    """Add Gaussian noise equal to ``level`` times the per-bin measurement-noise
    channel. Deterministic (fixed seed) so the same setting is stable on rerun."""
    if level <= 0.0:
        return _respawn(spectrum, spectrum.flux)
    rng = np.random.default_rng(seed)
    flux = np.asarray(spectrum.flux, dtype=np.float64)
    sigma = np.abs(np.asarray(spectrum.noise, dtype=np.float64)) * float(level)
    return _respawn(spectrum, flux + rng.standard_normal(flux.shape) * sigma)


def tilt(spectrum: RawSpectrum, slope: float) -> RawSpectrum:
    """Add a linear continuum slope across the band (as a fraction of the mean)."""
    flux = np.asarray(spectrum.flux, dtype=np.float64)
    ramp = np.linspace(-0.5, 0.5, flux.shape[0])
    return _respawn(spectrum, flux + ramp * float(slope) * float(flux.mean()))


def perturbed_record(record: PlanetRecord, *, depth: float, noise: float, slope: float) -> PlanetRecord:
    """Compose the perturbations into a fresh immutable PlanetRecord (aux + truth kept)."""
    spectrum = tilt(add_noise(scale_features(record.spectrum, depth), noise), slope)
    return PlanetRecord(spectrum=spectrum, aux=record.aux, truth=record.truth)
