"""Pure preprocessing for the quantum model's input.

Paradigm: pure functions. Each function maps input to output with no side
effects and no mutation of its arguments, so the transformation is testable.

The steps mirror exactly what the model saw at training time.
  * aux: log10 on 7 of 8 features (all but ``star_temperature``), then z-score
    with the *checkpoint's* scaler;
  * spectra: keep the 2 learned channels (flux, noise) channels-first; the
    spectral scaler z-scores them and appends the 2 fixed channels itself;
  * targets: model output is in z-score space, must be inverse-transformed.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import AUX_COLS, GASES, LOG10_AUX_COLS, AuxFeatures, RawSpectrum

# indices of aux features that get log10-scaled
_LOG10_IDX = tuple(i for i, col in enumerate(AUX_COLS) if col in LOG10_AUX_COLS)

# the 2 learned spectral channels, in model order (flux, then noise)
_LEARNED_CHANNELS = ("flux", "noise")


class Scaler(Protocol):
    def transform(self, values: np.ndarray) -> np.ndarray: ...


class InvertibleScaler(Protocol):
    def inverse_transform(self, values: np.ndarray) -> np.ndarray: ...


def log_scale_aux(aux: AuxFeatures) -> np.ndarray:
    """Apply log10 to the 7 log-scaled aux features; return a fresh (8,) array.

    Pure: the input ``aux.values`` is copied, never mutated.
    """
    out = aux.values.astype(np.float32, copy=True)
    for idx in _LOG10_IDX:
        out[idx] = np.log10(np.clip(out[idx], 1.0e-12, None))
    return out


def to_channels_first(spectrum: RawSpectrum) -> np.ndarray:
    """Stack the 2 learned channels into a (2, 52) array (channels-first)."""
    return np.stack([getattr(spectrum, ch) for ch in _LEARNED_CHANNELS], axis=0).astype(np.float32)


def normalize_sample_spectra(spectra_2ch: np.ndarray) -> np.ndarray:
    """Divide both learned channels by the per-sample mean of the *spectrum*
    channel (channel 0) — the ``divide_by_sample_mean`` step from training,
    applied before standardization (``models/taurex_exobiome/dataset.py``).

    Accepts (2, 52) or (N, 2, 52); returns the same rank. Pure (copies input).
    """
    arr = np.asarray(spectra_2ch, dtype=np.float32)
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]
    reference = arr[:, 0, :]
    sample_mean = np.clip(reference.mean(axis=1, keepdims=True), 1.0e-12, None)
    out = (arr / sample_mean[:, None, :]).astype(np.float32)
    return out[0] if single else out


def standardize_aux(aux_log: np.ndarray, scaler: Scaler) -> np.ndarray:
    """z-score the (already log-scaled) aux features. Input/output 2D (N, 8)."""
    return scaler.transform(np.atleast_2d(aux_log).astype(np.float32))


def standardize_spectra(spectra_2ch: np.ndarray, scaler: Scaler) -> np.ndarray:
    """z-score the 2 learned channels; the scaler appends the fixed channels.

    Input (N, 2, 52) → output (N, 4, 52).
    """
    spectra_2ch = np.asarray(spectra_2ch, dtype=np.float32)
    if spectra_2ch.ndim == 2:  # (2, 52) -> (1, 2, 52)
        spectra_2ch = spectra_2ch[None, ...]
    return scaler.transform(spectra_2ch)


def denormalize_targets(pred_norm: np.ndarray, scaler: InvertibleScaler) -> np.ndarray:
    """Map model output from z-score space back to physical log VMR. (N, 5)."""
    return scaler.inverse_transform(np.atleast_2d(pred_norm).astype(np.float32))


def vmr_to_dict(row: np.ndarray) -> dict[str, float]:
    """Turn a (5,) physical-prediction row into a {gas: value} mapping (pure)."""
    return {gas: float(v) for gas, v in zip(GASES, np.asarray(row).ravel())}
