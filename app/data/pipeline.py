"""Functional composition utilities and the quantum input pipeline

Paradigm: higher-order functions, closures, decorators. ``pipe``/``compose``
take functions and return a new function; ``make_quantum_input`` closes over the
two scalers and returns a ready-to-use transformation.
"""

from __future__ import annotations

import time
from functools import reduce, wraps
from typing import Callable

import numpy as np

from . import preprocessing as pp
from .types import PlanetRecord

Func = Callable[..., object]


def pipe(*funcs: Func) -> Func:
    """Left-to-right function composition: ``pipe(f, g)(x) == g(f(x))``.
    HOF returning a closure over ``funcs``.
    """
    def run(value):
        return reduce(lambda acc, fn: fn(acc), funcs, value)
    return run


def compose(*funcs: Func) -> Func:
    """Right-to-left composition: ``compose(f, g)(x) == f(g(x))``."""
    return pipe(*reversed(funcs))


def with_timing(label: str | None = None) -> Callable[[Func], Func]:
    """Decorator that logs how long the wrapped function took"""
    def decorator(fn: Func) -> Func:
        name = label or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"[timing] {name}: {elapsed_ms:.1f} ms")
            return result

        return wrapper

    return decorator


def make_quantum_input(
    aux_scaler: pp.Scaler, spectral_scaler: pp.Scaler
) -> Callable[[PlanetRecord], tuple[np.ndarray, np.ndarray]]:
    """Build the model-ready input transform for a single planet.

    Returns a closure ``record -> (aux_n (1, 8), spectra_n (1, 4, 52))`` that
    captures the two checkpoint scalers. Composed from the pure preprocessing
    steps via ``pipe``.
    """
    aux_path = pipe(pp.log_scale_aux, lambda a: pp.standardize_aux(a, aux_scaler))
    spectra_path = pipe(
        pp.to_channels_first,
        pp.normalize_sample_spectra,
        lambda s: pp.standardize_spectra(s, spectral_scaler),
    )

    def transform(record: PlanetRecord) -> tuple[np.ndarray, np.ndarray]:
        return aux_path(record.aux), spectra_path(record.spectrum)

    return transform
