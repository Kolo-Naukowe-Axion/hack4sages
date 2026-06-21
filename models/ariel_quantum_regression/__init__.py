"""Compatibility shim for the historical ``models.ariel_quantum_regression`` package.

The current implementation lives in :mod:`models.taurex_exobiome`, but older
app, test, and Garnet-port call sites still import the historical dotted path.
Submodules are exposed as small lazy re-export files so importing
``models.ariel_quantum_regression.training`` does not also import optional
cross-validation dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["constants", "cross_validation", "dataset", "model", "training"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
