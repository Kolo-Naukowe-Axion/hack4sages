"""Compatibility shim for the historical ``models.ariel_quantum_regression`` package.

The original ``ariel_quantum_regression`` core package is no longer present in
the tree, but several call sites still import it under that name:

* ``models.garnet_ariel_quantum_regression.checkpoint`` (the frozen inference
  bridge that loads ``artifacts/ariel_quantum_best_v4_epoch6``),
* ``models.garnet_ariel_quantum_regression.runtime`` (the Garnet port runtime),
* the TauREx runners in ``models.taurex_exobiome``,
* the ``tests/test_ariel_quantum_*`` suites.

The current implementation lives in :mod:`models.taurex_exobiome` (same
checkpoint lineage: 8-qubit hybrid, ``lightning.gpu``). This package re-exports
that module's submodules under the historical dotted path so every
``from models.ariel_quantum_regression.<sub> import ...`` resolves to the exact
same objects (identity preserved) without editing each call site.

All re-exported submodules import their heavy/quantum dependencies lazily, so
importing this shim does NOT pull in PennyLane/TauREx — the lightweight
inference path keeps working with only ``torch``/``numpy``/``h5py``.
"""

from __future__ import annotations

import sys

from models.taurex_exobiome import (
    constants,
    cross_validation,
    dataset,
    model,
    training,
)

# Register the submodules under the old dotted path so that both
# ``import models.ariel_quantum_regression.<sub>`` and
# ``from models.ariel_quantum_regression.<sub> import X`` resolve correctly.
for _name, _module in (
    ("constants", constants),
    ("cross_validation", cross_validation),
    ("dataset", dataset),
    ("model", model),
    ("training", training),
):
    sys.modules[f"{__name__}.{_name}"] = _module

del _name, _module

__all__ = ["constants", "cross_validation", "dataset", "model", "training"]
