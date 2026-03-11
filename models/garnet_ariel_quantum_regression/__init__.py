"""IQM Garnet notebook-first port for Ariel quantum regression."""

from .backend import LayoutSelection, resolve_iqm_backend, select_garnet_layout
from .circuit import QiskitQuantumTemplate, add_measurements, bind_quantum_circuit, build_quantum_template

__all__ = [
    "LayoutSelection",
    "QiskitQuantumTemplate",
    "add_measurements",
    "bind_quantum_circuit",
    "build_quantum_template",
    "resolve_iqm_backend",
    "select_garnet_layout",
]

try:
    from .checkpoint import (
        CheckpointBundle,
        FrozenArielHybridBridge,
        build_frozen_bridge,
        load_checkpoint_bundle,
        load_default_checkpoint_bundle,
    )
    from .runtime import (
        GarnetPortConfig,
        PreparedGarnetRun,
        load_prepared_data,
        prepare_garnet_run,
        prepare_validation_split_run,
        resolve_split,
        run_iqm_execution,
        run_local_baseline,
        run_mock_evaluation,
    )
except ImportError:
    pass
else:
    __all__.extend(
        [
            "CheckpointBundle",
            "FrozenArielHybridBridge",
            "GarnetPortConfig",
            "PreparedGarnetRun",
            "build_frozen_bridge",
            "load_checkpoint_bundle",
            "load_default_checkpoint_bundle",
            "load_prepared_data",
            "prepare_garnet_run",
            "prepare_validation_split_run",
            "resolve_split",
            "run_iqm_execution",
            "run_local_baseline",
            "run_mock_evaluation",
        ]
    )
