"""Runtime helpers for preparing and evaluating Garnet Ariel quantum runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from models.ariel_quantum_regression.dataset import InferenceSplit, LabeledSplit, PreparedData, prepare_data

from .backend import LayoutSelection, resolve_iqm_backend, select_garnet_layout, transpile_circuits
from .checkpoint import CheckpointBundle, FrozenArielHybridBridge, build_frozen_bridge, load_checkpoint_bundle
from .circuit import (
    QiskitQuantumTemplate,
    add_measurements,
    bind_quantum_circuit,
    build_quantum_template,
    evaluate_statevector_expectations,
    expectations_from_counts,
)
from .constants import (
    DEFAULT_BACKEND_ALIAS,
    DEFAULT_BACKEND_URL,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_QUBITS,
)


@dataclass(frozen=True)
class GarnetPortConfig:
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR
    n_qubits: int = 8
    backend_mode: str = "fake"
    backend_url: str = DEFAULT_BACKEND_URL
    backend_alias: str = DEFAULT_BACKEND_ALIAS
    shots: int = 256
    optimization_level: int = 1
    layout_policy: str = "dynamic"
    selected_physical_qubits: Optional[tuple[int, ...]] = None
    submit_to_iqm: bool = False
    max_samples: Optional[int] = 8


@dataclass
class PreparedGarnetRun:
    config: GarnetPortConfig
    checkpoint_bundle: CheckpointBundle
    bridge: FrozenArielHybridBridge
    template: QiskitQuantumTemplate
    chosen_layout: Optional[LayoutSelection]
    backend_metadata: dict[str, Any]
    transpile_stats: dict[str, Any]
    observables: tuple[Any, ...]
    split_label: Optional[str] = None
    head_context: Optional[np.ndarray] = None
    fused: Optional[np.ndarray] = None
    quantum_inputs: Optional[np.ndarray] = None
    classical_predictions_scaled: Optional[np.ndarray] = None
    classical_predictions: Optional[np.ndarray] = None
    bound_circuits: list[Any] = field(default_factory=list)
    transpiled_circuits: list[Any] = field(default_factory=list)
    planet_ids: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    degraded: bool = False


def _require_supported_qubits(n_qubits: int) -> None:
    if n_qubits not in SUPPORTED_QUBITS:
        raise ValueError(f"Unsupported n_qubits={n_qubits}. Expected one of {SUPPORTED_QUBITS}.")


def _resolve_prediction_compatibility(config: GarnetPortConfig, bundle: CheckpointBundle, has_data_split: bool) -> None:
    if config.n_qubits == bundle.qnn_qubits:
        return
    if has_data_split:
        raise ValueError(
            f"Checkpoint-backed prediction mode only supports {bundle.qnn_qubits} qubits with the current best artifact; "
            f"received n_qubits={config.n_qubits}."
        )


def load_prepared_data(data_root: str | Path = DEFAULT_DATA_ROOT, limit: Optional[int] = None) -> PreparedData:
    return prepare_data(
        data_root=Path(data_root),
        output_dir=Path(DEFAULT_OUTPUT_DIR),
        prepared_cache_dir=None,
        seed=42,
        train_limit=limit,
        val_limit=limit,
        holdout_limit=limit,
        test_limit=limit,
    )


def resolve_split(prepared: PreparedData, split_name: str) -> LabeledSplit | InferenceSplit:
    normalized = split_name.strip().lower()
    if normalized == "train":
        return prepared.train
    if normalized == "val":
        return prepared.val
    if normalized == "holdout":
        return prepared.holdout
    if normalized == "testdata":
        return prepared.testdata
    raise ValueError(f"Unsupported split_name={split_name!r}.")


def prepare_validation_split_run(
    config: GarnetPortConfig,
    prepared: PreparedData,
    split_name: str = "val",
) -> PreparedGarnetRun:
    normalized = split_name.strip().lower()
    if normalized not in {"val", "validation", "holdout", "testdata"}:
        raise ValueError("Validation runs must use one of: val, validation, holdout, testdata.")
    resolved_name = "val" if normalized == "validation" else normalized
    return prepare_garnet_run(config, data_split=resolve_split(prepared, resolved_name), split_label=resolved_name)


def _tensor_to_numpy(tensor: Any) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _select_indices(rows: int, max_samples: Optional[int]) -> np.ndarray:
    if max_samples is None or max_samples >= rows:
        return np.arange(rows, dtype=np.int64)
    return np.arange(int(max_samples), dtype=np.int64)


def _backend_name(backend: Any) -> Any:
    name = getattr(backend, "name", None)
    return name() if callable(name) else name


def prepare_garnet_run(
    config: GarnetPortConfig,
    data_split: Optional[LabeledSplit | InferenceSplit] = None,
    split_label: Optional[str] = None,
) -> PreparedGarnetRun:
    _require_supported_qubits(config.n_qubits)
    bundle = load_checkpoint_bundle(config.checkpoint_dir)
    bridge = build_frozen_bridge(bundle=bundle)
    _resolve_prediction_compatibility(config, bundle, data_split is not None)

    template = build_quantum_template(config.n_qubits, bundle.qnn_depth if config.n_qubits == bundle.qnn_qubits else 2)
    backend = resolve_iqm_backend(config.backend_mode, config.backend_url, config.backend_alias)
    chosen_layout = (
        LayoutSelection(
            logical_to_physical=tuple(config.selected_physical_qubits),
            ring_supported=False,
            degraded=False,
            cz_error_score=0.0,
            coherence_score=0.0,
            reason="User supplied the physical qubit layout explicitly.",
        )
        if config.selected_physical_qubits is not None
        else select_garnet_layout(backend, config.n_qubits, config.layout_policy)
    )

    run = PreparedGarnetRun(
        config=config,
        checkpoint_bundle=bundle,
        bridge=bridge,
        template=template,
        chosen_layout=chosen_layout,
        backend_metadata={
            "backend_mode": config.backend_mode,
            "backend_name": _backend_name(backend),
            "backend_alias": config.backend_alias,
            "backend_url": config.backend_url,
        },
        transpile_stats={},
        observables=template.observables,
        split_label=split_label,
        degraded=bool(chosen_layout.degraded),
    )

    if data_split is None:
        demo_circuit = bind_quantum_circuit(template, np.zeros(config.n_qubits, dtype=np.float32), np.zeros(len(template.weight_parameters), dtype=np.float32))
        run.bound_circuits = [add_measurements(demo_circuit)]
        run.transpiled_circuits = transpile_circuits(
            run.bound_circuits,
            backend=backend,
            physical_layout=chosen_layout.logical_to_physical,
            optimization_level=config.optimization_level,
        )
        run.transpile_stats = _collect_transpile_stats(run.transpiled_circuits)
        return run

    indices = _select_indices(data_split.rows, config.max_samples)
    index_tensor = bridge.torch.as_tensor(indices, dtype=bridge.torch.int64)
    aux = _tensor_to_numpy(data_split.aux.index_select(0, index_tensor))
    spectra = _tensor_to_numpy(data_split.spectra.index_select(0, index_tensor))
    encoded = bridge.encode_features(aux, spectra)
    head_context = _tensor_to_numpy(encoded["head_context"])
    fused = _tensor_to_numpy(encoded["fused"])
    quantum_inputs = _tensor_to_numpy(bridge.project_quantum_angles(encoded["fused"]))
    classical_predictions_scaled = _tensor_to_numpy(bridge.classical_predict(encoded["head_context"]))
    classical_predictions = bundle.target_scaler.inverse_transform(classical_predictions_scaled)

    bound_circuits = [
        add_measurements(bind_quantum_circuit(template, input_angles, bundle.quantum_weights))
        for input_angles in quantum_inputs
    ]
    transpiled_circuits = transpile_circuits(
        bound_circuits,
        backend=backend,
        physical_layout=chosen_layout.logical_to_physical,
        optimization_level=config.optimization_level,
    )

    run.head_context = head_context
    run.fused = fused
    run.quantum_inputs = quantum_inputs
    run.classical_predictions_scaled = classical_predictions_scaled
    run.classical_predictions = classical_predictions
    run.bound_circuits = bound_circuits
    run.transpiled_circuits = transpiled_circuits
    run.planet_ids = data_split.planet_ids[indices]
    run.transpile_stats = _collect_transpile_stats(transpiled_circuits)
    return run


def _collect_transpile_stats(circuits: list[Any]) -> dict[str, Any]:
    if not circuits:
        return {"count": 0}
    depths = [int(circuit.depth()) for circuit in circuits]
    widths = [int(circuit.num_qubits) for circuit in circuits]
    size = [int(circuit.size()) for circuit in circuits]
    counts = circuits[0].count_ops()
    return {
        "count": len(circuits),
        "depth_min": min(depths),
        "depth_max": max(depths),
        "widths": widths,
        "ops_first_circuit": {str(key): int(value) for key, value in counts.items()},
        "size_min": min(size),
        "size_max": max(size),
    }


def run_local_baseline(run: PreparedGarnetRun) -> np.ndarray:
    if run.quantum_inputs is None or run.head_context is None:
        raise ValueError("Local baseline requires a prepared data split with checkpoint-compatible qubit count.")
    features = np.stack(
        [evaluate_statevector_expectations(circuit.remove_final_measurements(inplace=False), run.observables) for circuit in run.bound_circuits],
        axis=0,
    )
    combined_scaled = _tensor_to_numpy(run.bridge.combine_predictions(run.head_context, features))
    run.predictions = run.checkpoint_bundle.target_scaler.inverse_transform(combined_scaled)
    return run.predictions


def _backend_counts_to_features(job_result: Any, n_qubits: int, circuit_count: int) -> np.ndarray:
    features = []
    for circuit_index in range(circuit_count):
        counts = job_result.get_counts(circuit_index)
        features.append(expectations_from_counts(counts, n_qubits))
    return np.asarray(features, dtype=np.float32)


def run_mock_evaluation(run: PreparedGarnetRun, backend: Optional[Any] = None) -> np.ndarray:
    if run.quantum_inputs is None or run.head_context is None:
        raise ValueError("Mock evaluation requires a prepared data split with checkpoint-compatible qubit count.")
    active_backend = backend or resolve_iqm_backend("fake", run.config.backend_url, run.config.backend_alias)
    job = active_backend.run(run.transpiled_circuits, shots=run.config.shots)
    result = job.result()
    features = _backend_counts_to_features(result, run.template.n_qubits, len(run.transpiled_circuits))
    combined_scaled = _tensor_to_numpy(run.bridge.combine_predictions(run.head_context, features))
    run.predictions = run.checkpoint_bundle.target_scaler.inverse_transform(combined_scaled)
    return run.predictions


def run_iqm_execution(run: PreparedGarnetRun, backend: Optional[Any] = None) -> dict[str, Any]:
    if run.config.submit_to_iqm and run.split_label is not None and run.split_label.strip().lower() == "train":
        raise ValueError("Live IQM submission is disabled for training splits; use validation, holdout, or testdata only.")
    if not run.config.submit_to_iqm:
        return {
            "submitted": False,
            "backend_name": run.backend_metadata.get("backend_name"),
            "backend_mode": run.backend_metadata.get("backend_mode"),
            "circuit_count": len(run.transpiled_circuits),
            "shots": run.config.shots,
        }
    active_backend = backend or resolve_iqm_backend(
        run.config.backend_mode,
        run.config.backend_url,
        run.config.backend_alias,
    )
    job = active_backend.run(run.transpiled_circuits, shots=run.config.shots)
    return {"submitted": True, "job": job}
