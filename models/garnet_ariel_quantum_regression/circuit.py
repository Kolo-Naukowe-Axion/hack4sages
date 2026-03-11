"""Qiskit circuit utilities for the Garnet Ariel quantum regression port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


def import_qiskit() -> tuple[Any, Any, Any, Any]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp, Statevector
    except ImportError as exc:
        raise ImportError("Qiskit is required for the Garnet quantum port.") from exc
    return QuantumCircuit, ParameterVector, SparsePauliOp, Statevector


@dataclass(frozen=True)
class QiskitQuantumTemplate:
    n_qubits: int
    depth: int
    circuit: Any
    input_parameters: Any
    weight_parameters: Any
    observables: tuple[Any, ...]


def _z_observable_label(n_qubits: int, qubit: int) -> str:
    return "I" * (n_qubits - qubit - 1) + "Z" + "I" * qubit


def build_quantum_template(n_qubits: int, depth: int) -> QiskitQuantumTemplate:
    if depth % 2 != 0:
        raise ValueError("The Garnet quantum template expects an even depth.")
    QuantumCircuit, ParameterVector, SparsePauliOp, _ = import_qiskit()
    circuit = QuantumCircuit(n_qubits, name=f"ariel_qnn_{n_qubits}q_d{depth}")
    input_parameters = ParameterVector("x", n_qubits)
    weight_parameters = ParameterVector("w", 3 * n_qubits * (depth // 2))

    for qubit in range(n_qubits):
        circuit.ry(input_parameters[qubit], qubit)

    parameter_index = 0
    for _ in range(depth // 2):
        for qubit in range(n_qubits):
            circuit.ry(weight_parameters[parameter_index], qubit)
            parameter_index += 1
        for qubit in range(n_qubits):
            circuit.cx(qubit, (qubit + 1) % n_qubits)
        for qubit in range(n_qubits):
            circuit.rz(weight_parameters[parameter_index], qubit)
            parameter_index += 1
        for qubit in range(n_qubits):
            circuit.crx(weight_parameters[parameter_index], qubit, (qubit + 1) % n_qubits)
            parameter_index += 1

    observables = tuple(
        SparsePauliOp.from_list([(_z_observable_label(n_qubits, qubit), 1.0)]) for qubit in range(n_qubits)
    )
    return QiskitQuantumTemplate(
        n_qubits=n_qubits,
        depth=depth,
        circuit=circuit,
        input_parameters=input_parameters,
        weight_parameters=weight_parameters,
        observables=observables,
    )


def flatten_weights_to_parameter_map(
    flat_weights: np.ndarray | Iterable[float],
    template: QiskitQuantumTemplate,
) -> dict[Any, float]:
    values = np.asarray(list(flat_weights), dtype=np.float32)
    if values.shape != (len(template.weight_parameters),):
        raise ValueError(
            f"Expected {len(template.weight_parameters)} quantum weights for {template.n_qubits} qubits, got {values.shape}."
        )
    return {parameter: float(value) for parameter, value in zip(template.weight_parameters, values.tolist())}


def bind_quantum_circuit(
    template: QiskitQuantumTemplate,
    input_angles: np.ndarray | Iterable[float],
    flat_weights: np.ndarray | Iterable[float],
) -> Any:
    inputs = np.asarray(list(input_angles), dtype=np.float32)
    if inputs.shape != (template.n_qubits,):
        raise ValueError(f"Expected {template.n_qubits} input angles, got {inputs.shape}.")
    parameter_map = flatten_weights_to_parameter_map(flat_weights, template)
    parameter_map.update({parameter: float(value) for parameter, value in zip(template.input_parameters, inputs.tolist())})
    return template.circuit.assign_parameters(parameter_map, inplace=False)


def add_measurements(circuit: Any) -> Any:
    measured = circuit.copy()
    if getattr(measured, "num_clbits", 0) == 0:
        measured.measure_all()
    return measured


def evaluate_statevector_expectations(circuit: Any, observables: Iterable[Any]) -> np.ndarray:
    _, _, _, Statevector = import_qiskit()
    state = Statevector.from_instruction(circuit)
    return np.asarray([float(np.real(state.expectation_value(obs))) for obs in observables], dtype=np.float32)


def expectations_from_counts(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    totals = np.zeros(n_qubits, dtype=np.float64)
    shots = 0
    for raw_bitstring, count in counts.items():
        bitstring = raw_bitstring.replace(" ", "").rjust(n_qubits, "0")
        shots += int(count)
        for qubit in range(n_qubits):
            bit = bitstring[-(qubit + 1)]
            totals[qubit] += (1.0 if bit == "0" else -1.0) * int(count)
    if shots == 0:
        raise ValueError("Cannot compute expectations from empty shot counts.")
    return (totals / float(shots)).astype(np.float32)
