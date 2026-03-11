from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_QISKIT = importlib.util.find_spec("qiskit") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None
HAS_IQM = importlib.util.find_spec("iqm") is not None


class LayoutSelectionTests(unittest.TestCase):
    class _CouplingMap:
        def __init__(self, edges):
            self._edges = edges

        def get_edges(self):
            return list(self._edges)

    class _Properties:
        def gate_error(self, gate_name, qubits):
            pair = tuple(sorted(qubits))
            return {
                ("cz", (0, 1)): 0.01,
                ("cz", (1, 2)): 0.01,
                ("cz", (2, 3)): 0.01,
                ("cz", (0, 3)): 0.01,
                ("cz", (0, 2)): 0.20,
                ("cz", (1, 3)): 0.20,
            }.get((gate_name, pair), 0.05)

        def t1(self, qubit):
            return 10.0 + qubit

        def t2(self, qubit):
            return 20.0 + qubit

    class _Backend:
        def __init__(self, edges):
            self.coupling_map = LayoutSelectionTests._CouplingMap(edges)

        def properties(self):
            return LayoutSelectionTests._Properties()

    def test_select_garnet_layout_prefers_ring(self) -> None:
        from models.garnet_ariel_quantum_regression.backend import select_garnet_layout

        backend = self._Backend([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])
        selection = select_garnet_layout(backend, 4)
        self.assertTrue(selection.ring_supported)
        self.assertFalse(selection.degraded)
        self.assertEqual(set(selection.logical_to_physical), {0, 1, 2, 3})

    def test_select_garnet_layout_marks_degraded_without_cycle(self) -> None:
        from models.garnet_ariel_quantum_regression.backend import select_garnet_layout

        backend = self._Backend([(0, 1), (1, 2), (2, 3)])
        selection = select_garnet_layout(backend, 4)
        self.assertFalse(selection.ring_supported)
        self.assertTrue(selection.degraded)


@unittest.skipUnless(HAS_QISKIT and HAS_PENNYLANE, "qiskit and pennylane are required for circuit parity tests")
class CircuitParityTests(unittest.TestCase):
    def test_qiskit_mapping_matches_pennylane_statevector(self) -> None:
        import pennylane as qml

        from models.garnet_ariel_quantum_regression.circuit import (
            bind_quantum_circuit,
            build_quantum_template,
            evaluate_statevector_expectations,
        )

        n_qubits = 4
        depth = 2
        inputs = np.linspace(-0.3, 0.4, n_qubits, dtype=np.float32)
        weights = np.linspace(-0.2, 0.2, 3 * n_qubits * (depth // 2), dtype=np.float32)
        template = build_quantum_template(n_qubits, depth)
        qiskit_values = evaluate_statevector_expectations(
            bind_quantum_circuit(template, inputs, weights),
            template.observables,
        )

        device = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(device)
        def circuit(x, w):
            for qubit in range(n_qubits):
                qml.RY(x[qubit], wires=qubit)
            param_index = 0
            for _ in range(depth // 2):
                for qubit in range(n_qubits):
                    qml.RY(w[param_index], wires=qubit)
                    param_index += 1
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
                for qubit in range(n_qubits):
                    qml.RZ(w[param_index], wires=qubit)
                    param_index += 1
                for qubit in range(n_qubits):
                    qml.CRX(w[param_index], wires=[qubit, (qubit + 1) % n_qubits])
                    param_index += 1
            return [qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits)]

        pennylane_values = np.asarray(circuit(inputs, weights), dtype=np.float32)
        self.assertTrue(np.allclose(qiskit_values, pennylane_values, atol=1.0e-5))


@unittest.skipUnless(HAS_TORCH and HAS_QISKIT and HAS_IQM, "torch, qiskit, and iqm-client are required for Garnet runtime tests")
class GarnetRuntimeTests(unittest.TestCase):
    def _fake_split(self, rows: int = 2):
        import torch

        from models.ariel_quantum_regression.dataset import InferenceSplit

        return InferenceSplit(
            planet_ids=np.asarray([f"planet_{i}" for i in range(rows)], dtype="U32"),
            aux=torch.zeros((rows, 8), dtype=torch.float32),
            spectra=torch.zeros((rows, 4, 52), dtype=torch.float32),
        )

    def test_checkpoint_prediction_mode_works_for_eight_qubits(self) -> None:
        from models.garnet_ariel_quantum_regression.runtime import GarnetPortConfig, prepare_garnet_run

        run = prepare_garnet_run(
            GarnetPortConfig(n_qubits=8, backend_mode="fake", max_samples=2),
            data_split=self._fake_split(rows=2),
            split_label="val",
        )
        self.assertEqual(run.quantum_inputs.shape, (2, 8))
        self.assertEqual(run.classical_predictions.shape, (2, 5))
        self.assertEqual(len(run.transpiled_circuits), 2)

    def test_checkpoint_prediction_mode_rejects_twelve_qubits(self) -> None:
        from models.garnet_ariel_quantum_regression.runtime import GarnetPortConfig, prepare_garnet_run

        with self.assertRaises(ValueError):
            prepare_garnet_run(
                GarnetPortConfig(n_qubits=12, backend_mode="fake", max_samples=1),
                data_split=self._fake_split(rows=1),
                split_label="val",
            )

    def test_local_and_mock_prediction_shapes(self) -> None:
        from models.garnet_ariel_quantum_regression.runtime import (
            GarnetPortConfig,
            prepare_garnet_run,
            run_local_baseline,
            run_mock_evaluation,
        )

        run = prepare_garnet_run(
            GarnetPortConfig(n_qubits=8, backend_mode="fake", max_samples=2, shots=128),
            data_split=self._fake_split(rows=2),
            split_label="val",
        )
        local_predictions = run_local_baseline(run)
        mock_predictions = run_mock_evaluation(run)
        self.assertEqual(local_predictions.shape, (2, 5))
        self.assertEqual(mock_predictions.shape, (2, 5))

    def test_live_submission_rejects_train_split(self) -> None:
        from models.garnet_ariel_quantum_regression.runtime import GarnetPortConfig, prepare_garnet_run, run_iqm_execution

        run = prepare_garnet_run(
            GarnetPortConfig(n_qubits=8, backend_mode="fake", max_samples=1, submit_to_iqm=True),
            data_split=self._fake_split(rows=1),
            split_label="train",
        )
        with self.assertRaises(ValueError):
            run_iqm_execution(run)
