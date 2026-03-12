from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_ROOT = PROJECT_ROOT / "data" / "ariel-ml-dataset"
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_H5PY = importlib.util.find_spec("h5py") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


@unittest.skipUnless(HAS_TORCH, "torch is required for model shape tests")
class ModelShapeTests(unittest.TestCase):
    def test_forward_shape_classical_only(self) -> None:
        import torch

        from models.five_qubit_exobiome.model import ModelConfig, build_model

        device = torch.device("cpu")
        model = build_model(ModelConfig(classical_only=True, spectral_input_channels=4, use_amp=False), device)
        aux = torch.zeros((4, 8), dtype=torch.float32)
        spectra = torch.zeros((4, 4, 52), dtype=torch.float32)
        outputs = model(aux, spectra)
        self.assertEqual(tuple(outputs.shape), (4, 5))
        self.assertTrue(torch.isfinite(outputs).all().item())

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane is required for hybrid model tests")
    def test_forward_shape_hybrid_uses_five_qubits(self) -> None:
        import torch

        from models.five_qubit_exobiome.model import ModelConfig, build_model

        device = torch.device("cpu")
        model = build_model(
            ModelConfig(classical_only=False, spectral_input_channels=4, qnn_qubits=5, qnn_depth=2, use_amp=False),
            device,
        )
        aux = torch.zeros((2, 8), dtype=torch.float32)
        spectra = torch.zeros((2, 4, 52), dtype=torch.float32)
        outputs = model(aux, spectra)
        self.assertEqual(tuple(outputs.shape), (2, 5))
        self.assertTrue(torch.isfinite(outputs).all().item())
        self.assertEqual(model.quantum_block.n_qubits, 5)


@unittest.skipUnless(HAS_TORCH and HAS_H5PY, "torch and h5py are required for smoke training tests")
class SmokeTrainingTests(unittest.TestCase):
    def test_smoke_training_outputs_classical_only(self) -> None:
        from models.five_qubit_exobiome.training import TrainingConfig, run_training_experiment

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_training_experiment(
                TrainingConfig(
                    project_root=str(PROJECT_ROOT),
                    data_root=str(DATA_ROOT),
                    output_dir=str(output_dir),
                    seed=7,
                    batch_size=4,
                    eval_batch_size=8,
                    max_epochs=1,
                    early_stop_patience=1,
                    scheduler_patience=1,
                    loss_name="mse",
                    classical_only=True,
                    quantum_warmup_epochs=0,
                    use_amp=False,
                    train_limit=16,
                    val_limit=8,
                    holdout_limit=8,
                    test_limit=6,
                    log_every_batches=0,
                )
            )
            self.assertIn("summary", result)
            self.assertTrue((output_dir / "best_model.pt").exists())
            self.assertTrue((output_dir / "history.csv").exists())
            self.assertTrue((output_dir / "validation_metrics.json").exists())
            self.assertTrue((output_dir / "holdout_metrics.json").exists())
            self.assertTrue((output_dir / "testdata_predictions.csv").exists())

            predictions = np.genfromtxt(
                output_dir / "testdata_predictions.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            self.assertEqual(len(predictions), 6)
            self.assertEqual(
                list(predictions.dtype.names),
                ["planet_ID", "log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3"],
            )

            config = json.loads((output_dir / "config.json").read_text())
            self.assertEqual(int(config["qnn_qubits"]), 5)

            history_header = (output_dir / "history.csv").read_text().splitlines()[0]
            self.assertIn("val_rmse_mean", history_header)

    @unittest.skipUnless(HAS_PENNYLANE, "pennylane is required for hybrid smoke tests")
    def test_smoke_training_outputs_hybrid(self) -> None:
        from models.five_qubit_exobiome.training import TrainingConfig, run_training_experiment

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_training_experiment(
                TrainingConfig(
                    project_root=str(PROJECT_ROOT),
                    data_root=str(DATA_ROOT),
                    output_dir=str(output_dir),
                    seed=9,
                    batch_size=2,
                    eval_batch_size=4,
                    max_epochs=1,
                    early_stop_patience=1,
                    scheduler_patience=1,
                    loss_name="mse",
                    classical_only=False,
                    quantum_warmup_epochs=0,
                    use_amp=False,
                    qnn_qubits=5,
                    qnn_depth=2,
                    train_limit=8,
                    val_limit=4,
                    holdout_limit=4,
                    test_limit=3,
                    log_every_batches=0,
                )
            )
            self.assertIn("validation_metrics", result)
            self.assertTrue((output_dir / "best_model.pt").exists())
            metrics = json.loads((output_dir / "validation_metrics.json").read_text())
            self.assertTrue(np.isfinite(metrics["rmse_mean"]))


if __name__ == "__main__":
    unittest.main()
