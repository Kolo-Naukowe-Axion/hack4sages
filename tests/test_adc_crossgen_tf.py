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

from models.adc_crossgen_tf.constants import AUX_COLUMNS, TARGET_COLUMNS
from models.adc_crossgen_tf.dataset import build_aux_targets, load_aligned_dataset, prepare_data


DATA_ROOT = PROJECT_ROOT / "data" / "generated-data" / "crossgen_biosignatures_20260311"
HAS_TF = importlib.util.find_spec("tensorflow") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


class DatasetLoaderTests(unittest.TestCase):
    def test_full_dataset_contract(self) -> None:
        labels, spectra, sigma_ppm, wavelength_um = load_aligned_dataset(DATA_ROOT)
        aux, targets, _ = build_aux_targets(labels, sigma_ppm)

        self.assertEqual(len(labels), 42108)
        self.assertEqual(spectra.shape, (42108, 218))
        self.assertEqual(aux.shape, (42108, len(AUX_COLUMNS)))
        self.assertEqual(targets.shape, (42108, len(TARGET_COLUMNS)))
        self.assertEqual(len(wavelength_um), 218)

        counts = labels.groupby(["generator", "split"]).size().to_dict()
        self.assertEqual(int(counts[("tau", "train")]), 37281)
        self.assertEqual(int(counts[("tau", "val")]), 4142)
        self.assertEqual(int(counts[("poseidon", "test")]), 685)

    def test_prepare_data_subset_shapes(self) -> None:
        prepared = prepare_data(DATA_ROOT, augment_repeat=2, train_limit=6, val_limit=4, poseidon_limit=3)
        self.assertEqual(prepared.train.rows, 12)
        self.assertEqual(prepared.tau_val.rows, 4)
        self.assertEqual(prepared.poseidon.rows, 3)
        self.assertEqual(prepared.train.spectra.shape[1], 218)
        self.assertEqual(prepared.train.aux.shape[1], 5)
        self.assertEqual(prepared.train.targets.shape[1], 5)


@unittest.skipUnless(HAS_TF and HAS_PENNYLANE, "tensorflow and pennylane are required for model tests")
class TensorFlowModelTests(unittest.TestCase):
    def test_forward_and_mc_shapes(self) -> None:
        from models.adc_crossgen_tf.model import build_model
        from models.adc_crossgen_tf.training import _predict_mc

        model = build_model(
            spectrum_length=218,
            aux_dim=5,
            target_dim=5,
            dropout=0.1,
            qnn_qubits=2,
            qnn_depth=1,
        )
        spectra = np.zeros((4, 218), dtype=np.float32)
        aux = np.zeros((4, 5), dtype=np.float32)
        outputs = model([spectra, aux], training=False)
        self.assertEqual(tuple(outputs.shape), (4, 5))

        dummy_split = type("DummySplit", (), {"rows": 4, "spectra": spectra, "aux": aux})()

        mc = _predict_mc(model, dummy_split, mc_samples=3, batch_size=2)
        self.assertEqual(mc.shape, (3, 4, 5))
        self.assertTrue(np.isfinite(mc).all())

    def test_smoke_training_outputs(self) -> None:
        from models.adc_crossgen_tf.training import TrainingConfig, train_and_evaluate

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            config = TrainingConfig(
                project_root=str(PROJECT_ROOT),
                data_root=str(DATA_ROOT),
                output_dir=str(output_dir),
                epochs=1,
                batch_size=8,
                lr=1.0e-3,
                augment_repeat=1,
                mc_samples=2,
                qnn_qubits=2,
                qnn_depth=1,
                train_limit=16,
                val_limit=8,
                poseidon_limit=6,
                patience=1,
            )
            result = train_and_evaluate(config)
            self.assertIn("tau_val_metrics", result)
            self.assertTrue((output_dir / "best_model.keras").exists())
            self.assertTrue((output_dir / "history.csv").exists())
            self.assertTrue((output_dir / "tau_val_metrics.json").exists())
            self.assertTrue((output_dir / "poseidon_metrics.json").exists())

            tau_predictions = np.genfromtxt(output_dir / "tau_val_predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            poseidon_predictions = np.genfromtxt(
                output_dir / "poseidon_predictions.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            self.assertEqual(len(tau_predictions), 8)
            self.assertEqual(len(poseidon_predictions), 6)

            summary = json.loads((output_dir / "run_summary.json").read_text())
            self.assertEqual(int(summary["dataset"]["tau_val_rows"]), 8)
            self.assertEqual(int(summary["dataset"]["poseidon_rows"]), 6)


if __name__ == "__main__":
    unittest.main()
