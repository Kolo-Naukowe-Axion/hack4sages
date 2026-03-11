from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.sbi_ariel_crossgen.dataset import load_datasets
from models.sbi_ariel_crossgen.prepare_dataset import prepare_dataset
from models.sbi_ariel_crossgen.training import train_model


def _source_dataset_dir() -> Path:
    return PROJECT_ROOT / "data" / "generated-data" / "crossgen_biosignatures_20260311"


def _tiny_settings(prepared_dir: Path) -> dict:
    return {
        "seed": 7,
        "task": {"name": "crossgen_biosignature_regression"},
        "dataset": {
            "type": "CrossGenRealScalarNoiseNormalizedDataset",
            "path": str(prepared_dir),
            "train_split": "tau_train",
            "validation_split": "tau_val",
            "holdout_split": "poseidon_holdout",
        },
        "model": {
            "posterior_model_type": "flow_matching",
            "prior": {"type": "StandardNormal"},
            "posterior_kwargs": {
                "type": "DenseResidualNet",
                "activation": "gelu",
                "batch_norm": True,
                "context_with_glu": False,
                "dropout": 0.0,
                "hidden_dims_layers": [32, 32],
                "multiplicative_factor_layers": 1,
                "sigma_min": 0.001,
                "theta_with_glu": True,
                "time_prior_exponent": -0.5,
            },
        },
        "training": {
            "device": "cpu",
            "batch_size": 4,
            "eval_batch_size": 4,
            "epochs": 1,
            "patience": 5,
            "checkpoint_every_batches": 1,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "optimizer": {"type": "adam", "lr": 1.0e-3},
            "scheduler": {"type": "cosine", "T_max": 10},
            "max_steps": None,
        },
        "evaluation": {
            "posterior_samples": 8,
            "context_batch_size": 2,
        },
        "logging": {
            "use_wandb": False,
            "project": "unit-test",
        },
    }


class PreparedDatasetTests(unittest.TestCase):
    def test_prepare_dataset_emits_expected_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared_dir = Path(temp_dir) / "prepared"
            manifest = prepare_dataset(
                source_dir=_source_dataset_dir(),
                output_dir=prepared_dir,
                limit_tau_train=16,
                limit_tau_val=6,
                limit_poseidon=5,
            )
            self.assertEqual(manifest["context_dim"], 223)
            self.assertEqual(manifest["theta_dim"], 5)
            self.assertEqual(manifest["split_counts"]["tau_train"], 16)
            self.assertEqual(manifest["split_counts"]["tau_val"], 6)
            self.assertEqual(manifest["split_counts"]["poseidon_holdout"], 5)

            settings = _tiny_settings(prepared_dir)
            datasets = load_datasets(settings)
            self.assertEqual(tuple(datasets["train"].context.shape), (16, 223))
            self.assertEqual(tuple(datasets["train"].theta.shape), (16, 5))
            self.assertEqual(tuple(datasets["validation"].theta.shape), (6, 5))


@unittest.skipUnless(__import__("importlib").util.find_spec("dingo") is not None, "dingo-gw is required for training smoke tests")
class TrainingSmokeTests(unittest.TestCase):
    def test_training_resume_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prepared_dir = root / "prepared"
            run_dir = root / "run"
            prepare_dataset(
                source_dir=_source_dataset_dir(),
                output_dir=prepared_dir,
                limit_tau_train=16,
                limit_tau_val=6,
                limit_poseidon=5,
            )

            settings = _tiny_settings(prepared_dir)
            settings["training"]["max_steps"] = 2
            first_summary = train_model(settings=settings, run_dir=run_dir, resume_mode="never")
            self.assertEqual(first_summary["status"], "stopped_max_steps")
            self.assertTrue((run_dir / "resume_latest.pt").exists())

            settings = _tiny_settings(prepared_dir)
            resumed_summary = train_model(settings=settings, run_dir=run_dir, resume_mode="auto")
            self.assertIn(resumed_summary["status"], {"completed", "stopped_early"})
            self.assertTrue((run_dir / "history.txt").exists())
            self.assertTrue((run_dir / "tau_val_regression_metrics.json").exists())
            self.assertTrue((run_dir / "tau_val_regression_predictions.csv").exists())

            steps = []
            with (run_dir / "batch_metrics.jsonl").open() as handle:
                for line in handle:
                    steps.append(json.loads(line)["global_step"])
            self.assertEqual(sorted(steps), steps)
            self.assertEqual(len(steps), len(set(steps)))

            metrics = json.loads((run_dir / "tau_val_regression_metrics.json").read_text())
            self.assertEqual(metrics["posterior_samples"], 8)
            self.assertEqual(metrics["target_columns"], [
                "log10_vmr_h2o",
                "log10_vmr_co2",
                "log10_vmr_co",
                "log10_vmr_ch4",
                "log10_vmr_nh3",
            ])


if __name__ == "__main__":
    unittest.main()
