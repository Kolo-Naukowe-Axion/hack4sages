from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATA_ROOT = PROJECT_ROOT / "data" / "full-ariel"
WORKFLOW_ROOT = PROJECT_ROOT / "models" / "sbi_ariel_adc2023"
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_H5PY = importlib.util.find_spec("h5py") is not None


class DatasetPreflightTests(unittest.TestCase):
    def test_workflow_folder_is_self_contained(self) -> None:
        forbidden = ("models.ariel_quantum_regression", "models.old_model", "crossgen_hybrid_training")
        for path in WORKFLOW_ROOT.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, msg=f"{path} still references {token}")

    def test_missing_dataset_files_raise_clear_error(self) -> None:
        from models.sbi_ariel_adc2023.prepare_dataset import prepare_dataset

        with tempfile.TemporaryDirectory() as temp_root, tempfile.TemporaryDirectory() as temp_out:
            root = Path(temp_root)
            (root / "TrainingData").mkdir(parents=True)
            with self.assertRaises(FileNotFoundError) as ctx:
                prepare_dataset(data_root=root, output_dir=temp_out, overwrite=True, train_limit=1, validation_limit=1)
            self.assertIn("ADC2023 dataset is incomplete", str(ctx.exception))

    @unittest.skipUnless(HAS_TORCH, "torch is required for prepared dataset loader tests")
    def test_load_datasets_from_minimal_prepared_dir(self) -> None:
        from models.sbi_ariel_adc2023.constants import (
            CONTEXT_DIM,
            CONTEXT_FILENAME_TEMPLATE,
            DATASET_TYPE,
            HOLDOUT_SPLIT,
            MANIFEST_FILENAME,
            METADATA_FILENAME_TEMPLATE,
            NORMALIZATION_FILENAME,
            NORMALIZATION_MODE,
            RAW_TARGET_FILENAME_TEMPLATE,
            TARGET_COLS,
            TARGET_FILENAME_TEMPLATE,
            TESTDATA_SPLIT,
            THETA_DIM,
            TRAIN_SPLIT,
            VALIDATION_SPLIT,
        )
        from models.sbi_ariel_adc2023.dataset import load_datasets

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = {
                "dataset_type": DATASET_TYPE,
                "normalization_mode": NORMALIZATION_MODE,
            }
            (root / MANIFEST_FILENAME).write_text(json.dumps(manifest))
            np.savez(
                root / NORMALIZATION_FILENAME,
                target_mean=np.zeros(THETA_DIM, dtype=np.float32),
                target_scale=np.ones(THETA_DIM, dtype=np.float32),
            )
            for split_name, rows in ((TRAIN_SPLIT, 4), (VALIDATION_SPLIT, 2), (HOLDOUT_SPLIT, 2)):
                np.save(root / CONTEXT_FILENAME_TEMPLATE.format(split_name=split_name), np.zeros((rows, CONTEXT_DIM), dtype=np.float32))
                np.save(root / TARGET_FILENAME_TEMPLATE.format(split_name=split_name), np.zeros((rows, THETA_DIM), dtype=np.float32))
                np.save(root / RAW_TARGET_FILENAME_TEMPLATE.format(split_name=split_name), np.zeros((rows, THETA_DIM), dtype=np.float32))
                pd.DataFrame({"planet_ID": [f"Planet_{i}" for i in range(rows)], "source_row_index": np.arange(rows)}).to_csv(
                    root / METADATA_FILENAME_TEMPLATE.format(split_name=split_name),
                    index=False,
                )
            np.save(root / CONTEXT_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT), np.zeros((3, CONTEXT_DIM), dtype=np.float32))
            pd.DataFrame({"planet_ID": [f"Planet_test_{i}" for i in range(3)], "source_row_index": np.arange(3)}).to_csv(
                root / METADATA_FILENAME_TEMPLATE.format(split_name=TESTDATA_SPLIT),
                index=False,
            )

            settings = {"dataset": {"path": str(root), "train_split": TRAIN_SPLIT, "validation_split": VALIDATION_SPLIT, "holdout_split": HOLDOUT_SPLIT}}
            datasets = load_datasets(settings)
            self.assertEqual(len(datasets["train"]), 4)
            self.assertEqual(len(datasets["validation"]), 2)
            self.assertEqual(len(datasets["holdout"]), 2)
            self.assertEqual(len(datasets["testdata"]), 3)
            self.assertEqual(settings["task"]["dim_theta"], THETA_DIM)
            self.assertEqual(settings["task"]["dim_x"], CONTEXT_DIM)
            self.assertEqual(settings["task"]["target_columns"], TARGET_COLS)


@unittest.skipUnless(HAS_TORCH and HAS_H5PY and DATA_ROOT.exists(), "real ADC2023 dataset is required")
class DatasetContractTests(unittest.TestCase):
    def test_prepare_dataset_subset_shapes(self) -> None:
        from models.sbi_ariel_adc2023.prepare_dataset import prepare_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = prepare_dataset(
                data_root=DATA_ROOT,
                output_dir=temp_dir,
                overwrite=True,
                seed=42,
                train_limit=16,
                validation_limit=8,
                holdout_limit=8,
                test_limit=6,
            )
            self.assertEqual(manifest["split_counts"]["train"], 16)
            self.assertEqual(manifest["split_counts"]["validation"], 8)
            self.assertEqual(manifest["split_counts"]["holdout"], 8)
            self.assertEqual(manifest["split_counts"]["testdata"], 6)
            self.assertEqual(manifest["context_dim"], 112)
            self.assertEqual(manifest["theta_dim"], 5)


if __name__ == "__main__":
    unittest.main()
