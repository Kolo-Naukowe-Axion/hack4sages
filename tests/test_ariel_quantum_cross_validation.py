from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from models.ariel_quantum_regression.constants import TARGET_COLUMNS
from models.ariel_quantum_regression.cross_validation import (
    CrossValidationConfig,
    aggregate_cross_validation_outputs,
    build_cross_validation_folds,
    compute_regression_metrics,
    select_cross_validation_folds,
)


class CrossValidationFoldTests(unittest.TestCase):
    def test_build_cross_validation_folds_covers_each_row_once(self) -> None:
        rows = 60
        targets = np.zeros((rows, len(TARGET_COLUMNS)), dtype=np.float32)
        targets[:20, :] = -7.0
        targets[20:40, :3] = -7.0
        targets[40:, 0] = -7.0

        folds = build_cross_validation_folds(targets, num_folds=5, seed=17, val_fraction=0.2)
        self.assertEqual(len(folds), 5)

        holdout_union = np.concatenate([fold.holdout_indices for fold in folds])
        self.assertEqual(len(np.unique(holdout_union)), rows)
        self.assertEqual(set(holdout_union.tolist()), set(range(rows)))

        for fold in folds:
            train_set = set(fold.train_indices.tolist())
            val_set = set(fold.val_indices.tolist())
            holdout_set = set(fold.holdout_indices.tolist())
            self.assertTrue(train_set.isdisjoint(val_set))
            self.assertTrue(train_set.isdisjoint(holdout_set))
            self.assertTrue(val_set.isdisjoint(holdout_set))
            self.assertGreater(len(fold.train_indices), 0)
            self.assertGreater(len(fold.val_indices), 0)
            self.assertGreater(len(fold.holdout_indices), 0)

    def test_select_cross_validation_folds_filters_by_1_based_fold_number(self) -> None:
        rows = 40
        targets = np.zeros((rows, len(TARGET_COLUMNS)), dtype=np.float32)
        targets[:20, :] = -7.0
        targets[20:, 0] = -7.0

        folds = build_cross_validation_folds(targets, num_folds=5, seed=13, val_fraction=0.2)
        selected = select_cross_validation_folds(folds, selected_folds=(2, 4), num_folds=5)

        self.assertEqual([fold.fold_index for fold in selected], [1, 3])


class RegressionMetricTests(unittest.TestCase):
    def test_compute_regression_metrics_matches_expected_values(self) -> None:
        true_values = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ],
            dtype=np.float32,
        )
        pred_values = np.asarray(
            [
                [0.0, 2.0, 1.0, 3.0, 7.0],
                [2.0, 2.0, 4.0, 6.0, 5.0],
            ],
            dtype=np.float32,
        )

        metrics = compute_regression_metrics(true_values, pred_values)
        self.assertEqual(metrics["rows"], 2)
        self.assertAlmostEqual(metrics["rmse"]["log_H2O"], 0.70710677, places=6)
        self.assertAlmostEqual(metrics["rmse"]["log_CO2"], 0.70710677, places=6)
        self.assertAlmostEqual(metrics["rmse"]["log_CO"], 1.0, places=6)
        self.assertAlmostEqual(metrics["rmse"]["log_CH4"], 1.4142135, places=6)
        self.assertAlmostEqual(metrics["rmse"]["log_NH3"], 2.1213202, places=6)
        self.assertAlmostEqual(metrics["mae"]["log_CH4"], 1.0, places=6)
        self.assertAlmostEqual(metrics["rmse_mean"], np.mean(list(metrics["rmse"].values())), places=6)


class CrossValidationAggregationTests(unittest.TestCase):
    def test_aggregate_cross_validation_outputs_merges_completed_folds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            for fold_number, offset in ((1, 0.0), (2, 0.5)):
                fold_dir = output_root / f"fold_{fold_number:02d}"
                fold_dir.mkdir(parents=True)
                run_summary = {
                    "fold": fold_number,
                    "best_epoch": 3 + fold_number,
                    "best_val_rmse_mean": 0.4 + offset,
                    "validation_rmse_mean": 0.5 + offset,
                    "validation_mae_mean": 0.3 + offset,
                    "holdout_rmse_mean": 0.6 + offset,
                    "holdout_mae_mean": 0.2 + offset,
                    "testdata_rows": 2,
                    "output_dir": str(fold_dir),
                    "dataset": {"fold_index": fold_number - 1},
                }
                (fold_dir / "run_summary.json").write_text(json.dumps(run_summary) + "\n")

                holdout_frame = pd.DataFrame({"planet_ID": [f"p{fold_number}a", f"p{fold_number}b"]})
                test_frame = pd.DataFrame({"planet_ID": ["t1", "t2"]})
                for target_index, target_name in enumerate(TARGET_COLUMNS):
                    true_values = np.asarray([offset + target_index, offset + target_index + 1.0], dtype=np.float32)
                    pred_values = true_values + 0.1
                    holdout_frame[f"true_{target_name}"] = true_values
                    holdout_frame[f"pred_{target_name}"] = pred_values
                    test_frame[target_name] = np.asarray(
                        [offset + target_index + 10.0, offset + target_index + 20.0],
                        dtype=np.float32,
                    )
                holdout_frame.to_csv(fold_dir / "holdout_predictions.csv", index=False)
                test_frame.to_csv(fold_dir / "testdata_predictions.csv", index=False)

            summary = aggregate_cross_validation_outputs(
                output_root,
                cv_config=CrossValidationConfig(num_folds=3, val_fraction=0.2),
            )

            self.assertEqual(summary["num_folds"], 2)
            self.assertEqual(summary["configured_num_folds"], 3)
            self.assertEqual(summary["completed_folds"], [1, 2])
            self.assertEqual(summary["missing_folds"], [3])

            oof_frame = pd.read_csv(output_root / "oof_predictions.csv")
            self.assertEqual(sorted(oof_frame["fold"].unique().tolist()), [1, 2])

            ensemble_frame = pd.read_csv(output_root / "testdata_predictions_ensemble.csv")
            self.assertEqual(ensemble_frame["planet_ID"].tolist(), ["t1", "t2"])
            self.assertIn("log_H2O_std", ensemble_frame.columns)


if __name__ == "__main__":
    unittest.main()
