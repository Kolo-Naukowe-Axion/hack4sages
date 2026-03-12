from __future__ import annotations

import unittest

import numpy as np

from models.ariel_quantum_regression.constants import TARGET_COLUMNS
from models.ariel_quantum_regression.cross_validation import build_cross_validation_folds, compute_regression_metrics


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


if __name__ == "__main__":
    unittest.main()
