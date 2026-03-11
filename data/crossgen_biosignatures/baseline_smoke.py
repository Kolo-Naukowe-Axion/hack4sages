"""Simple baseline smoke test for the cross-generator biosignature dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DATASET_PATHS, PRESENCE_THRESHOLD_LOG10_VMR
from .dataset_io import read_labels_parquet, read_spectra_h5


def _standardize(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return ((values - mean) / scale), mean, scale


def _ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float64), x_train.astype(np.float64)], axis=1)
    lhs = x_aug.T @ x_aug
    rhs = x_aug.T @ y_train.astype(np.float64)
    regularizer = np.eye(lhs.shape[0], dtype=np.float64) * float(alpha)
    regularizer[0, 0] = 0.0
    return np.linalg.pinv(lhs + regularizer).dot(rhs)


def _ridge_predict(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((values.shape[0], 1), dtype=np.float64), values.astype(np.float64)], axis=1)
    return x_aug.dot(weights)


def _rmse_by_column(true_values: np.ndarray, pred_values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred_values - true_values) ** 2, axis=0))


def run_baseline_smoke(output_root: Path, alpha: float = 1.0) -> dict[str, Any]:
    """Train a small ridge baseline on TauREx and infer on POSEIDON."""

    labels = read_labels_parquet(output_root).sort_values(["generator", "sample_id"]).reset_index(drop=True)
    spectra = read_spectra_h5(output_root)
    target_columns = [column for column in labels.columns if column.startswith("log10_vmr_")]
    binary_columns = [column for column in labels.columns if column.startswith("present_")]

    feature_matrix = np.concatenate(
        [
            spectra["transit_depth_noisy"].astype(np.float64),
            np.log10(np.clip(spectra["sigma_ppm"].astype(np.float64), 1.0, None)).reshape(-1, 1),
        ],
        axis=1,
    )
    y = labels[target_columns].to_numpy(dtype=np.float64)
    y_binary = labels[binary_columns].to_numpy(dtype=np.int64)

    train_mask = (labels["generator"] == "tau") & (labels["split"] == "train")
    val_mask = (labels["generator"] == "tau") & (labels["split"] == "val")
    test_mask = labels["generator"] == "poseidon"

    x_train = feature_matrix[train_mask.to_numpy()]
    x_val = feature_matrix[val_mask.to_numpy()]
    x_test = feature_matrix[test_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    y_val = y[val_mask.to_numpy()]
    y_test = y[test_mask.to_numpy()]
    y_binary_val = y_binary[val_mask.to_numpy()]
    y_binary_test = y_binary[test_mask.to_numpy()]

    x_train_scaled, mean, scale = _standardize(x_train, x_train)
    x_val_scaled = (x_val - mean) / scale
    x_test_scaled = (x_test - mean) / scale

    weights = _ridge_fit(x_train_scaled, y_train, alpha=alpha)
    pred_val = _ridge_predict(weights, x_val_scaled)
    pred_test = _ridge_predict(weights, x_test_scaled)

    pred_binary_val = (pred_val >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)
    pred_binary_test = (pred_test >= PRESENCE_THRESHOLD_LOG10_VMR).astype(np.int64)

    summary = {
        "feature_dim": int(feature_matrix.shape[1]),
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
        "test_rows": int(len(x_test)),
        "target_columns": target_columns,
        "val_rmse": {column: float(value) for column, value in zip(target_columns, _rmse_by_column(y_val, pred_val))},
        "test_rmse": {column: float(value) for column, value in zip(target_columns, _rmse_by_column(y_test, pred_test))},
        "val_binary_accuracy": {
            column: float((pred_binary_val[:, idx] == y_binary_val[:, idx]).mean())
            for idx, column in enumerate(binary_columns)
        },
        "test_binary_accuracy": {
            column: float((pred_binary_test[:, idx] == y_binary_test[:, idx]).mean())
            for idx, column in enumerate(binary_columns)
        },
    }

    (output_root / DATASET_PATHS.baseline_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    predictions = pd.DataFrame(pred_test, columns=[f"pred_{column}" for column in target_columns])
    predictions.insert(0, "sample_id", labels.loc[test_mask, "sample_id"].tolist())
    predictions.to_csv(output_root / DATASET_PATHS.baseline_predictions_csv, index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_baseline_smoke(args.output_root, alpha=args.alpha)
    print("Baseline smoke summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
