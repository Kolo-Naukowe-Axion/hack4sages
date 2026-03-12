"""Evaluation helpers and CLI for the Ariel winner-family rerun package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from .constants import FIVE_GAS_TARGET_COLUMNS, TARGET_COLUMNS
from .dataset import PreparedData, inverse_targets_batch, load_prepared_data
from .model import IndependentNSF, ModelConfig


def _resolve_device(requested: str | None) -> torch.device:
    if requested in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if device.type == "mps" and not (getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but unavailable.")
    return device


def _ks_statistic_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    support = np.concatenate([x, y], axis=0)
    support.sort()
    cdf_x = np.searchsorted(x, support, side="right") / max(1, x.size)
    cdf_y = np.searchsorted(y, support, side="right") / max(1, y.size)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def _sample_ground_truth(split, indices: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    if split.trace_targets_raw is None or split.trace_weights is None:
        raise ValueError("Distribution metrics require a labeled split with tracedata.")
    sampled = []
    targets = split.trace_targets_raw.index_select(0, torch.as_tensor(indices, dtype=torch.long)).cpu().numpy()
    weights = split.trace_weights.index_select(0, torch.as_tensor(indices, dtype=torch.long)).cpu().numpy()
    for row_targets, row_weights in zip(targets, weights):
        valid = row_weights > 0
        valid_targets = row_targets[valid]
        valid_weights = row_weights[valid]
        valid_weights = valid_weights / np.clip(valid_weights.sum(), 1.0e-12, None)
        choice = rng.choice(valid_targets.shape[0], size=num_samples, replace=True, p=valid_weights)
        sampled.append(valid_targets[choice])
    return np.stack(sampled, axis=0).astype(np.float32)


def _sample_predictions(model: IndependentNSF, context: torch.Tensor, num_samples: int) -> np.ndarray:
    return model.sample(context, num_samples=num_samples).detach().cpu().numpy().astype(np.float32)


def evaluate_bundle(
    model: IndependentNSF,
    split,
    data: PreparedData,
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    point_estimate: str = "median",
    trace_seed: int = 42,
    log_prefix: str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if split.fm_targets_raw is None:
        raise ValueError("Evaluation requires a labeled split.")

    rng = np.random.default_rng(trace_seed)
    predictions = []
    indices_seen = []
    ks_per_target = np.zeros(len(TARGET_COLUMNS), dtype=np.float64)
    total_rows = 0

    model.eval()
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, split.rows, batch_size), start=1):
            stop = min(start + batch_size, split.rows)
            rows = np.arange(start, stop, dtype=np.int64)
            if log_prefix:
                print(f"{log_prefix}: batch {batch_number}/{int(np.ceil(split.rows / batch_size))}", flush=True)
            context = split.context.index_select(0, torch.as_tensor(rows, dtype=torch.long)).to(device)
            pred_scaled = _sample_predictions(model, context, num_samples=num_samples)
            pred_raw = inverse_targets_batch(pred_scaled, split.radius_reference[rows].cpu().numpy(), data.scalers)
            truth_samples = _sample_ground_truth(split, rows, num_samples=num_samples, rng=rng)
            for row_pred, row_truth in zip(pred_raw, truth_samples):
                for target_index in range(len(TARGET_COLUMNS)):
                    ks_per_target[target_index] += _ks_statistic_1d(row_truth[:, target_index], row_pred[:, target_index])
            predictions.append(pred_raw)
            indices_seen.append(rows)
            total_rows += len(rows)

    pred_raw = np.concatenate(predictions, axis=0)
    rows = np.concatenate(indices_seen, axis=0)
    if point_estimate == "mean":
        point_pred = pred_raw.mean(axis=1)
    elif point_estimate == "median":
        point_pred = np.median(pred_raw, axis=1)
    else:
        raise ValueError(f"Unsupported point_estimate={point_estimate!r}")

    truth_fm = split.fm_targets_raw[rows]
    truth_q2 = split.quartiles_raw[rows, :, 1] if split.quartiles_raw is not None else None
    rmse_fm = _rmse(truth_fm, point_pred)
    rmse_q2 = _rmse(truth_q2, point_pred) if truth_q2 is not None else None
    five_gas_indices = [TARGET_COLUMNS.index(name) for name in FIVE_GAS_TARGET_COLUMNS]

    metrics: dict[str, Any] = {
        "rows": int(total_rows),
        "num_samples": int(num_samples),
        "point_estimate": point_estimate,
        "ks_mean": float((ks_per_target / max(1, total_rows)).mean()),
        "ks": {name: float(value / max(1, total_rows)) for name, value in zip(TARGET_COLUMNS, ks_per_target)},
        "rmse_mean_all7_vs_fm": float(rmse_fm.mean()),
        "rmse_mean_5gas_vs_fm": float(rmse_fm[five_gas_indices].mean()),
        "rmse_vs_fm": {name: float(value) for name, value in zip(TARGET_COLUMNS, rmse_fm)},
    }
    if rmse_q2 is not None:
        metrics["rmse_mean_all7_vs_q2"] = float(rmse_q2.mean())
        metrics["rmse_mean_5gas_vs_q2"] = float(rmse_q2[five_gas_indices].mean())
        metrics["rmse_vs_q2"] = {name: float(value) for name, value in zip(TARGET_COLUMNS, rmse_q2)}

    frame = pd.DataFrame({"planet_ID": split.planet_id[rows]})
    for column_index, target_name in enumerate(TARGET_COLUMNS):
        frame[f"pred_{target_name}"] = point_pred[:, column_index]
        frame[f"truth_fm_{target_name}"] = truth_fm[:, column_index]
        if truth_q2 is not None:
            frame[f"truth_q2_{target_name}"] = truth_q2[:, column_index]

    return metrics, frame


def evaluate_testdata_medians(
    model: IndependentNSF,
    split,
    data: PreparedData,
    *,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    point_estimate: str = "median",
) -> pd.DataFrame:
    predictions = []
    ids = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, split.rows, batch_size):
            stop = min(start + batch_size, split.rows)
            rows = np.arange(start, stop, dtype=np.int64)
            context = split.context.index_select(0, torch.as_tensor(rows, dtype=torch.long)).to(device)
            pred_scaled = _sample_predictions(model, context, num_samples=num_samples)
            pred_raw = inverse_targets_batch(pred_scaled, split.radius_reference[rows].cpu().numpy(), data.scalers)
            if point_estimate == "mean":
                point_pred = pred_raw.mean(axis=1)
            elif point_estimate == "median":
                point_pred = np.median(pred_raw, axis=1)
            else:
                raise ValueError(f"Unsupported point_estimate={point_estimate!r}")
            predictions.append(point_pred)
            ids.append(split.planet_id[rows])
    pred = np.concatenate(predictions, axis=0)
    planet_ids = np.concatenate(ids, axis=0)
    frame = pd.DataFrame({"planet_ID": planet_ids})
    for column_index, target_name in enumerate(TARGET_COLUMNS):
        frame[f"pred_{target_name}"] = pred[:, column_index]
    return frame


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def _load_model(run_dir: Path, settings: dict[str, Any], device: torch.device) -> IndependentNSF:
    model = IndependentNSF(ModelConfig(**settings["model"])).to(device)
    bundle = torch.load(run_dir / "best_independent_bundle.pt", map_location=device)
    model.load_state_dict(bundle["model"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a finished Ariel winner-family rerun.")
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    settings = yaml.safe_load(args.settings.read_text())
    device = _resolve_device(args.device)
    data = load_prepared_data(args.prepared_data)
    model = _load_model(args.run_dir, settings, device)

    evaluation_settings = settings["evaluation"]
    batch_size = int(settings["training"]["eval_batch_size"])
    num_samples = int(evaluation_settings["final_num_samples"])
    point_estimate = str(evaluation_settings.get("point_estimate", "median"))
    trace_seed = int(evaluation_settings.get("trace_seed", 42))

    validation_metrics, validation_predictions = evaluate_bundle(
        model,
        data.validation,
        data,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        point_estimate=point_estimate,
        trace_seed=trace_seed,
        log_prefix="validation final",
    )
    holdout_metrics, holdout_predictions = evaluate_bundle(
        model,
        data.holdout,
        data,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        point_estimate=point_estimate,
        trace_seed=trace_seed,
        log_prefix="holdout final",
    )
    test_predictions = evaluate_testdata_medians(
        model,
        data.testdata,
        data,
        device=device,
        num_samples=num_samples,
        batch_size=batch_size,
        point_estimate=point_estimate,
    )

    save_metrics(args.run_dir / "validation_metrics.json", validation_metrics)
    save_metrics(args.run_dir / "holdout_metrics.json", holdout_metrics)
    validation_predictions.to_csv(args.run_dir / "validation_predictions.csv", index=False)
    holdout_predictions.to_csv(args.run_dir / "holdout_predictions.csv", index=False)
    test_predictions.to_csv(args.run_dir / "testdata_predictions.csv", index=False)

    summary = {
        "validation": validation_metrics,
        "holdout": holdout_metrics,
        "device": str(device),
        "run_dir": str(args.run_dir.expanduser().resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

