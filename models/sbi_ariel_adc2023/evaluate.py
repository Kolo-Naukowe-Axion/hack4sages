"""Posterior-based evaluation and prediction export for ADC2023 five-gas FMPE runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .constants import POINT_ESTIMATE_CHOICES, TARGET_COLS
from .dataset import load_datasets
from .dingo_compat import load_posterior_model


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=0)


def select_row_indices(num_rows: int, max_rows: Optional[int] = None, seed: int = 42) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or max_rows >= num_rows:
        return np.arange(num_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_rows, size=int(max_rows), replace=False).astype(np.int64))


def resolve_evaluation_device(settings: dict[str, Any]) -> torch.device:
    requested = settings.get("evaluation", {}).get("device")
    if requested:
        device = torch.device(requested)
    else:
        training_device = torch.device(settings["training"]["device"])
        device = torch.device("cpu") if training_device.type == "mps" else training_device

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    if device.type == "mps":
        raise RuntimeError(
            "MPS evaluation is unsupported for FMPE posterior sampling because the ODE solver requires float64 tensors. "
            "Use evaluation.device=cpu."
        )
    return device


def resolve_checkpoint_path(run_path: Path, settings: dict[str, Any]) -> Path:
    requested = settings.get("evaluation", {}).get("checkpoint")
    if requested:
        checkpoint = Path(requested).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Requested checkpoint does not exist: {checkpoint}")
        return checkpoint
    candidates = [
        run_path / "best_model_by_mrmse.pt",
        run_path / "best_model.pt",
        run_path / "best_model_by_loss.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No trained checkpoint found in {run_path}.")


def _sample_predictions(
    model: Any,
    context_tensor: torch.Tensor,
    *,
    device: torch.device,
    posterior_samples: int,
    context_batch_size: int,
    row_indices: np.ndarray,
    progress_label: Optional[str] = None,
    progress_every_batches: int = 0,
    sampling_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    mean_predictions: list[np.ndarray] = []
    median_predictions: list[np.ndarray] = []
    total_batches = int(np.ceil(len(row_indices) / max(context_batch_size, 1)))

    if progress_label is not None:
        print(
            f"{progress_label}: evaluating {len(row_indices)} rows with {posterior_samples} posterior samples in "
            f"{total_batches} batches on {device}.",
            flush=True,
        )

    fork_devices = [device.index or 0] if device.type == "cuda" else []
    model.network.eval()
    with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
        if sampling_seed is not None:
            torch.manual_seed(int(sampling_seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(sampling_seed))
        for batch_number, start in enumerate(range(0, len(row_indices), context_batch_size), start=1):
            batch_indices = torch.as_tensor(row_indices[start : start + context_batch_size], dtype=torch.long)
            context = context_tensor.index_select(0, batch_indices)
            context = context.to(device, non_blocking=device.type == "cuda")
            if progress_label is not None and progress_every_batches > 0 and (
                batch_number % progress_every_batches == 0 or batch_number == total_batches
            ):
                print(f"{progress_label}: starting batch {batch_number}/{total_batches}", flush=True)
            sampler = getattr(model, "sample", None)
            if callable(sampler):
                samples = sampler(context, num_samples=posterior_samples)
            else:
                samples, _ = model.sample_and_log_prob(context, num_samples=posterior_samples)
            batch_size = samples.shape[0]
            samples_np = samples.detach().cpu().numpy().reshape(batch_size, posterior_samples, len(TARGET_COLS))
            mean_predictions.append(samples_np.mean(axis=1))
            if posterior_samples == 1:
                median_predictions.append(samples_np[:, 0, :])
            else:
                median_predictions.append(np.median(samples_np, axis=1))
            if progress_label is not None and progress_every_batches > 0 and (
                batch_number % progress_every_batches == 0 or batch_number == total_batches
            ):
                print(f"{progress_label}: batch {batch_number}/{total_batches}", flush=True)

    return np.concatenate(mean_predictions, axis=0), np.concatenate(median_predictions, axis=0)


def evaluate_split(
    model: Any,
    dataset,
    device: torch.device,
    posterior_samples: int,
    context_batch_size: int,
    include_predictions: bool = True,
    max_rows: Optional[int] = None,
    row_selection_seed: int = 42,
    progress_label: Optional[str] = None,
    progress_every_batches: int = 0,
    sampling_seed: Optional[int] = None,
) -> tuple[dict[str, Any], Optional[pd.DataFrame]]:
    row_indices = select_row_indices(len(dataset), max_rows=max_rows, seed=row_selection_seed)
    pred_mean_norm, pred_median_norm = _sample_predictions(
        model,
        dataset.context,
        device=device,
        posterior_samples=posterior_samples,
        context_batch_size=context_batch_size,
        row_indices=row_indices,
        progress_label=progress_label,
        progress_every_batches=progress_every_batches,
        sampling_seed=sampling_seed,
    )
    pred_mean = dataset.inverse_transform_theta(pred_mean_norm)
    pred_median = dataset.inverse_transform_theta(pred_median_norm)
    truth = dataset.targets_raw[row_indices]

    metrics = {
        "num_rows": int(len(row_indices)),
        "full_num_rows": int(len(dataset)),
        "posterior_samples": int(posterior_samples),
        "target_columns": TARGET_COLS,
        "mean": {
            "rmse": {name: float(value) for name, value in zip(TARGET_COLS, _rmse(truth, pred_mean))},
            "mae": {name: float(value) for name, value in zip(TARGET_COLS, _mae(truth, pred_mean))},
        },
        "median": {
            "rmse": {name: float(value) for name, value in zip(TARGET_COLS, _rmse(truth, pred_median))},
            "mae": {name: float(value) for name, value in zip(TARGET_COLS, _mae(truth, pred_median))},
        },
    }
    metrics["mean"]["rmse_mean"] = float(np.mean(list(metrics["mean"]["rmse"].values())))
    metrics["mean"]["mae_mean"] = float(np.mean(list(metrics["mean"]["mae"].values())))
    metrics["median"]["rmse_mean"] = float(np.mean(list(metrics["median"]["rmse"].values())))
    metrics["median"]["mae_mean"] = float(np.mean(list(metrics["median"]["mae"].values())))

    if not include_predictions:
        return metrics, None

    frame = dataset.metadata.iloc[row_indices].copy().reset_index(drop=True)
    for idx, name in enumerate(TARGET_COLS):
        frame[f"true_{name}"] = truth[:, idx]
        frame[f"pred_mean_{name}"] = pred_mean[:, idx]
        frame[f"pred_median_{name}"] = pred_median[:, idx]
    return metrics, frame


def choose_point_estimate(point_estimate: str, validation_metrics: dict[str, Any]) -> str:
    if point_estimate not in POINT_ESTIMATE_CHOICES:
        raise ValueError(f"Unsupported point estimate '{point_estimate}'.")
    if point_estimate != "auto":
        return point_estimate
    mean_rmse = float(validation_metrics["mean"]["rmse_mean"])
    median_rmse = float(validation_metrics["median"]["rmse_mean"])
    return "mean" if mean_rmse <= median_rmse else "median"


def save_test_predictions(
    model: Any,
    dataset,
    device: torch.device,
    posterior_samples: int,
    context_batch_size: int,
    point_estimate: str,
    output_path: Path,
    *,
    progress_every_batches: int = 0,
) -> None:
    row_indices = np.arange(len(dataset), dtype=np.int64)
    pred_mean_norm, pred_median_norm = _sample_predictions(
        model,
        dataset.context,
        device=device,
        posterior_samples=posterior_samples,
        context_batch_size=context_batch_size,
        row_indices=row_indices,
        progress_label="testdata prediction export",
        progress_every_batches=progress_every_batches,
    )
    predicted_norm = pred_mean_norm if point_estimate == "mean" else pred_median_norm
    predicted = dataset.inverse_transform_theta(predicted_norm)
    frame = dataset.metadata.copy().reset_index(drop=True)
    for idx, name in enumerate(TARGET_COLS):
        frame[name] = predicted[:, idx]
    frame.to_csv(output_path, index=False)


def run_regression_evaluation(
    settings: dict[str, Any],
    run_dir: str | Path,
    prepared_data_override: Optional[str | Path] = None,
) -> dict[str, str]:
    run_path = Path(run_dir).expanduser().resolve()
    device = resolve_evaluation_device(settings)
    checkpoint_path = resolve_checkpoint_path(run_path, settings)

    datasets = load_datasets(settings, prepared_data_override=prepared_data_override)
    model = load_posterior_model(checkpoint_path, device=device)
    posterior_samples = int(settings["evaluation"]["posterior_samples"])
    context_batch_size = int(settings["evaluation"].get("context_batch_size", 16))
    progress_every_batches = int(
        settings["evaluation"].get(
            "progress_every_batches",
            25 if device.type == "cpu" else 0,
        )
        or 0
    )

    validation_metrics, validation_predictions = evaluate_split(
        model=model,
        dataset=datasets["validation"],
        device=device,
        posterior_samples=posterior_samples,
        context_batch_size=context_batch_size,
        include_predictions=True,
        progress_label="validation final evaluation",
        progress_every_batches=progress_every_batches,
    )
    point_estimate_mode = choose_point_estimate(
        settings.get("evaluation", {}).get("point_estimate", "auto"),
        validation_metrics,
    )

    holdout_metrics, holdout_predictions = evaluate_split(
        model=model,
        dataset=datasets["holdout"],
        device=device,
        posterior_samples=posterior_samples,
        context_batch_size=context_batch_size,
        include_predictions=True,
        progress_label="holdout final evaluation",
        progress_every_batches=progress_every_batches,
    )

    outputs = {
        "checkpoint_path": str(checkpoint_path),
        "evaluation_device": str(device),
    }
    payloads = [
        ("validation", validation_metrics, validation_predictions),
        ("holdout", holdout_metrics, holdout_predictions),
    ]
    for split_name, metrics, predictions in payloads:
        metrics_path = run_path / f"{split_name}_regression_metrics.json"
        predictions_path = run_path / f"{split_name}_regression_predictions.csv"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        predictions.to_csv(predictions_path, index=False)
        outputs[f"{split_name}_metrics"] = str(metrics_path)
        outputs[f"{split_name}_predictions"] = str(predictions_path)

    selection_payload = {
        "checkpoint_path": str(checkpoint_path),
        "configured_point_estimate": settings.get("evaluation", {}).get("point_estimate", "auto"),
        "selected_point_estimate": point_estimate_mode,
        "validation_mean_rmse_mean": float(validation_metrics["mean"]["rmse_mean"]),
        "validation_median_rmse_mean": float(validation_metrics["median"]["rmse_mean"]),
    }
    selection_path = run_path / "posterior_selection.json"
    selection_path.write_text(json.dumps(selection_payload, indent=2, sort_keys=True) + "\n")
    outputs["posterior_selection"] = str(selection_path)

    test_predictions_path = run_path / "testdata_predictions.csv"
    save_test_predictions(
        model=model,
        dataset=datasets["testdata"],
        device=device,
        posterior_samples=posterior_samples,
        context_batch_size=context_batch_size,
        point_estimate=point_estimate_mode,
        output_path=test_predictions_path,
        progress_every_batches=progress_every_batches,
    )
    outputs["testdata_predictions"] = str(test_predictions_path)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate posterior mean/median regression metrics from a trained ADC2023 FMPE run.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing checkpoints and settings.yaml.")
    parser.add_argument("--settings", default=None, help="Optional explicit settings YAML path. Defaults to <run-dir>/settings.yaml.")
    parser.add_argument("--prepared-data", default=None, help="Optional explicit prepared dataset directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    settings_path = Path(args.settings).expanduser().resolve() if args.settings else run_dir / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    outputs = run_regression_evaluation(
        settings=settings,
        run_dir=run_dir,
        prepared_data_override=args.prepared_data,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
