"""Posterior-based regression evaluation for the explicit validation and holdout splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from .constants import TARGET_COLS
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
        # Dingo's ODE sampling path relies on float64 tolerances; MPS does not support that.
        device = torch.device("cpu") if training_device.type == "mps" else training_device

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable.")
    if device.type == "mps":
        raise RuntimeError(
            "MPS evaluation is unsupported for FMPE posterior sampling because the ODE solver "
            "requires float64 tensors. Use evaluation.device=cpu."
        )
    return device


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
) -> tuple[dict[str, Any], Optional[pd.DataFrame]]:
    row_indices = select_row_indices(len(dataset), max_rows=max_rows, seed=row_selection_seed)
    mean_predictions: list[np.ndarray] = []
    median_predictions: list[np.ndarray] = []
    total_batches = int(np.ceil(len(row_indices) / max(context_batch_size, 1)))

    if progress_label is not None:
        print(
            f"{progress_label}: evaluating {len(row_indices)}/{len(dataset)} rows "
            f"with {posterior_samples} posterior samples in {total_batches} batches on {device}.",
            flush=True,
        )

    model.network.eval()
    with torch.no_grad():
        for batch_number, start in enumerate(range(0, len(row_indices), context_batch_size), start=1):
            batch_indices = torch.as_tensor(row_indices[start : start + context_batch_size], dtype=torch.long)
            context = dataset.context.index_select(0, batch_indices)
            context = context.to(device, non_blocking=device.type == "cuda")
            samples, _ = model.sample_and_log_prob(context, num_samples=posterior_samples)
            batch_size = samples.shape[0]
            samples_np = samples.detach().cpu().numpy().reshape(batch_size * posterior_samples, len(TARGET_COLS))
            samples_orig = dataset.inverse_transform_theta(samples_np).reshape(batch_size, posterior_samples, len(TARGET_COLS))
            mean_predictions.append(samples_orig.mean(axis=1))
            median_predictions.append(np.median(samples_orig, axis=1))
            if progress_label is not None and progress_every_batches > 0 and (
                batch_number % progress_every_batches == 0 or batch_number == total_batches
            ):
                print(f"{progress_label}: batch {batch_number}/{total_batches}", flush=True)

    pred_mean = np.concatenate(mean_predictions, axis=0)
    pred_median = np.concatenate(median_predictions, axis=0)
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


def run_regression_evaluation(
    settings: dict[str, Any],
    run_dir: str | Path,
    prepared_data_override: Optional[str | Path] = None,
    include_poseidon_eval: bool = False,
) -> dict[str, str]:
    run_path = Path(run_dir).expanduser().resolve()
    device = resolve_evaluation_device(settings)

    datasets = load_datasets(settings, prepared_data_override=prepared_data_override)
    model = load_posterior_model(run_path / "best_model.pt", device=device)
    posterior_samples = int(settings["evaluation"]["posterior_samples"])
    context_batch_size = int(settings["evaluation"].get("context_batch_size", 8))
    progress_every_batches = int(
        settings["evaluation"].get(
            "progress_every_batches",
            25 if device.type == "cpu" else 0,
        )
        or 0
    )

    outputs: dict[str, str] = {}
    split_names = [("tau_val", datasets["validation"])]
    if include_poseidon_eval:
        split_names.append(("poseidon_holdout", datasets["holdout"]))

    for split_name, dataset in split_names:
        metrics, predictions = evaluate_split(
            model=model,
            dataset=dataset,
            device=device,
            posterior_samples=posterior_samples,
            context_batch_size=context_batch_size,
            include_predictions=True,
            progress_label=f"{split_name} final evaluation",
            progress_every_batches=progress_every_batches,
        )
        metrics_path = run_path / f"{split_name}_regression_metrics.json"
        predictions_path = run_path / f"{split_name}_regression_predictions.csv"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        predictions.to_csv(predictions_path, index=False)
        outputs[f"{split_name}_metrics"] = str(metrics_path)
        outputs[f"{split_name}_predictions"] = str(predictions_path)
    outputs["evaluation_device"] = str(device)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate posterior mean/median regression metrics from a trained FMPE run.")
    parser.add_argument("--run-dir", required=True, help="Training run directory containing best_model.pt and settings.yaml.")
    parser.add_argument("--settings", default=None, help="Optional explicit settings YAML path. Defaults to <run-dir>/settings.yaml.")
    parser.add_argument("--prepared-data", default=None, help="Optional explicit prepared dataset directory.")
    parser.add_argument("--include-poseidon-eval", action="store_true", help="Also evaluate the POSEIDON holdout split.")
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
        include_poseidon_eval=args.include_poseidon_eval,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
