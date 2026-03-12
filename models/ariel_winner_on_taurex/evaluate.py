"""Evaluation helpers for the winner-style five-gas independent NSF model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import TARGET_COLUMNS
from .dataset import LoadedSplit, build_context_batch, inverse_targets_batch


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def evaluate_point_metric(
    model,
    split: LoadedSplit,
    scalers,
    *,
    device: torch.device,
    num_samples: int,
    point_estimate: str,
    batch_size: int,
    max_rows: int | None = None,
    row_seed: int = 42,
    sample_noise: bool = True,
    noise_seed: int = 42,
    log_prefix: str | None = None,
) -> dict[str, Any]:
    if split.targets_raw is None:
        raise ValueError("Point-metric evaluation requires labeled targets.")
    row_count = split.rows if not max_rows or max_rows <= 0 else min(int(max_rows), split.rows)
    if row_count == split.rows:
        chosen = np.arange(split.rows, dtype=np.int64)
    else:
        rng = np.random.default_rng(row_seed)
        chosen = np.sort(rng.choice(split.rows, size=row_count, replace=False).astype(np.int64))

    predictions = []
    total_batches = int(np.ceil(len(chosen) / batch_size))
    noise_generator = torch.Generator(device=device.type)
    noise_generator.manual_seed(int(noise_seed))

    model.eval()
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, len(chosen), batch_size), start=1):
            if log_prefix:
                print(f"{log_prefix}: batch {batch_number}/{total_batches}", flush=True)
            batch_rows = torch.as_tensor(chosen[start : start + batch_size], dtype=torch.long)
            context = build_context_batch(
                split,
                batch_rows,
                scalers,
                device=device,
                sample_noise=sample_noise,
                noise_generator=noise_generator,
            )
            samples = model.sample(context, num_samples=num_samples).detach().cpu().numpy()
            if point_estimate == "mean":
                point = samples.mean(axis=1)
            elif point_estimate == "median":
                point = np.median(samples, axis=1)
            else:
                raise ValueError(f"Unsupported point_estimate={point_estimate!r}")
            predictions.append(point)

    pred_scaled = np.concatenate(predictions, axis=0)
    pred = inverse_targets_batch(pred_scaled, scalers)
    truth = split.targets_raw[chosen]
    rmse = _rmse(truth, pred)
    return {
        "rows": int(len(chosen)),
        "num_samples": int(num_samples),
        "point_estimate": point_estimate,
        "rmse_mean": float(rmse.mean()),
        "rmse": {name: float(value) for name, value in zip(TARGET_COLUMNS, rmse)},
    }


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.write_text(json.dumps(metrics, indent=2) + "\n")

