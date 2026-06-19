"""Model-comparison transforms.
Pure functions over in-memory results. Takes the
predictions both models produced live (plus optional ground truth) and turns
them into per-gas comparison rows and aggregate errors. Because it operates on
Prediction objects rather than files, it stays a clean functional pipeline
"""

from __future__ import annotations

import math
from typing import Mapping

from .types import GASES, ComparisonRow, DataError, GroundTruth, Prediction


def build_comparison(
    truth: GroundTruth | None, preds: Mapping[str, Prediction]
) -> list[ComparisonRow]:
    """One row per gas with each model's prediction and (if truth is known) error"""
    for name, pred in preds.items():
        missing = [g for g in GASES if g not in pred.log_vmr]
        if missing:
            raise DataError(f"Prediction '{name}' is missing gases: {missing}.")
    rows: list[ComparisonRow] = []
    for gas in GASES:
        per_model = {name: float(p.log_vmr[gas]) for name, p in preds.items()}
        true_val = None if truth is None else float(truth.log_vmr[gas])
        errors = (
            {} if true_val is None
            else {name: abs(value - true_val) for name, value in per_model.items()}
        )
        rows.append(ComparisonRow(gas=gas, true=true_val, preds=per_model, errors=errors))
    return rows


def aggregate(rows: list[ComparisonRow]) -> dict[str, dict[str, float]]:
    """Per-model RMSE across gases (only when ground truth is present).
    Returns {model_name: {"rmse_mean": ...}}; empty if no truth. This is the
    single-planet RMSE noisy by design"""
    if not rows or rows[0].true is None:
        return {}
    models = list(rows[0].preds.keys())
    out: dict[str, dict[str, float]] = {}
    for name in models:
        squared = [row.errors[name] ** 2 for row in rows if name in row.errors]
        if squared:
            out[name] = {"rmse_mean": math.sqrt(sum(squared) / len(squared))}
    return out
