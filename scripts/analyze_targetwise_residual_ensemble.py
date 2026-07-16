"""Build validation-selected targetwise ensembles from completed residual runs."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = PROJECT_ROOT / "reports" / "residual_architecture_suite_rtx4090_remote"
CONTROL_DIR = PROJECT_ROOT / "reports" / "controlled_quantum_residual_rtx4090_remote"
TARGETS = ("log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3")


VARIANT_DIRS = {
    "controlled_gated_quantum_residual": (CONTROL_DIR, "gated_quantum_residual"),
    "controlled_classical_residual": (CONTROL_DIR, "classical_residual"),
    "mlp_residual": (ARCH_DIR, "mlp_residual"),
    "nystroem_ridge_residual": (ARCH_DIR, "nystroem_ridge_residual"),
    "quantum_reupload_ridge_residual": (ARCH_DIR, "quantum_reupload_ridge_residual"),
    "pca_ridge_residual": (ARCH_DIR, "pca_ridge_residual"),
    "rff_ridge_residual": (ARCH_DIR, "rff_ridge_residual"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(payload: dict[str, Any]) -> None:
    with (PROJECT_ROOT / "_script_manifest.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std_sample": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "sem": float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_dir(seed: int, variant: str) -> Path:
    base, subdir = VARIANT_DIRS[variant]
    return base / f"seed_{seed}" / subdir


def mrmse_from_frame(
    frame: pd.DataFrame,
) -> tuple[float, dict[str, float], dict[str, float]]:
    rmse = {}
    mae = {}
    for target in TARGETS:
        err = frame[f"pred_{target}"].to_numpy() - frame[f"true_{target}"].to_numpy()
        rmse[target] = float(np.sqrt(np.mean(np.square(err))))
        mae[target] = float(np.mean(np.abs(err)))
    return float(np.mean(list(rmse.values()))), rmse, mae


def main() -> None:
    start = time.time()
    seeds = (42, 43, 44)
    selection_rows = []
    eval_rows = []
    prediction_dir = ARCH_DIR / "targetwise_ensemble"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        validation_metrics = {
            variant: read_json(run_dir(seed, variant) / "validation_metrics.json")
            for variant in VARIANT_DIRS
        }
        selected = {}
        for target in TARGETS:
            best_variant = min(
                VARIANT_DIRS,
                key=lambda variant: validation_metrics[variant]["rmse"][target],
            )
            selected[target] = best_variant
            selection_rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "selected_variant": best_variant,
                    "validation_rmse": validation_metrics[best_variant]["rmse"][target],
                }
            )

        first_variant = next(iter(VARIANT_DIRS))
        assembled = pd.read_csv(
            run_dir(seed, first_variant) / "holdout_predictions.csv"
        )[["planet_ID"]].copy()
        for target in TARGETS:
            source = pd.read_csv(
                run_dir(seed, selected[target]) / "holdout_predictions.csv"
            )
            assembled[f"true_{target}"] = source[f"true_{target}"]
            assembled[f"pred_{target}"] = source[f"pred_{target}"]
        out_path = (
            prediction_dir
            / f"seed_{seed}_targetwise_validation_selected_predictions.csv"
        )
        assembled.to_csv(out_path, index=False)
        rmse_mean, rmse, mae = mrmse_from_frame(assembled)
        eval_rows.append(
            {
                "seed": seed,
                "variant": "targetwise_validation_selected",
                "holdout_mrmse": rmse_mean,
                "holdout_mae_mean": float(np.mean(list(mae.values()))),
                **{f"rmse_{target}": rmse[target] for target in TARGETS},
                "prediction_path": str(out_path),
            }
        )

    summary = {
        "task": "Validation-selected per-target ensemble from completed residual variants",
        "selection_rule": "For each seed and target, select the variant with the lowest TauREx validation RMSE for that target, then evaluate assembled predictions on POSEIDON holdout.",
        "leakage_note": "POSEIDON holdout labels are not used for target/variant selection.",
        "seeds": list(seeds),
        "selection_rows": selection_rows,
        "eval_rows": eval_rows,
        "aggregate": {
            "targetwise_validation_selected_holdout_mrmse": summarize(
                [row["holdout_mrmse"] for row in eval_rows]
            ),
        },
        "comparison_sources": {
            "architecture_suite": str(ARCH_DIR / "analysis" / "aggregate_ranked.csv"),
            "controlled_summary": str(
                CONTROL_DIR / "controlled_aggregate_summary.json"
            ),
        },
    }
    out_dir = prediction_dir / "analysis"
    save_json(out_dir / "targetwise_ensemble_summary.json", summary)
    write_csv(out_dir / "selection_rows.csv", selection_rows)
    write_csv(out_dir / "targetwise_eval_rows.csv", eval_rows)
    append_manifest(
        {
            "timestamp_unix": time.time(),
            "command": " ".join([sys.executable, *sys.argv]),
            "cwd": str(PROJECT_ROOT),
            "duration_seconds": time.time() - start,
            "outputs": [str(out_dir / "targetwise_ensemble_summary.json")],
            "status": "ok",
        }
    )
    print(
        json.dumps(
            {"summary": str(out_dir / "targetwise_ensemble_summary.json")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
