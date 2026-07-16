"""Analyze existing TauREx-to-POSEIDON residual generalization artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


TARGETS = ("log_H2O", "log_CO2", "log_CO", "log_CH4", "log_NH3")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUANTUM_DIR = PROJECT_ROOT / "reports" / "ariel_quantum_taurex_snapshot_20260312_1003"
NOQUANT_DIR = (
    PROJECT_ROOT / "reports" / "taurex_noquant_taurex_snapshot_20260312_133054"
)
DATA_ROOT = PROJECT_ROOT / "data" / "TauREx set"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = ["planet_ID"]
    for target in TARGETS:
        expected.extend([f"true_{target}", f"pred_{target}"])
    missing = [col for col in expected if col not in frame.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {missing}")
    return frame.loc[:, expected].copy()


def rmse_by_target(frame: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in TARGETS:
        err = frame[f"pred_{target}"].to_numpy() - frame[f"true_{target}"].to_numpy()
        out[target] = float(np.sqrt(np.mean(np.square(err))))
    return out


def mae_by_target(frame: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for target in TARGETS:
        err = frame[f"pred_{target}"].to_numpy() - frame[f"true_{target}"].to_numpy()
        out[target] = float(np.mean(np.abs(err)))
    return out


def mean_row_abs_error(frame: pd.DataFrame) -> np.ndarray:
    values = []
    for target in TARGETS:
        values.append(
            np.abs(
                frame[f"pred_{target}"].to_numpy() - frame[f"true_{target}"].to_numpy()
            )
        )
    return np.stack(values, axis=1).mean(axis=1)


def bootstrap_mrmse_difference(
    quantum: pd.DataFrame,
    noquant: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    q_true = np.stack([quantum[f"true_{t}"].to_numpy() for t in TARGETS], axis=1)
    q_pred = np.stack([quantum[f"pred_{t}"].to_numpy() for t in TARGETS], axis=1)
    c_true = np.stack([noquant[f"true_{t}"].to_numpy() for t in TARGETS], axis=1)
    c_pred = np.stack([noquant[f"pred_{t}"].to_numpy() for t in TARGETS], axis=1)
    if not np.allclose(q_true, c_true, atol=1.0e-6):
        raise RuntimeError(
            "Quantum and noquant prediction files have different truths."
        )

    n = q_true.shape[0]
    diffs = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        rows = rng.integers(0, n, size=n)
        q_rmse = np.sqrt(np.mean(np.square(q_pred[rows] - q_true[rows]), axis=0)).mean()
        c_rmse = np.sqrt(np.mean(np.square(c_pred[rows] - c_true[rows]), axis=0)).mean()
        diffs[i] = c_rmse - q_rmse

    observed = float(
        np.sqrt(np.mean(np.square(c_pred - c_true), axis=0)).mean()
        - np.sqrt(np.mean(np.square(q_pred - q_true), axis=0)).mean()
    )
    return {
        "definition": "noquant_mrmse_minus_quantum_mrmse; positive means quantum has lower mRMSE",
        "observed": observed,
        "iterations": int(iterations),
        "seed": int(seed),
        "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
        "p_bootstrap_le_zero": float(np.mean(diffs <= 0.0)),
        "bootstrap_mean": float(np.mean(diffs)),
        "bootstrap_std": float(np.std(diffs, ddof=1)),
    }


def param_counts(checkpoint_path: Path, quantum: bool) -> dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    tensor_items = {k: v for k, v in state.items() if torch.is_tensor(v)}
    total = int(sum(v.numel() for v in tensor_items.values()))
    if quantum:
        residual_keys = ("projector", "quantum_block", "quantum_head", "quantum_gate")
    else:
        residual_keys = ("refinement_projector", "refinement_block", "refinement_head")
    residual = int(
        sum(
            v.numel()
            for k, v in tensor_items.items()
            if any(token in k for token in residual_keys)
        )
    )
    return {
        "total_parameters_in_state": total,
        "residual_parameters_in_state": residual,
    }


def split_and_leakage_checks() -> dict[str, Any]:
    labels_path = DATA_ROOT / "labels.parquet"
    spectra_path = DATA_ROOT / "spectra.h5"
    labels = pd.read_parquet(labels_path)
    counts = labels.groupby(["generator", "split"]).size().to_dict()
    train_ids = set(
        labels.loc[
            (labels.generator == "tau") & (labels.split == "train"), "sample_id"
        ].astype(str)
    )
    val_ids = set(
        labels.loc[
            (labels.generator == "tau") & (labels.split == "val"), "sample_id"
        ].astype(str)
    )
    holdout_ids = set(
        labels.loc[
            (labels.generator == "poseidon") & (labels.split == "test"), "sample_id"
        ].astype(str)
    )

    with h5py.File(spectra_path, "r") as handle:
        h5_sample_ids = np.array(handle["sample_id"][:]).astype(str)
        h5_generators = np.array(handle["generator"][:]).astype(str)
        h5_splits = np.array(handle["split"][:]).astype(str)
        aligned = bool(
            np.array_equal(labels["sample_id"].astype(str).to_numpy(), h5_sample_ids)
            and np.array_equal(
                labels["generator"].astype(str).to_numpy(), h5_generators
            )
            and np.array_equal(labels["split"].astype(str).to_numpy(), h5_splits)
        )

    return {
        "labels_path": str(labels_path),
        "spectra_path": str(spectra_path),
        "labels_sha256": sha256(labels_path),
        "spectra_h5_sha256": sha256(spectra_path),
        "counts_by_generator_split": {
            f"{k[0]}:{k[1]}": int(v) for k, v in counts.items()
        },
        "sample_id_overlap_train_val": int(len(train_ids & val_ids)),
        "sample_id_overlap_train_poseidon_holdout": int(len(train_ids & holdout_ids)),
        "sample_id_overlap_val_poseidon_holdout": int(len(val_ids & holdout_ids)),
        "labels_hdf5_alignment_ok": aligned,
        "scalers_fit_scope_from_code": "prepare_data fits aux/target/spectral scalers on TauREx train split only, then transforms val and POSEIDON holdout",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "quantum_residual_generalization_analysis",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    quantum_predictions_path = QUANTUM_DIR / "poseidon_holdout_predictions.csv"
    noquant_predictions_path = NOQUANT_DIR / "poseidon_holdout_predictions.csv"
    quantum = load_predictions(quantum_predictions_path)
    noquant = load_predictions(noquant_predictions_path)
    if not quantum["planet_ID"].equals(noquant["planet_ID"]):
        raise RuntimeError("Prediction rows are not paired by planet_ID.")

    q_rmse = rmse_by_target(quantum)
    c_rmse = rmse_by_target(noquant)
    q_mae = mae_by_target(quantum)
    c_mae = mae_by_target(noquant)
    q_row_mae = mean_row_abs_error(quantum)
    c_row_mae = mean_row_abs_error(noquant)

    val_quantum = read_json(QUANTUM_DIR / "training_state.json")["best_val_rmse_mean"]
    val_noquant = read_json(NOQUANT_DIR / "training_state.json")["best_val_rmse_mean"]
    bootstrap = bootstrap_mrmse_difference(
        quantum, noquant, iterations=args.bootstrap, seed=args.seed
    )

    summary = {
        "task": "gated quantum residual vs classical residual cross-simulator generalization",
        "created_unix": time.time(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": int(args.seed),
        "paired_rows": int(len(quantum)),
        "targets": list(TARGETS),
        "input_artifacts": {
            "quantum_predictions": str(quantum_predictions_path),
            "quantum_predictions_sha256": sha256(quantum_predictions_path),
            "noquant_predictions": str(noquant_predictions_path),
            "noquant_predictions_sha256": sha256(noquant_predictions_path),
        },
        "models": {
            "gated_quantum_residual": {
                "source_dir": str(QUANTUM_DIR),
                "validation_tau_mrmse": float(val_quantum),
                "poseidon_holdout_mrmse": float(np.mean(list(q_rmse.values()))),
                "poseidon_holdout_mae_mean": float(np.mean(list(q_mae.values()))),
                "rmse_by_target": q_rmse,
                "mae_by_target": q_mae,
                "checkpoint_state_counts": param_counts(
                    QUANTUM_DIR / "stage2_best_model_epoch005.pt", quantum=True
                ),
                "config": read_json(QUANTUM_DIR / "config.json"),
                "training_state": read_json(QUANTUM_DIR / "training_state.json"),
            },
            "classical_residual_refiner": {
                "source_dir": str(NOQUANT_DIR),
                "validation_tau_mrmse": float(val_noquant),
                "poseidon_holdout_mrmse": float(np.mean(list(c_rmse.values()))),
                "poseidon_holdout_mae_mean": float(np.mean(list(c_mae.values()))),
                "rmse_by_target": c_rmse,
                "mae_by_target": c_mae,
                "checkpoint_state_counts": param_counts(
                    NOQUANT_DIR / "best_model_epoch059.pt", quantum=False
                ),
                "config": read_json(NOQUANT_DIR / "config.json"),
                "training_state": read_json(NOQUANT_DIR / "training_state.json"),
            },
        },
        "comparisons": {
            "poseidon_mrmse_noquant_minus_quantum": float(
                np.mean(list(c_rmse.values())) - np.mean(list(q_rmse.values()))
            ),
            "tau_validation_mrmse_noquant_minus_quantum": float(
                val_noquant - val_quantum
            ),
            "poseidon_gap_quantum": float(np.mean(list(q_rmse.values())) - val_quantum),
            "poseidon_gap_noquant": float(np.mean(list(c_rmse.values())) - val_noquant),
            "paired_row_mean_abs_error_quantum_better_fraction": float(
                np.mean(q_row_mae < c_row_mae)
            ),
            "paired_row_mean_abs_error_noquant_better_fraction": float(
                np.mean(c_row_mae < q_row_mae)
            ),
            "paired_row_mean_abs_error_tie_fraction": float(
                np.mean(np.isclose(q_row_mae, c_row_mae))
            ),
            "bootstrap": bootstrap,
        },
        "split_and_leakage_checks": split_and_leakage_checks(),
        "limitations": [
            "This analysis uses existing single-seed snapshots, not a freshly rerun multi-seed training campaign.",
            "The noquant refiner is a parameter-matched classical residual path, but it is not an exact architectural isomorph of the quantum residual gate.",
            "The quantum snapshot was captured at epoch 5 while remote training was still continuing; the noquant run trained for 60 epochs.",
        ],
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    target_rows = []
    for target in TARGETS:
        target_rows.append(
            {
                "target": target,
                "quantum_rmse": q_rmse[target],
                "noquant_rmse": c_rmse[target],
                "noquant_minus_quantum_rmse": c_rmse[target] - q_rmse[target],
                "quantum_mae": q_mae[target],
                "noquant_mae": c_mae[target],
                "noquant_minus_quantum_mae": c_mae[target] - q_mae[target],
            }
        )
    write_csv(args.out_dir / "per_target_metrics.csv", target_rows)

    manifest_record = {
        "timestamp_unix": time.time(),
        "command": " ".join([sys.executable, *sys.argv]),
        "cwd": str(PROJECT_ROOT),
        "duration_seconds": time.time() - start,
        "outputs": [str(summary_path), str(args.out_dir / "per_target_metrics.csv")],
        "status": "ok",
    }
    append_manifest(PROJECT_ROOT / "_script_manifest.jsonl", manifest_record)
    print(
        json.dumps({"summary": str(summary_path), "manifest_appended": True}, indent=2)
    )


if __name__ == "__main__":
    main()
