"""Aggregate controlled RTX 4090 quantum-residual campaign outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("stage1_classical", "gated_quantum_residual", "classical_residual")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def append_manifest(project_root: Path, payload: dict[str, Any]) -> None:
    with (project_root / "_script_manifest.jsonl").open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    out: dict[str, Any] = {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std_sample": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
    if arr.size > 1:
        out["sem"] = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return out


def bootstrap_seed_ci(
    values: list[float], *, seed: int = 20260706, iterations: int = 10000
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return {
        "seed": seed,
        "iterations": iterations,
        "ci95_percentile": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_mean": float(means.mean()),
        "p_mean_le_zero": float(np.mean(means <= 0.0)),
    }


def validate_split(summary: dict[str, Any], *, variant: str, seed: int) -> list[str]:
    issues = []
    dataset = summary.get("dataset", {})
    selectors = dataset.get("split_selectors", {})
    holdout = selectors.get("holdout", {})
    train = selectors.get("train", {})
    val = selectors.get("val", {})
    if holdout.get("generator") != "poseidon" or holdout.get("split") != "test":
        issues.append(
            f"seed {seed} {variant}: holdout selector is {holdout}, expected poseidon/test"
        )
    if train.get("generator") != "tau" or train.get("split") != "train":
        issues.append(
            f"seed {seed} {variant}: train selector is {train}, expected tau/train"
        )
    if val.get("generator") != "tau" or val.get("split") != "val":
        issues.append(f"seed {seed} {variant}: val selector is {val}, expected tau/val")
    if int(dataset.get("holdout_rows", -1)) != 685:
        issues.append(
            f"seed {seed} {variant}: holdout_rows={dataset.get('holdout_rows')}, expected 685"
        )
    if int(dataset.get("train_rows", -1)) != 37281:
        issues.append(
            f"seed {seed} {variant}: train_rows={dataset.get('train_rows')}, expected 37281"
        )
    if int(dataset.get("val_rows", -1)) != 4142:
        issues.append(
            f"seed {seed} {variant}: val_rows={dataset.get('val_rows')}, expected 4142"
        )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-dir",
        type=Path,
        default=Path("reports/controlled_quantum_residual_rtx4090_remote"),
    )
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    campaign_dir = (
        (project_root / args.campaign_dir).resolve()
        if not args.campaign_dir.is_absolute()
        else args.campaign_dir
    )
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    start = time.time()

    rows: list[dict[str, Any]] = []
    split_issues: list[str] = []
    for seed in seeds:
        for variant in VARIANTS:
            run_dir = campaign_dir / f"seed_{seed}" / variant
            summary_path = run_dir / "run_summary.json"
            metrics_path = run_dir / "holdout_metrics.json"
            if not summary_path.exists():
                raise FileNotFoundError(summary_path)
            summary = read_json(summary_path)
            holdout = read_json(metrics_path) if metrics_path.exists() else {}
            split_issues.extend(validate_split(summary, variant=variant, seed=seed))
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "best_epoch": summary.get("best_epoch"),
                    "validation_rmse_mean": summary.get("validation_rmse_mean"),
                    "holdout_rmse_mean": summary.get("holdout_rmse_mean"),
                    "holdout_mae_mean": summary.get("holdout_mae_mean"),
                    "holdout_rows": holdout.get(
                        "rows", summary.get("dataset", {}).get("holdout_rows")
                    ),
                    "run_dir": str(run_dir),
                    "run_summary": str(summary_path),
                    "holdout_metrics": str(metrics_path),
                }
            )

    by_seed = {
        seed: {row["variant"]: row for row in rows if row["seed"] == seed}
        for seed in seeds
    }
    paired_rows = []
    for seed, variants in by_seed.items():
        stage1 = variants["stage1_classical"]["holdout_rmse_mean"]
        quantum = variants["gated_quantum_residual"]["holdout_rmse_mean"]
        classical = variants["classical_residual"]["holdout_rmse_mean"]
        paired_rows.append(
            {
                "seed": seed,
                "stage1_holdout_mrmse": stage1,
                "quantum_holdout_mrmse": quantum,
                "classical_residual_holdout_mrmse": classical,
                "classical_minus_quantum_holdout_mrmse": classical - quantum,
                "stage1_minus_quantum_holdout_mrmse": stage1 - quantum,
                "stage1_minus_classical_holdout_mrmse": stage1 - classical,
            }
        )

    diffs = [row["classical_minus_quantum_holdout_mrmse"] for row in paired_rows]
    quantum_vals = [row["quantum_holdout_mrmse"] for row in paired_rows]
    classical_vals = [row["classical_residual_holdout_mrmse"] for row in paired_rows]
    stage1_vals = [row["stage1_holdout_mrmse"] for row in paired_rows]
    result = {
        "task": "Controlled RTX 4090 TauREx-to-POSEIDON gated quantum residual vs classical residual",
        "metric": "mean RMSE across five gas log-abundance targets on POSEIDON holdout; lower is better",
        "holdout_rows_per_seed": 685,
        "train_rows": 37281,
        "validation_rows": 4142,
        "seeds": seeds,
        "rows": rows,
        "paired_rows": paired_rows,
        "summary": {
            "stage1_holdout_mrmse": summarize(stage1_vals),
            "quantum_holdout_mrmse": summarize(quantum_vals),
            "classical_residual_holdout_mrmse": summarize(classical_vals),
            "classical_minus_quantum_holdout_mrmse": summarize(diffs),
            "classical_minus_quantum_bootstrap_seed_ci": bootstrap_seed_ci(diffs),
            "quantum_better_seed_count": int(sum(value > 0 for value in diffs)),
            "classical_better_seed_count": int(sum(value < 0 for value in diffs)),
        },
        "split_audit": {
            "status": "pass" if not split_issues else "fail",
            "issues": split_issues,
            "expected": "train=tau/train, val=tau/val, holdout=poseidon/test, holdout_rows=685",
        },
        "known_issue_fixed": "An earlier seed-42 classical residual run used TauREx val as holdout; it was moved aside and rerun. The included seed-42 classical_residual has excluded_generators=[] and holdout=poseidon/test.",
        "provenance_notes": [
            "The campaign was completed in two launches: seed 42 first, then seeds 43 and 44. The copied campaign_config.json reflects the later 43,44 launch; this aggregate is the authoritative three-seed rollup.",
            "The original seed-42 log still contains an invalid classical-residual summary with TauREx-val holdout. That run was moved aside on the remote host and replaced by seed_42/classical_residual, whose split manifest passes the POSEIDON holdout audit.",
        ],
        "remote_workspace": "/home/iwo/hack4sages-quantum-residual-20260706",
        "local_campaign_dir": str(campaign_dir),
        "generated_unix": time.time(),
    }

    out_path = campaign_dir / "controlled_aggregate_summary.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    append_manifest(
        project_root,
        {
            "timestamp_unix": time.time(),
            "command": " ".join([sys.executable, *sys.argv]),
            "cwd": str(project_root),
            "duration_seconds": time.time() - start,
            "outputs": [str(out_path)],
            "status": "ok",
        },
    )
    print(
        json.dumps(
            {"summary": str(out_path), "split_audit": result["split_audit"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
