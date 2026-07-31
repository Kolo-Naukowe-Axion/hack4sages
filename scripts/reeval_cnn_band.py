"""CNN baseline band (option 3): re-run the CNN notebook N times, report mean +/- std.

Why it is worth running: 0.650037 is quoted from `reports/model_comparison/rmse/cnn_metrics.json`
with no prediction file behind it. `audit/a02_trivial_baseline.py` marks THREE sources that way --
this one (`a02:46`), "winner on TauREx" 3.4531 (`a02:52`) and the H200 row 2.8946 (`a02:54`); the
CNN row is the only one of the three with positive skill, not the only one marked. (`a13` counts
differently again -- it classifies this number as `backed_summary`, since the vendored directory
does hold weights and code; see K10(b).) This script cannot reproduce the digit (see below); run it
and it WOULD replace an unbacked scalar with a measured band.

STATUS: NOT YET RUN. There is no `cnn_holdout_band` directory under `reports/reeval/`, so as of
today the CNN row remains an unreproduced scalar and nothing here has superseded it. Do not cite
this file as if the band existed.

The CNN notebook RE-TRAINS the net (`model.fit` + `save_weights`) and predicts via MC-Dropout
averaging, so a single holdout mRMSE is stochastic (training variance + dropout). The result is
therefore judged against a band derived from this run's own spread: the manifest records
`tolerance_kind: "seed-band"` and status `within-seed-band`, never `verified`.

The notebook and its env live OUTSIDE this repo, so the location must be supplied explicitly --
there is no portable default and a reviewer cannot run this without the external checkout:

    CNN_DIR=/path/to/ADC2023-baseline
    "$CNN_DIR/.venv/bin/python" scripts/reeval_cnn_band.py --cnn-dir "$CNN_DIR" --runs 3

Needs the CNN env (numpy 2.4.3 + TF 2.21 + keras 3). Each run re-trains the CNN (minutes), so
--runs 3 is the practical default. Writes reports/reeval/<UTC-date>/cnn_holdout_band/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import reeval_lib as R  # scripts/ is sys.path[0] when run as a script

NOTEBOOK = "ariel_cnn.ipynb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-dir", type=Path, default=os.environ.get("CNN_DIR"),
                    help="external ADC2023-baseline checkout holding %s (or set CNN_DIR)" % NOTEBOOK)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--reported", type=float, default=0.650037)
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    if args.cnn_dir is None:
        ap.error("--cnn-dir (or CNN_DIR) is required: the notebook lives outside this repo")
    cnn_dir = Path(args.cnn_dir).expanduser().resolve()
    if not (cnn_dir / NOTEBOOK).is_file():
        ap.error(f"{NOTEBOOK} not found in {cnn_dir}")
    metrics_path = cnn_dir / "cnn_metrics.json"

    means: list[float] = []
    per_gas_runs: list[dict] = []
    rows_seen: set[int] = set()
    for i in range(1, args.runs + 1):
        print(f"[run {i}/{args.runs}] executing {NOTEBOOK} (re-trains CNN)...", flush=True)
        before = R.sha256(metrics_path)
        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
             f"--ExecutePreprocessor.timeout={args.timeout}",
             "--output", f"ariel_cnn_reeval_run{i}.ipynb", NOTEBOOK],
            cwd=str(cnn_dir), check=True,
        )
        after = R.sha256(metrics_path)
        if after is None:
            raise SystemExit(f"run {i}: {metrics_path} does not exist after executing the notebook")
        if i > 1 and after == before:
            raise SystemExit(
                f"run {i}: {metrics_path} is byte-identical to the previous run (sha256 {after[:12]}). "
                f"The notebook did not rewrite it, so the remaining runs would re-read a stale file "
                f"and the 'band' would be one run counted {args.runs} times."
            )
        payload = json.loads(metrics_path.read_text())
        means.append(float(payload["rmse_mean"]))
        per_gas_runs.append(payload.get("rmse") or {})
        if payload.get("rows") is not None:
            rows_seen.add(int(payload["rows"]))
        print(f"[run {i}] holdout mRMSE = {means[-1]:.6f}", flush=True)

    mean, std = float(np.mean(means)), float(np.std(means))
    # Per-gas as the mean over runs, so `rmse` decomposes `rmse_mean`
    gases = sorted({g for d in per_gas_runs for g in d})
    per_gas = {g: float(np.mean([d[g] for d in per_gas_runs if g in d])) for g in gases}
    rows = rows_seen.pop() if len(rows_seen) == 1 else None
    metrics = {"rmse": per_gas, "rmse_mean": mean, "rows": rows,
               "rmse_per_gas_per_run": per_gas_runs,
               "rmse_mean_per_run": means, "rmse_mean_std": std, "runs": args.runs,
               "rows_disagreed_across_runs": sorted(rows_seen) if len(rows_seen) > 1 else None,
               "aggregation": "rmse and rmse_mean are both means over the same runs; "
                              "mean(rmse.values()) == rmse_mean up to summation order"}
    if rows is None:
        print(f"WARNING: the notebook's cnn_metrics.json carries no consistent 'rows' field "
              f"({sorted(rows_seen) or 'absent'}); the band will not record its row set.", flush=True)
    run_dir, status, delta = R.write_reeval_run(
        "cnn_holdout_band",
        metrics,
        method=f"re-run notebook x{args.runs} (retrain + MC-Dropout avg); mean±std over runs",
        reported=args.reported,
        tolerance=max(2.0 * std, 5e-3),
        tolerance_kind="seed-band",
        data_root="data/ariel-ml-dataset",
        seed=None,
        quantum_scale=None,
        notes=(f"CNN baseline is stochastic (model.fit + MC-Dropout averaging). "
               f"band mean={mean:.6f} std={std:.6f} over {args.runs} runs {means}. "
               f"Report as mean±σ in the inter-class benchmark; the legacy 0.650037 is one run "
               f"and is the 'NO backing artifact' row in a02_trivial_baseline. "
               f"Notebook checkout: {cnn_dir} (outside this repo)."),
    )
    print(f"\nCNN holdout mRMSE = {mean:.6f} ± {std:.6f}  (n={args.runs})  status={status}  -> {run_dir}")


if __name__ == "__main__":
    main()
