"""Non-invasive re-eval of the SOTA model (ADC2023 winner-style independent NSF).

Loads the trained checkpoint and re-runs the SAME evaluation the training script does
in its final-eval block (`models/adc_winner_on_ariel/train.py`) — it imports the package
and calls `evaluate_point_metric`; it does NOT modify any model/train code and does NOT
train. This preserves the methodological purity of the ported winner model.

Prerequisite — build the prepared dataset once, using the exact saved split
(`data/val_dataset`, identical to trained_run/saved_split_manifest.json):

    . .venv-qml/bin/activate
    python -m models.adc_winner_on_ariel.prepare_dataset -h          # confirm arg names
    python -m models.adc_winner_on_ariel.prepare_dataset \
        --data-root data/ariel-ml-dataset \
        --split-source data/val_dataset \
        --output data/generated-data/ariel_winner_nf_prepared

Then:

    .venv-qml/bin/python scripts/reeval_sota.py            # 5 noise seeds, the shipped configuration

Writes reports/reeval/<UTC-date>/sota_{validation,holdout}_{median,mean}/{metrics,manifest}.json.

As-reported (median estimator): holdout 0.552288, val 0.552812. This eval samples observation
noise, so a re-run does NOT land on those digits -- the 5-seed bands come out around 0.5545
holdout / 0.5522 val. The comparison is therefore judged against a band derived from this run's
own seed spread, and the manifests record `tolerance_kind: "seed-band"` with status
`within-seed-band`, NOT `verified`. Do not quote such a run as a reproduction.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

import reeval_lib as R  # scripts/ is sys.path[0] when run as a script

REPO = R.REPO_ROOT
sys.path.insert(0, str(REPO))  # so `import models.adc_winner_on_ariel...` resolves

from models.adc_winner_on_ariel.dataset import load_prepared_data, move_prepared_data_to_device  # noqa: E402
from models.adc_winner_on_ariel.evaluate import evaluate_point_metric  # noqa: E402
from models.adc_winner_on_ariel.model import IndependentNSF, ModelConfig  # noqa: E402

REPORTED = {"validation": 0.5528115034103394, "holdout": 0.5522884130477905}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=REPO / "models/adc_winner_on_ariel/trained_run")
    ap.add_argument("--prepared-data", type=Path, default=REPO / "data/generated-data/ariel_winner_nf_prepared")
    ap.add_argument("--checkpoint", type=Path, default=None, help="default: <run-dir>/best_model_by_mrmse.pt")
    # Default is the 5-seed set the shipped records were produced with. It must NOT be a single
    # seed: with one seed np.std == 0, the band below collapses to its hard floor, and the run would
    # still be labelled `seed-band` while containing no seed information at all. `main` rejects that
    # configuration outright a few lines down.
    ap.add_argument("--noise-seeds", default="42,1,2,3,4",
                    help="comma-separated noise seeds; >=2 required, the band is 2 sigma over them")
    ap.add_argument("--point-estimates", default="median,mean",
                    help="comma list. plan §6(B) mandates 'mean' for the inter-class benchmark; "
                         "'median' reproduces the as-reported 0.552288.")
    args = ap.parse_args()
    noise_seeds = [int(s) for s in str(args.noise_seeds).split(",") if s.strip()]
    point_estimates = [s.strip() for s in str(args.point_estimates).split(",") if s.strip()]
    if len(noise_seeds) < 2:
        ap.error(f"--noise-seeds needs at least 2 seeds to define a band, got {noise_seeds}. "
                 f"With one seed the spread is 0 and `tolerance_kind='seed-band'` would be a label "
                 f"on the hard floor, not on a measured spread.")

    checkpoint = args.checkpoint or (args.run_dir / "best_model_by_mrmse.pt")
    settings = yaml.safe_load((args.run_dir / "settings_resolved.yaml").read_text())
    ev = settings["evaluation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = move_prepared_data_to_device(load_prepared_data(args.prepared_data), device)

    model = IndependentNSF(ModelConfig(**settings["model"])).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])

    import numpy as np

    def eval_once(split_name: str, noise_seed: int, point_estimate: str) -> dict:
        return evaluate_point_metric(
            model,
            getattr(data, split_name),
            data.scalers,
            device=device,
            num_samples=int(ev["final_num_samples"]),
            point_estimate=point_estimate,
            batch_size=int(settings["training"]["eval_batch_size"]),
            max_rows=None,
            row_seed=int(ev.get("row_seed", 42)),
            sample_noise=True,
            noise_seed=noise_seed,
        )

    print(f"device={device.type}  noise_seeds={noise_seeds}  point_estimates={point_estimates}")
    print(f"{'split/est':22} {'mean':>10} {'std':>10} {'reported':>10} {'in ±2σ?':>9}  status")
    print("-" * 78)
    for split_name in ("validation", "holdout"):
        for est in point_estimates:
            runs = [eval_once(split_name, s, est) for s in noise_seeds]
            means = [float(r["rmse_mean"]) for r in runs]
            mean, std = float(np.mean(means)), float(np.std(means))
            # 'reported' 0.552x was produced with MEDIAN; only compare like-for-like.
            reported = REPORTED[split_name] if est == "median" else None
            band = max(2.0 * std, 1e-3)
            in_band = (reported is not None and abs(mean - reported) <= band)
            # `rmse` must decompose `rmse_mean`. Taking per-gas from runs[0] while rmse_mean was
            # the mean over seeds put the two 1.4e-3..2.7e-3 apart, so anyone summing the per-gas
            # block failed to recover the headline. Both are now means over the same seeds; the
            # first-seed block stays available under its own name.
            per_gas = {g: float(np.mean([r["rmse"][g] for r in runs])) for g in R.GASES}
            metrics = {"rmse": per_gas, "rmse_mean": mean, "rows": runs[0]["rows"],
                       "point_estimate": est,
                       "rmse_per_gas_first_seed": runs[0]["rmse"],
                       "rmse_per_gas_std": {g: float(np.std([r["rmse"][g] for r in runs]))
                                            for g in R.GASES},
                       "rmse_mean_per_seed": dict(zip(map(str, noise_seeds), means)),
                       "rmse_mean_std": std,
                       "aggregation": "rmse and rmse_mean are both means over the same noise seeds; "
                                      "mean(rmse.values()) == rmse_mean up to summation order "
                                      "(observed |diff| ~1e-8, float64 associativity)"}
            run_dir, status, delta = R.write_reeval_run(
                f"sota_{split_name}_{est}",
                metrics,
                method=f"reeval-from-weights (evaluate_point_metric, num_samples=128, {est}; {len(noise_seeds)} noise seed(s))",
                reported=reported,
                tolerance=band,
                tolerance_kind="seed-band",
                checkpoint=checkpoint,
                data_root="data/ariel-ml-dataset",
                split_manifest="data/val_dataset/manifest.json",
                seed=int(settings.get("seed", 42)),
                quantum_scale=None,
                notes=(f"ADC2023 winner-style INDEPENDENT NSF (5 gas marginals only), team reimplementation, "
                       f"NOT the winner's released weights and NOT the joint 7-target posterior. "
                       f"point_estimate={est} (plan §6B mandates 'mean' for the inter-class benchmark; "
                       f"'median' reproduces the as-reported number). "
                       f"Eval samples observation noise -> stochastic; device={device.type} (original: cuda). "
                       f"band mean={mean:.6f} std={std:.6f} over seeds {noise_seeds}"
                       + (f"; reported {'IN' if in_band else 'OUTSIDE'} ±2σ." if reported is not None else "; no as-reported mean baseline.")),
            )
            tag = f"{split_name}/{est}"
            rep_s = f"{reported:10.6f}" if reported is not None else f"{'—':>10}"
            print(f"{tag:22} {mean:10.6f} {std:10.6f} {rep_s} {str(in_band):>9}  {status}  -> {run_dir.relative_to(REPO)}")


if __name__ == "__main__":
    main()
