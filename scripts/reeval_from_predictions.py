"""Re-eval by recomputing metrics from SAVED prediction CSVs.

No weights, no model import, no data loading -> works in any env with pandas/numpy
(e.g. .venv-qml). Covers every reported number whose prediction file exists in the repo:

  - #8 quantum snapshot  -> Poseidon holdout mRMSE
  - #9 noquant snapshot  -> Poseidon holdout mRMSE  (checkpoint ABSENT; predictions are
                            the only verification source)
  - #1/#2/#3 ExoBiome epoch6 -> holdout / val / mac-holdout

quantum_scale note. An earlier version of this file asserted that the artifact/table numbers are
at `quantum_scale = 1.0`. That is not what `audit/a04_quantum_scale_provenance.py` found, and the
assertion has been withdrawn from the two records it affected. What a04 actually reports is:

  - `scales_reproducing_the_published_number = []` on BOTH splits -- no swept scale reproduces the
    committed 0.299376 / 0.293614;
  - scale 1.0 reproduces only the MAC RE-EVALUATION (0.298693), and `a04:34-36` says in as many
    words that matching only that "does not establish the provenance of the number in the report";
  - the earlier `docs/VERIFICATION.md:65` claim of a "native ramp scale ~0.5" does not hold either:
    0.5 gives 0.295552 holdout / 0.292237 validation.

So the provenance of the committed numbers is UNRESOLVED (K4), CUDA+AMP at scale 1.0 being the
leading hypothesis rather than a finding. Only `exobiome_epoch6_mac_holdout` asserts a scale,
because there the sweep matches to the last digit; the other two carry `quantum_scale: null`.

Run:
    .venv-qml/bin/python scripts/reeval_from_predictions.py

Writes reports/reeval/<UTC-date>/<model>/{metrics.json,manifest.json}.
"""
from __future__ import annotations

import pandas as pd

import reeval_lib as R  # same directory (added to sys.path[0] when run as a script)

REPO = R.REPO_ROOT

# Independent label sources, deliberately NOT `audit_lib` -- reeval must not share
# machinery with the audit it cross-checks. float32 round-trip through the saved CSVs puts the
# expected residual at ~1e-6, so 1e-5 separates "same labels" from "different labels" with an order
# of margin while staying far below any effect being measured.
TRUTH_TOL = 1e-5
ADC_LABELS = REPO / "data/ariel-ml-dataset/TrainingData/Ground Truth Package/FM_Parameter_Table.csv"
XGEN_LABELS = REPO / "data/TauREx set/labels.parquet"
XGEN_COL = {"log_H2O": "log10_vmr_h2o", "log_CO2": "log10_vmr_co2", "log_CO": "log10_vmr_co",
            "log_CH4": "log10_vmr_ch4", "log_NH3": "log10_vmr_nh3"}


def verify_truth_column(df, source: str) -> dict:
    """Compare a prediction CSV's own `true_*` columns against an independent load of the labels.

    Returns the comparison for the manifest. Raises if any id is unmatched or any gas disagrees by
    more than TRUTH_TOL -- a CSV whose truth column is not the truth would otherwise recompute to a
    perfectly self-consistent, perfectly meaningless scalar.
    """
    import numpy as np

    ids = df["planet_ID"].astype(str)
    if source == "adc":
        ref = pd.read_csv(ADC_LABELS).set_index("planet_ID").reindex(ids)
        cols, path = {g: g for g in R.GASES}, ADC_LABELS
    elif source == "crossgen":
        ref = pd.read_parquet(XGEN_LABELS).set_index("sample_id").reindex(ids)
        cols, path = XGEN_COL, XGEN_LABELS
    else:
        raise ValueError(f"unknown label source {source!r}")

    n_unmatched = int(ref[cols[R.GASES[0]]].isna().sum())
    per_gas = {
        g: float(np.nanmax(np.abs(df[f"true_{g}"].to_numpy(float) - ref[cols[g]].to_numpy(float))))
        for g in R.GASES
    }
    worst = max(per_gas.values())
    out = {
        "label_source": str(path.relative_to(REPO)),
        "label_source_sha256": R.sha256(path),
        "n_rows_compared": int(len(df)),
        "n_ids_unmatched": n_unmatched,
        "max_abs_diff_per_gas": per_gas,
        "max_abs_diff": worst,
        "tolerance": TRUTH_TOL,
        "verified": bool(n_unmatched == 0 and worst <= TRUTH_TOL),
        "what_this_establishes": "the CSV's true_* column equals an independent load of the labels; "
                                 "without it, recomputation only proves internal consistency of the CSV",
    }
    if not out["verified"]:
        raise ValueError(f"truth-column check FAILED: {n_unmatched} unmatched id(s), "
                         f"max |diff| = {worst:.3e} > {TRUTH_TOL:.0e}")
    return out


CASES = [
    dict(
        model="quantum_snapshot_poseidon",
        csv=REPO / "reports/ariel_quantum_taurex_snapshot_20260312_1003/poseidon_holdout_predictions.csv",
        reported=3.2156150341033936,
        method="recompute-from-predictions",
        checkpoint=REPO / "reports/ariel_quantum_taurex_snapshot_20260312_1003/stage2_best_model_epoch005.pt",
        data_root="data/TauREx set",
        labels="crossgen",
        seed=42,
        quantum_scale=0.625,
        notes="#8 cross-gen Poseidon side. TauREx-val side (1.449002) confirmed against training log.",
    ),
    dict(
        model="noquant_snapshot_poseidon",
        csv=REPO / "reports/taurex_noquant_taurex_snapshot_20260312_133054/poseidon_holdout_predictions.csv",
        reported=3.2795588970184326,
        method="recompute-from-predictions",
        checkpoint=None,
        data_root="data/TauREx set",
        labels="crossgen",
        seed=42,
        quantum_scale=None,
        notes=(
            "#9 - checkpoint weights ABSENT from repo (best_model_epoch059.pt); predictions are the "
            "only verification source. Corroborated independently: a14_importability lists that exact "
            "path under missing_paths."
        ),
    ),
    dict(
        model="exobiome_epoch6_val",
        csv=REPO / "artifacts/ariel_quantum_best_v4_epoch6/validation_predictions.csv",
        reported=0.29361358284950256,
        method="recompute-from-artifact-predictions",
        checkpoint=REPO / "artifacts/ariel_quantum_best_v4_epoch6/best_model.pt",
        data_root="data/ariel-ml-dataset",
        split_manifest="artifacts/ariel_quantum_best_v4_epoch6/split_manifest.json",
        labels="adc",
        seed=42,
        quantum_scale=None,
        notes=(
            "#2. What is recomputed here is the committed scalar 0.293614 from the committed CSV; "
            "the eval scale behind that number is NOT established and this record does not assert "
            "one. a04 swept the checkpoint at five scales and reports "
            "scales_reproducing_the_published_number = [] on this split: 0.294821 at scale 1.0 and "
            "0.292237 at 0.5 both differ from 0.293614. CUDA+AMP at scale 1.0 is the most likely "
            "explanation for the ~1.2e-3 gap and stays a HYPOTHESIS (K4); a04:34-36 states outright "
            "that matching only the Mac re-eval does not establish the provenance of the published "
            "number. quantum_scale is therefore left null rather than guessed."
        ),
    ),
    dict(
        model="exobiome_epoch6_holdout",
        csv=REPO / "artifacts/ariel_quantum_best_v4_epoch6/holdout_predictions.csv",
        reported=0.2993761897087097,
        method="recompute-from-artifact-predictions",
        checkpoint=REPO / "artifacts/ariel_quantum_best_v4_epoch6/best_model.pt",
        data_root="data/ariel-ml-dataset",
        split_manifest="artifacts/ariel_quantum_best_v4_epoch6/split_manifest.json",
        labels="adc",
        seed=42,
        quantum_scale=None,
        notes=(
            "#1. What is recomputed here is the committed scalar 0.299376 from the committed CSV; "
            "the eval scale behind that number is NOT established and this record does not assert "
            "one. a04 swept the checkpoint at five scales and reports "
            "scales_reproducing_the_published_number = [] on this split: 0.298693 at scale 1.0 and "
            "0.295552 at 0.5 both differ from 0.299376. CUDA+AMP at scale 1.0 is the most likely "
            "explanation for the ~6.8e-4 gap and stays a HYPOTHESIS (K4); a04:34-36 states outright "
            "that matching only the Mac re-eval does not establish the provenance of the published "
            "number. quantum_scale is therefore left null rather than guessed."
        ),
    ),
    dict(
        model="exobiome_epoch6_mac_holdout",
        csv=REPO / "artifacts/ariel_quantum_best_v4_epoch6/mac_eval_20260312/mac_holdout_predictions.csv",
        reported=0.29869264364242554,
        method="recompute-from-artifact-predictions",
        checkpoint=REPO / "artifacts/ariel_quantum_best_v4_epoch6/best_model.pt",
        data_root="data/ariel-ml-dataset",
        labels="adc",
        seed=42,
        quantum_scale=1.0,
        notes=(
            "#3 mac re-eval holdout. This is the one artifact that pins the scale exactly, which is "
            "why this is the only case here that asserts a quantum_scale: "
            "0.29869264364242554 equals the a04 sweep at 1.0 to the last digit "
            "(matches_mac_reeval = true at 1.0 and at no other scale)."
        ),
    ),
]


def main() -> None:
    print(f"{'model':34} {'rederived':>11} {'reported':>11} {'delta':>12}  {'truth':>9}  status")
    print("-" * 96)
    for c in CASES:
        df = pd.read_csv(c["csv"])
        truth = verify_truth_column(df, c["labels"])
        metrics = R.rmse_from_predictions(df)
        run_dir, status, delta = R.write_reeval_run(
            c["model"],
            metrics,
            method=c["method"],
            inputs={"source_predictions": c["csv"]},
            reported=c["reported"],
            checkpoint=c.get("checkpoint"),
            data_root=c.get("data_root"),
            split_manifest=c.get("split_manifest"),
            seed=c.get("seed"),
            quantum_scale=c.get("quantum_scale"),
            notes=c.get("notes"),
            truth_check=truth,
        )
        print(
            f"{c['model']:34} {metrics['rmse_mean']:11.6f} {c['reported']:11.6f} "
            f"{delta:12.6f}  {truth['max_abs_diff']:9.1e}  {status}  -> {run_dir.relative_to(REPO)}"
        )


if __name__ == "__main__":
    main()
