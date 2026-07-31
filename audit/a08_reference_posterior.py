"""A08 — Is the reported accuracy physically achievable? (information ceiling)

Proves/disproves finding K8. data/ariel-ml-dataset/TrainingData/Ground Truth Package/ ships
Tracedata.hdf5 (full nested-sampling posterior traces, 2884 x 7 with weights, for all 41,423
planets) and QuartilesTable.csv. That is a reference Bayesian retrieval performed with the true
forward model on the same spectra.

The reference is ONE concrete MultiNest run with the challenge's own priors and forward model —
NOT "the exact Bayesian solution", and the earlier premise "no estimator can beat the posterior"
was therefore false as stated. Measured on the 663 holdout planets that carry a reference
retrieval: its median is biased by -0.7936 dex and its IQR contains the truth 0.7897 of the time
against a nominal 0.50, i.e. it OVER-covers by 1.579x. A biased, over-wide posterior is beatable
by a point estimator, so the strong premise proves nothing. The narrower claim is still decisive
for K8: a regressor whose RMSE is a multiple BELOW the RMSE of the retrieval the challenge itself
published is not solving the problem that retrieval solved — it is inverting a (near-)noiseless
simulator. Calibration is measured and reported, but it deliberately does NOT enter the criterion:
mixing "does the model beat the published run" with "is the published run calibrated" into one
status would make the verdict unattributable to either question.

PASS criterion: model mRMSE >= reference-run mRMSE, both computed on the SAME planets (the 663
that have a reference retrieval).

This check also hands the project the assets it has been treating as unavailable:
per-planet posterior width, multimodality labels, and an information ceiling per gas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a08_reference_posterior",
    finding="K8 — the reported 0.30 dex beats the reference nested-sampling retrieval ~3.8x; the metric measures simulator inversion, not retrieval",
    question="How does the model's RMSE compare with the reference Bayesian posterior on the same planets?",
    criterion="model mRMSE >= reference-run mRMSE, both on the same 663 planets that carry a reference retrieval",
)

# Nominal coverage of the interquartile range. Used ONLY to diagnose the reference's calibration,
# which is reported alongside the verdict and does not enter it.
NOMINAL_IQR_COVERAGE = 0.50


def main() -> None:
    import h5py
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--predictions", default=str(A.EXOBIOME_ARTIFACT / "holdout_predictions.csv"))
    ap.add_argument("--trace-planets", type=int, default=200,
                    help="how many planets to open in Tracedata.hdf5 for the multimodality probe")
    ap.add_argument("--trace-seed", type=int, default=42,
                    help="seed for the RANDOM draw of probed planets (the split order is not random)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = A.adc_root() / "TrainingData/Ground Truth Package"
    q = pd.read_csv(gt / "QuartilesTable.csv").set_index("planet_ID")
    fm = A.load_adc_targets()
    ids = A.adc_split_ids(args.split)
    qq, y = q.reindex(ids), fm.loc[ids]

    # Common mask of planets with a reference retrieval, computed once before anything else.
    # Previously: the reference on the non-NaN mask (663 planets), the model on all 4143 CSV rows —
    # two different sets despite criterion= "on the same data". It matters: the reference/model ratio
    # is this finding's quoted number (4.789 on the mismatched sets vs 3.803 on the common one).
    # Conjunction over gases, because the per-gas masks happen to be identical today (663 each) —
    # should they diverge, the criterion stays unambiguous without a further fix.
    have = np.ones(len(ids), dtype=bool)
    for t in A.TARGETS:
        for suf in ("q1", "q2", "q3"):
            have &= ~np.isnan(qq[f"{t}_{suf}"].to_numpy(float))
    ids_ref = ids[have]

    per_gas = {}
    ref_rmse, ref_sigma, ref_bias, ref_cov = [], [], [], []
    for t in A.TARGETS:
        q1, q2, q3 = (qq[f"{t}_q1"].to_numpy(float), qq[f"{t}_q2"].to_numpy(float), qq[f"{t}_q3"].to_numpy(float))
        truth = y[t].to_numpy(float)
        m_gas = ~(np.isnan(q1) | np.isnan(q2) | np.isnan(q3))
        m = have                                   # every number on the COMMON mask
        sig = (q3[m] - q1[m]) / 1.349
        r = float(np.sqrt(np.mean((q2[m] - truth[m]) ** 2)))
        bias = float(np.mean(q2[m] - truth[m]))
        cov = float(np.mean((truth[m] >= q1[m]) & (truth[m] <= q3[m])))
        per_gas[t] = {"n_with_reference_retrieval": int(m_gas.sum()),
                      "n_scored_on_common_mask": int(m.sum()),
                      "reference_rmse_median_vs_truth": r,
                      "reference_bias": bias,
                      "reference_sigma_median": float(np.median(sig)),
                      "reference_sigma_mean": float(np.mean(sig)),
                      "iqr_coverage_of_truth": cov,
                      "prior_support_in_fm_table": [float(fm[t].min()), float(fm[t].max())]}
        ref_rmse.append(r)
        ref_sigma.append(float(np.median(sig)))
        ref_bias.append(bias)
        ref_cov.append(cov)

    pred = pd.read_csv(args.predictions)
    pred["planet_ID"] = pred["planet_ID"].astype(str)
    pred = pred.set_index("planet_ID")
    missing = [p for p in ids_ref if p not in pred.index]
    assert not missing, (f"{len(missing)} planet_ID z maski referencyjnej nie ma w {args.predictions} "
                         f"(pierwsze 5: {missing[:5]})")
    sub = pred.loc[list(ids_ref)]
    yt = sub[[f"true_{t}" for t in A.TARGETS]].to_numpy(float)
    yp = sub[[f"pred_{t}" for t in A.TARGETS]].to_numpy(float)
    model_m = A.mrmse(yt, yp)
    base_m, _ = A.constant_predictor_mrmse(yt)

    # The same quantity on the FULL CSV — only so the difference against the old run is visible
    # directly instead of having to be reconstructed. Does NOT enter the verdict.
    yt_all = pred[[f"true_{t}" for t in A.TARGETS]].to_numpy(float)
    yp_all = pred[[f"pred_{t}" for t in A.TARGETS]].to_numpy(float)

    # multimodality probe: reference traces are available, so "degenerate vs unimodal" needs no new campaign
    multimodal = None
    trace_path = gt / "Tracedata.hdf5"
    if trace_path.exists():
        counts = {t: 0 for t in A.TARGETS}
        n_ok = 0
        with h5py.File(trace_path, "r") as h:
            # RANDOM sample from the full list of trace-bearing planets. Previously:
            # ids[:trace_planets] — because adc_split_ids sorts numerically, the first 200 positions
            # are the lowest-numbered edge of the holdout, where 52% of planets carry a trace against
            # 16.0% globally; that was quoted as a property of the whole holdout. Enumeration reads
            # only shapes (h5py does not load data for .shape), so sweeping all 4143 keys is cheap.
            avail = []
            for pid in ids:
                key = f"Planet_{pid}"
                if key not in h or "tracedata" not in h[key]:
                    continue
                shape = h[key]["tracedata"].shape
                if shape == () or len(shape) != 2 or shape[1] < 7:
                    continue
                avail.append(pid)
            rng = np.random.default_rng(args.trace_seed)
            n_draw = len(avail) if args.trace_planets <= 0 else min(args.trace_planets, len(avail))
            picked = rng.choice(np.asarray(avail, dtype=object), size=n_draw, replace=False) \
                if avail else np.asarray([], dtype=object)
            for pid in picked:
                key = f"Planet_{pid}"
                # Split into two conditions. Previously: `if key not in h or "tracedata" in h[key] and
                # h[key]["tracedata"].shape == ()` — `or` binds looser than `and`, so a missing
                # "tracedata" key made the whole test False, `continue` never ran, and a KeyError
                # followed (the same pattern as a24:141, which gets it right). Unreachable on today's
                # data: in the holdout 0 of 4143 groups lack "tracedata" (3480 hold it as a scalar,
                # 663 as a matrix), so no number moves — but the code should be correct regardless.
                if key not in h or "tracedata" not in h[key]:
                    continue
                if h[key]["tracedata"].shape == ():
                    continue
                tr = np.asarray(h[key]["tracedata"])
                w = np.asarray(h[key]["weights"])
                if tr.ndim != 2 or tr.shape[1] < 7:
                    continue
                n_ok += 1
                for gi, t in enumerate(A.TARGETS):
                    col = tr[:, 2 + gi]
                    hist, edges = np.histogram(col, bins=40, weights=w, density=True)
                    # count interior local maxima above 20% of the global max -> crude multimodality flag
                    peaks = [i for i in range(1, len(hist) - 1)
                             if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.2 * hist.max()]
                    if len(peaks) >= 2:
                        counts[t] += 1
        multimodal = {"planets_probed": n_ok,
                      "n_available": len(avail),
                      "n_requested": int(args.trace_planets),
                      "sampling": (f"uniform without replacement from all {len(avail)} holdout planets that "
                                   f"carry a >=7-column trace, numpy default_rng(seed={args.trace_seed}); "
                                   f"NOT the first {args.trace_planets} split positions, which are the "
                                   f"lowest-numbered planets and 52% trace-bearing vs 16.0% globally"),
                      "seed": int(args.trace_seed),
                      "fraction_with_>=2_modes": {t: (counts[t] / n_ok if n_ok else None) for t in A.TARGETS},
                      "note": "crude weighted-histogram peak count; enough to show the labels are obtainable "
                              "from assets already in the repo (contradicts plan_problems_log P8)"}

    payload = {
        "split": args.split, "per_gas": per_gas,
        "n_planets_compared": int(len(ids_ref)),
        "n_planets_in_split": int(len(ids)),
        "n_rows_in_predictions_csv": int(len(pred)),
        "comparison_set": ("the 663 planets with a non-NaN reference quartile on all 5 gases; "
                           "reference AND model are both scored on exactly these rows"),
        "reference_posterior_mrmse": float(np.mean(ref_rmse)),
        "reference_posterior_sigma_mean_over_gases": float(np.mean(ref_sigma)),
        "model_mrmse": model_m, "constant_predictor_mrmse": base_m,
        "reference_over_model_ratio": float(np.mean(ref_rmse) / model_m),
        "model_mrmse_all_rows_not_in_verdict": A.mrmse(yt_all, yp_all),
        "ratio_on_mismatched_sets_not_in_verdict": float(np.mean(ref_rmse) / A.mrmse(yt_all, yp_all)),
        "ratio_key_note": ("`reference_over_model_ratio` = reference / model, >1 means the model reports a "
                           "SMALLER error than the reference run. The previous key was named "
                           "`model_over_reference_ratio` while holding exactly this quantity — inverted "
                           "relative to its contents, and the report quoted the key name"),
        # Reference calibration: measured, reported, EXPLICITLY outside the criterion. It justifies
        # weakening the claim from "exact Bayesian solution" to "one concrete MultiNest run".
        "reference_calibration_not_in_criterion": {
            "mean_bias_dex": float(np.mean(ref_bias)),
            "mean_iqr_coverage_of_truth": float(np.mean(ref_cov)),
            "nominal_iqr_coverage": NOMINAL_IQR_COVERAGE,
            "over_coverage_factor": float(np.mean(ref_cov) / NOMINAL_IQR_COVERAGE),
            "why_not_in_criterion": ("A run this biased and this over-wide is not 'the exact posterior', so "
                                     "'no estimator can beat the posterior' cannot carry the argument. The "
                                     "criterion is therefore a comparison against a NAMED artefact (the "
                                     "published MultiNest run), which is what K8 actually needs. Folding "
                                     "calibration into the same status would make a FAIL unattributable "
                                     "between 'model too good' and 'reference miscalibrated'"),
        },
        "multimodality_probe": multimodal,
        "interpretation": (
            "The regressor reports an accuracy several times finer than the reference MultiNest retrieval on "
            "the same spectra and the same planets. That is only possible if the input is (near-)noiseless, "
            "i.e. the task is simulator inversion. Under the observational noise these spectra were generated "
            "for, the achievable accuracy is the reference sigma (0.3-3 dex), not 0.3 dex."),
    }
    print(f"  planets compared (reference AND model)            = {payload['n_planets_compared']} "
          f"of {payload['n_planets_in_split']} in the split")
    print(f"  reference nested-sampling mRMSE (median vs truth) = {payload['reference_posterior_mrmse']:.4f} dex")
    print(f"  reference posterior sigma, mean over gases        = {payload['reference_posterior_sigma_mean_over_gases']:.4f} dex")
    print(f"  model mRMSE on those same planets                = {model_m:.4f} dex")
    print(f"  model mRMSE on all {len(pred)} CSV rows (not in verdict) = "
          f"{payload['model_mrmse_all_rows_not_in_verdict']:.4f} dex")
    print(f"  reference/model = {payload['reference_over_model_ratio']:.3f}x on the common set "
          f"(was {payload['ratio_on_mismatched_sets_not_in_verdict']:.3f}x on mismatched sets)")
    rc = payload["reference_calibration_not_in_criterion"]
    print(f"  reference calibration (NOT in criterion): bias={rc['mean_bias_dex']:+.4f} dex, "
          f"IQR coverage={rc['mean_iqr_coverage_of_truth']:.4f} vs nominal {NOMINAL_IQR_COVERAGE:.2f} "
          f"-> over-covers {rc['over_coverage_factor']:.3f}x")
    for t, d in per_gas.items():
        print(f"    {t:8} n={d['n_scored_on_common_mask']:5} ref RMSE={d['reference_rmse_median_vs_truth']:.3f} "
              f"ref sigma(med)={d['reference_sigma_median']:.3f} IQR cov={d['iqr_coverage_of_truth']:.3f}")
    if multimodal:
        print(f"  multimodality probe: {multimodal['planets_probed']} planets drawn at random from "
              f"{multimodal['n_available']} available (seed {multimodal['seed']})")
        for t, v in multimodal["fraction_with_>=2_modes"].items():
            print(f"    {t:8} fraction with >=2 modes = {v:.4f}" if v is not None else f"    {t:8} n/a")
    CHECK.emit("FAIL" if payload["reference_over_model_ratio"] > 1.0 else "PASS", payload,
               inputs=[gt / "QuartilesTable.csv"], out=args.out)


if __name__ == "__main__":
    main()
