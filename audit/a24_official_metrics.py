"""A24 — Score the models on the metrics the benchmarks actually use.

The repo compares models on mRMSE. No Ariel Data Challenge used mRMSE:
  * ADC2022 "light track": relative error on the 16/50/84 percentiles of each marginal.
  * ADC2022 "regular track": Wasserstein-2 distance on the joint conditional distribution.
  * ADC2023: 0.8 x (two-sample KS on the 7 marginals vs MultiNest reference) + 0.2 x spectral score.

The reference posteriors those metrics compare against ship with the dataset
(`Ground Truth Package/Tracedata.hdf5`, weighted nested-sampling samples, 7 columns), so the
distributional part of every one of these metrics is computable here, today, without training
anything. This check computes them for:
  * ExoBiome  — a point estimate, which enters as a Dirac delta;
  * the winner-style NSF — a real posterior;
  * the PRIOR — the training-set marginal for each gas, identical for every planet, ignoring the
    spectrum entirely. This is the distributional analogue of the constant predictor in a02: the
    score of a model that has learned nothing. Without it a distributional number is as
    uninterpretable as an mRMSE without a trivial baseline.
  * reference-vs-itself (trace split in half) — the finite-sample floor of each metric,
    i.e. what a PERFECT model would score.

The spectral component of the ADC2023 score is NOT computed: it needs a forward model.
Temperature and radius are NOT scored: neither model predicts them (see a15).

PASS criterion: the point model has positive skill against the PRIOR arm on the ADC2023 KS
component, i.e. it beats a model that ignores the spectrum on the challenge's own metric.

The previous criterion ("within a factor 2 of the finite-sample floor") was structurally dead and
the docstring said so itself: a delta has KS >= 0.5 against any continuous reference BY
CONSTRUCTION, while the floor is ~0.044, so 2x floor can never reach 0.5. Measured over all 3315
gas-planet pairs: min KS = 0.5001, max 2x floor = 0.1679 (0.2309 with the single-split floor the
retired criterion actually ran on), pairs satisfying the criterion = 0 either way. That made the
check a theorem about the metric wearing the costume of a measurement of the model. The floor
comparison is kept, computed and printed, but as an explicitly labelled statement about the metric
that does not enter the verdict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a24_official_metrics",
    finding="K10 — scored on the metrics the challenge actually used, a point estimate is at or near the worst attainable value",
    question="What do ExoBiome and the NSF baseline score on the ADC2022/ADC2023 distributional metrics?",
    criterion="point model has positive KS skill against the PRIOR arm (beats a model that ignores the spectrum)",
)

GAS_TRACE_COL = {g: 2 + i for i, g in enumerate(A.TARGETS)}  # verified column order, see report K9

# Number of independent 50/50 splits of each planet's trace that the floor is averaged over.
# Previously one split per planet, so the floor carried the full variance of a single draw.
# 5 splits cut the spread ~2.2x (1/sqrt(5)) at 5x the cost. The splits are CORRELATED (same
# column), so they are not an independent sample and no standard error is derived from them.
FLOOR_SPLITS = 5

# Seeds torch's global RNG before the NSF posterior is sampled.
#
# `IndependentNSF.sample` -> `flow(context).sample(...)` (models/adc_winner_on_ariel/model.py:54)
# takes no generator, so it draws from torch's global RNG. The "rng_seed": 0 field in the floor
# payload covers the numpy splits ONLY, not the posterior. Without this line the NSF arm was
# irreproducible between runs (KS nsf moved in the fourth decimal). Seeded once, before the batch
# loop, so with a fixed `rows` order the whole arm's draw stream is deterministic. The number is
# still one Monte Carlo draw of `--nsf-samples` samples — the seed makes it reproducible, not
# more accurate.
POSTERIOR_SAMPLE_SEED = 42


def wcdf(x: np.ndarray, w: np.ndarray, grid: np.ndarray) -> np.ndarray:
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = np.cumsum(ws) / ws.sum()
    return np.interp(grid, xs, c, left=0.0, right=1.0)


def wquantile(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(q, c, xs)


def ks_vs_reference(ref_x, ref_w, mod_x, mod_w=None) -> float:
    """Two-sample KS between a weighted reference sample and a model sample (delta if len==1)."""
    grid = np.unique(np.concatenate([ref_x, mod_x]))
    grid = np.concatenate([grid - 1e-9, grid + 1e-9])
    grid.sort()
    f_ref = wcdf(ref_x, ref_w, grid)
    if mod_w is None:
        mod_w = np.ones(len(mod_x))
    f_mod = wcdf(mod_x, mod_w, grid) if len(mod_x) > 1 else (grid >= mod_x[0]).astype(float)
    return float(np.max(np.abs(f_ref - f_mod)))


def wasserstein1(ref_x, ref_w, mod_x, mod_w=None) -> float:
    q = np.linspace(0.001, 0.999, 999)
    a = wquantile(ref_x, ref_w, q)
    b = wquantile(mod_x, mod_w if mod_w is not None else np.ones(len(mod_x)), q) if len(mod_x) > 1 \
        else np.full_like(q, mod_x[0])
    return float(np.mean(np.abs(a - b)))


def light_track(ref_x, ref_w, mod_x, mod_w=None) -> float:
    """ADC2022 light-track style: mean relative error on the 16/50/84 percentiles."""
    q = np.array([0.16, 0.50, 0.84])
    a = wquantile(ref_x, ref_w, q)
    b = wquantile(mod_x, mod_w if mod_w is not None else np.ones(len(mod_x)), q) if len(mod_x) > 1 \
        else np.full_like(q, mod_x[0])
    return float(np.mean(np.abs(a - b) / np.maximum(np.abs(a), 1.0)))


def main() -> None:
    import h5py
    import pandas as pd
    ap = argparse.ArgumentParser()
    # None = every planet with a reference posterior (663 in the ADC holdout). Previously 400, so
    # the cap dropped 263 planets (39.7%) non-randomly — adc_split_ids returns an ascending list,
    # leaving a low-numbered prefix — and the cap was NOT recorded in the payload. Effect on the
    # cited numbers: W1 1.0733 -> 1.2293 (+14.5%), light_track 0.1291 -> 0.1620 (+25.5%),
    # skill vs prior on light_track +0.6185 -> +0.5384.
    ap.add_argument("--max-planets", type=int, default=None,
                    help="None = wszystkie dostepne (663); wartosc >0 ogranicza probke do prefiksu")
    ap.add_argument("--nsf-samples", type=int, default=128)
    ap.add_argument("--skip-nsf", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = A.adc_root() / "TrainingData/Ground Truth Package"
    exo = pd.read_csv(A.EXOBIOME_ARTIFACT / "holdout_predictions.csv").set_index("planet_ID")

    # ---- reference traces
    refs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(gt / "Tracedata.hdf5", "r") as h:
        for pid in A.adc_split_ids("holdout"):
            k = f"Planet_{pid}"
            if k not in h or "tracedata" not in h[k]:
                continue
            tr = np.asarray(h[k]["tracedata"])
            w = np.asarray(h[k]["weights"], dtype=np.float64)
            if tr.ndim != 2 or tr.shape[1] != 7 or len(w) != len(tr) or w.sum() <= 0:
                continue
            refs[pid] = (tr, w / w.sum())
    # Enumeration is always FULL, the cap applied only afterwards — otherwise `n_available` could
    # not be reported, and it is the only number telling a reader how much the sample skips.
    n_available = len(refs)
    ids = list(refs)
    if args.max_planets:
        ids = ids[: args.max_planets]
    print(f"planets with reference posteriors: {n_available} available, {len(ids)} scored"
          f"{'' if not args.max_planets else f' (cap --max-planets={args.max_planets})'}")

    # ---- NSF posterior samples for the same planets
    nsf: dict[str, np.ndarray] = {}
    if not args.skip_nsf:
        import torch
        import yaml
        sys.path.insert(0, str(A.REPO))
        from models.adc_winner_on_ariel.dataset import load_prepared_data, move_prepared_data_to_device
        from models.adc_winner_on_ariel.dataset import build_context_batch
        from models.adc_winner_on_ariel.model import IndependentNSF, ModelConfig
        from models.adc_winner_on_ariel.dataset import inverse_targets_batch
        run = A.REPO / "models/adc_winner_on_ariel/trained_run"
        prepared = A.REPO / "data/generated-data/ariel_winner_nf_prepared"
        if prepared.exists():
            settings = yaml.safe_load((run / "settings_resolved.yaml").read_text())
            dev = torch.device("cpu")
            data = move_prepared_data_to_device(load_prepared_data(prepared), dev)
            model = IndependentNSF(ModelConfig(**settings["model"])).to(dev)
            model.load_state_dict(torch.load(run / "best_model_by_mrmse.pt", map_location=dev)["model"])
            model.eval()
            split = data.holdout
            # the prepared split stores planet_id as a bare integer (18); the ground-truth tables
            # use the challenge id ("train18") — normalise to the challenge form
            raw = np.asarray(split.planet_id) if hasattr(split, "planet_id") else None
            split_ids = None if raw is None else [
                (str(s) if str(s).startswith("train") else f"train{int(s)}") for s in raw.tolist()]
            if split_ids is None:
                print("  (prepared split carries no ids; skipping NSF)")
            else:
                pos = {p: i for i, p in enumerate(split_ids)}
                rows = [pos[p] for p in ids if p in pos]
                got = [p for p in ids if p in pos]
                print(f"  NSF: matched {len(rows)}/{len(ids)} planets; sampling {args.nsf_samples} draws each")
                torch.manual_seed(POSTERIOR_SAMPLE_SEED)
                with torch.inference_mode():
                    for s in range(0, len(rows), 256):
                        idx = torch.as_tensor(rows[s:s + 256], dtype=torch.long)
                        ctx = build_context_batch(split, idx, data.scalers, device=dev,
                                                  sample_noise=False, noise_generator=None)
                        smp = model.sample(ctx, num_samples=args.nsf_samples).cpu().numpy()
                        for j, p in enumerate(got[s:s + 256]):
                            nsf[p] = inverse_targets_batch(smp[j], data.scalers)
        else:
            print(f"  (prepared data missing at {prepared}; skipping NSF)")

    # ---- score
    # trivial distributional baseline: the training-set marginal, identical for every planet
    fm = A.load_adc_targets()
    train_ids = A.adc_split_ids("train")
    rng0 = np.random.default_rng(1)
    prior = {}
    n_train_pool = 0
    for g in A.TARGETS:
        col = fm.loc[train_ids, g].to_numpy(float)
        n_train_pool = len(col)
        prior[g] = rng0.choice(col, size=min(2000, len(col)), replace=False)
    n_prior_draws = len(next(iter(prior.values())))
    # The 2000-of-33138 training-planet cap used to go to stdout only, not to the payload, so in
    # the record the PRIOR arm was a number with no stated sample size behind it.
    print(f"  prior arm: training marginal per gas, {n_prior_draws} draws of {n_train_pool} available, "
          "identical for every planet (ignores the spectrum)")

    res = {m: {g: {"exobiome": [], "nsf": [], "prior": [], "floor": []} for g in A.TARGETS}
           for m in ("ks", "w1_dex", "light_track")}
    rng = np.random.default_rng(0)
    for pid in ids:
        tr, w = refs[pid]
        for g in A.TARGETS:
            col = tr[:, GAS_TRACE_COL[g]]
            point = np.array([float(exo.loc[pid, f"pred_{g}"])])
            # Finite-sample floor, averaged over FLOOR_SPLITS 50/50 splits.
            # Previously ONE split per planet, so every floor value carried the full variance of a
            # single draw — and the floor is reported per gas and compared against the model arms,
            # so its noise fed straight through to the conclusion.
            #
            # What this does NOT and cannot fix: both halves hold ~len(col)/2 samples (median
            # holdout trace is 3940, so ~1970), while the model arms are scored against the FULL
            # reference (~3940). Two-sample KS error scales as sqrt(1/n1 + 1/n2), so for that reason
            # the floor stays INFLATED — a conservative direction for the old criterion (an inflated
            # floor made PASS EASIER, so it did not overstate the finding). Recorded in the payload
            # as `floor_definition`.
            fk, fw, fl = [], [], []
            for _ in range(FLOOR_SPLITS):
                half = rng.random(len(col)) < 0.5
                if half.sum() < 10 or (~half).sum() < 10:
                    continue
                fk.append(ks_vs_reference(col[half], w[half], col[~half], w[~half]))
                fw.append(wasserstein1(col[half], w[half], col[~half], w[~half]))
                fl.append(light_track(col[half], w[half], col[~half], w[~half]))
            if fk:
                floor_ks, floor_w1, floor_lt = float(np.mean(fk)), float(np.mean(fw)), float(np.mean(fl))
            else:
                floor_ks = floor_w1 = floor_lt = np.nan
            res["ks"][g]["exobiome"].append(ks_vs_reference(col, w, point))
            res["w1_dex"][g]["exobiome"].append(wasserstein1(col, w, point))
            res["light_track"][g]["exobiome"].append(light_track(col, w, point))
            res["ks"][g]["prior"].append(ks_vs_reference(col, w, prior[g]))
            res["w1_dex"][g]["prior"].append(wasserstein1(col, w, prior[g]))
            res["light_track"][g]["prior"].append(light_track(col, w, prior[g]))
            res["ks"][g]["floor"].append(floor_ks)
            res["w1_dex"][g]["floor"].append(floor_w1)
            res["light_track"][g]["floor"].append(floor_lt)
            if pid in nsf:
                s = nsf[pid][:, A.TARGETS.index(g)]
                res["ks"][g]["nsf"].append(ks_vs_reference(col, w, s))
                res["w1_dex"][g]["nsf"].append(wasserstein1(col, w, s))
                res["light_track"][g]["nsf"].append(light_track(col, w, s))

    payload = {"n_planets": len(ids), "n_available": int(n_available),
               "max_planets": (int(args.max_planets) if args.max_planets else None),
               "n_planets_dropped_by_cap": int(n_available - len(ids)),
               "n_nsf": len(nsf), "nsf_samples": args.nsf_samples,
               "posterior_sample_seed": POSTERIOR_SAMPLE_SEED,
               "posterior_sample_seed_note":
                   "global torch RNG seeded before the NSF sampling loop; the flow's sample() takes "
                   "no generator (model.py:54), so the floor's rng_seed does NOT cover it. Makes the "
                   "NSF arm reproducible, not more accurate: still one draw of nsf_samples samples.",
               "prior_draws": int(n_prior_draws), "prior_pool_size": int(n_train_pool),
               "floor_definition": {
                   "method": f"mean over {FLOOR_SPLITS} independent random 50/50 splits of each "
                             f"planet's reference trace, one KS/W1/light_track per split",
                   "splits_per_planet": FLOOR_SPLITS,
                   "was": "one split per planet, so each floor value carried the full variance of a single draw",
                   "residual_bias": "each half holds ~len(trace)/2 (~1442) samples against the ~2884 the "
                                    "model arms are scored on, so the floor remains INFLATED; that direction "
                                    "made the old PASS branch easier, not harder",
                   "rng_seed": 0,
               },
               "scored_parameters": A.TARGETS,
               "not_scored": ["planet_radius", "planet_temp", "ADC2023 spectral component"],
               "metrics": {}}
    for m in res:
        payload["metrics"][m] = {"per_gas": {}, "mean": {}}
        for arm in ("exobiome", "nsf", "prior", "floor"):
            vals = []
            for g in A.TARGETS:
                v = np.array(res[m][g][arm], dtype=float)
                v = v[np.isfinite(v)]
                if len(v):
                    payload["metrics"][m]["per_gas"].setdefault(g, {})[arm] = float(np.mean(v))
                    vals.append(np.mean(v))
            if vals:
                payload["metrics"][m]["mean"][arm] = float(np.mean(vals))

    names = {"ks": "ADC2023 posterior score component — 2-sample KS (lower better, 0..1)",
             "w1_dex": "ADC2022 regular-track spirit — Wasserstein-1 [dex] (lower better)",
             "light_track": "ADC2022 light-track spirit — rel. err. on q16/q50/q84 (lower better)"}
    for m in ("ks", "w1_dex", "light_track"):
        d = payload["metrics"][m]
        print(f"\n  {names[m]}")
        print(f"    {'gas':10} {'ExoBiome (point)':>18} {'NSF (posterior)':>18} "
              f"{'PRIOR (no info)':>17} {'floor (perfect)':>17}")
        for g in A.TARGETS:
            r = d["per_gas"].get(g, {})
            print(f"    {g:10} {r.get('exobiome', float('nan')):18.4f} "
                  f"{r.get('nsf', float('nan')):18.4f} {r.get('prior', float('nan')):17.4f} "
                  f"{r.get('floor', float('nan')):17.4f}")
        print(f"    {'MEAN':10} {d['mean'].get('exobiome', float('nan')):18.4f} "
              f"{d['mean'].get('nsf', float('nan')):18.4f} {d['mean'].get('prior', float('nan')):17.4f} "
              f"{d['mean'].get('floor', float('nan')):17.4f}")

    ks = payload["metrics"]["ks"]["mean"]
    payload["skill_vs_prior"] = {
        m: {arm: (1.0 - payload["metrics"][m]["mean"][arm] / payload["metrics"][m]["mean"]["prior"])
            for arm in ("exobiome", "nsf") if arm in payload["metrics"][m]["mean"]}
        for m in payload["metrics"] if "prior" in payload["metrics"][m]["mean"]}
    payload["interpretation"] = (
        f"A point estimate enters a KS test as a Dirac delta, so its KS statistic against any "
        f"continuous reference is >= 0.5 by construction; measured mean {ks.get('exobiome', float('nan')):.3f} "
        f"vs finite-sample floor {ks.get('floor', float('nan')):.3f}. mRMSE ranks the point model first; "
        f"every distributional metric the challenge actually used ranks it at or near the worst "
        f"attainable value. This is not a tuning gap — it is a category difference in the output object."
    )
    # The old PASS branch was STRUCTURALLY DEAD — measured here pair by pair so the claim does not
    # stay a mere claim. `is not None` instead of a truthiness test on a float: `ks.get("floor")
    # and ...` was falsy for floor == 0.0, i.e. FAIL because "the floor is zero" — yet a zero floor
    # means a perfect metric and should give the strictest threshold, not a failure. Unreachable
    # today (the lowest per-gas floor is ~0.044), but telling "no value" from "the value zero" must
    # not depend on the data. Pairs are kept PAIRED and the non-finite filter applied to both at
    # once; filtering each list separately would shift the indices and match one planet's KS to
    # another planet's floor.
    pair_ks, pair_fl = [], []
    for g in A.TARGETS:
        for a, b in zip(res["ks"][g]["exobiome"], res["ks"][g]["floor"]):
            if np.isfinite(a) and np.isfinite(b):
                pair_ks.append(a)
                pair_fl.append(b)
    n_pairs_ok = sum(1 for a, b in zip(pair_ks, pair_fl) if a <= 2.0 * b)
    old_ok = (ks.get("exobiome") is not None and ks.get("floor") is not None and
              ks["exobiome"] <= 2.0 * ks["floor"])
    payload["floor_comparison_is_a_statement_about_the_metric"] = {
        "retired_criterion": "point model's distributional score within 2x of the finite-sample floor",
        "would_pass": bool(old_ok),
        "n_gas_planet_pairs": len(pair_ks),
        "min_ks_exobiome_over_pairs": float(np.min(pair_ks)) if pair_ks else None,
        "max_two_times_floor_over_pairs": float(2.0 * np.max(pair_fl)) if pair_fl else None,
        "n_pairs_satisfying_the_retired_criterion": int(n_pairs_ok),
        "why_retired": ("a Dirac delta has KS >= 0.5 against any continuous reference BY CONSTRUCTION, "
                        "while the floor is ~0.04, so 2x floor cannot reach 0.5 for any data. The branch "
                        "could not fire, which makes the check a theorem about the KS statistic rather "
                        "than a measurement of the model. Retained as a reported quantity, removed from "
                        "the verdict."),
    }
    # New criterion: skill against the PRIOR arm on the ADC2023 KS component.
    # Falsifiable in BOTH directions on the same data in the same run: the NSF arm reaches positive
    # skill (PASS), the point arm negative (FAIL). This measures the model, not the metric: it asks
    # whether a point prediction beats, distributionally, a model that never sees the spectrum.
    skill_ks = payload["skill_vs_prior"].get("ks", {}).get("exobiome")
    payload["status_terms"] = {
        "criterion": "skill_vs_prior['ks']['exobiome'] > 0",
        "ks_skill_vs_prior_exobiome": skill_ks,
        "ks_skill_vs_prior_nsf": payload["skill_vs_prior"].get("ks", {}).get("nsf"),
        "falsifiable_both_ways": ("yes — the NSF arm is scored on the identical references in this same run "
                                  "and lands on the other side of zero"),
        "evaluable": skill_ks is not None,
    }
    if skill_ks is None:
        status = "INFO"
        payload["criterion_not_evaluable"] = "no PRIOR arm in the KS mean; skill against prior undefined"
    else:
        status = "PASS" if skill_ks > 0.0 else "FAIL"
    print(f"\n  criterion: KS skill vs PRIOR, exobiome = "
          f"{skill_ks if skill_ks is None else round(skill_ks, 4)} "
          f"(NSF = {payload['status_terms']['ks_skill_vs_prior_nsf']}) -> {status}")
    fc = payload["floor_comparison_is_a_statement_about_the_metric"]
    print(f"  retired floor criterion (reported, not in verdict): min KS = {fc['min_ks_exobiome_over_pairs']:.4f}, "
          f"max 2x floor = {fc['max_two_times_floor_over_pairs']:.4f}, "
          f"pairs satisfying it = {fc['n_pairs_satisfying_the_retired_criterion']}/{fc['n_gas_planet_pairs']}")
    CHECK.emit(status, payload, out=args.out)


if __name__ == "__main__":
    main()
