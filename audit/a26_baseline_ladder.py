"""A26 — The baseline ladder: where does the reported skill actually come from?

Implements task A0.4. `a02` establishes rung 0 (a constant predictor) and shows that every
cross-generator model falls below it. That answers "is there any skill". It does not answer the
next question: **which input carries the skill.** A model can look strong on mRMSE while taking
most of its information from the auxiliary table rather than from the spectrum — and on ADC2023 we
already know the aux table determines the atmospheric temperature to 48.5 K (see a15), so this is
not a hypothetical concern.

Rungs, each ignoring strictly less information than the next:

  0  constant          the training-set mean vector, same for every planet        -> defines zero skill
  1  aux only          learned model on the auxiliary columns, NO spectrum        -> non-spectral skill
  2  spectrum only     learned model on the spectrum (+ its noise), aux TABLE
                       withheld — but NOT free of aux information, see below      -> spectral skill
  2b spectrum without  as rung 2 with the log10(mean transit depth) column
     the scale column  dropped; quantifies the instructed removal                 -> control for rung 2
  3  aux + spectrum    both                                                       -> ceiling for this model class
  -  reported models   from the repo's own tables (a02)                           -> for comparison

Two learners per rung so the result is not an artifact of one estimator: ridge (mirrors the team's
own `data/crossgen_biosignatures/baseline_smoke.py`) and gradient boosting.

The headline number is `skill_share_of_aux = skill(rung 1) / skill(rung 3)` — the fraction of the
achievable skill that needs no spectrum at all. Numerator and denominator are taken from the SAME
learner (see `headline_learner`); the earlier version mixed them, which changed the headline by 3x.

IMPORTANT CAVEAT on rung 2 (added 2026-07-28, after measuring): "aux withheld" means the aux TABLE
is not supplied, not that the spectral block is free of aux information. The measured ridge R^2 for
recovering the aux columns from the spectral block alone is in the payload
(`aux_recoverable_from_spectrum_block`), and it is high: `log_g_cgs` 0.67 from the spectral SHAPE
alone on tau_val, `star_temperature` 0.75 from the noise vector alone on ADC. So
`skill_share_of_spectrum_only` must NOT be read as "this much of the skill is purely spectroscopic".

PASS criterion: for every dataset the aux-only rung carries less than 20 % of the aux+spectrum skill
(compared against the UPPER end of the bootstrap CI, not the point estimate), and every dataset has
a positive aux+spectrum skill in the first place — a dataset with nothing to divide cannot support
the claim that the models are doing spectroscopy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a26_baseline_ladder",
    finding="K2 (extension) — how much of the reported skill survives when the spectrum is removed",
    question="Constant vs aux-only vs spectrum-only vs both: where does the skill come from?",
    criterion="upper end of the CI on the aux-only skill share < 20 % of the aux+spectrum skill, "
              "on every dataset, and every dataset has a positive aux+spectrum skill to divide",
)

# THRESHOLD DERIVATION — this used to be `AUX_SHARE_LIMIT = 0.20` with no justification at all.
#
# The only known, documented route from the aux table to the scored targets (the gases): aux fixes
# the equilibrium temperature, which is coupled to the abundances. Both links are measured in `a15`
# on ADC2023: `aux_leakage_of_temperature.rmse_Teq_K = 48.50 K` (aux pins T to that accuracy) and
# `conditioning_value.relative_improvement = 0.00259` (the true T and Rp improve the gas mRMSE by
# 0.26 %, so the route is worth ~0.3 % of the achievable skill). The 0.20 limit leaves ~77x margin
# over that route: a share above it CANNOT be explained by the aux -> T -> abundance coupling and
# means an undocumented shortcut. A lower limit (say 0.05) would catch the same physical effect and
# FAIL correct physics; a higher one (0.50) would pass a model taking half its skill from the star
# catalogue.
#
# TODAY: the measured share is about -0.002 (ADC) and -0.004 (tau_val) — 50-100x below the limit and
# with the wrong sign (rung 1 is WORSE than the constant). So the limit is not binding today and its
# resolving power near the boundary is untested on these data (`threshold.binding_today = false`).
AUX_SHARE_LIMIT = 0.20
AUX_SHARE_LIMIT_BASIS = {
    "limit": AUX_SHARE_LIMIT,
    "derived_from": "a15 ADC2023: aux pins Teq to 48.50 K, and conditioning on the TRUE T and Rp "
                    "improves the scored gas mRMSE by 0.259 % (conditioning_value."
                    "relative_improvement = 0.0025897)",
    "documented_aux_to_gas_route_share": 0.0025897,
    "margin_over_documented_route": AUX_SHARE_LIMIT / 0.0025897,
    "reading": "a share above the limit cannot be explained by the aux -> Teq -> abundance coupling "
               "and would mean an undocumented shortcut",
    "compared_against": "upper end of the 95 % bootstrap CI over evaluation rows, not the point "
                        "estimate — a point estimate cannot be below a limit 'significantly'",
}

# Bootstrap resamples for the CI on the share. 1000 is the standard minimum for the 2.5/97.5
# percentiles (Monte Carlo error at the tails ~1 pp at n_eval of order 4k); the cost is negligible
# because we resample FINISHED predictions instead of refitting the model.
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 26

# Fraction of resamples above which the denominator counts as numerically indistinguishable from
# zero and the share as UNDEFINED. Without this guard, a CI on a ratio whose denominator sits at
# zero is arithmetically correct and substantively useless.
MAX_NONPOSITIVE_DENOMINATOR_FRACTION = 0.05


def ridge_fit_predict(xa, ya, xb, alpha=1.0):
    """Same recipe as data/crossgen_biosignatures/baseline_smoke.py: standardise, ridge, predict."""
    mean, scale = xa.mean(0), np.where(xa.std(0) == 0, 1.0, xa.std(0))
    xa_s, xb_s = (xa - mean) / scale, (xb - mean) / scale
    xa_aug = np.hstack([np.ones((len(xa_s), 1)), xa_s])
    xb_aug = np.hstack([np.ones((len(xb_s), 1)), xb_s])
    reg = alpha * np.eye(xa_aug.shape[1])
    reg[0, 0] = 0.0
    w = np.linalg.pinv(xa_aug.T @ xa_aug + reg) @ (xa_aug.T @ ya)
    return xb_aug @ w


def gbm_fit_predict(xa, ya, xb, max_iter=200):
    from sklearn.ensemble import HistGradientBoostingRegressor
    out = np.empty((len(xb), ya.shape[1]))
    for j in range(ya.shape[1]):
        out[:, j] = HistGradientBoostingRegressor(max_iter=max_iter, random_state=0).fit(xa, ya[:, j]).predict(xb)
    return out


LEARNERS = ("ridge", "gbm")


def aux_recoverable(x_train, x_eval, aux_idx, spectral_groups, aux_names) -> dict:
    """Ridge R^2 for recovering each aux column from the spectral block and from each of its parts.

    Added 2026-07-28. Rung `2_spectrum_only` was described as "aux withheld", but `spec` contains
    `np.log10(ref)`, from which `star_radius_rsun` is recoverable on tau_val (r = -0.8935,
    R^2 = 0.798). Measuring ALL parts of the spectral block, not just that column, shows it is not
    the only leak: on tau_val the shape S/ref alone recovers log_g_cgs at R^2 = 0.666; on ADC the
    noise vector N/ref alone recovers star_temperature at R^2 = 0.749 and star_distance at 0.693.

    Decision this forces: neither dropping `log10(ref)` from the spectral block (log_g stays at
    0.67) nor moving it to the aux block (which would mislabel a spectral measurement — mean
    transit depth — as "aux table", inflating the headline numerator) makes rung 2 aux-free.
    `log10(ref)` therefore stays in the spectral block; the rung's description now says "aux TABLE
    withheld", the leak is measured and reported here, and the numeric effect of dropping the
    column alone is computed separately as rung `2b`.
    """
    out = {"note": "ridge R^2 on the evaluation split; > 0 means the auxiliary column is partly "
                   "reconstructible from spectral columns alone, so no rung here is aux-free",
           "by_source": {}}
    y_tr, y_ev = x_train[:, aux_idx], x_eval[:, aux_idx]
    denom = ((y_ev - y_ev.mean(0)) ** 2).sum(0)
    for src, idx in spectral_groups.items():
        if len(idx) == 0:
            continue
        pred = ridge_fit_predict(x_train[:, idx], y_tr, x_eval[:, idx])
        r2 = np.where(denom > 0, 1.0 - ((y_ev - pred) ** 2).sum(0) / np.where(denom > 0, denom, 1.0),
                      np.nan)
        out["by_source"][src] = {n: float(v) for n, v in zip(aux_names, r2)}
    both = {n: max((g[n] for g in out["by_source"].values()), default=float("nan"))
            for n in aux_names}
    out["max_r2_over_sources"] = both
    out["worst_leak"] = max(both, key=lambda k: (both[k] if np.isfinite(both[k]) else -np.inf))
    out["worst_leak_r2"] = both[out["worst_leak"]]
    return out


def bootstrap_share(y_eval, const, pred_num, pred_den, n_boot, seed) -> dict:
    """Confidence interval on the share skill(numerator)/skill(denominator), resampling eval rows.

    Compared against the UPPER end of this interval, not the point estimate: `AUX_SHARE_LIMIT` is
    meant to assert "the share is reliably below 20%", which a point estimate alone cannot do.
    """
    rng = np.random.default_rng(seed)
    n = len(y_eval)
    cb = np.broadcast_to(np.asarray(const, float), y_eval.shape)
    vals, nonpos = [], 0
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        b = A.mrmse(y_eval[i], cb[i])
        if not np.isfinite(b) or b <= 0:
            nonpos += 1
            continue
        s_den = A.skill(A.mrmse(y_eval[i], pred_den[i]), b)
        if s_den <= 0:
            nonpos += 1
            continue
        vals.append(A.skill(A.mrmse(y_eval[i], pred_num[i]), b) / s_den)
    frac = nonpos / n_boot
    res = {"n_resamples": int(n_boot), "n_resamples_with_nonpositive_denominator": int(nonpos),
           "fraction_with_nonpositive_denominator": float(frac),
           "denominator_unusable": bool(frac > MAX_NONPOSITIVE_DENOMINATOR_FRACTION)}
    if vals and not res["denominator_unusable"]:
        v = np.asarray(vals)
        res["ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        res["median"] = float(np.median(v))
    else:
        res["ci95"] = None
        res["median"] = None
    return res


def evaluate_rungs(name, x_train, y_train, x_eval, y_eval, blocks, targets, gbm_iters,
                   aux_names, spectral_groups) -> dict:
    """blocks: {'aux': slice, 'spectrum': slice} into the column axis of x."""
    const = y_train.mean(axis=0)
    base_m, base_gas = A.constant_predictor_mrmse(y_eval, const)
    res = {"dataset": name, "n_train": int(len(x_train)), "n_eval": int(len(y_eval)),
           "rungs": {"0_constant": {"mrmse": base_m, "skill": 0.0,
                                    "per_gas": dict(zip(targets, base_gas.tolist()))}}}
    ncol = x_train.shape[1]
    aux_idx = np.arange(*blocks["aux"].indices(ncol))
    spec_idx = np.arange(*blocks["spectrum"].indices(ncol))
    scale = set(int(i) for i in spectral_groups.get("log10_mean_transit_depth", []))
    spec_no_scale = np.array([c for c in spec_idx if int(c) not in scale], dtype=int)
    rung_idx = {
        "1_aux_only": aux_idx,
        "2_spectrum_only": spec_idx,
        "2b_spectrum_without_scale_column": spec_no_scale,
        "3_aux_plus_spectrum": np.concatenate([aux_idx, spec_idx]),
    }
    preds: dict[tuple[str, str], np.ndarray] = {}
    for rung, idx in rung_idx.items():
        entry: dict = {}
        for learner, fn in (("ridge", ridge_fit_predict),
                            ("gbm", lambda a, b, c: gbm_fit_predict(a, b, c, gbm_iters))):
            pred = fn(x_train[:, idx], y_train, x_eval[:, idx])
            preds[(rung, learner)] = pred
            m = A.mrmse(y_eval, pred)
            entry[learner] = {"mrmse": m, "skill": A.skill(m, base_m),
                              "per_gas": dict(zip(targets, A.per_gas_rmse(y_eval, pred).tolist()))}
        entry["n_columns"] = int(len(idx))
        entry["best_learner"] = min(LEARNERS, key=lambda k: entry[k]["mrmse"])
        entry["mrmse"] = entry[entry["best_learner"]]["mrmse"]
        entry["skill"] = entry[entry["best_learner"]]["skill"]
        res["rungs"][rung] = entry

    # NUMERATOR AND DENOMINATOR FROM THE SAME ESTIMATOR.
    #
    # Previously both `skill` values came from a `best_learner` picked INDEPENDENTLY per rung — on
    # `adc`/`tau_val` the numerator came out of ridge and the denominator out of gbm (and the other
    # way round on `poseidon_test`). The ADC headline was then -0.000290; consistent ridge/ridge
    # gives -0.000860 (3x more) — a ratio of two different estimators is not the "share" of anything.
    #
    # The selection rule is DECLARED, not picked after the fact: take the learner with the lowest
    # rung-3 mRMSE, because that rung defines the "achievable skill" (the denominator). Both learners
    # are reported separately anyway in `skill_share_of_aux_by_learner`.
    head = res["rungs"]["3_aux_plus_spectrum"]["best_learner"]
    res["headline_learner"] = head
    res["headline_learner_rule"] = ("the learner with the lowest rung-3 mRMSE; rung 3 defines the "
                                    "achievable skill, hence the denominator")

    def share(num_rung, den_rung, learner):
        sd = res["rungs"][den_rung][learner]["skill"]
        sn = res["rungs"][num_rung][learner]["skill"]
        if sd <= 0:
            return {"value": None, "denominator_nonpositive": True,
                    "numerator_skill": sn, "denominator_skill": sd,
                    "why": "the aux+spectrum rung has no positive skill over the constant "
                           "predictor, so there is no achievable skill to take a share of"}
        return {"value": float(sn / sd), "denominator_nonpositive": False,
                "numerator_skill": sn, "denominator_skill": sd}

    res["skill_share_of_aux_by_learner"] = {
        k: share("1_aux_only", "3_aux_plus_spectrum", k) for k in LEARNERS}
    res["skill_share_of_spectrum_only_by_learner"] = {
        k: share("2_spectrum_only", "3_aux_plus_spectrum", k) for k in LEARNERS}
    res["skill_share_of_spectrum_without_scale_by_learner"] = {
        k: share("2b_spectrum_without_scale_column", "3_aux_plus_spectrum", k) for k in LEARNERS}

    head_aux = res["skill_share_of_aux_by_learner"][head]
    res["denominator_nonpositive"] = head_aux["denominator_nonpositive"]
    res["skill_share_of_aux"] = head_aux["value"]
    res["skill_share_of_spectrum_only"] = res["skill_share_of_spectrum_only_by_learner"][head]["value"]
    # The name `skill_share_of_spectrum_only` is kept for continuity with the 20260727 record, but
    # now carries a caveat: the reported 1.0014 / 0.9930 (above unity) was itself a signal that
    # rung 2 is not what its name claims.
    res["skill_share_of_spectrum_only_caveat"] = (
        "rung 2 withholds the auxiliary TABLE, not auxiliary INFORMATION; see "
        "aux_recoverable_from_spectrum_block. A value at or above 1.0 means rung 2 matched or beat "
        "rung 3, which is itself evidence that the aux table adds nothing the spectrum lacks.")

    # A DATASET WITH NO SKILL AT ALL CANNOT SUPPORT A PASS.
    #
    # Previously `share = ... if s3 > 0 else None` with `main()` filtering on `is not None`, so
    # `poseidon_test` (rung 1 -0.0000, rung 2 -0.0194, rung 3 -0.0262 — no skill to divide) dropped
    # out of `offenders` and SILENTLY SUPPORTED PASS. The check asserted "the models do
    # spectroscopy" on a dataset where they do nothing at all. Now it is a separate, explicit state.
    if res["denominator_nonpositive"]:
        res["verdict_contribution"] = "no_achievable_skill_to_divide"
        res["skill_share_of_aux_bootstrap"] = None
    else:
        res["skill_share_of_aux_bootstrap"] = bootstrap_share(
            y_eval, const, preds[("1_aux_only", head)], preds[("3_aux_plus_spectrum", head)],
            N_BOOTSTRAP, BOOTSTRAP_SEED)
        bs = res["skill_share_of_aux_bootstrap"]
        if bs["denominator_unusable"]:
            res["verdict_contribution"] = "denominator_unstable_under_resampling"
        elif bs["ci95"][1] >= AUX_SHARE_LIMIT:
            res["verdict_contribution"] = "aux_dominates"
        else:
            res["verdict_contribution"] = "spectroscopy_confirmed"

    res["aux_recoverable_from_spectrum_block"] = aux_recoverable(
        x_train, x_eval, aux_idx, spectral_groups, aux_names)
    return res


def build_adc(n_train: int):
    import h5py
    import pandas as pd
    fm = A.load_adc_targets()
    aux = pd.read_csv(A.adc_root() / "TrainingData/AuxillaryTable.csv")
    aux = aux.drop(columns=[c for c in aux.columns if c.startswith("Unnamed:")]).set_index("planet_ID")
    sys.path.insert(0, str(A.REPO))
    from models.ariel_exobiome.constants import AUX_COLUMNS, LOG10_AUX_COLUMNS

    def block(split, limit=None):
        ids = A.adc_split_ids(split)
        # Was `if limit:` — a truthiness test on an int, so `--n-train 0` truncated NOTHING and
        # silently meant "all", while `build_crossgen` on the same `0` did `[:0]`, i.e. an EMPTY
        # training set. Two builders, two opposite meanings for one flag. `is not None` removes the
        # special case; `main()` rejects 0 outright.
        if limit is not None:
            ids = ids[:limit]
        a = aux.loc[ids]
        af = a[AUX_COLUMNS].to_numpy(float).copy()
        for j, c in enumerate(AUX_COLUMNS):
            if c in LOG10_AUX_COLUMNS:
                af[:, j] = np.log10(np.clip(af[:, j], 1e-12, None))
        with h5py.File(A.adc_root() / "TrainingData/SpectralData.hdf5", "r") as h:
            S = np.stack([h[f"Planet_{p}"]["instrument_spectrum"][:] for p in ids]).astype(float)
            N = np.stack([h[f"Planet_{p}"]["instrument_noise"][:] for p in ids]).astype(float)
        ref = S.mean(axis=1, keepdims=True)
        # `log10(ref)` STAYS in the spectral block: mean transit depth is a spectral measurement, not
        # an aux-table entry. What changed is the wording around it — the rung's name and description
        # now say "aux TABLE withheld", and `spectral_groups` lets us measure how much aux each part
        # of the block recovers (on ADC the worst leak is N/ref, not log10(ref)).
        spec = np.hstack([S / ref, N / ref, np.log10(ref)])
        nb = S.shape[1]
        groups = {"shape_normalised_spectrum": list(range(af.shape[1], af.shape[1] + nb)),
                  "noise_vector": list(range(af.shape[1] + nb, af.shape[1] + 2 * nb)),
                  "log10_mean_transit_depth": [af.shape[1] + 2 * nb]}
        return (np.hstack([af, spec]), fm.loc[ids, A.TARGETS].to_numpy(float),
                af.shape[1], spec.shape[1], groups)

    xtr, ytr, na, ns, groups = block("train", n_train)
    xev, yev, _, _, _ = block("holdout")
    return ("ADC2023 holdout", xtr, ytr, xev, yev,
            {"aux": slice(0, na), "spectrum": slice(na, na + ns)}, A.TARGETS,
            list(AUX_COLUMNS), groups)


def build_crossgen(n_train: int, eval_split: str):
    import h5py
    import pandas as pd
    lab = pd.read_parquet(A.REPO / "data/TauREx set/labels.parquet")
    aux_cols = ["planet_radius_rjup", "log_g_cgs", "star_radius_rsun"]  # the live ones; see a21
    with h5py.File(A.REPO / "data/TauREx set/spectra.h5", "r") as f:
        gen = np.array([g.decode() for g in f["generator"][:]])
        spl = np.array([g.decode() for g in f["split"][:]])
        want = {"tau_val": (gen == "tau") & (spl == "val"),
                "poseidon_test": (gen == "poseidon") & (spl == "test")}[eval_split]
        itr = np.sort(np.where((gen == "tau") & (spl == "train"))[0][:n_train])
        iev = np.sort(np.where(want)[0])

        def block(idx):
            S = f["transit_depth_noisy"][idx].astype(float)
            sg = f["sigma_ppm"][idx].astype(float)[:, None]
            ref = S.mean(axis=1, keepdims=True)
            spec = np.hstack([S / ref, np.log10(ref), np.log10(np.clip(sg, 1.0, None))])
            af = lab.iloc[idx][aux_cols].to_numpy(float)
            y = lab.iloc[idx][A.CROSSGEN_TARGETS].to_numpy(float)
            na_, nb = af.shape[1], S.shape[1]
            groups = {"shape_normalised_spectrum": list(range(na_, na_ + nb)),
                      "log10_mean_transit_depth": [na_ + nb],
                      "log10_sigma_ppm": [na_ + nb + 1]}
            return np.hstack([af, spec]), y, na_, spec.shape[1], groups
        xtr, ytr, na, ns, groups = block(itr)
        xev, yev, _, _, _ = block(iev)
    return (f"crossgen {eval_split}", xtr, ytr, xev, yev,
            {"aux": slice(0, na), "spectrum": slice(na, na + ns)}, A.CROSSGEN_TARGETS,
            list(aux_cols), groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--gbm-iters", type=int, default=200)
    ap.add_argument("--datasets", default="adc,tau_val,poseidon_test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # `--n-train 0` has no valid meaning: zero rows train nothing, and it used to produce two
    # DIFFERENT behaviours in the two builders (see the comment in `build_adc.block`).
    if args.n_train <= 0:
        ap.error("--n-train must be positive; 0 trains on nothing and used to mean two different "
                 "things in build_adc and build_crossgen")

    payload: dict = {"rung_definitions": {
        "0_constant": "training-set mean vector, identical for every planet",
        "1_aux_only": "learned on the auxiliary columns, spectrum withheld",
        "2_spectrum_only": "learned on the spectrum and its noise; the auxiliary TABLE is withheld "
                           "but auxiliary INFORMATION is not — see aux_recoverable_from_spectrum_"
                           "block, where log_g_cgs reaches R^2 0.67 from the spectral shape alone "
                           "on tau_val and star_temperature R^2 0.75 from the noise vector on ADC",
        "2b_spectrum_without_scale_column": "as rung 2 with the log10(mean transit depth) column "
                                            "removed; isolates how much of rung 2 rides on the "
                                            "one column that most directly encodes stellar radius "
                                            "(R^2 0.80 for star_radius_rsun on tau_val)",
        "3_aux_plus_spectrum": "both"},
        "learners": ["ridge (same recipe as data/crossgen_biosignatures/baseline_smoke.py)",
                     "HistGradientBoostingRegressor"],
        "threshold": dict(AUX_SHARE_LIMIT_BASIS),
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
                      "resamples": "evaluation rows, with replacement; predictions are reused, the "
                                   "learners are not refitted",
                      "max_nonpositive_denominator_fraction": MAX_NONPOSITIVE_DENOMINATOR_FRACTION},
        "datasets": {}}

    builders = {"adc": lambda: build_adc(args.n_train),
                "tau_val": lambda: build_crossgen(args.n_train, "tau_val"),
                "poseidon_test": lambda: build_crossgen(args.n_train, "poseidon_test")}
    for key in [d.strip() for d in args.datasets.split(",")]:
        name, xtr, ytr, xev, yev, blocks, targets, aux_names, groups = builders[key]()
        print(f"\n=== {name}   train {xtr.shape}  eval {xev.shape} ===")
        r = evaluate_rungs(name, xtr, ytr, xev, yev, blocks, targets, args.gbm_iters,
                           aux_names, groups)
        payload["datasets"][key] = r
        print(f"  {'rung':34} {'mRMSE (best)':>13} {'skill':>8}   {'ridge':>8} {'gbm':>8}")
        for rung, e in r["rungs"].items():
            extra = (f"{e['ridge']['mrmse']:8.4f} {e['gbm']['mrmse']:8.4f}" if "ridge" in e else "")
            print(f"  {rung:34} {e['mrmse']:13.4f} {e['skill']:8.3f}   {extra}")
        lk = r["headline_learner"]
        if r["skill_share_of_aux"] is None:
            print(f"  -> UNDEFINED: rung 3 skill is {r['rungs']['3_aux_plus_spectrum'][lk]['skill']:+.4f} "
                  f"({lk}); there is no achievable skill to take a share of -> cannot support PASS")
        else:
            ci = r["skill_share_of_aux_bootstrap"]["ci95"]
            print(f"  -> [{lk}] aux alone carries {r['skill_share_of_aux']*100:6.3f} % of the achievable "
                  f"skill (CI95 {ci[0]*100:+.3f} .. {ci[1]*100:+.3f} %); "
                  f"spectrum alone {r['skill_share_of_spectrum_only']*100:6.2f} %")
        print(f"     worst aux leak into the spectrum block: "
              f"{r['aux_recoverable_from_spectrum_block']['worst_leak']} "
              f"R^2 = {r['aux_recoverable_from_spectrum_block']['worst_leak_r2']:.3f}")

    # Three disjoint states instead of a single `is not None` filter. A dataset with no positive
    # rung-3 skill no longer falls silently into the PASS-supporting pool — it yields WARN.
    offenders = [k for k, r in payload["datasets"].items()
                 if r["verdict_contribution"] == "aux_dominates"]
    no_skill = [k for k, r in payload["datasets"].items()
                if r["verdict_contribution"] in ("no_achievable_skill_to_divide",
                                                 "denominator_unstable_under_resampling")]
    payload["datasets_where_aux_dominates"] = offenders
    payload["datasets_with_no_achievable_skill_to_divide"] = no_skill
    payload["threshold"]["binding_today"] = bool(offenders)
    payload["verdict_contributions"] = {k: r["verdict_contribution"]
                                        for k, r in payload["datasets"].items()}
    # The threshold rationale used to appeal to TEMPERATURE (48.5 K from a15), which this check does
    # not have among its targets — `a24:22` says outright "Temperature and radius are NOT scored".
    # Reworded in terms of the gases, i.e. what is actually scored here.
    payload["interpretation"] = (
        "A high aux share means the reported accuracy is not spectroscopy. The targets scored here "
        "are the five gas abundances only (a24: temperature and radius are not scored), so the "
        "threshold is anchored on the gases: on ADC2023 the aux table pins the equilibrium "
        "temperature to 48.5 K (a15), yet conditioning on the TRUE temperature and radius improves "
        "the scored gas mRMSE by just 0.259 % (a15 conditioning_value). The documented aux -> gas "
        "route is therefore worth ~0.3 % of the achievable skill, and the 20 % limit sits ~77x above "
        "it: a share above the limit would mean the benchmark rewards reading the star catalogue "
        "through some path nobody has described. Measured today the share is ~-0.2 % (ADC) and "
        "~-0.4 % (tau_val) — the aux-only rung is WORSE than the constant predictor, so the limit is "
        "far from binding and its resolving power near the boundary is untested on these data. "
        "poseidon_test has no positive aux+spectrum skill at all, so it cannot support the claim.")
    status = "FAIL" if offenders else ("WARN" if no_skill else "PASS")
    CHECK.emit(status, payload, out=args.out)


if __name__ == "__main__":
    main()
