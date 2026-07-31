"""A12 — Does the claimed statistical evidence support an architectural claim?

Proves/disproves finding P7 and the Test-07 half of P4. Every "significance" statement in the repo
resamples TEST ROWS of a SINGLE checkpoint. That estimates test-set sampling precision for one
trained model; it says nothing about whether a re-trained model would show the effect. Seed variance
is the missing dimension, and it is the one an architectural claim needs.

This check:
  * recomputes the cross-generator delta with a paired row bootstrap (reproducing the repo's method),
  * states explicitly what that interval does and does not cover,
  * reports the minimum detectable effect for the set sizes actually used (n=685, n=64), naming the
    set each sd was measured on,
  * DERIVES the seed inventory from the repository's own files instead of asserting it,
  * counts how many comparisons are made without multiplicity control.

The earlier version of this docstring promised "the numeric noise floor from a13 for comparison".
No such quantity exists: a13's payload has keys claims / documents / legend / missing_documents /
note / unbacked_or_mismatched / untracked_documents and zero hits for noise or floor, and a12 never
computed one either. The promise is removed from both the docstring and the criterion rather than
left standing without cover — a criterion naming a term nobody computes cannot be evaluated, and an
unevaluable conjunct silently makes the whole criterion decorative.

PASS criterion: any published effect claim is backed by across-seed variance (n_seeds >= 5) and the
effect exceeds the across-seed spread.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a12_significance_power",
    finding="P7 / P4 — all 'significance' is row bootstrap on one checkpoint; no seed variance, no multiplicity control",
    question="Is the resampling unit the right one for the claim, and can the set sizes resolve the claimed effects?",
    criterion="effect claims backed by >=5 seeds (derived from repo files) and larger than the across-seed spread",
)

PAIR = ("reports/ariel_quantum_taurex_snapshot_20260312_1003/poseidon_holdout_predictions.csv",
        "reports/taurex_noquant_taurex_snapshot_20260312_133054/poseidon_holdout_predictions.csv")

SEEDS_REQUIRED = 5   # the threshold from criterion=; declared here, not picked after seeing the result

# Where to look for seeds. Only files WRITTEN by a training or evaluation run count, not code that
# could take a seed as an argument — otherwise we would be measuring intent, not history.
SEED_SCAN_GLOBS = ("**/settings_resolved.yaml", "**/*_metrics.json", "**/config.json")
# Worktrees are copies of the same repo and would double the counts without contributing a new seed;
# reports/audit holds this audit's own output, so counting it would be circular.
SEED_SCAN_EXCLUDE = (".git/", ".claude/worktrees/", "node_modules/", "reports/audit/")


def _walk_seed_fields(obj, path: str = ""):
    """Recursively yield (path, value) pairs for keys containing 'seed'.

    Bools are rejected before the int check — bool is an int subclass, so
    `deterministic_seed: true` would otherwise enter the inventory as seed 1.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and "seed" in k.lower() and not isinstance(v, bool) \
                    and isinstance(v, (int, float)):
                yield f"{path}/{k}", int(v)
            yield from _walk_seed_fields(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_seed_fields(v, f"{path}/{i}")


def scan_seed_inventory() -> dict:
    """Derive the seed inventory from a scan of repository artefacts (not a literal).

    Previously: `n_seeds_found = 1` was a literal in the source reported as a measurement, and
    `seed_inventory` was the same literal under another name — circular, with no actual scan. If the
    result is still 1, it is now a MEASUREMENT: N files scanned, M seed fields found, all with the
    same value.
    """
    import json as _json
    import yaml
    files, seen = [], set()
    for pat in SEED_SCAN_GLOBS:
        # `glob` was imported at :20 and NEVER used — the import implied a scan that did not exist.
        for f in glob.glob(str(A.REPO / pat), recursive=True):
            rel = str(Path(f).relative_to(A.REPO))
            if any(x in f"{rel}/" for x in SEED_SCAN_EXCLUDE) or rel in seen:
                continue
            seen.add(rel)
            files.append((rel, Path(f)))
    seeds: dict[int, list[str]] = {}
    n_fields = n_unreadable = 0
    for rel, path in sorted(files):
        try:
            txt = path.read_text()
            doc = yaml.safe_load(txt) if path.suffix in (".yaml", ".yml") else _json.loads(txt)
        except Exception:
            n_unreadable += 1
            continue
        for key, val in _walk_seed_fields(doc):
            n_fields += 1
            seeds.setdefault(val, []).append(f"{rel}{key}={val}")
    return {
        "method": "derived by scanning repository artefacts; NOT a literal in the source",
        "globs": list(SEED_SCAN_GLOBS),
        "excluded_path_fragments": list(SEED_SCAN_EXCLUDE),
        "n_files_scanned": len(files),
        "n_files_unparseable": n_unreadable,
        "n_seed_fields_found": n_fields,
        "distinct_seed_values": sorted(seeds),
        "n_distinct_seeds": len(seeds),
        "occurrences_per_seed": {str(s): len(v) for s, v in sorted(seeds.items())},
        "examples": [e for s in sorted(seeds) for e in seeds[s][:3]],
    }


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a, b = (pd.read_csv(A.REPO / p) for p in PAIR)
    yt = a[[f"true_{t}" for t in A.TARGETS]].to_numpy(float)
    ytb = b[[f"true_{t}" for t in A.TARGETS]].to_numpy(float)
    assert np.allclose(yt, ytb), "the two prediction files disagree on ground truth; comparison invalid"
    pa = a[[f"pred_{t}" for t in A.TARGETS]].to_numpy(float)
    pb = b[[f"pred_{t}" for t in A.TARGETS]].to_numpy(float)

    boot = A.paired_bootstrap(yt, pa, pb, n_boot=args.n_boot, seed=0)
    # sign test + per-row spread, for the MDE
    se_a = ((yt - pa) ** 2).mean(axis=1)
    se_b = ((yt - pb) ** 2).mean(axis=1)
    diff = np.sqrt(se_a) - np.sqrt(se_b)
    base, _ = A.constant_predictor_mrmse(yt)

    inv = scan_seed_inventory()
    n_seeds_found = inv["n_distinct_seeds"]
    seed_inventory = [str(s) for s in inv["distinct_seed_values"]]

    # Aggregate counterpart for n=64: the bootstrap draws 64 rows with replacement from the same 685,
    # so it estimates the mRMSE spread for a set of size 64 from THIS population — still not the
    # spread of fair_small_experiment_cpu, which the repo does not store per row.
    rng64 = np.random.default_rng(64)
    d64 = np.empty(args.n_boot)
    for i in range(args.n_boot):
        k = rng64.integers(0, len(yt), 64)
        d64[i] = A.mrmse(yt[k], pa[k]) - A.mrmse(yt[k], pb[k])

    payload = {
        "comparison": "quantum snapshot vs noquant snapshot on POSEIDON test (n=685)",
        "row_bootstrap": boot,
        "sign_test": {"a_worse_rows": int((diff > 0).sum()), "b_worse_rows": int((diff < 0).sum()),
                      "fraction_a_worse": float((diff > 0).mean())},
        "per_row_sd_of_paired_difference": float(diff.std(ddof=1)),
        # two routes to the MDE, because they bound TWO DIFFERENT quantities:
        "mde_per_row_estimand_n685": A.min_detectable_effect(diff.std(ddof=1), 685),
        "mde_per_row_estimand_n64": A.min_detectable_effect(diff.std(ddof=1), 64),
        "mde_aggregate_mrmse_n685": A.mde_from_bootstrap(boot["boot_sd"]),
        "mde_aggregate_mrmse_n64": A.mde_from_bootstrap(float(d64.std(ddof=1))),
        "mde_note": ("mde_per_row_* ogranicza SREDNIA PO WIERSZACH roznicy per-wierszowego RMSE. "
                     "mde_aggregate_* ogranicza roznice AGREGATOWEGO mRMSE, czyli te wielkosc, "
                     "ktora raporty faktycznie cytuja. To rozne funkcjonaly (Jensen); porownuj "
                     "cytowane efekty z ta druga."),
        # Where each sd came from. Without this field mde_per_row_estimand_n64 reads as a measurement
        # of the n=64 set's spread, whereas it transplants the sd from 685 POSEIDON rows onto a
        # different set whose spread was never measured.
        "mde_sd_source": {
            "sd_value": float(diff.std(ddof=1)),
            "measured_on": "685 rows of the POSEIDON holdout pair, sd of the per-row paired RMSE difference",
            "n685_is_the_matching_set": True,
            "n64_is_the_matching_set": False,
            "n64_refers_to": "fair_small_experiment_cpu data_64 cells; the repo stores no per-row "
                             "predictions for them, so their own spread is not measurable here",
            "consequence": "mde_per_row_estimand_n64 is INDICATIVE: correct arithmetic on a borrowed sd. "
                           "mde_aggregate_mrmse_n64 bootstraps 64-row draws from the POSEIDON population, "
                           "which is the right functional but still the wrong population",
            "boot_sd_n64": float(d64.std(ddof=1)),
            "n_boot_n64": int(args.n_boot),
        },
        "trivial_baseline_on_the_same_rows": base,
        "both_models_below_baseline": bool(A.mrmse(yt, pa) > base and A.mrmse(yt, pb) > base),
        "seeds_used_in_the_repo": n_seeds_found,
        "seed_inventory": seed_inventory,
        "seed_inventory_provenance": inv,
        "what_the_interval_covers": "test-row sampling error for two fixed checkpoints",
        "what_it_does_not_cover": ["training/seed variance", "checkpoint-selection variance",
                                   "environment/numeric variance (see a13)",
                                   "the fact that both arms have negative skill (see a02)"],
        "numeric_noise_floor": {
            "available": False,
            "was_promised_by": "this check's own docstring and criterion, 'the numeric noise floor from a13'",
            "reality": "a13's payload has no noise/floor field (keys: claims, documents, legend, "
                       "missing_documents, note, unbacked_or_mismatched, untracked_documents) and a12 "
                       "never computed one; the term is removed from the criterion rather than left "
                       "as an unevaluable conjunct",
        },
        "multiplicity": {"corrections_found_in_repo": 0,
                         "comparisons_made_without_correction": [
                             "3 rankings over the same 3 models (taurex_model_comparison.md)",
                             "5 per-gas t-tests + 1 overall + 5 per-gas CIs (qat Test 07)",
                             "11 runtime x data cells (fair_small_experiment_cpu)"]},
        "verdict_text": ("The interval is narrow and correct for what it measures, and irrelevant to the "
                         "claim it is used for. Replace with: >=5 training seeds per arm, delta computed "
                         "per seed, CI over seeds, plus a pre-declared multiplicity correction."),
    }
    print(f"  delta (a-b) = {boot['delta']:+.6f}   95% CI [{boot['ci95_lo']:+.4f}, {boot['ci95_hi']:+.4f}]  "
          f"P(delta>0)={boot['p_delta_gt_0']:.3f}")
    print(f"  sign test: a worse on {payload['sign_test']['a_worse_rows']}/685 rows "
          f"({payload['sign_test']['fraction_a_worse']:.1%})")
    print(f"  MDE per-wierszowy estymand: n=685 -> {payload['mde_per_row_estimand_n685']:.4f}"
          f"   n=64 -> {payload['mde_per_row_estimand_n64']:.4f}")
    print(f"  MDE dla AGREGATOWEGO mRMSE (z bootstrapu, wlasciwy do porownan): "
          f"{payload['mde_aggregate_mrmse_n685']:.4f}")
    print(f"  MDE dla AGREGATOWEGO mRMSE, n=64 (bootstrap 64 wierszy): "
          f"{payload['mde_aggregate_mrmse_n64']:.4f}")
    print(f"  trivial baseline on the same rows = {base:.4f}; both models below baseline: "
          f"{payload['both_models_below_baseline']}")
    print(f"  seed scan: {inv['n_files_scanned']} plikow, {inv['n_seed_fields_found']} pol seedowych, "
          f"{n_seeds_found} rozna wartosc(i): {seed_inventory}")

    # Status derived from the criterion. Previously: `CHECK.emit("FAIL", ...)` with the status
    # hard-coded regardless of the data — the code tested neither of the two conjuncts of criterion=.
    # The former third conjunct ("numeric noise floor") is removed, not merely left unevaluated: it
    # exists neither in a13 nor here (payload["numeric_noise_floor"]).
    enough_seeds = n_seeds_found >= SEEDS_REQUIRED
    # Across-seed spread needs >=2 seeds and a per-seed delta; the repo has neither, so this conjunct
    # is explicitly unevaluated instead of being silently resolved either way.
    seed_spread_measurable = n_seeds_found >= 2
    effect_exceeds_seed_spread = None
    payload["status_terms"] = {
        "seeds_required": SEEDS_REQUIRED,
        "n_seeds_found": n_seeds_found,
        "n_seeds_source": "scan_seed_inventory() over repository artefacts",
        "enough_seeds": bool(enough_seeds),
        "seed_spread_measurable": bool(seed_spread_measurable),
        "effect_exceeds_seed_spread": effect_exceeds_seed_spread,
        "numeric_noise_floor_in_criterion": False,
    }
    if n_seeds_found == 0:
        status = "INFO"
        payload["criterion_not_evaluable"] = (
            f"the scan of {inv['n_files_scanned']} repository artefacts found no seed field at all, so "
            f"neither conjunct of the criterion can be evaluated; this is a gap in the evidence, not a "
            f"finding about the model")
        payload["status_terms"]["decisive"] = "no seed evidence found"
    elif not enough_seeds:
        status = "FAIL"
        payload["status_terms"]["decisive"] = (
            f"seed count: {n_seeds_found} distinct seed(s) measured across {inv['n_seed_fields_found']} "
            f"seed fields in {inv['n_files_scanned']} artefacts, criterion requires {SEEDS_REQUIRED}")
    elif effect_exceeds_seed_spread is None:
        status = "INFO"
        payload["criterion_not_evaluable"] = (
            f"{n_seeds_found} seeds are present but the repo stores no per-seed delta, so the across-seed "
            f"spread cannot be computed and the second conjunct is unevaluable")
        payload["status_terms"]["decisive"] = "seed spread not computable"
    else:
        status = "PASS" if effect_exceeds_seed_spread else "FAIL"
        payload["status_terms"]["decisive"] = "effect vs seed spread"
    print(f"  status wyliczony: {status} :: {payload['status_terms']['decisive']}")
    CHECK.emit(status, payload, inputs=[A.REPO / PAIR[0], A.REPO / PAIR[1]], out=args.out)


if __name__ == "__main__":
    main()
