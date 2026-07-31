"""A29 — Recompute the team's own smoke baseline from the artefacts it shipped in.

Proves/disproves finding K1(b). The bundle `data/TauREx set/` contains `baseline_smoke.json` and
`baseline_poseidon_predictions.csv`, both produced by the team's own `baseline_smoke.py` before the
dataset was handed over. They already carry the answer: the ridge baseline learns something modest on
the TauREx split and nothing at all on POSEIDON. This check turns the numbers into a record. Six
tests:

    T1  per-gas RMSE recomputed from the predictions CSV == the RMSE recorded in the JSON
    T2  the scalar mRMSE behind each generator (the JSON stores per-gas dicts, not scalars)
    T3  skill against a constant predictor, in both baseline variants a02 reports
    T4  spread of the predictions against the spread of the labels
    T5  whether the predictions are functionally the prior mean
    T6  whether anything outside the bundle, this harness and the report reads the artefacts by path

T1 is the falsification test named in the report: if the CSV does not reproduce the JSON, the two
files are not from one run.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a29_smoke_baseline_recovery",
    finding="K1(b) — the team's own ridge smoke baseline ran, shipped in the bundle, and records no skill on POSEIDON",
    question="Do the two smoke artefacts agree per gas, what skill is behind them, and how wide are the predictions?",
    criterion="the finding is REFUTED (PASS) if an artefact is missing, or the CSV does not reproduce "
              "the JSON within 1e-9, or there is no skill contrast between the two splits (sign test, "
              "guarded by the n=685 MDE so a |skill| below detectability is read as 'no skill' rather "
              "than as an agreeing sign), or any file outside the bundle / audit/ / docs/ reads the "
              "artefacts by path; it STANDS (FAIL) otherwise",
)

BUNDLE = Path("data/TauREx set")
PRIOR_LO, PRIOR_HI = -12.0, -2.0
CONSISTENCY_TOL = 1e-9

# Detectability floor for skill at n=685, copied from `a12.mde_aggregate_mrmse_n685` (80 % power,
# alpha=0.05, bootstrap of the mRMSE aggregate over the 685 POSEIDON rows). Used ONLY as a guard on
# condition (c) — see the comment at `no_contrast`. It enters no reported number.
MDE_SKILL_N685 = 0.043621087128020715

# Which of the two computed skills decides the contrast. `skill_vs_train_mean` — because that is the
# variant a02 reports (constant vector = mean of the tau/train labels), i.e. the honest trivial
# baseline available to the model. The `skill_vs_oracle_constant` variant uses the mean of the
# EVALUATED ROWS and therefore sees the test labels; it is an upper bound on model credit and stays
# reported, but it cannot decide the verdict.
DECISIVE_SKILL = "skill_vs_train_mean"

ARTEFACT_NAMES = ("baseline_smoke.json", "baseline_poseidon_predictions.csv")

EXCLUDED_PATH_PREFIXES = ("data/TauREx set", "audit", "docs", "reports/audit", ".git",
                          # Git worktrees are COPIES of the same repo. Without this exclusion the
                          # consumer count grows by the whole file set times the number of
                          # attached worktrees (measured: 8 -> 51 with one worktree), so the
                          # headline number stops being reproducible and depends on the state of
                          # the directory rather than of the repo. `a12` excludes
                          # `.claude/worktrees/` for the same reason.
                          ".claude/worktrees")
EXCLUDED_DIR_GLOBS = (".venv*", "__pycache__", "*.egg-info", "node_modules")
EXCLUDED_FILES = (".gitignore",)
# Files larger than this are not scanned.
MAX_SCAN_BYTES = 8_000_000


def bundle_paths() -> dict[str, Path]:
    root = A.REPO / BUNDLE
    return {"smoke": root / "baseline_smoke.json",
            "predictions": root / "baseline_poseidon_predictions.csv",
            "labels": root / "labels.parquet"}


def git_tracked(path: Path) -> bool:
    """READ-ONLY. Whether git has this path in the index — the mechanism behind K1(b)."""
    try:
        proc = subprocess.run(["git", "-C", str(A.REPO), "ls-files", "--error-unmatch", str(path)],
                              capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def git_added_in(path: Path) -> str:
    """READ-ONLY. Commit that first added this path, `"<sha> <date>"` or `""` if untracked. Needed to
    tell a consumer that predates the hand-over from a file the audit itself produced."""
    try:
        rel = str(path.relative_to(A.REPO))
    except ValueError:
        return ""
    try:
        proc = subprocess.run(["git", "-C", str(A.REPO), "log", "--diff-filter=A", "--format=%h %ad",
                               "--date=short", "-1", "--", rel], capture_output=True, text=True)
        return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    except Exception:
        return ""


def _excluded(rel: str, is_dir: bool) -> bool:
    if any(rel == pref or rel.startswith(pref + "/") for pref in EXCLUDED_PATH_PREFIXES):
        return True
    name = rel.rsplit("/", 1)[-1]
    if is_dir:
        return any(fnmatch.fnmatch(name, g) for g in EXCLUDED_DIR_GLOBS)
    return rel in EXCLUDED_FILES


def find_consumers() -> dict:
    """T6 — READ-ONLY byte scan for files that name either artefact, outside the exclusions above."""
    hits: list[dict] = []
    n_scanned = n_skipped_large = 0
    for root, dirs, files in os.walk(A.REPO):
        rel_root = os.path.relpath(root, A.REPO).replace(os.sep, "/")
        rel_root = "" if rel_root == "." else rel_root
        dirs[:] = sorted(d for d in dirs if not _excluded(f"{rel_root}/{d}".lstrip("/"), True))
        for fn in sorted(files):
            rel = f"{rel_root}/{fn}".lstrip("/")
            if _excluded(rel, False):
                continue
            p = A.REPO / rel
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                if p.stat().st_size > MAX_SCAN_BYTES:
                    n_skipped_large += 1
                    continue
                blob = p.read_bytes()
            except OSError:
                continue
            n_scanned += 1
            named = [n for n in ARTEFACT_NAMES if n.encode() in blob]
            if named:
                hits.append({"path": rel, "artefacts_referenced": named,
                             "git_tracked": git_tracked(p), "added_in": git_added_in(p)})
    return {
        "consumers": hits,
        "n_consumers": len(hits),
        "consumed": bool(hits),
        "artefact_names_searched": list(ARTEFACT_NAMES),
        "excluded_path_prefixes": list(EXCLUDED_PATH_PREFIXES),
        "excluded_dir_globs": list(EXCLUDED_DIR_GLOBS),
        "excluded_files": list(EXCLUDED_FILES),
        "n_files_scanned": n_scanned,
        "n_files_skipped_over_size_cap": n_skipped_large,
        "size_cap_bytes": MAX_SCAN_BYTES,
        "match_semantics": "bare filename appearing in the file's bytes; a reference, not a proven read",
    }


def file_provenance(paths: dict[str, Path]) -> dict[str, dict]:
    out = {}
    for key, p in paths.items():
        if not p.is_file():
            out[key] = {"path": str(p.relative_to(A.REPO)), "exists": False}
            continue
        st = p.stat()
        out[key] = {
            "path": str(p.relative_to(A.REPO)),
            "exists": True,
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "git_tracked": git_tracked(p),
            "added_in": git_added_in(p),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pandas as pd

    paths = bundle_paths()

    # Criterion (a) must be settled BEFORE any read: is either of the two artefacts absent?
    consumption = find_consumers()
    missing_artefacts = sorted(k for k in ("smoke", "predictions") if not paths[k].is_file())
    if missing_artefacts:
        CHECK.emit("PASS", {
            "resolved_by": "criterion (a) — artefact absent from the bundle",
            "missing": [str(paths[k].relative_to(A.REPO)) for k in missing_artefacts],
            "t6_consumption": consumption,
            "provenance": file_provenance(paths),
            "note": "the smoke baseline cannot be recovered from this bundle, so K1(b) as worded "
                    "(the run happened and its result was shipped) is not supported by these files",
        }, out=args.out)
        return
    if not paths["labels"].is_file():
        CHECK.emit("INFO", {
            "resolved_by": "labels.parquet absent — no ground truth, so T1..T5 cannot be computed",
            "t6_consumption": consumption,
            "provenance": file_provenance(paths),
        }, out=args.out)
        return

    smoke = json.loads(paths["smoke"].read_text())
    preds = pd.read_csv(paths["predictions"])
    labels = pd.read_parquet(paths["labels"])

    gases = list(smoke["target_columns"])
    assert gases == A.CROSSGEN_TARGETS, (gases, A.CROSSGEN_TARGETS)

    test = labels[(labels.generator == "poseidon") & (labels.split == "test")].set_index("sample_id")
    val = labels[(labels.generator == "tau") & (labels.split == "val")]
    train = labels[(labels.generator == "tau") & (labels.split == "train")]
    train_constant = train[gases].to_numpy(dtype=np.float64).mean(axis=0)

    known = preds["sample_id"].isin(set(test.index))
    unmatched = sorted(set(preds.loc[~known, "sample_id"]))
    preds_matched = preds.loc[known]
    aligned = test.loc[preds_matched["sample_id"]]
    y_true = aligned[gases].to_numpy(dtype=np.float64)
    y_pred = preds_matched[[f"pred_{g}" for g in gases]].to_numpy(dtype=np.float64)
    assert len(y_true) == len(y_pred), (len(y_true), len(y_pred))

    # T1 — does the CSV reproduce the RMSE the JSON recorded?
    recomputed = A.per_gas_rmse(y_true, y_pred)
    t1_per_gas, max_diff = {}, 0.0
    for g, r in zip(gases, recomputed):
        recorded = float(smoke["test_rmse"][g])
        diff = abs(float(r) - recorded)
        max_diff = max(max_diff, diff)
        t1_per_gas[g] = {"recomputed_from_csv": float(r), "recorded_in_json": recorded, "abs_diff": diff}
    consistent = max_diff < CONSISTENCY_TOL

    # T2 — the JSON stores per-gas dicts; the scalar the report quotes is their unweighted mean.
    mrmse_test_recorded = float(np.mean([smoke["test_rmse"][g] for g in gases]))
    mrmse_val_recorded = float(np.mean([smoke["val_rmse"][g] for g in gases]))
    mrmse_test_from_csv = A.mrmse(y_true, y_pred)

    # T3 — skill, in both baseline variants a02 carries.
    # oracle is a constant predictor that scores the best possible constant, hence an upper bound on model credit.
    y_val = val[gases].to_numpy(dtype=np.float64)
    skills = {}
    for key, y, reported in (("taurex_val", y_val, mrmse_val_recorded),
                             ("poseidon_test", y_true, mrmse_test_recorded)):
        oracle, oracle_per_gas = A.constant_predictor_mrmse(y)
        train_mean, train_mean_per_gas = A.constant_predictor_mrmse(y, train_constant)
        skills[key] = {
            "n_rows": int(len(y)),
            "smoke_mrmse": reported,
            "baseline_train_mean_mrmse": train_mean,
            "baseline_train_mean_per_gas": train_mean_per_gas.tolist(),
            "baseline_oracle_constant_mrmse": oracle,
            "baseline_oracle_per_gas": oracle_per_gas.tolist(),
            "skill_vs_train_mean": A.skill(reported, train_mean),
            "skill_vs_oracle_constant": A.skill(reported, oracle),
        }

    # T4 — how much of the label spread do the predictions span?
    prior_sd = (PRIOR_HI - PRIOR_LO) / np.sqrt(12.0)
    prior_mean = 0.5 * (PRIOR_LO + PRIOR_HI)
    t4_per_gas, ratios = {}, []
    for i, g in enumerate(gases):
        sd_pred = float(y_pred[:, i].std(ddof=1))
        sd_label = float(y_true[:, i].std(ddof=1))
        ratios.append(sd_pred / sd_label)
        t4_per_gas[g] = {"sd_pred": sd_pred, "sd_label": sd_label, "ratio": sd_pred / sd_label,
                         "label_sd_vs_uniform_prior_sd": sd_label / float(prior_sd)}

    # T5 — is the predictor functionally the constant at the prior mean?
    const_prior, _ = A.constant_predictor_mrmse(y_true, np.full(len(gases), prior_mean))

    payload = {
        "t1_artefact_consistency": {
            "per_gas": t1_per_gas,
            "max_abs_diff": max_diff,
            "tolerance": CONSISTENCY_TOL,
            "consistent": consistent,
            "n_rows_csv": int(len(preds)),
            "n_rows_csv_matched_to_labels": int(len(preds_matched)),
            "n_rows_labels": int(len(test)),
            "sample_ids_in_csv_absent_from_labels": unmatched,
            "n_sample_ids_absent": len(unmatched),
            "rmse_computed_on": "only the rows matched to labels; any unmatched id is a FAIL reason",
        },
        "t2_mrmse": {
            "taurex_val": {"per_gas": {g: float(smoke["val_rmse"][g]) for g in gases},
                           "mrmse": mrmse_val_recorded,
                           "independently_recomputable": False,
                           "note": "the bundle ships no prediction file for the TauREx split, so this "
                                   "aggregate rests on the recorded per-gas values alone"},
            "poseidon_test": {"per_gas": {g: float(smoke["test_rmse"][g]) for g in gases},
                              "mrmse": mrmse_test_recorded,
                              "mrmse_recomputed_from_csv": mrmse_test_from_csv,
                              "independently_recomputable": True},
        },
        "t3_skill": skills,
        "t4_prediction_spread": {
            "per_gas": t4_per_gas,
            "mean_ratio": float(np.mean(ratios)),
            "mean_prediction_all_gases": float(y_pred.mean()),
            "prior_mean": prior_mean,
            "prior_sd": float(prior_sd),
            "sd_convention": "ddof=1",
        },
        "t5_collapse_to_prior_mean": {
            "constant_at_prior_mean_mrmse": const_prior,
            "smoke_mrmse": mrmse_test_recorded,
            "abs_diff": abs(const_prior - mrmse_test_recorded),
        },
        "t6_consumption": consumption,
        "provenance": file_provenance(paths),
        "definitions": {
            "mrmse": "unweighted mean over gases of per-gas RMSE over the sample axis (audit_lib.mrmse)",
            "smoke_rmse": "baseline_smoke.py:_rmse_by_column = sqrt(mean((pred-true)**2, axis=0)), "
                          "bit-for-bit the same functional as audit_lib.per_gas_rmse",
            "baseline_train_mean": "constant vector = mean of the tau/train labels, evaluated on the "
                                   "rows in question — the honest trivial baseline, and the variant a02 reports",
            "baseline_oracle_constant": "constant vector = mean of the evaluation rows themselves — the "
                                        "most favourable constant, hence an upper bound on model credit",
            "prior": f"crossgen prior is U({PRIOR_LO:.0f}, {PRIOR_HI:.0f}) for all five gases",
        },
    }

    # Four disjoint routes to PASS, each refuting a different clause of finding K1(b); none of them
    # looks at the MAGNITUDE of the skills (that belongs to K1/a01 and the ladders a02, a26):
    #   (a) artefact missing, (b) the CSV does not reproduce the JSON, (c) no contrast between the
    #   skill signs, (d) a consumer exists. (c) looks at the SIGNS, not the values — the finding
    #   says "modest on tau, nothing on POSEIDON", i.e. it is about the RELATION of two signs; were
    #   both equal, the bundle alone would not tell the generators apart.
    skill_tau = float(skills["taurex_val"][DECISIVE_SKILL])
    skill_pos = float(skills["poseidon_test"][DECISIVE_SKILL])

    # MDE GUARD. A bare `sign(a) == sign(b)` would be a sign test on a quantity that can sit far
    # below the detectability floor: skill on POSEIDON is about -5e-4, while the MDE for n=685 is
    # 0.0436 (a12.mde_aggregate_mrmse_n685), i.e. ~92x larger. The sign of such a quantity is not
    # measurable — another draw would have given +3e-4, equally indistinguishable from zero, and
    # condition (c) would then refute the WHOLE of K1(b) on noise.
    # The finding says "modest skill on tau, NO skill on POSEIDON", not "negative skill on
    # POSEIDON". So the contrast counts only when the tau side is DETECTABLY positive; if |skill| on
    # either side sinks into the MDE, that side's sign stays out of the algebra.
    # This does NOT change today's verdict: skill_tau = +0.0938 > MDE, skill_pos ~ 0 -> no contrast
    # = False, exactly as with the bare sign test.
    tau_detectable = abs(skill_tau) >= MDE_SKILL_N685
    pos_detectable = abs(skill_pos) >= MDE_SKILL_N685
    if tau_detectable and pos_detectable:
        no_contrast = bool(np.sign(skill_tau) == np.sign(skill_pos))
    else:
        # At least one side is below the floor. The contrast exists if and only if tau is detectably
        # positive and POSEIDON is not detectably positive.
        no_contrast = not (tau_detectable and skill_tau > 0 and not (pos_detectable and skill_pos > 0))

    cond = {
        "a_artefact_missing": False,
        "b_csv_disagrees_with_json": bool(not consistent),
        "c_no_skill_sign_contrast": bool(no_contrast),
        "d_result_was_consumed": bool(consumption["consumed"]),
    }
    # A separate key at payload level, NOT inside `t3_skill`: that dict is iterated per split below
    # (`for key, s in payload["t3_skill"].items()`), so an extra key of a different shape would
    # break the printout.
    payload["skill_detectability"] = {
        "mde_skill_n685": MDE_SKILL_N685,
        "mde_source": "a12.mde_aggregate_mrmse_n685 — MDE for the aggregate mRMSE functional, n=685",
        "skill_taurex_val_detectable": bool(tau_detectable),
        "skill_poseidon_test_detectable": bool(pos_detectable),
        "why": "condition (c) is a sign test; a sign is only meaningful above the detectability "
               "floor, so a |skill| below MDE is read as 'no skill', never as 'agreeing sign'",
    }
    refuting = [k for k, v in cond.items() if v]
    stands = {
        "both_artefacts_present": True,
        "artefacts_agree_within_tol": bool(consistent),
        "positive_skill_on_taurex_val": bool(skill_tau > 0),
        "no_skill_on_poseidon_test": bool(skill_pos <= 0),
        "no_consumer_outside_audit": bool(not consumption["consumed"]),
        "all_sample_ids_matched": bool(not unmatched),
    }
    if unmatched:
        stands["all_sample_ids_matched"] = False

    if refuting:
        status, resolved_by = "PASS", refuting
    elif all(stands.values()):
        status, resolved_by = "FAIL", []
    else:
        # Gap between the two criteria (e.g. skill_tau == 0 with skill_pos < 0): neither refuted nor
        # fully confirmed by the `stands` conjunction above.
        status, resolved_by = "WARN", []

    reasons = []
    if cond["b_csv_disagrees_with_json"]:
        reasons.append(f"(b) the two artefacts disagree by up to {max_diff:.2e} per gas "
                       f"(tol {CONSISTENCY_TOL:.0e}) — not one run on this bundle")
    if cond["c_no_skill_sign_contrast"]:
        reasons.append(f"(c) no skill sign contrast: {DECISIVE_SKILL} is {skill_tau:+.6f} on "
                       f"taurex_val and {skill_pos:+.6f} on poseidon_test — the shipped numbers do "
                       f"not single out POSEIDON on their own")
    if cond["d_result_was_consumed"]:
        reasons.append(f"(d) the result WAS consumed: {consumption['n_consumers']} file(s) outside the "
                       f"bundle / audit/ / docs/ reference the artefacts by name, e.g. "
                       + ", ".join(h["path"] for h in consumption["consumers"][:3]))
    if unmatched:
        reasons.append(f"{len(unmatched)} sample_id(s) in the CSV are absent from the labels, so T1 "
                       f"was evaluated on {len(preds_matched)}/{len(preds)} rows only: "
                       + ", ".join(map(str, unmatched[:5])))

    payload["verdict"] = {
        "status": status,
        "refuting_conditions_met": refuting,
        "refuting_conditions": cond,
        "finding_stands_conjunction": stands,
        "decisive_skill_variant": DECISIVE_SKILL,
        "skill_taurex_val": skill_tau,
        "skill_poseidon_test": skill_pos,
        "magnitudes_are_reported_not_scored": (
            "neither |skill| enters the PASS/FAIL algebra; baseline quality is K1 (a01) and the "
            "ladders (a02, a26), not K1(b)"),
    }
    payload["verdict_reasons"] = reasons

    print(f"T1 artefact consistency — max |CSV - JSON| per gas = {max_diff:.2e} "
          f"({'consistent' if consistent else 'INCONSISTENT'}), {len(preds)} rows, "
          f"{len(unmatched)} unmatched sample_id")
    for g, d in t1_per_gas.items():
        print(f"     {g:16} csv {d['recomputed_from_csv']:.9f}   json {d['recorded_in_json']:.9f}")

    print(f"\nT2/T3 skill of the team's own smoke baseline")
    print(f"     {'set':14} {'mRMSE':>9} {'base(train)':>12} {'skill':>9} {'base(oracle)':>13} {'skill':>9}")
    for key, s in skills.items():
        print(f"     {key:14} {s['smoke_mrmse']:9.4f} {s['baseline_train_mean_mrmse']:12.4f} "
              f"{s['skill_vs_train_mean']:+9.4f} {s['baseline_oracle_constant_mrmse']:13.4f} "
              f"{s['skill_vs_oracle_constant']:+9.4f}")

    print(f"\nT4 prediction spread (ddof=1)")
    print(f"     {'gas':16} {'sd(pred)':>9} {'sd(label)':>10} {'ratio':>7} {'sd(label)/prior_sd':>19}")
    for g, d in t4_per_gas.items():
        print(f"     {g:16} {d['sd_pred']:9.4f} {d['sd_label']:10.4f} {d['ratio']:7.4f} "
              f"{d['label_sd_vs_uniform_prior_sd']:19.4f}")
    print(f"     mean ratio {np.mean(ratios):.4f}  ·  mean prediction {y_pred.mean():.4f}  ·  "
          f"prior mean {prior_mean:.1f}  ·  prior sd {prior_sd:.4f}")

    print(f"\nT5 constant at the prior mean scores {const_prior:.6f}; the smoke baseline scores "
          f"{mrmse_test_recorded:.6f} (|d| = {abs(const_prior - mrmse_test_recorded):.6f})")

    for k, v in payload["provenance"].items():
        print(f"\nprovenance {k:12} git_tracked={v.get('git_tracked')} added_in={v.get('added_in')!r}")

    print(f"\nT6 consumption — {consumption['n_consumers']} referencing file(s) "
          f"({consumption['n_files_scanned']} scanned, {consumption['n_files_skipped_over_size_cap']} "
          f"over the {MAX_SCAN_BYTES/1e6:.0f} MB cap)")
    for h in consumption["consumers"]:
        print(f"     {h['path']}  tracked={h['git_tracked']} added_in={h['added_in']!r}")

    print(f"\nverdict {status} — refuting conditions met: {refuting or 'none'}")
    for r in reasons:
        print(f"     {r}")

    CHECK.emit(status, payload,
               inputs=[paths["smoke"], paths["predictions"], paths["labels"]], out=args.out)


if __name__ == "__main__":
    main()
