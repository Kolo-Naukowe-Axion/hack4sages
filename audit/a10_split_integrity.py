"""A10 — Is every "holdout" actually held out?

Proves/disproves findings P1 / P2 / U4:
  * models/taurex_fmpe/constants.py:18-22 points validation and holdout at the SAME selector
    (tau/val); prepare_dataset.py records "holdout_mirrors_validation": true. Early stopping and
    best_model_by_mrmse selection then run on the rows later reported as holdout.
  * models/taurex_exobiome_without_quant with taurex_ignore_poseidon=true copies val -> holdout.
  * models/taurex_exobiome maps testdata -> the same rows as holdout.
  * there is no in-distribution TauREx test split at all.

Static + empirical: reads the selectors from the packages, then (where data is present) checks row
overlap between the splits actually produced.

PASS criterion: for every package, validation and holdout selectors differ, holdout ∩ validation = 0,
and a split used for early stopping is never reported as a test metric.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a10_split_integrity",
    finding="P1 / P2 / U4 — taurex_fmpe holdout == validation; noquant copies val->holdout; testdata == holdout; no in-domain TauREx test split",
    question="Does any package report as 'holdout' rows that were used for early stopping or model selection?",
    criterion="validation and holdout selectors differ AND row overlap is zero in every package",
)

# Previously: the list held only TAUREX_VALIDATION_GENERATOR/SPLIT, which do not exist in the repo —
# the code uses TAUREX_VAL_GENERATOR/SPLIT (taurex_exobiome/dataset.py:55-56,
# _without_quant/dataset.py:58-59). Effect: validation_selector came out [None, None] for both flagship
# packages, the condition at :123 was never evaluated, and the payload showed "issues: []" — the test
# looked passed but was empty. Hence the explicit "undecidable selector -> problem" branch below.
SELECTOR_KEYS = ["SOURCE_TRAIN_GENERATOR", "SOURCE_TRAIN_SPLIT", "SOURCE_VALIDATION_GENERATOR",
                 "SOURCE_VALIDATION_SPLIT", "SOURCE_HOLDOUT_GENERATOR", "SOURCE_HOLDOUT_SPLIT",
                 "TAUREX_TRAIN_GENERATOR", "TAUREX_TRAIN_SPLIT",
                 "TAUREX_VAL_GENERATOR", "TAUREX_VAL_SPLIT",
                 "TAUREX_VALIDATION_GENERATOR", "TAUREX_VALIDATION_SPLIT",
                 "TAUREX_HOLDOUT_GENERATOR", "TAUREX_HOLDOUT_SPLIT",
                 "EXCLUDED_GENERATORS"]

# Missing selectors are a fact to report, not a reason to skip.
# Previously: `if not found: continue` dropped 10 of 13 directories with no trace in the payload.
SELECTOR_ALIASES = {
    "validation_generator": ["SOURCE_VALIDATION_GENERATOR", "TAUREX_VAL_GENERATOR",
                             "TAUREX_VALIDATION_GENERATOR"],
    "validation_split": ["SOURCE_VALIDATION_SPLIT", "TAUREX_VAL_SPLIT", "TAUREX_VALIDATION_SPLIT"],
    "holdout_generator": ["SOURCE_HOLDOUT_GENERATOR", "TAUREX_HOLDOUT_GENERATOR"],
    "holdout_split": ["SOURCE_HOLDOUT_SPLIT", "TAUREX_HOLDOUT_SPLIT"],
}


def literals(path: Path) -> dict:
    """Extract module-level string/tuple constants without importing (no side effects)."""
    out = {}
    try:
        tree = ast.parse(path.read_text())
    except Exception:
        return out
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in SELECTOR_KEYS:
                try:
                    out[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def pick(found: dict, role: str):
        for k in SELECTOR_ALIASES[role]:
            if found.get(k) is not None:
                return found[k], k
        return None, None

    packages, problems, skipped_packages = {}, [], []
    for pkg in sorted((A.REPO / "models").glob("*/")):
        found = {}
        for f in ("constants.py", "dataset.py"):
            p = pkg / f
            if p.exists():
                found.update(literals(p))
        if not found:
            # report what was NOT examined — a traceless skip is indistinguishable from a passed test
            skipped_packages.append({"package": pkg.name,
                                     "reason": "brak stalych selektorow w constants.py / dataset.py",
                                     "files_present": [f for f in ("constants.py", "dataset.py")
                                                       if (pkg / f).exists()]})
            continue
        vg, vg_key = pick(found, "validation_generator")
        vs, vs_key = pick(found, "validation_split")
        hg, hg_key = pick(found, "holdout_generator")
        hs, hs_key = pick(found, "holdout_split")
        val, hold = (vg, vs), (hg, hs)
        entry = {"selectors": found, "validation_selector": val, "holdout_selector": hold,
                 "selector_keys_used": {"validation_generator": vg_key, "validation_split": vs_key,
                                        "holdout_generator": hg_key, "holdout_split": hs_key},
                 "issues": []}
        undecidable = [role for role, v in (("validation_generator", vg), ("validation_split", vs),
                                           ("holdout_generator", hg), ("holdout_split", hs))
                       if v is None]
        if undecidable:
            # An undecidable selector is NO ANSWER, not the answer "they differ". Previously
            # `if val[0] is not None` silently skipped the whole test for such a package.
            entry["issues"].append(
                f"selektory nierozstrzygalne ({', '.join(undecidable)}) — nie da sie orzec, czy "
                f"validation i holdout sa rozne; odczytane stale: {sorted(found)}")
        elif val == hold:
            entry["issues"].append(f"validation and holdout use the SAME selector {val} -> the reported "
                                   "holdout metric is a validation metric (selection-contaminated)")
        packages[pkg.name] = entry
        if entry["issues"]:
            problems.append({"package": pkg.name, "issues": entry["issues"]})

    # grep for the val->holdout copy and the testdata==holdout copy
    greps = []
    for rel, needle, note in (
        ("models/taurex_exobiome_without_quant/dataset.py", "holdout_source_indices = val_source_indices.copy()",
         "taurex_ignore_poseidon=true copies validation into holdout"),
        ("models/taurex_exobiome/dataset.py", "test_indices = holdout_indices.copy()",
         "testdata is a copy of holdout"),
        ("models/taurex_fmpe/prepare_dataset.py", "holdout_mirrors_validation",
         "package self-documents that holdout mirrors validation"),
    ):
        p = A.REPO / rel
        if p.exists():
            hit = needle in p.read_text()
            greps.append({"file": rel, "pattern": needle, "present": hit, "meaning": note})
            if hit:
                problems.append({"package": rel, "issues": [note]})

    # empirical: which committed configs have the flag on?
    flags = []
    for cfg in sorted(A.REPO.glob("reports/**/config.json")):
        try:
            d = json.loads(cfg.read_text())
        except Exception:
            continue
        if d.get("taurex_ignore_poseidon"):
            flags.append(str(cfg.relative_to(A.REPO)))

    # empirical overlap on the one dataset present
    overlap = None
    lab = A.REPO / "data/TauREx set/labels.parquet"
    if lab.exists():
        df = pd.read_parquet(lab)
        ids = {k: set(g["sample_id"]) for k, g in df.groupby(["generator", "split"])}
        keys = list(ids)
        overlap = {"split_sizes": {str(k): len(v) for k, v in ids.items()},
                   "pairwise_overlap": {f"{a}|{b}": len(ids[a] & ids[b])
                                        for i, a in enumerate(keys) for b in keys[i + 1:]},
                   "duplicate_sample_ids": int(df["sample_id"].duplicated().sum()),
                   "in_distribution_test_split_exists": bool(
                       any(k[0] == "tau" and k[1] == "test" for k in ids))}
        # Previously: computed but never compared against zero, even though criterion= requires it —
        # a measurement that cannot change the verdict is not a test. Now: 3 overlaps = 0, 0 duplicates.
        nonzero = {k: v for k, v in overlap["pairwise_overlap"].items() if v > 0}
        overlap["pairwise_overlap_all_zero"] = not nonzero
        overlap["nonzero_pairs"] = nonzero
        if nonzero:
            problems.append({"package": "data/TauREx set",
                             "issues": [f"niezerowe przeciecie splitow: {nonzero} — criterion wymaga zera "
                                        "w kazdym pakiecie"]})
        if overlap["duplicate_sample_ids"] > 0:
            problems.append({"package": "data/TauREx set",
                             "issues": [f"{overlap['duplicate_sample_ids']} zduplikowanych sample_id w "
                                        "labels.parquet — przypisanie wiersza do splitu nie jest jednoznaczne"]})
        if not overlap["in_distribution_test_split_exists"]:
            problems.append({"package": "data/TauREx set",
                             "issues": ["no tau/test split exists: tau/val is both the early-stopping set "
                                        "and the reported in-distribution score"]})

    payload = {"packages": packages, "n_packages_examined": len(packages),
               "packages_skipped_no_selectors": skipped_packages,
               "n_packages_skipped": len(skipped_packages),
               "n_package_dirs_in_models": len(list((A.REPO / "models").glob("*/"))),
               "code_patterns": greps, "configs_with_ignore_poseidon": flags,
               "crossgen_overlap": overlap, "problems": problems}
    for pr in problems:
        print(f"  [{pr['package']}]")
        for i in pr["issues"]:
            print(f"     - {i}")
    print(f"  zbadano {len(packages)} pakietow, pominieto {len(skipped_packages)} "
          f"(brak stalych selektorow): {[s['package'] for s in skipped_packages]}")
    CHECK.emit("FAIL" if problems else "PASS", payload, out=args.out)


if __name__ == "__main__":
    main()
