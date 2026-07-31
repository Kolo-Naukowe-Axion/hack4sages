"""A21 — Dead input dimensions, degenerate channels, and informative columns nobody reads.

Proves/disproves findings U2 / U3, and it is the check that would have flagged K9(d) automatically:
on the cross-generator set `_build_taurex_auxiliary_frame` hard-codes 4 of the 8 auxiliary features
(and derives a 5th from two of them), so after standardisation those dimensions are exactly zero —
while `temperature_k`, which varies 500-1800 K and sets the atmospheric scale height, is present in
labels.parquet and read by nobody.

Also flags channels that are a single scalar broadcast across the spectral axis (the crossgen noise
channel, and 218 of FMPE's 444 context dims).

PASS criterion: zero zero-variance input dimensions after the package's own standardisation, no
scalar-broadcast channel, and no label column that varies in the data but is read by no arm of a
comparison.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a21_dead_features",
    finding="U2 / U3 / K9(d) — 5 of 8 crossgen aux features are constants; temperature_k varies 500-1800 K and is read by nobody",
    question="Which model inputs carry no information, and which informative label columns are never read?",
    criterion="no zero-variance input dims, no scalar-broadcast channel, no unread varying label column",
)

# Unique-value threshold above which a label column enters the "who reads this" test.
#
# Previously `> 100` — arbitrary and never reported in the payload. It silenced the five
# `present_h2o/co2/co/ch4/nh3` columns (each `nunique == 2`), which appear nowhere under `models/`
# (grep over the whole tree: 0 hits) — i.e. it hid exactly the columns this check looks for.
# `> 1` has a substantive justification: a single-valued column cannot carry information, two or
# more values already discriminate. The threshold is now in the payload (`label_column_filter`).
MIN_NUNIQUE_TO_BE_INFORMATIVE = 1


def hardcoded_aux(pkg: str) -> dict:
    """Find aux fields that are constants in the TauREx aux builder.

    Resolves two forms: a literal `np.full(...)` in the dict, and a dict value that is a bare name
    bound to an `np.full(...)` earlier in the same function (the builder does both).
    """
    p = A.REPO / "models" / pkg / "dataset.py"
    if not p.exists():
        return {}
    tree = ast.parse(p.read_text())
    out: dict = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and "auxiliary_frame" in node.name):
            continue
        const_locals: dict[str, str] = {}
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                src = ast.unparse(stmt.value)
                if src.startswith("np.full("):
                    const_locals[stmt.targets[0].id] = src
        for d in ast.walk(node):
            if not isinstance(d, ast.Dict):
                continue
            for k, v in zip(d.keys, d.values):
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                    continue
                src = ast.unparse(v)
                if src.startswith("np.full("):
                    out[k.value] = {"constant": True, "how": "np.full in place", "expr": src}
                elif isinstance(v, ast.Name) and v.id in const_locals:
                    out[k.value] = {"constant": True, "how": f"bound to constant local `{v.id}`",
                                    "expr": const_locals[v.id]}
                elif "_taurex_orbital_period_days" in src:
                    out[k.value] = {"constant": "derived from two constants", "how": "derived", "expr": src}
                else:
                    base = src.split(".astype")[0]
                    if isinstance(v, ast.Attribute) or "astype" in src:
                        for nm, expr in const_locals.items():
                            if base.strip() == nm:
                                out[k.value] = {"constant": True, "how": f"bound to constant local `{nm}`",
                                                "expr": expr}
    return out


def main() -> None:
    import h5py
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    problems: list[dict] = []
    payload: dict = {}

    # ---- 1. crossgen aux: constants by construction
    aux_pkgs = {}
    for pkg in ("taurex_exobiome", "taurex_exobiome_without_quant"):
        const = hardcoded_aux(pkg)
        aux_pkgs[pkg] = const
        if const:
            problems.append({"where": f"models/{pkg}/dataset.py",
                             "issue": f"{len(const)} of 8 auxiliary features are constants by construction "
                                      f"({sorted(const)}) -> exactly zero after standardisation"})
    payload["crossgen_constant_aux"] = aux_pkgs

    # ---- 2. empirical zero-variance check on the crossgen labels the aux frame is built from
    lab = pd.read_parquet(A.REPO / "data/TauREx set/labels.parquet")
    varying = {c: {"nunique": int(lab[c].nunique()), "min": float(lab[c].min()), "max": float(lab[c].max())}
               for c in lab.columns if lab[c].dtype.kind in "fi"}
    payload["crossgen_label_columns"] = varying

    # ---- 3. which varying label columns does each package actually read?
    readers: dict[str, dict] = {}
    interesting = [c for c, v in varying.items()
                   if v["nunique"] > MIN_NUNIQUE_TO_BE_INFORMATIVE and not c.startswith("log10_vmr")]
    # Threshold and its effect stated in the payload — the previous version filtered at `> 100`
    # silently, so the record gave no way to see that five `present_*` columns never entered the test.
    payload["label_column_filter"] = {
        "min_nunique_to_be_informative": MIN_NUNIQUE_TO_BE_INFORMATIVE,
        "rationale": "a column with a single value cannot carry information, so nobody reading it is "
                     "not a defect; two or more values already discriminate",
        "excluded_as_log10_vmr_target": sorted(c for c in varying if c.startswith("log10_vmr")),
        "excluded_as_uninformative": sorted(
            c for c, v in varying.items()
            if v["nunique"] <= MIN_NUNIQUE_TO_BE_INFORMATIVE and not c.startswith("log10_vmr")),
        "columns_tested": sorted(interesting),
        "previous_threshold_that_was_never_reported": 100,
        "columns_this_change_adds": sorted(
            c for c, v in varying.items()
            if MIN_NUNIQUE_TO_BE_INFORMATIVE < v["nunique"] <= 100 and not c.startswith("log10_vmr")),
    }
    for pkg in ("taurex_exobiome", "taurex_exobiome_without_quant", "ariel_winner_on_taurex", "taurex_fmpe"):
        d = A.REPO / "models" / pkg
        if not d.exists():
            continue
        text = "\n".join(f.read_text() for f in d.glob("*.py"))
        # a column is a real INPUT only if it lands in one of the declared column lists
        declared: set[str] = set()
        for f in d.glob("*.py"):
            try:
                tr = ast.parse(f.read_text())
            except Exception:
                continue
            for st in tr.body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                        and st.targets[0].id.endswith(("AUX_COLUMNS", "TARGET_COLUMNS", "FEATURE_COLUMNS",
                                                       "LABEL_FEATURE_COLUMNS")):
                    try:
                        vals = ast.literal_eval(st.value)
                        declared.update(v for v in vals if isinstance(v, str))
                    except Exception:
                        pass
        readers[pkg] = {}
        for c in interesting:
            readers[pkg][c] = {"mentioned_anywhere": c in text,
                               "declared_as_feature_or_target": c in declared}
    payload["who_reads_which_column"] = readers
    for c in interesting:
        mentions = {p: readers[p][c]["declared_as_feature_or_target"] for p in readers}
        if not any(mentions.values()):
            problems.append({"where": "all TauREx packages",
                             "issue": f"`{c}` varies over [{varying[c]['min']:.4g}, {varying[c]['max']:.4g}] "
                                      f"({varying[c]['nunique']} unique values) and is a declared "
                                      "feature/target in NO package"
                                      + (" (it is mentioned in "
                                         + str([p for p in readers if readers[p][c]['mentioned_anywhere']])
                                         + " but only as a required-column name)"
                                         if any(readers[p][c]['mentioned_anywhere'] for p in readers) else "")})
        elif not all(mentions.values()):
            problems.append({"where": "comparison fairness",
                             "issue": f"`{c}` is a declared feature/target in {[p for p,v in mentions.items() if v]} "
                                      f"but not in {[p for p,v in mentions.items() if not v]} -> asymmetric "
                                      "inputs between compared arms"})

    # ---- 4. scalar-broadcast channels
    #
    # Three defects fixed here at once, all the same kind — assertion instead of measurement:
    #
    # (a) `n = min(2000, ...)` computed `sigma_range_ppm` from the first 2000 of 42,108 rows (95% of
    #     the data skipped), and the cap was NOT in the payload. The sample gave [20.1313, 99.9924];
    #     the whole set gives [20.0005, 99.9999]. `sigma_ppm` is 42k float32 scalars (~165 kB), so
    #     there is nothing to cap: the cap is gone and `n_rows_*` is now in the payload.
    #
    # (b) `"distinct_values_per_row": 1` was HARD-CODED, not measured — one of this check's problems
    #     quoted in the report was a claim read off the source, not a measurement. Now the noise
    #     matrix is built with the package's OWN function and `np.unique(row).size` is counted per row.
    #
    # (c) `problems.append({...218 spectral bins...})` was UNCONDITIONAL while the status is
    #     `"FAIL" if problems else "PASS"` — so PASS was impossible for ANY data. Same pattern as the
    #     tautology fixed in `a15:381`. The problem is now appended only when the measurement confirms
    #     it, and 218 / 444 are read from the package constants rather than from memory.
    sys.path.insert(0, str(A.REPO))
    from models.taurex_exobiome.dataset import _build_taurex_noise_matrix
    from models.taurex_fmpe.constants import AUX_FEATURE_COLS, CONTEXT_DIM, SPECTRAL_LENGTH
    from models.taurex_fmpe.raw_dataset import build_noise_matrix

    with h5py.File(A.REPO / "data/TauREx set/spectra.h5", "r") as f:
        n_rows_available = int(f["sigma_ppm"].shape[0])
        sigma = f["sigma_ppm"][:]                       # all rows, no cap
        n_bins = int(f["transit_depth_noisy"].shape[1])

    def distinct_per_row(mat: np.ndarray) -> dict:
        """How many DISTINCT values the model sees in one row of the noise channel — measured, not assumed."""
        d = np.array([np.unique(row).size for row in mat])
        return {"min": int(d.min()), "max": int(d.max()), "mean": float(d.mean()),
                "rows_with_exactly_one_distinct_value": int((d == 1).sum()),
                "n_rows": int(len(d)), "n_columns": int(mat.shape[1])}

    broadcast = {
        "models/taurex_exobiome/dataset.py::_build_taurex_noise_matrix":
            distinct_per_row(_build_taurex_noise_matrix(sigma, n_bins)),
        "models/taurex_fmpe/raw_dataset.py::build_noise_matrix":
            distinct_per_row(build_noise_matrix(sigma, SPECTRAL_LENGTH)),
    }
    payload["noise_channel"] = {
        "stored_as": "one scalar sigma_ppm per sample",
        "fed_to_model_as": f"np.repeat(sigma, {n_bins}) -> {n_bins} numbers per row",
        "n_rows_measured": int(len(sigma)),
        "n_rows_available": n_rows_available,
        "spectral_bins": n_bins,
        "measured_distinct_values_per_row": broadcast,
        "sigma_range_ppm": [float(sigma.min()), float(sigma.max())],
        "fmpe_context_total": int(CONTEXT_DIM),
        "fmpe_context_breakdown": f"{SPECTRAL_LENGTH} spectrum + {SPECTRAL_LENGTH} noise "
                                  f"+ {len(AUX_FEATURE_COLS)} aux = {CONTEXT_DIM}",
        "fmpe_context_dims_from_the_noise_channel": int(SPECTRAL_LENGTH),
    }
    for where, m in broadcast.items():
        if m["n_columns"] <= 1:
            continue                                   # a single-column channel is not a broadcast
        if m["max"] == 1:
            extra = (f"; for taurex_fmpe that is {SPECTRAL_LENGTH} of {CONTEXT_DIM} context "
                     "dimensions carrying one number" if "fmpe" in where else "")
            problems.append({"where": where,
                             "issue": f"a single scalar is broadcast across {m['n_columns']} spectral bins "
                                      f"— measured {m['rows_with_exactly_one_distinct_value']} of "
                                      f"{m['n_rows']} rows with exactly one distinct value" + extra})
        elif m["rows_with_exactly_one_distinct_value"]:
            problems.append({"where": where,
                             "issue": f"{m['rows_with_exactly_one_distinct_value']} of {m['n_rows']} rows are a "
                                      f"single scalar broadcast across {m['n_columns']} bins (max distinct in any "
                                      f"row: {m['max']}) — degenerate for part of the set, not all of it"})

    # ---- 5. ADC side: is any aux feature constant there?
    aux = pd.read_csv(A.adc_root() / "TrainingData/AuxillaryTable.csv")
    aux = aux.drop(columns=[c for c in aux.columns if c.startswith("Unnamed:")])
    adc_const = [c for c in aux.columns if c != "planet_ID" and aux[c].nunique() <= 1]
    payload["adc_constant_aux"] = adc_const
    if adc_const:
        problems.append({"where": "data/ariel-ml-dataset AuxillaryTable.csv",
                         "issue": f"constant aux columns on ADC: {adc_const}"})

    for p in problems:
        print(f"  [{p['where']}]\n     - {p['issue']}")
    payload["problems"] = problems
    CHECK.emit("FAIL" if problems else "PASS", payload, out=args.out)


if __name__ == "__main__":
    main()
