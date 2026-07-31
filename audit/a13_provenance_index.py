"""A13 — Does every published number have a backing artifact?

Proves/disproves findings P6 and U8. This is the core of the repo inventory: for each number
that appears in a report, figure script or README, resolve it to
(config + seed + weights or predictions) or mark it unbacked.

Rules applied:
  backed_full      -> predictions CSV present, and recomputing from it reproduces the number
  backed_summary   -> only a metrics JSON present (number is transcribed, not re-derivable).
                      Since 2026-07-28 this is a FAIL reason, not a pass: the criterion below says
                      "recomputable", and a transcribed number is not.
  unbacked         -> no artifact at all
  untracked_doc    -> the document asserting it is not under version control

PASS criterion: every number presented as a result is backed_full, and every document asserting a
result is tracked in git.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a13_provenance_index",
    finding="P6 / U8 — several headline numbers have no backing artifact; the verification doc and plan-of-record are not in git",
    question="Can each published number be re-derived from a committed artifact?",
    criterion="every result number is recomputable from committed predictions, and its document is tracked",
)

CLAIMS = [
    {"id": "exobiome_holdout", "value": 0.2993761897087097,
     "metrics": "artifacts/ariel_quantum_best_v4_epoch6/holdout_metrics.json",
     "predictions": "artifacts/ariel_quantum_best_v4_epoch6/holdout_predictions.csv",
     "weights": "artifacts/ariel_quantum_best_v4_epoch6/best_model.pt", "cited_in": "README.md"},
    {"id": "exobiome_val", "value": 0.29361358284950256,
     "metrics": "artifacts/ariel_quantum_best_v4_epoch6/validation_metrics.json",
     "predictions": "artifacts/ariel_quantum_best_v4_epoch6/validation_predictions.csv",
     "weights": "artifacts/ariel_quantum_best_v4_epoch6/best_model.pt", "cited_in": "README.md"},
    {"id": "exobiome_mac_holdout", "value": 0.29869264364242554,
     "metrics": "artifacts/ariel_quantum_best_v4_epoch6/mac_eval_20260312/mac_holdout_metrics.json",
     "predictions": "artifacts/ariel_quantum_best_v4_epoch6/mac_eval_20260312/mac_holdout_predictions.csv",
     "weights": "artifacts/ariel_quantum_best_v4_epoch6/best_model.pt",
     "cited_in": "reports/model_comparison/rmse/exobiome_metrics.json + figure 1"},
    {"id": "sota_holdout_median", "value": 0.5522884130477905,
     "metrics": "models/adc_winner_on_ariel/trained_run/holdout_metrics.json",
     "predictions": None, "weights": "models/adc_winner_on_ariel/trained_run/best_model_by_mrmse.pt",
     "cited_in": "reports/model_comparison/rmse/sota_metrics.json + figure 1"},
    {"id": "cnn_holdout", "value": 0.65003745144915,
     "metrics": "models/adc_baseline/cnn_metrics.json",
     "predictions": None, "weights": "models/adc_baseline/cnn_whole_ariel_new.weights.h5",
     "cited_in": ("figure 1; identical JSON also at reports/model_comparison/rmse/cnn_metrics.json. "
                  "This is the OFFICIAL ADC2023 baseline (ucl-exoplanets/ADC2023-baseline), vendored "
                  "with code + weights + notebook -> re-derivable, needs the TF env")},
    {"id": "random_forest_holdout", "value": 0.75724,
     "metrics": "reports/model_comparison/rmse/random_forest_metrics.json",
     "predictions": None, "weights": None, "cited_in": "figure 1 (removed 2026-07-23)"},
    {"id": "quantum_poseidon", "value": 3.215615,
     "metrics": "reports/ariel_quantum_taurex_snapshot_20260312_1003/poseidon_holdout_metrics.json",
     "predictions": "reports/ariel_quantum_taurex_snapshot_20260312_1003/poseidon_holdout_predictions.csv",
     "weights": "reports/ariel_quantum_taurex_snapshot_20260312_1003/stage2_best_model_epoch005.pt",
     "cited_in": "reports/taurex_model_comparison.md"},
    {"id": "noquant_poseidon", "value": 3.279559,
     "metrics": "reports/taurex_noquant_taurex_snapshot_20260312_133054/poseidon_holdout_metrics.json",
     "predictions": "reports/taurex_noquant_taurex_snapshot_20260312_133054/poseidon_holdout_predictions.csv",
     "weights": "reports/taurex_noquant_taurex_snapshot_20260312_133054/best_model_epoch059.pt",
     "cited_in": "reports/taurex_model_comparison.md"},
    {"id": "winner_on_taurex_poseidon", "value": 3.453121, "metrics": None, "predictions": None,
     "weights": None, "cited_in": "reports/taurex_model_comparison.md (1 of 3 rows)"},
    {"id": "h200_poseidon", "value": 2.894607, "metrics": None, "predictions": None, "weights": None,
     "cited_in": "reports/taurex_noquant_h200_...md (excluded from ranking)"},
    {"id": "garnet_hardware", "value": 0.278756, "metrics": None, "predictions": None, "weights": None,
     "cited_in": "scripts/presentation/render_scientific_figures.py figure 4"},
]

DOCS = ["docs/VERIFICATION.md", "docs/publication_plan.tex", "docs/exobiome_context.md",
        "docs/delta_margin_methodology.tex", "docs/raport_lipiec_szkic.tex",
        "reports/taurex_model_comparison.md", "README.md"]

# Previously: the DOCS list was the ONLY source of knowledge about documents — in a check whose
# subject is completeness. Two concrete effects:
#   (1) 4 of 5 FAIL reasons were files that DO EXIST (3x .tex + docs/exobiome_context.md), just in
#       another worktree (.claude/worktrees/elastic-blackwell-f063b8/docs/) — i.e. worktree
#       bookkeeping reported as "missing_documents";
#   (2) conversely, docs/METHODOLOGICAL_AUDIT.md (the main report) and docs/LITERATURA.md sat
#       outside DOCS, so the check's own criterion never covered them; LITERATURA.md is still
#       untracked, METHODOLOGICAL_AUDIT.md is now in git.
# Now documents are ENUMERATED from disk, and DOCS remains only as a "must exist" set.
DOC_SCAN_GLOBS = ("docs/**/*.md", "docs/**/*.tex", "reports/**/*.md", "reports/**/*.tex")
# Exclusions are explicit, because each one is a decision rather than an oversight:
#   reports/audit/ = this harness's own records, overwritten every run — a measurement log, not a
#   document asserting a claim; holding them to the "tracked" criterion would mean the check can
#   never pass after running itself.
DOC_SCAN_EXCLUDE_PREFIXES = ("reports/audit/",)


def tracked(rel: str, root: Path | None = None) -> bool:
    r = subprocess.run(["git", "-C", str(root or A.REPO), "ls-files", "--error-unmatch", rel],
                       capture_output=True, text=True)
    return r.returncode == 0


def worktree_roots() -> list[Path]:
    """A.REPO first, then the repo's other worktrees.

    Document paths are resolved EXPLICITLY against A.REPO (the main checkout). When a file is
    absent there, we look for it in the other worktrees and report that — otherwise "missing"
    would conflate an absent document with one that merely lives on a different working branch.
    """
    roots = [A.REPO]
    r = subprocess.run(["git", "-C", str(A.REPO), "worktree", "list", "--porcelain"],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("worktree "):
            p = Path(line[len("worktree "):]).resolve()
            if p not in roots:
                roots.append(p)
    return roots


def enumerate_docs(roots: list[Path]) -> list[dict]:
    """All documents found on disk, from A.REPO AND from the worktree this audit code lives in.

    The second root exists for the case where this file runs from a worktree other than the
    main checkout: some report may only exist there, and without scanning it the criterion would
    not cover the audit's own top-level report. When run from the main checkout (as here), the
    two roots coincide and this is a no-op.
    """
    audit_root = A.AUDIT_DIR.parent.resolve()
    scan = [A.REPO] + ([audit_root] if audit_root != A.REPO.resolve() else [])
    rows, seen = [], set()
    for root in scan:
        for g in DOC_SCAN_GLOBS:
            for p in sorted(root.glob(g)):
                if not p.is_file():
                    continue
                rel = p.relative_to(root).as_posix()
                if rel.startswith(DOC_SCAN_EXCLUDE_PREFIXES) or p.name.startswith("."):
                    continue
                key = (str(root), rel)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"doc": rel, "exists": True,
                             "tracked_in_git": tracked(rel, root),
                             "resolved_root": str(root),
                             "is_main_checkout": root.resolve() == A.REPO.resolve(),
                             "source": "enumerated"})
    return rows


def recompute(pred_rel: str) -> float | None:
    import pandas as pd
    p = A.REPO / pred_rel
    if not p.exists():
        return None
    df = pd.read_csv(p)
    tset = A.TARGETS if f"true_{A.TARGETS[0]}" in df.columns else A.CROSSGEN_TARGETS
    try:
        yt = df[[f"true_{t}" for t in tset]].to_numpy(float)
        yp = df[[f"pred_{t}" for t in tset]].to_numpy(float)
    except KeyError:
        return None
    return A.mrmse(yt, yp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, unbacked, summary_only, weights_absent = [], [], [], []
    print(f"{'claim':28} {'value':>10} {'recomputed':>11} {'status':16} artifacts")
    print("-" * 104)
    for c in CLAIMS:
        have = {k: (bool((A.REPO / c[k]).exists()) if c.get(k) else False)
                for k in ("metrics", "predictions", "weights")}
        rec = recompute(c["predictions"]) if have["predictions"] else None
        if rec is not None and abs(rec - c["value"]) < args.tol:
            status = "backed_full"
        elif rec is not None:
            status = "MISMATCH"
        elif have["metrics"]:
            status = "backed_summary"
        else:
            status = "UNBACKED"
        if status in ("UNBACKED", "MISMATCH"):
            unbacked.append(c["id"])
        if status == "backed_summary":
            summary_only.append(c["id"])
        # Previously: have["weights"] was collected and never used. Effect: noquant_poseidon got
        # backed_full with weights=False, while a14 reports EXACTLY that file
        # (reports/taurex_noquant_.../best_model_epoch059.pt) as a missing path -> two checks said
        # two different things about one file. Without the weights the forward pass cannot be
        # repeated, so the number is reproducible only from the stored CSV, not from the model.
        if c.get("weights") and not have["weights"]:
            weights_absent.append({"id": c["id"], "weights": c["weights"],
                                   "also_reported_by": "a14_importability.missing_paths"})
        rows.append({**c, "artifacts_present": have, "recomputed": rec, "status": status,
                     "delta": (rec - c["value"]) if rec is not None else None})
        print(f"{c['id']:28} {c['value']:10.6f} "
              f"{(f'{rec:11.6f}' if rec is not None else '          -')} {status:16} "
              f"{''.join(k[0].upper() if v else '-' for k, v in have.items())}")

    roots = worktree_roots()
    doc_rows = enumerate_docs(roots)
    enumerated_keys = {(d["resolved_root"], d["doc"]) for d in doc_rows}

    # DOCS stays as the "this document MUST exist" set, but each path is resolved explicitly
    # against A.REPO; when the file is not there, we look for it in the other worktrees.
    elsewhere = []
    for d in DOCS:
        p = A.REPO / d
        if p.exists():
            if (str(A.REPO), d) not in enumerated_keys:
                doc_rows.append({"doc": d, "exists": True, "tracked_in_git": tracked(d),
                                 "resolved_root": str(A.REPO), "is_main_checkout": True,
                                 "source": "DOCS (reczna lista)"})
            continue
        hits = [{"worktree": str(r), "tracked_in_that_worktree": tracked(d, r)}
                for r in roots[1:] if (r / d).exists()]
        if hits:
            elsewhere.append({"doc": d, "found_in": hits})
            for h in hits:
                if (h["worktree"], d) not in enumerated_keys:
                    doc_rows.append({"doc": d, "exists": True,
                                     "tracked_in_git": h["tracked_in_that_worktree"],
                                     "resolved_root": h["worktree"], "is_main_checkout": False,
                                     "source": "DOCS (reczna lista), znaleziony w innym worktree"})
        else:
            doc_rows.append({"doc": d, "exists": False, "tracked_in_git": False,
                             "resolved_root": str(A.REPO), "is_main_checkout": True,
                             "source": "DOCS (reczna lista)"})

    untracked = sorted({f"{d['doc']} @ {d['resolved_root']}"
                        for d in doc_rows if d["exists"] and not d["tracked_in_git"]})
    missing = sorted({d["doc"] for d in doc_rows if not d["exists"]})
    print("\n  documents:")
    for d in sorted(doc_rows, key=lambda x: (not x["is_main_checkout"], x["doc"])):
        mark = "" if d["is_main_checkout"] else "  [inny worktree]"
        print(f"    {d['doc']:52} exists={d['exists']!s:5} tracked={d['tracked_in_git']}{mark}")
    if elsewhere:
        print("\n  w A.REPO nie ma, ale istnieja w innym worktree (to NIE jest brak dokumentu):")
        for e in elsewhere:
            print(f"    {e['doc']:52} -> {e['found_in'][0]['worktree']}")

    payload = {"claims": rows, "unbacked_or_mismatched": unbacked,
               # DECISION: backed_summary DOES count towards FAIL, because criterion= says
               # "recomputable from committed predictions" and backed_summary is a number
               # TRANSCRIBED from a JSON. Without this the check would PASS while figure 1 carries
               # two non-recomputable numbers (sota_holdout_median 0.5523, cnn_holdout 0.6500).
               # Relaxing criterion= to "has any artifact at all" was rejected — it would weaken
               # P6, which is precisely about reproducibility.
               "backed_summary_not_recomputable": summary_only,
               "declared_weights_absent": weights_absent,
               "documents": doc_rows, "n_documents": len(doc_rows),
               "document_scan_globs": list(DOC_SCAN_GLOBS),
               "document_scan_excluded_prefixes": list(DOC_SCAN_EXCLUDE_PREFIXES),
               "worktree_roots": [str(r) for r in roots],
               "untracked_documents": untracked, "missing_documents": missing,
               "documents_absent_in_main_checkout_present_in_other_worktree": elsewhere,
               "legend": "artifacts column: M=metrics json, P=predictions csv, W=weights",
               "note": ("backed_summary means the number was transcribed from a summary JSON and cannot be "
                        "re-derived without a fresh forward pass. For a publication every result must be "
                        "backed_full — and since 2026-07-28 backed_summary is a FAIL reason, not a pass.")}
    print(f"\n  {len(unbacked)} UNBACKED/MISMATCH, {len(summary_only)} backed_summary (nieodtwarzalne), "
          f"{len(weights_absent)} z brakiem zadeklarowanych wag, {len(untracked)} nietrackowanych "
          f"dokumentow, {len(missing)} brakujacych")
    CHECK.emit("FAIL" if unbacked or summary_only or weights_absent or untracked or missing else "PASS",
               payload, out=args.out)


if __name__ == "__main__":
    main()
