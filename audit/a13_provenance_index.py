"""A13 — Does every published number have a backing artifact?

Proves/disproves findings P6 and U8. This is the core of the repo inventory ("inwentaryzacja"):
for each number that appears in a report, figure script or README, resolve it to
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

# Bylo: lista DOCS byla JEDYNYM zrodlem wiedzy o dokumentach — w checku, ktorego tematem jest
# kompletnosc. Dwa konkretne skutki:
#   (1) 4 z 5 powodow FAIL to byly pliki .tex, ktore ISTNIEJA, tylko w innym worktree
#       (.claude/worktrees/elastic-blackwell-f063b8/docs/) — czyli bookkeeping worktree'ow
#       raportowany jako "missing_documents";
#   (2) odwrotnie: docs/METHODOLOGICAL_AUDIT.md (raport glowny), STATUS.md, TODO_UMOWA.md,
#       HANDOFF_PROMPT.md, LITERATURA.md, DZIENNIK_KONSULTACJI.md sa nietrackowane i byly POZA
#       lista DOCS, wiec raport glowny nie byl objety wlasnym kryterium.
# Teraz dokumenty sa ENUMEROWANE z dysku, a lista DOCS zostaje tylko jako zestaw "musi istniec".
DOC_SCAN_GLOBS = ("docs/**/*.md", "docs/**/*.tex", "reports/**/*.md", "reports/**/*.tex")
# Wykluczenia jawne, bo kazde jest decyzja, nie przeoczeniem:
#   reports/audit/ = wlasne rekordy tego harnessu, nadpisywane co przebieg — to zapis pomiaru,
#   nie dokument stawiajacy teze; objecie ich kryterium "tracked" znaczyloby, ze check nie moze
#   przejsc nigdy po wlasnym uruchomieniu.
DOC_SCAN_EXCLUDE_PREFIXES = ("reports/audit/",)


def tracked(rel: str, root: Path | None = None) -> bool:
    r = subprocess.run(["git", "-C", str(root or A.REPO), "ls-files", "--error-unmatch", rel],
                       capture_output=True, text=True)
    return r.returncode == 0


def worktree_roots() -> list[Path]:
    """A.REPO na pierwszym miejscu, potem pozostale worktree tego repo.

    Sciezki dokumentow rozwiazujemy JAWNIE wobec A.REPO (glowny checkout). Gdy plik tam nie
    istnieje, szukamy go w pozostalych worktree i to raportujemy — inaczej "missing" myli brak
    dokumentu z tym, ze lezy w innej galezi roboczej.
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
    """Wszystkie dokumenty z dysku, z A.REPO ORAZ z worktree, w ktorym lezy ten kod audytu.

    Drugi root jest konieczny: raport glowny (docs/METHODOLOGICAL_AUDIT.md) istnieje wylacznie
    tam i bez tego nie byl objety kryterium wlasnego checku.
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
        # Bylo: have["weights"] zbierane i nigdy nieuzyte. Skutek: noquant_poseidon dostawal
        # backed_full przy weights=False, a a14 zglasza DOKLADNIE ten plik
        # (reports/taurex_noquant_.../best_model_epoch059.pt) jako brakujaca sciezke -> dwa checki
        # mowily o tym samym pliku dwie rozne rzeczy. Bez wag nie da sie powtorzyc forward passa,
        # wiec liczba jest odtwarzalna tylko z zapisanego CSV, nie z modelu.
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

    # Lista DOCS zostaje jako zbior "ten dokument MUSI istniec", ale sciezka jest rozwiazywana
    # jawnie wobec A.REPO, a gdy pliku tam nie ma — szukamy go w pozostalych worktree.
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
               # WYBOR: backed_summary WCHODZI do FAIL. Uzasadnienie: criterion= mowi "every result
               # number is recomputable from committed predictions", a backed_summary znaczy dokladnie,
               # ze liczba jest PRZEPISANA z JSON-a i nieodtwarzalna. Bez tego, po domknieciu 4
               # UNBACKED, check dawalby PASS majac dwie nieprzeliczalne liczby na figurze 1
               # (sota_holdout_median 0,5523 i cnn_holdout 0,6500) — czyli PASS przeczacy wlasnemu
               # kryterium. Alternatywa (zlagodzenie criterion= do "ma jakikolwiek artefakt") zostala
               # odrzucona, bo oslabia teze P6, ktora wlasnie o odtwarzalnosc jest.
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
