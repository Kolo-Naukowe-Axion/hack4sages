"""A14 — Do the documented entrypoints actually run?

Proves/disproves finding U6. Seven files import `models.ariel_quantum_regression`, a module renamed
to `taurex_exobiome`, so the documented CLI for the flagship model and the whole IQM Garnet port are
non-executable as committed. `scripts/analyze_quantum_residual_evidence.py` loads a checkpoint that
is not in the repo, so the bootstrap evidence cannot be reproduced from this checkout.

Static (AST import extraction + file existence) — safe, no code executed.

PASS criterion: every module referenced by an entrypoint or test resolves, and every file path
hard-coded in an analysis script exists.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a14_importability",
    finding="U6 — flagship CLI entrypoints and the Garnet port import a non-existent module; the evidence script loads a missing checkpoint",
    question="Do all documented entrypoints, tests and analysis scripts resolve their imports and file paths?",
    criterion="all referenced modules resolve and all hard-coded artifact paths exist",
)

SCAN_DIRS = ["models", "scripts", "tests"]

ARTIFACT_SUFFIXES = (".pt", ".csv", ".h5", ".hdf5", ".parquet", ".npz", ".json", ".pkl", ".pth")
# Stringi z tymi znakami to szablony/teksty pomocy, nie sciezki (np. help="default:
# <run-dir>/best_model_by_mrmse.pt"). Wykluczenie jest jawne, bo inaczej check zglasza
# nieistniejaca sciezke tam, gdzie autor nigdy jej nie podal.
PLACEHOLDER_CHARS = "<>*?{}%$"
# Korzenie, wobec ktorych rozwiazujemy stala sciezke. "" = sam A.REPO. Pozostale to korzenie
# danych, wobec ktorych pakiety skladaja swoje stale w runtime
# (models/ariel_winner_trace_nf/constants.py:60-66 trzyma "TrainingData/Ground Truth Package/..."
# i dokleja DEFAULT_DATA_ROOT). Bez tego 7 poprawnych stalych wyszloby jako zepsute sciezki.
ARTIFACT_PATH_ROOTS = ["", "data/ariel-ml-dataset", "data/full-ariel", "data/TauREx set"]


def module_exists(dotted: str) -> bool:
    parts = dotted.split(".")
    if parts[0] not in {"models", "scripts", "tests", "data"}:
        return True  # third-party / stdlib: out of scope here
    p = A.REPO.joinpath(*parts)
    return p.with_suffix(".py").exists() or (p / "__init__.py").exists() or p.is_dir()


def resolve_artifact_path(rel: str) -> str | None:
    """Pierwszy korzen z ARTIFACT_PATH_ROOTS, wobec ktorego `rel` istnieje; None = nigdzie."""
    if rel.startswith("/"):
        return "<absolute>" if Path(rel).exists() else None
    for root in ARTIFACT_PATH_ROOTS:
        base = A.REPO / root if root else A.REPO
        if (base / rel).exists():
            return root or "<repo>"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    broken_imports, broken_paths = [], []
    path_consts: dict[str, dict] = {}
    scanned = 0
    for d in SCAN_DIRS:
        for f in sorted((A.REPO / d).rglob("*.py")):
            if "archive/" in str(f) or ".venv" in str(f):
                continue
            scanned += 1
            try:
                tree = ast.parse(f.read_text())
            except Exception as exc:
                broken_imports.append({"file": str(f.relative_to(A.REPO)), "error": f"parse: {exc}"})
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    mods = [node.module]
                for m in mods:
                    if m.startswith(("models.", "scripts.", "tests.")) and not module_exists(m):
                        broken_imports.append({"file": str(f.relative_to(A.REPO)), "line": node.lineno,
                                               "missing_module": m})
                # Hard-coded artifact paths as string constants.
                # BYLO: martwy kod. Jedyna galezia bylo `... and "/" not in v ...: continue`, a
                # `continue` bylo OSTATNIA instrukcja ciala petli — czyli zadna sciezka nigdy nie
                # trafiala do broken_paths. Druga polowa criterion= ("all hard-coded artifact paths
                # exist") byla realizowana WYLACZNIE 6-elementowa lista reczna nizej.
                # JEST: zbieramy stale zawierajace "/" i sufiks artefaktu i sprawdzamy istnienie.
                # Nazwy bez "/" zostaja pominiete swiadomie (skladane w runtime z katalogiem), ale
                # teraz sa policzone w payloadzie, zeby pominiecie bylo widoczne.
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    v = node.value
                    if not v.endswith(ARTIFACT_SUFFIXES) or len(v) <= 8:
                        continue
                    where = f"{f.relative_to(A.REPO)}:{node.lineno}"
                    if any(ch in v for ch in PLACEHOLDER_CHARS):
                        kind = "placeholder_or_help_text"
                    elif "/" not in v:
                        kind = "bare_filename_resolved_at_runtime"
                    else:
                        kind = "path"
                    e = path_consts.setdefault(v, {"value": v, "kind": kind, "referenced_at": []})
                    e["referenced_at"].append(where)

    for v, e in sorted(path_consts.items()):
        if e["kind"] != "path":
            continue
        root = resolve_artifact_path(v)
        e["resolved_against"] = root
        e["exists"] = root is not None
        if root is None:
            broken_paths.append({"path": v, "referenced_at": e["referenced_at"],
                                 "roots_tried": ARTIFACT_PATH_ROOTS,
                                 "why_it_matters": "stala sciezka artefaktu w kodzie nie istnieje "
                                                   "pod zadnym ze sprawdzanych korzeni"})

    # explicit spot-checks that matter for reproducing published evidence
    for rel, why in (
        ("reports/taurex_noquant_taurex_snapshot_20260312_133054/best_model_epoch059.pt",
         "loaded by scripts/analyze_quantum_residual_evidence.py -> bootstrap evidence not reproducible"),
        ("reports/quantum_residual_generalization_analysis",
         "output dir of the same script: never produced"),
        ("models/12q_taurex_exobiome", "documented in docs/12q_taurex_exobiome_architecture.md and tested"),
        ("outputs/fair_small_experiment_cpu", "location claimed by README.md for the CPU sweep"),
        ("data/full-ariel", "data_root of the committed mac re-eval (mac_run_summary.json)"),
        ("data/val_dataset/manifest.json", "split manifest referenced by every reeval manifest"),
    ):
        if not (A.REPO / rel).exists():
            broken_paths.append({"path": rel, "source": "manual spot-check list", "why_it_matters": why})

    kinds: dict[str, int] = {}
    for e in path_consts.values():
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    payload = {"files_scanned": scanned, "broken_imports": broken_imports, "missing_paths": broken_paths,
               "n_broken_imports": len(broken_imports), "n_missing_paths": len(broken_paths),
               "hardcoded_artifact_constants": sorted(path_consts.values(), key=lambda e: e["value"]),
               "n_hardcoded_artifact_constants": len(path_consts),
               "hardcoded_constant_kinds": kinds,
               "artifact_path_roots_tried": ARTIFACT_PATH_ROOTS}
    for b in broken_imports:
        print(f"  IMPORT  {b.get('file')}:{b.get('line')} -> {b.get('missing_module', b.get('error'))}")
    for b in broken_paths:
        print(f"  PATH    {b['path']}  ({b['why_it_matters']})")
    print(f"  stale sciezkowe: {len(path_consts)} lacznie {kinds}; "
          f"{sum(1 for e in path_consts.values() if e.get('exists') is False)} nierozwiazywalnych")
    CHECK.emit("FAIL" if broken_imports or broken_paths else "PASS", payload, out=args.out)


if __name__ == "__main__":
    main()
