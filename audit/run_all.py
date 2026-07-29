"""Run the whole audit suite and assemble the inventory table.

    cd /Users/mariaplatek/projects/AXION/hack4sages
    ./.venv-qml/bin/python .claude/worktrees/<wt>/audit/run_all.py            # everything
    ./.venv-qml/bin/python .claude/worktrees/<wt>/audit/run_all.py --fast     # skip model forward passes
    ./.venv-qml/bin/python .claude/worktrees/<wt>/audit/run_all.py --only a01,a02

Writes reports/audit/<UTC-date>/{summary.json,summary.md}. Those JSON records are the ONLY files
the harness creates; redirect them with --out or EXOBIOME_AUDIT_OUT if you want them elsewhere.

The harness is read-only with respect to the repo: git is only ever queried (`rev-parse`,
`status --porcelain`, `ls-files`), and no model, data or report file is modified.

Exit code 1 if any check FAILs, so it can gate a commit or a report build.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_lib as A  # noqa: E402

# (script, needs a model forward pass / heavy, fixed args always passed to this check)
#
# a27 i d01 byly POZA suita. To nie byla drobnostka: a27 podpiera A0.2, na ktorym stoja K3, K4 i K9,
# wiec trzy ustalenia krytyczne opieraly sie na checku, ktory nigdy nie przeszedl przez bramke
# exit-code i istnial tylko jako pojedynczy przebieg reczny (reports/audit/20260727/, 11:45, czyli
# JUZ PO summary.json z 10:45 — dokladnie ten mechanizm, ktory zamknieto w run_all:73-84).
#
# d01 dostaje na stale `--stage 0`, a nie heavy=True, i to jest wybor:
#   * stage 1 i 2 wymagaja importowalnego POSEIDON-a, ktorego w .venv-qml NIE MA (sprawdzone:
#     ModuleNotFoundError). Przy `--stage auto` check i tak konczy na stage 0 ze statusem INFO,
#     ale wynik zalezy wtedy od tego, co akurat jest zainstalowane — czyli suita przestaje byc
#     powtarzalna miedzy maszynami;
#   * stage 2 dodatkowo wymaga 72,1 GB opacities (zenodo 16107813), wiec nie moze byc czescia
#     rutynowego biegu w zadnym srodowisku;
#   * `--stage 0` jest tani (import + jeden parquet), nie laduje modelu, wiec przechodzi tez przy
#     `--fast`; suita rejestruje wtedy DIAGNOZE SRODOWISKA, a nie przypadkowy poziom stage'a.
# Stage 1 i 2 uruchamia sie recznie w scratch-venvie z POSEIDON-em; d01 zwraca INFO, wiec nie
# wplywa na exit code w zadnym wariancie.
SUITE = [
    ("a01_spectral_variation.py", False, ()),
    ("a02_trivial_baseline.py", False, ()),
    ("a03_input_convention.py", True, ()),
    ("a04_quantum_scale_provenance.py", True, ()),
    ("a05_training_completeness.py", False, ()),
    ("a06_param_accounting.py", True, ()),
    ("a07_gate_dynamics.py", True, ()),
    ("a08_reference_posterior.py", False, ()),
    ("a09_noise_realization.py", False, ()),
    ("a10_split_integrity.py", False, ()),
    ("a11_pairing_audit.py", False, ()),
    ("a12_significance_power.py", False, ()),
    ("a13_provenance_index.py", False, ()),
    ("a14_importability.py", False, ()),
    ("a15_target_completeness.py", True, ()),
    ("a21_dead_features.py", False, ()),
    ("a24_official_metrics.py", True, ()),
    ("a26_baseline_ladder.py", True, ()),
    ("a27_pipeline_fidelity.py", True, ()),
    ("a29_smoke_baseline_recovery.py", False, ()),
    ("d01_poseidon_diagnosis.py", False, ("--stage", "0")),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip checks that load models")
    ap.add_argument("--only", default=None, help="comma list of check prefixes, e.g. a01,a04")
    ap.add_argument("--out", default=None)
    ap.add_argument("--extra", default="", help="extra args forwarded to every check")
    args = ap.parse_args()

    out = A.out_dir(args.out)

    def wanted(s: str, heavy: bool) -> str | None:
        """None = brany; string = powod pominiecia. Powody ida do summary.json (patrz nizej)."""
        if args.fast and heavy:
            return "--fast: check laduje model"
        if args.only and not any(s.startswith(p.strip()) for p in args.only.split(",")):
            return f"--only {args.only}: prefiks nie pasuje"
        return None

    picked, skipped = [], []
    for s, heavy, fixed in SUITE:
        why = wanted(s, heavy)
        if why is None:
            picked.append((s, fixed))
        else:
            skipped.append({"check": s[:-3], "reason": why})

    run_started_utc = datetime.now(timezone.utc).isoformat()
    results = []
    for script, fixed in picked:
        print("=" * 100)
        print(f"RUN {script}" + (f" {' '.join(fixed)}" if fixed else ""))
        print("=" * 100)
        cmd = [sys.executable, str(HERE / script), "--out", str(out)] + list(fixed) + \
              ([a for a in args.extra.split(" ") if a] if args.extra else [])
        proc = subprocess.run(cmd, cwd=str(A.REPO))
        name = script[:-3]
        jf = out / f"{name}.json"
        # Rekord liczy sie jako wynik TEGO biegu tylko wtedy, gdy check wyszedl z zerem I zapisal
        # rekord PO starcie biegu. Bez tych dwoch warunkow padniety check byl raportowany jako
        # sukces na podstawie STAREGO rekordu lezacego w --out. Zweryfikowane empirycznie:
        # podrzucony rekord PASS z 2020-01-01 + a06 padniete na argparse dawalo
        # "PASS a06_param_accounting", counts {PASS: 1, ERROR: 0}, exit 0.
        # Tak samo powstaly re-runy a12/a13/a27 do katalogu 20260727 juz po summary.json.
        stale = None
        if jf.exists():
            rec = json.loads(jf.read_text())
            rec_ts = rec.get("timestamp_utc", "")
            if proc.returncode != 0:
                stale = f"check zwrocil returncode={proc.returncode}"
            elif rec_ts <= run_started_utc:
                stale = (f"rekord nie zostal nadpisany w tym biegu "
                         f"(timestamp {rec_ts or 'BRAK'} <= start {run_started_utc})")
        if jf.exists() and stale is None:
            rec = json.loads(jf.read_text())
            results.append({"check": name, "status": rec["status"], "finding": rec["finding"],
                            "question": rec["question"], "criterion": rec["pass_criterion"],
                            "timestamp_utc": rec.get("timestamp_utc"),
                            "json": str(jf.relative_to(A.REPO)) if str(jf).startswith(str(A.REPO)) else str(jf)})
        elif jf.exists():
            results.append({"check": name, "status": "ERROR",
                            "finding": f"rekord odrzucony jako nieaktualny: {stale}",
                            "question": "", "criterion": "", "json": str(jf),
                            "returncode": proc.returncode,
                            "stale_record_timestamp": rec.get("timestamp_utc")})
        else:
            results.append({"check": name, "status": "ERROR", "finding": "check did not emit a record",
                            "question": "", "criterion": "", "json": None,
                            "returncode": proc.returncode})

    results.sort(key=lambda r: (A.STATUS_ORDER.get(r["status"], 9), r["check"]))
    # summary.json jest nadpisywane bezwarunkowo, takze przy --only. Wczesniej nie bylo w nim ANI
    # SLOWA o tym, ze bieg byl czesciowy — plik z jednym checkiem wygladal identycznie jak plik z
    # pelnej suity, tylko krocej. Teraz `invocation` mowi, co uruchomiono, `skipped` — czego nie i
    # dlaczego, a `suite_size` daje mianownik.
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "git_revision": A.git_revision(), "repo": str(A.REPO), "env": A.env_versions(),
               "invocation": {"fast": bool(args.fast), "only": args.only, "extra": args.extra,
                              "argv": sys.argv[1:]},
               "suite_size": len(SUITE), "n_picked": len(picked), "n_skipped": len(skipped),
               "is_partial_run": bool(skipped),
               "picked": [{"check": s[:-3], "fixed_args": list(f)} for s, f in picked],
               "skipped": skipped,
               "counts": {s: sum(1 for r in results if r["status"] == s)
                          for s in ("FAIL", "WARN", "INFO", "PASS", "ERROR")},
               "checks": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = ["# ExoBiome audit — inventory summary", "",
          f"- generated: `{summary['generated_utc']}`",
          f"- revision: `{summary['git_revision']}`",
          f"- counts: " + ", ".join(f"**{k}** {v}" for k, v in summary["counts"].items() if v),
          f"- uruchomiono {len(picked)} z {len(SUITE)} checkow"
          + (f", pominieto {len(skipped)}: " + ", ".join(f"`{s['check']}` ({s['reason']})"
                                                         for s in skipped)
             if skipped else " (pelna suita)"),
          "", "| status | check | finding | pass criterion | record |", "|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['status']} | `{r['check']}` | {r['finding']} | {r['criterion']} | "
                  f"{'`' + r['json'] + '`' if r['json'] else '—'} |")
    md += ["", "Re-run any single check with:", "",
           "```bash", "./.venv-qml/bin/python audit/<check>.py", "```", ""]
    (out / "summary.md").write_text("\n".join(md) + "\n")

    print("\n" + "=" * 100)
    for r in results:
        print(f"  {r['status']:6} {r['check']}")
    print(f"\nwrote {out/'summary.json'} and {out/'summary.md'}")
    sys.exit(1 if summary["counts"]["FAIL"] or summary["counts"]["ERROR"] else 0)


if __name__ == "__main__":
    main()
