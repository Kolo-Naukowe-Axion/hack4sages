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
# d01 gets a fixed `--stage 0` rather than heavy=True: stages 1/2 need POSEIDON, which is not in
# .venv-qml (ModuleNotFoundError, verified), and stage 2 additionally needs 72.1 GB of opacities
# (zenodo 16107813) — no variant is fit for a routine run. `--stage 0` is cheap (an import plus one
# parquet), so it runs under `--fast` too and records the environment diagnosis rather than an
# arbitrary stage. Stages 1/2 are run by hand in a scratch venv with POSEIDON
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
        """None = check is run; a string = the reason it was skipped. Reasons go into summary.json (see below)."""
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
    # summary.json is overwritten unconditionally, including under --only
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
