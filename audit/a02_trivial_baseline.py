"""A02 — Is every reported number better than a constant predictor?

Proves/disproves finding K2.
For every (dataset, reported mRMSE) pair the check computes:
    baseline = mRMSE of the TRAIN-SPLIT mean vector, evaluated on that evaluation set
    skill    = 1 - reported/baseline
PASS criterion: skill > 0 for every reported number that is presented as a model result.

The baseline that decides the verdict is deliberately the *honest* constant: a single vector that a
pipeline could have emitted knowing only the training split. The oracle variant — the per-column
mean of the evaluation rows themselves, i.e. the best constant achievable on the very set being
scored — is reported alongside as `baseline_oracle_constant_mrmse` FOR INFORMATION ONLY and never
enters the status, because it reads the evaluation targets.

`--reported` accepts a JSON file (list of {model, value, set, source}) so the check can be re-run
over whatever the paper claims; the schema is validated before any data is loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a02_trivial_baseline",
    finding="K2 — every cross-generator model is worse than a constant predictor; no table reports a baseline",
    question="For each reported mRMSE, what is the trivial-baseline mRMSE on the same rows, and is the skill positive?",
    criterion="skill = 1 - reported/(train-mean baseline) > 0 for every number presented as a model result",
)

# Numbers as published, with the evaluation set each one is computed on.
DEFAULT_REPORTED = [
    {"model": "exobiome hybrid (scale 1.0)", "value": 0.2993761897087097, "set": "adc_holdout",
     "source": "artifacts/ariel_quantum_best_v4_epoch6/holdout_metrics.json"},
    {"model": "exobiome hybrid (val)", "value": 0.29361358284950256, "set": "adc_validation",
     "source": "artifacts/ariel_quantum_best_v4_epoch6/validation_metrics.json"},
    {"model": "winner-style NSF (median, +noise)", "value": 0.5522884130477905, "set": "adc_holdout",
     "source": "models/adc_winner_on_ariel/trained_run/holdout_metrics.json"},
    {"model": "CNN baseline", "value": 0.65003745144915, "set": "adc_holdout",
     "source": "reports/model_comparison/rmse/cnn_metrics.json (NO backing artifact)"},
    {"model": "quantum snapshot", "value": 3.215615, "set": "poseidon_test",
     "source": "reports/ariel_quantum_taurex_snapshot_20260312_1003/poseidon_holdout_metrics.json"},
    {"model": "noquant snapshot", "value": 3.279559, "set": "poseidon_test",
     "source": "reports/taurex_noquant_taurex_snapshot_20260312_133054/poseidon_holdout_metrics.json"},
    {"model": "winner on TauREx", "value": 3.453121, "set": "poseidon_test",
     "source": "reports/ariel_winner_on_taurex_20260312_112940_results_summary.md (NO backing artifact)"},
    {"model": "H200 noquant (excluded as 'underfit')", "value": 2.894607, "set": "poseidon_test",
     "source": "reports/taurex_noquant_h200_...md (NO backing artifact)"},
    {"model": "quantum snapshot", "value": 1.449002, "set": "taurex_val",
     "source": "reports/taurex_model_comparison.md"},
    {"model": "noquant snapshot", "value": 1.423032, "set": "taurex_val",
     "source": "reports/taurex_model_comparison.md"},
]


REQUIRED_REPORTED_KEYS = ("model", "value", "set")


def load_reported(path: str | None) -> list[dict]:
    """Validate the --reported file's schema BEFORE anything gets computed."""
    if path is None:
        return DEFAULT_REPORTED
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--reported {p}: plik nie jest poprawnym JSON-em ({exc})") from exc
    if not isinstance(raw, list):
        raise SystemExit(f"--reported {p}: oczekiwano listy obiektow JSON, jest {type(raw).__name__}. "
                         f"Format: [{{\"model\": str, \"value\": float, \"set\": str, \"source\": str}}, ...]")
    if not raw:
        raise SystemExit(f"--reported {p}: lista jest pusta — nie ma czego porownac z baseline'em")
    for i, row in enumerate(raw):
        where = f"--reported {p}: wiersz {i}"
        if not isinstance(row, dict):
            raise SystemExit(f"{where} nie jest obiektem JSON, jest {type(row).__name__}")
        missing = [k for k in REQUIRED_REPORTED_KEYS if k not in row]
        if missing:
            raise SystemExit(f"{where} nie ma wymaganego klucza: {', '.join(missing)}. "
                             f"Wymagane: {', '.join(REQUIRED_REPORTED_KEYS)}. "
                             f"Obecne: {', '.join(sorted(map(str, row))) or '(brak)'}")
        try:
            float(row["value"])
        except (TypeError, ValueError):
            raise SystemExit(f"{where} (model={row['model']!r}) ma value={row['value']!r}, "
                             "czego nie da sie odczytac jako liczby")
        row.setdefault("source", "(not stated in --reported file)")
    return raw


def check_sets_known(reported: list[dict], sets: dict[str, dict], path: str | None) -> None:
    """Set membership is checked separately because it needs the evaluation sets already built."""
    where = f"--reported {path}" if path else "DEFAULT_REPORTED"
    for i, row in enumerate(reported):
        if row["set"] not in sets:
            raise SystemExit(f"{where}: wiersz {i} (model={row['model']!r}) wskazuje set="
                             f"{row['set']!r}, ktorego ten check nie zna. Dostepne zbiory "
                             f"ewaluacyjne: {', '.join(sorted(sets))}")


def eval_sets() -> dict[str, dict]:
    """Ground-truth target matrices for every evaluation set used in the repo's tables."""
    import pandas as pd
    sets: dict[str, dict] = {}

    fm = A.load_adc_targets()
    for split in ("holdout", "validation"):
        ids = A.adc_split_ids(split)
        sets[f"adc_{split}"] = {
            "y": fm.loc[ids, A.TARGETS].to_numpy(dtype=np.float64),
            "train_constant": fm.loc[A.adc_split_ids("train"), A.TARGETS].to_numpy(dtype=np.float64).mean(axis=0),
            "prior_note": "ADC2023 FM_Parameter_Table; per-gas prior support differs per gas",
        }

    lab = pd.read_parquet(A.REPO / "data/TauREx set/labels.parquet")
    tr = lab[(lab.generator == "tau") & (lab.split == "train")][A.CROSSGEN_TARGETS].to_numpy(dtype=np.float64)
    for key, mask in (("taurex_val", (lab.generator == "tau") & (lab.split == "val")),
                      ("poseidon_test", (lab.generator == "poseidon") & (lab.split == "test"))):
        sets[key] = {"y": lab[mask][A.CROSSGEN_TARGETS].to_numpy(dtype=np.float64),
                     "train_constant": tr.mean(axis=0),
                     "prior_note": "crossgen prior is U(-12,-2) for all five gases"}
    return sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reported", default=None, help="JSON list of {model,value,set,source}")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reported = load_reported(args.reported)
    sets = eval_sets()
    check_sets_known(reported, sets, args.reported)

    rows, negative = [], []
    max_variant_gap = 0.0
    print(f"{'set':16} {'model':38} {'reported':>9} {'baseline':>9} {'skill':>7}  status")
    print("-" * 96)
    for r in reported:
        s = sets[r["set"]]
        base_self, base_self_gas = A.constant_predictor_mrmse(s["y"])            # oracle, informational
        base_train, _ = A.constant_predictor_mrmse(s["y"], s["train_constant"])  # honest, decides status
        sk = A.skill(r["value"], base_train)
        sk_oracle = A.skill(r["value"], base_self)
        max_variant_gap = max(max_variant_gap, abs(sk - sk_oracle))
        status = "ok" if sk > 0 else "NEGATIVE SKILL"
        if sk <= 0:
            negative.append({**r, "baseline_train_mean": base_train, "skill": sk})
        rows.append({**r, "n_rows": int(len(s["y"])),
                     "baseline_train_mean_mrmse": base_train,
                     "baseline_oracle_constant_mrmse": base_self,
                     "baseline_oracle_per_gas": base_self_gas.tolist(),
                     "skill_vs_train_mean": sk,
                     "skill_vs_oracle_constant_informational_only": sk_oracle,
                     "status": status})
        print(f"{r['set']:16} {r['model'][:38]:38} {r['value']:9.4f} {base_train:9.4f} {sk:7.3f}  {status}")

    payload = {"rows": rows, "negative_skill": negative,
               "note": ("skill <= 0 means the model carries no information beyond a single constant vector. "
                        "Ranking models in that regime orders their failure modes."),
               "definition": ("baseline that decides the status = mRMSE of the TRAIN-split mean vector "
                              "evaluated on the reported set's rows (`skill_vs_train_mean`)"),
               "oracle_variant": ("`baseline_oracle_constant_mrmse` / "
                                  "`skill_vs_oracle_constant_informational_only` use the per-column mean of "
                                  "the EVALUATION rows themselves. Reported for information only — it peeks "
                                  "at the evaluation targets and never enters the verdict."),
               "max_abs_skill_difference_between_baseline_variants": max_variant_gap,
               "variant_choice_changes_no_classification": bool(
                   all((r["skill_vs_train_mean"] > 0) ==
                       (r["skill_vs_oracle_constant_informational_only"] > 0) for r in rows))}
    CHECK.emit("FAIL" if negative else "PASS", payload, out=args.out)


if __name__ == "__main__":
    main()
