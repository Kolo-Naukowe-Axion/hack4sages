"""A05 — Was every reported checkpoint produced by a converged run?

Proves/disproves findings K5 (ExoBiome stopped at 8/30 with a diverging trajectory) and the
baseline-undertraining half of K3 (NSF stopped at 79/300 on val-NLL while its mRMSE was still
falling, checkpoint chosen from 256 rows x 16 posterior samples every 10th epoch).

Pure log parsing — no model loading, runs in a second, so it can be a pre-commit gate.

PASS criterion for each run:
  (a) the run terminated by its own stopping rule (early stop fired or max_epochs reached);
  (b) the reported metric had plateaued: its minimum is NOT within the last
      `IMPROVING_TAIL_MEASUREMENTS` measurements (there is no `--tail` flag; the constant is
      declared below with its justification);
  (c) the selection metric is the same quantity as the reported metric;
  (d) selection used the full validation split.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a05_training_completeness",
    finding="K5 / K3 — the flagship checkpoint comes from an 8/30-epoch aborted run; the baseline stopped at 79/300 on a different metric",
    question="Did each reported run converge, on the metric it is compared on, selected on the full split?",
    criterion="terminated by its own rule AND reported metric plateaued AND selection metric == reported metric AND full-split selection",
)


# Iloraz, ponizej ktorego uznajemy, ze harmonogram LR faktycznie zgasil uczenie. 100 zadeklarowane
# z gory i z marginesem: w ramieniu NSF LR spada 1e-3 -> 3.90625e-6, czyli iloraz 256, a krok
# wielkosci 1/100 poczatkowego oznacza, ze pozostale epoki nie moga juz przesunac wag na skale
# porownywalna z pierwszymi. Prog nie jest dobrany po wyniku: gdyby wynosil 300, NSF by go nie
# przekroczyl, wiec wybor MA znaczenie i musi byc jawny.
LR_COLLAPSE_FACTOR = 100.0

# Ile ostatnich pomiarow metryki liczy sie jako "minimum na koncu trajektorii", czyli brak plateau.
# 2, bo pomiar co 10 epok daje w NSF tylko 7 punktow: minimum na ostatnim albo przedostatnim z nich
# oznacza, ze co najmniej 10-20 epok z 300 zaplanowanych wciaz poprawialo metryke i nie ma dowodu,
# ze trajektoria sie wyplaszczyla. Przy 1 test wykrywalby tylko doslownie ostatni punkt i przegapil
# faktyczna sytuacje NSF (minimum na 6 z 7 pomiarow).
IMPROVING_TAIL_MEASUREMENTS = 2


def audit_exobiome() -> dict:
    import pandas as pd
    hist = pd.read_csv(A.EXOBIOME_ARTIFACT / "history.csv")
    cfg = json.loads((A.EXOBIOME_ARTIFACT / "config.json").read_text())
    state = json.loads((A.EXOBIOME_ARTIFACT / "training_state.json").read_text())
    val = hist["val_rmse_mean"].to_numpy()
    best_epoch = int(state["best_epoch"])
    last_epoch = int(hist["epoch"].max())

    # Wczesniej trajektoria byla ciezta POZYCYJNIE: `val[best_epoch - 1]` jako wartosc w najlepszej
    # epoce i `val[best_epoch:]` jako ogon po niej. Dzialalo tylko dlatego, ze w tym konkretnym
    # history.csv epoki sa numerowane 1..8, ciagle i bez brakow (val[5] = 0.29081112 = wartosc dla
    # epoki 6). Gdyby logowanie zaczynalo sie od 0, to przy best_epoch == 0 wyrazenie val[-1]
    # odczytalo by CICHO OSTATNIA epoke jako "najlepsza" — czyli check zglaszalby divergencje
    # wzgledem konca treningu zamiast wzgledem wybranego checkpointu, i to bez zadnego bledu.
    # Ponizej te same wielkosci przez maski po kolumnie `epoch`, plus asercja ciaglosci, zeby
    # niejawne zalozenie o numeracji przestalo byc niejawne.
    epochs = hist["epoch"].to_numpy(dtype=np.int64)
    expected = np.arange(int(epochs.min()), int(epochs.max()) + 1, dtype=np.int64)
    if not np.array_equal(epochs, expected):
        raise SystemExit(
            f"{A.EXOBIOME_ARTIFACT / 'history.csv'}: kolumna `epoch` nie jest ciagla rosnaca "
            f"sekwencja bez powtorzen. Odczytano {epochs.tolist()}, oczekiwano "
            f"{expected.tolist()}. Analiza trajektorii zaklada jeden wiersz na epoke; napraw log "
            "albo dopisz obsluge brakow, ale NIE indeksuj pozycyjnie.")
    sel_rows = hist.loc[hist["epoch"] == best_epoch, "val_rmse_mean"]
    if sel_rows.empty:
        raise SystemExit(
            f"{A.EXOBIOME_ARTIFACT / 'training_state.json'}: best_epoch={best_epoch} nie wystepuje "
            f"w history.csv (epoki {int(epochs.min())}..{int(epochs.max())})")
    best_val = float(sel_rows.iloc[0])
    post = hist.loc[hist["epoch"] > best_epoch, "val_rmse_mean"].to_numpy()

    issues = []
    if last_epoch < int(cfg["max_epochs"]) and last_epoch - best_epoch < int(cfg["early_stop_patience"]):
        issues.append(f"run ended at epoch {last_epoch} of max_epochs={cfg['max_epochs']} without "
                      f"exhausting early_stop_patience={cfg['early_stop_patience']} -> manual stop")
    if best_epoch == int(cfg["quantum_backbone_freeze_epochs"]):
        issues.append(f"best epoch ({best_epoch}) is exactly the last frozen-backbone epoch; the "
                      "reported model never trained with an unfrozen backbone")
    if len(post) and post.max() > best_val * 1.05:
        issues.append(f"validation diverged after the selected epoch: {best_val:.5f} -> "
                      f"{post.max():.5f} (+{100*(post.max()/best_val-1):.1f}%)")
    max_scale = float(hist["quantum_scale"].max())
    if max_scale < 1.0:
        issues.append(f"quantum_scale never reached 1.0 (max {max_scale:.3f}, ramp={cfg['quantum_ramp_epochs']} "
                      "epochs) yet metrics are reported at scale 1.0 — see a04")
    init = cfg.get("init_checkpoint_path")
    if init and not Path(init).exists():
        issues.append(f"stage-1 init checkpoint absent from the repo ({init}) -> total training compute "
                      "undocumented; no compute-matched claim is possible")
    return {"run": "exobiome_stage2_v4", "epochs_run": last_epoch, "max_epochs": int(cfg["max_epochs"]),
            "best_epoch": best_epoch, "best_val_rmse_mean": best_val,
            "epoch_column_contiguous": True, "epoch_range": [int(epochs.min()), int(epochs.max())],
            "post_best_val_trajectory": post.tolist(), "val_trajectory": val.tolist(),
            "quantum_scale_trajectory": hist["quantum_scale"].tolist(),
            "backbone_frozen_trajectory": hist["backbone_frozen"].tolist(),
            "selection_metric": "val mRMSE (full split, every epoch)",
            "reported_metric": "val/holdout mRMSE at quantum_scale=1.0",
            "issues": issues}


def audit_nsf() -> dict:
    run = A.REPO / "models/adc_winner_on_ariel/trained_run"
    if not run.exists():
        return {"run": "adc_winner_nsf", "issues": ["run dir absent"]}
    import yaml
    settings = yaml.safe_load((run / "settings_resolved.yaml").read_text())
    log = (run / "train.log").read_text() if (run / "train.log").exists() else ""
    hist = [json.loads(l) for l in (run / "history.jsonl").read_text().splitlines() if l.strip()]
    mrmse_pts = [(int(m.group(1)), float(m.group(2))) for m in
                 re.finditer(r"Epoch (\d+) validation mRMSE \| mean=([0-9.]+)", log)]
    issues = []
    last = max(h["epoch"] for h in hist)
    if last < int(settings["training"]["epochs"]):
        issues.append(f"stopped at epoch {last} of epochs={settings['training']['epochs']}")
    if "due to validation NLL patience" in log:
        issues.append("early stopping was driven by val NLL, NOT by the reported comparison metric (mRMSE)")
    best_i = None
    metric_still_improving = False
    if mrmse_pts:
        vals = [v for _, v in mrmse_pts]
        best_i = int(np.argmin(vals))
        metric_still_improving = bool(best_i >= len(vals) - IMPROVING_TAIL_MEASUREMENTS)
        if metric_still_improving:
            issues.append(f"the reported metric was still at/near its minimum at the last measurement "
                          f"({mrmse_pts[best_i]}) -> not converged on that metric")
        issues.append(f"mRMSE measured only {len(mrmse_pts)} times (every "
                      f"{settings['evaluation']['metric_every_epochs']} epochs)")
    if int(settings["evaluation"]["metric_max_rows"]) > 0:
        issues.append(f"checkpoint selected on {settings['evaluation']['metric_max_rows']} rows x "
                      f"{settings['evaluation']['metric_num_samples']} posterior samples, then reported "
                      f"on the full split with {settings['evaluation']['final_num_samples']} samples "
                      "-> selection estimator != reported estimator")
    if settings["evaluation"].get("point_estimate") == "median":
        issues.append("point estimate is the posterior MEDIAN while the metric is RMSE; the MSE-optimal "
                      "summary is the MEAN (evaluate.py supports it) -> baseline penalised by convention")
    lrs = [h["lr"] for h in hist]
    # Poprzednio warunkiem bylo TYLKO `lrs[-1] < lrs[0]/100`, a komunikat oskarzal o wygaszenie LR
    # "while the comparison metric was still improving" — czyli twierdzil rzecz, ktorej warunek
    # wcale nie sprawdzal. Zarzut mogl byc wiec podniesiony przeciw runowi, ktory zdazyl osiagnac
    # plateau: wygaszenie LR po zbieznosci jest poprawne, a nie wada. Teraz oba czlony musza
    # zachodzic naraz, a metryka jest ta sama, na ktorej run jest raportowany (mRMSE z train.log).
    # Dla NSF zarzut zostaje PRAWDZIWY: LR 1e-3 -> 3.90625e-6 (iloraz 256 > 100), a minimum mRMSE
    # wypada na 6 z 7 pomiarow (epoka 60 z 79 przebiegnietych z 300), czyli metryka wciaz spadala.
    lr_collapsed = bool(lrs and lrs[-1] < lrs[0] / LR_COLLAPSE_FACTOR)
    lr_ratio = (lrs[0] / lrs[-1]) if (lrs and lrs[-1]) else None
    if lr_collapsed and metric_still_improving:
        issues.append(f"LR annealed {lrs[0]:.0e} -> {lrs[-1]:.1e} (factor {lr_ratio:.0f} > "
                      f"{LR_COLLAPSE_FACTOR:.0f}, {len(set(lrs))} distinct values) WHILE the comparison "
                      f"metric was still improving: minimum of {len(mrmse_pts)} mRMSE measurements at "
                      f"{mrmse_pts[best_i]}, i.e. measurement {best_i + 1}/{len(mrmse_pts)}")
    return {"run": "adc_winner_nsf", "epochs_run": last, "max_epochs": int(settings["training"]["epochs"]),
            "mrmse_measurements": mrmse_pts, "lr_first_last": [lrs[0], lrs[-1]] if lrs else None,
            "lr_collapse_factor_threshold": LR_COLLAPSE_FACTOR, "lr_anneal_factor": lr_ratio,
            "lr_collapsed": lr_collapsed, "comparison_metric_still_improving": metric_still_improving,
            "best_mrmse_measurement_index": best_i,
            "n_mrmse_measurements": len(mrmse_pts),
            "improving_tail_measurements": IMPROVING_TAIL_MEASUREMENTS,
            "selection_metric": f"mRMSE on {settings['evaluation']['metric_max_rows']} rows x "
                                f"{settings['evaluation']['metric_num_samples']} samples every "
                                f"{settings['evaluation']['metric_every_epochs']} epochs",
            "stopping_metric": "val NLL", "reported_metric": "full-split mRMSE, 128 samples, median",
            "issues": issues}


def audit_snapshots() -> list[dict]:
    out = []
    for name, d in (("quantum_taurex_snapshot", "reports/ariel_quantum_taurex_snapshot_20260312_1003"),
                    ("noquant_taurex_snapshot", "reports/taurex_noquant_taurex_snapshot_20260312_133054")):
        p = A.REPO / d / "config.json"
        if not p.exists():
            continue
        cfg = json.loads(p.read_text())
        st = A.REPO / d / "training_state.json"
        state = json.loads(st.read_text()) if st.exists() else {}
        issues = []
        be, me = state.get("best_epoch"), cfg.get("max_epochs")
        if be is not None and me and be <= 0.25 * me:
            issues.append(f"best_epoch={be} of max_epochs={me} -> compared as a finished model")
        readme = A.REPO / d / "README.md"
        if readme.exists() and "continued after this snapshot" in readme.read_text():
            issues.append("README states the run continued after this snapshot: it is a mid-flight artifact")
        if cfg.get("taurex_ignore_poseidon"):
            issues.append("taurex_ignore_poseidon=true -> this run's own holdout_metrics.json is validation data")
        out.append({"run": name, "best_epoch": be, "max_epochs": me,
                    "batch_size": cfg.get("batch_size"), "classical_lr": cfg.get("classical_lr"),
                    "init_checkpoint_path": cfg.get("init_checkpoint_path"),
                    "early_stop_patience": cfg.get("early_stop_patience"), "issues": issues})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [audit_exobiome(), audit_nsf(), *audit_snapshots()]
    for r in runs:
        print(f"\n  == {r['run']}: epochs {r.get('epochs_run') or r.get('best_epoch')}/{r.get('max_epochs')}")
        for i in r["issues"]:
            print(f"     - {i}")
    payload = {"runs": runs, "n_runs_with_issues": sum(1 for r in runs if r["issues"])}
    CHECK.emit("FAIL" if payload["n_runs_with_issues"] else "PASS", payload, out=args.out)


if __name__ == "__main__":
    main()
