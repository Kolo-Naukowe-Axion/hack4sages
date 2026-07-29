"""A04 — At which quantum_scale were the reported metrics computed?

Proves/disproves finding K4. models/*/training.py calls evaluate_labeled_split() for the final
validation/holdout metrics WITHOUT passing quantum_scale, so it silently takes the signature
default 1.0 — while epoch selection ran at the ramped scale (0.5 at the selected epoch).
docs/VERIFICATION.md asserts the opposite ("artifact numbers -> native ramp scale ~= 0.5").

This check sweeps the scale, finds which value reproduces each published number, and reports the
exact classical ablation (scale=0) in dex, i.e. the honest size of the whole quantum pathway.

PASS criterion: SOME swept scale reproduces a published number AND that scale equals the scale at
which the checkpoint was selected (read from history.csv / training_state.json). Both halves are
required: a sweep in which nothing matches has not established provenance, it has failed to find it.
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
    name="a04_quantum_scale_provenance",
    finding="K4 — reported metrics use quantum_scale=1.0 (never validated); selection ran at 0.5",
    question="Which quantum_scale reproduces holdout_metrics.json / validation_metrics.json, and what is the gate-off value?",
    criterion="some swept scale reproduces a published number AND it equals the selection-time scale from history.csv",
)

# REPORTED = liczby faktycznie opublikowane (artifact holdout/validation_metrics.json).
# MAC_REPORTED = ponowna ewaluacja tego samego checkpointu na Macu; NIE jest liczba opublikowana.
# Trafienie tylko w MAC_REPORTED nie jest ustaleniem prowenansu liczby z raportu.
REPORTED = {"holdout": 0.2993761897087097, "validation": 0.29361358284950256}
MAC_REPORTED = {"holdout": 0.29869264364242554, "validation": 0.29482102394104004}


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.0,0.25,0.5,0.6666666666666666,1.0")
    ap.add_argument("--tol", type=float, default=2e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    hist = pd.read_csv(A.EXOBIOME_ARTIFACT / "history.csv")
    state = json.loads((A.EXOBIOME_ARTIFACT / "training_state.json").read_text())
    best_epoch = int(state["best_epoch"])
    sel = hist[hist.epoch == best_epoch].iloc[0]
    selection_scale = float(sel["quantum_scale"])
    selection_val = float(sel["val_rmse_mean"])

    aux_scaler, target_scaler, spectral_scaler = A.exobiome_scalers()
    model, _ = A.load_exobiome()

    payload = {"best_epoch": best_epoch, "selection_time_quantum_scale": selection_scale,
               "selection_time_val_mrmse": selection_val, "splits": {}}
    mismatches = []
    not_reproduced = []
    for split in ("holdout", "validation"):
        ids, aux_raw, spec_raw, y = A.load_adc_raw(split)
        aux, spectra = A.exobiome_inputs(spec_raw, aux_raw, aux_scaler, spectral_scaler, None)
        table = {}
        for s in scales:
            pred = A.exobiome_predict(model, aux, spectra, target_scaler, s)
            m = A.mrmse(y, pred)
            # Klucze formatowane na 6 miejsc PO ROZWAZENIU zarzutu o zaokraglenie i ODRZUCENIU go:
            # sprawdzane skale sa ilorazami malych liczb calkowitych (0, 1/4, 1/2, 2/3, 1), wiec
            # blad zaokraglenia do 6 miejsc wynosi <= 5e-7, a jedyne porownanie kluczy uzywa
            # tolerancji 1e-6 (linia ponizej z `abs(float(h) - selection_scale) < 1e-6`).
            # 5e-7 < 1e-6 zawsze, wiec kolizji ani chybienia byc nie moze. NIE zmieniaj na .12f
            # "dla bezpieczenstwa" — zmieni klucze w kazdym zapisanym payloadzie bez zysku.
            table[f"{s:.6f}"] = {"mrmse": m,
                                 "per_gas": dict(zip(A.TARGETS, A.per_gas_rmse(y, pred).tolist())),
                                 "matches_reported": bool(abs(m - REPORTED[split]) < args.tol),
                                 "matches_mac_reeval": bool(abs(m - MAC_REPORTED[split]) < args.tol)}
            print(f"  {split:11} scale={s:.4f} mRMSE={m:.6f}"
                  + ("  <== reported" if table[f'{s:.6f}']['matches_reported'] else "")
                  + ("  <== mac re-eval" if table[f'{s:.6f}']['matches_mac_reeval'] else ""))
        # Poprzednio jedno pole `scales_matching_a_published_number` scalalo trafienia w REPORTED
        # i w MAC_REPORTED. Bylo mylace az do przeklamania: `matches_reported` jest False dla
        # WSZYSTKICH pieciu skal na oba splity, wiec kazde trafienie na tej liscie pochodzi
        # wylacznie z ponownej ewaluacji na Macu — a nazwa pola sugerowala, ze liczba z raportu
        # zostala odtworzona. Rozdzielone, zeby brak odtworzenia liczby OPUBLIKOWANEJ byl widoczny
        # wprost w payloadzie, bez porownywania piecioelementowych slownikow.
        scales_reproducing_published = [k for k, v in table.items() if v["matches_reported"]]
        scales_reproducing_mac_only = [k for k, v in table.items()
                                       if v["matches_mac_reeval"] and not v["matches_reported"]]
        hits = scales_reproducing_published + scales_reproducing_mac_only
        gate_off = table[f"{0.0:.6f}"]["mrmse"] if f"{0.0:.6f}" in table else None
        # Straznik na skale 1.0, tak jak przy 0.0 i selection_scale. --scales jest parametrem CLI,
        # wiec `--scales 0.0,0.5` bez tego wywalalo KeyError: '1.000000' juz po pelnym sweepie.
        one_key = f"{1.0:.6f}"
        best_scale = min(table, key=lambda k: table[k]["mrmse"])
        # `gate_off is not None`, nie `if gate_off`: mRMSE == 0.0 to falsy float, wiec test
        # prawdziwosciowy raportowalby None ("nie zmierzono") dla modelu doskonalego przy scale=0,
        # czyli dokladnie w przypadku, w ktorym wklad sciezki kwantowej bylby najciekawszy.
        payload["splits"][split] = {
            "n_rows": int(len(ids)), "reported": REPORTED[split], "mac_reported": MAC_REPORTED[split],
            "sweep": table,
            "scales_reproducing_the_published_number": scales_reproducing_published,
            "published_number_reproduced_by_any_swept_scale": bool(scales_reproducing_published),
            "scales_reproducing_only_the_mac_reevaluation_not_published": scales_reproducing_mac_only,
            "gate_off_mrmse": gate_off,
            "quantum_pathway_contribution_at_scale_1":
                (gate_off - table[one_key]["mrmse"]) if (gate_off is not None and one_key in table) else None,
            "quantum_pathway_contribution_at_selection_scale":
                (gate_off - table[f"{selection_scale:.6f}"]["mrmse"])
                if (gate_off is not None and f"{selection_scale:.6f}" in table) else None,
            "best_scale_on_this_split": best_scale,
            "reported_is_best": bool(abs(table[best_scale]["mrmse"] - REPORTED[split]) < args.tol),
        }
        if not hits:
            # Poprzednio pusta `hits` po prostu nie dopisywala nic do `mismatches`, wiec sweep, w
            # ktorym ZADNA skala nie odtwarza zadnej opublikowanej liczby, konczyl sie PASS —
            # check "potwierdzal prowenans", nie znajdujac go. To poprawka teoretyczna: ostatni
            # przebieg dal FAIL z innego powodu (skala 1.0 odtwarza re-ewaluacje mac, a selekcja
            # szla na 0.5), ale rozgalezienie na pustej liscie jest wlasnie tym rodzajem cichego
            # PASS, ktorego audyt ma nie produkowac.
            not_reproduced.append({
                "split": split, "swept_scales": sorted(table),
                "reported": REPORTED[split], "mac_reported": MAC_REPORTED[split],
                "tolerance": args.tol,
                "closest_scale": best_scale,
                "closest_mrmse": table[best_scale]["mrmse"],
                "reason": ("no swept quantum_scale reproduces either the published number or the mac "
                           "re-evaluation within tol; the provenance of the reported metric is "
                           "UNESTABLISHED, which cannot be reported as a PASS"),
            })
        elif not any(abs(float(h) - selection_scale) < 1e-6 for h in hits):
            mismatches.append({"split": split, "selection_scale": selection_scale,
                               "reproducing_scales": hits,
                               "reproduces_published_number": bool(scales_reproducing_published)})

    payload["verification_md_claim"] = ("docs/VERIFICATION.md: 'artifact / table numbers -> native ramp "
                                       "scale ~= 0.5'. This check tests that claim directly.")
    payload["code_site"] = "models/ariel_exobiome/training.py final eval omits quantum_scale -> default 1.0"
    payload["mismatches"] = mismatches
    payload["splits_where_no_scale_reproduces_a_published_number"] = not_reproduced
    payload["published_numbers_reproduced_on_all_splits"] = all(
        payload["splits"][s]["published_number_reproduced_by_any_swept_scale"]
        for s in payload["splits"])
    CHECK.emit("FAIL" if (mismatches or not_reproduced) else "PASS", payload,
               inputs=[A.EXOBIOME_ARTIFACT / "best_model.pt", A.EXOBIOME_ARTIFACT / "history.csv"],
               out=args.out)


if __name__ == "__main__":
    main()
