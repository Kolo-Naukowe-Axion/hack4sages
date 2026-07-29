"""A03 — Are compared models fed the SAME input? (the 2x2 noise x model table)

Proves/disproves finding K3, and simultaneously tests the "noise robustness" hypothesis.

The published comparison is:
    ExoBiome           0.2994   <- instrument_spectrum as stored
    winner-style NSF   0.5523   <- the `median` point-estimate arm (trained_run/holdout_metrics.json
                                   records point_estimate="median" next to rmse_mean=0.5522884;
                                   `rmse_mean` is the mean over gases, not the `mean` arm), on
                                   instrument_spectrum + N(0, instrument_noise), because
                                   scripts/reeval_sota.py calls evaluate_point_metric(sample_noise=True)
                                   and models/adc_winner_on_ariel/preprocessing.py:149 does
                                   torch.normal(mean=spectra, std=noise)
i.e. the two models solve different inference problems. This check evaluates BOTH models under
BOTH conventions on the same split, so the comparison becomes well-posed.

PASS criterion: the sign of (ExoBiome - baseline) is the same under both input conventions.
If it flips, the published ranking is an artifact of the input convention.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

# Artefakt, ktory ROZSTRZYGA, ktore ramie punktowe zespol faktycznie opublikowal jako 0.5523.
# Pole "point_estimate" w tym pliku ma wartosc "median"; klucz "rmse_mean" obok niego NIE oznacza
# ramienia "mean" — to srednia po gazach, tak nazywa ja evaluate_point_metric. Ta zbieznosc nazw
# jest zrodlem bledu: poprzednia wersja robila `est = "mean" if "mean" in base["clean"] else ...`,
# czyli porownywala ramie, ktorego zespol nigdy nie podal. Skutek nie byl kosmetyczny — dla mean
# ratio_clean wychodzi 1.3485, dla publikowanego median 1.2884, wiec audyt zarzucalby zespolowi
# przewage o 6 pp inna niz ta, ktora zespol ogloszil. Ramie ustalamy z artefaktu, nie z zgadywania,
# i zapisujemy w payloadzie sciezke, z ktorej to wiadomo.
PUBLISHED_METRICS_REL = Path("models/adc_winner_on_ariel/trained_run/holdout_metrics.json")
# Uzywane TYLKO gdy artefaktu nie ma na dysku. Wartosc nie jest domysleniem: taka jest wartosc
# pola point_estimate w wersji pliku towarzyszacej publikowanej tabeli.
PUBLISHED_ESTIMATOR_FALLBACK = "median"

# Seed globalnego RNG torcha przed KAZDYM wywolaniem evaluate_point_metric.
#
# Dlaczego to jest potrzebne: estymator punktowy NSF to mediana z `final_num_samples` losowan
# posterioru. Losowanie idzie przez `IndependentNSF.sample` -> `flow(context).sample(...)`
# (models/adc_winner_on_ariel/model.py:54), ktore NIE przyjmuje generatora, wiec czerpie z
# globalnego RNG torcha. Argumenty `row_seed` i `noise_seed`, ktore evaluate_point_metric
# przyjmuje, seeduja wybor wierszy i szum WEJSCIOWY — nie sam posterior. Bez tej linii kazde
# uruchomienie a03 dawalo inne liczby NSF (rzad 1e-3 na agregacie, do 5e-3 per gaz), co bylo
# jedynym zrodlem wariancji przebieg-do-przebiegu w calym harnessie.
#
# Czego to NIE naprawia: liczba pozostaje jednym losowaniem Monte Carlo z `final_num_samples`
# probek, wiec jej odleglosc od wartosci granicznej sie nie zmienia — zmienia sie tylko to, ze
# jest ODTWARZALNA. Zmniejszenie samego bledu MC wymaga wiekszej liczby probek, nie seeda.
#
# Kod zespolu pozostaje nietkniety: seedujemy globalny RNG PRZED wywolaniem zaimportowanej
# funkcji, dokladnie tak, jak sam evaluate.py:46-47 robi to dla szumu wejsciowego.
POSTERIOR_SAMPLE_SEED = 42


def published_estimator() -> dict:
    """READ-ONLY. Ktore ramie punktowe stoi za publikowana liczba 0.5523 — czytane z artefaktu."""
    p = A.REPO / PUBLISHED_METRICS_REL
    out = {"source": str(PUBLISHED_METRICS_REL), "source_exists": p.is_file(),
           "field": "point_estimate",
           "why_this_field": "the sibling key is named `rmse_mean`, but that is the mean over gases "
                             "(evaluate_point_metric's naming), NOT the `mean` point-estimate arm; "
                             "`point_estimate` is the only field that names the arm"}
    if not p.is_file():
        out.update({"estimator": PUBLISHED_ESTIMATOR_FALLBACK, "read_from_artefact": False})
        return out
    d = json.loads(p.read_text())
    out.update({"estimator": str(d.get("point_estimate", PUBLISHED_ESTIMATOR_FALLBACK)),
                "read_from_artefact": "point_estimate" in d,
                "published_mrmse": float(d["rmse_mean"]) if "rmse_mean" in d else None,
                "published_rows": int(d["rows"]) if "rows" in d else None})
    return out


CHECK = A.Check(
    name="a03_input_convention",
    finding="K3 — the ExoBiome-vs-SOTA comparison mixes two input conventions; the ranking reverses when they are equalised",
    question="What is each model's mRMSE with and without eval-time N(0,sigma) injection, on the same split?",
    criterion="sign(ExoBiome - baseline) is invariant to the input convention",
)


def eval_exobiome(split: str, seeds: list[int], scales: list[float]) -> dict:
    ids, aux_raw, spec_raw, y = A.load_adc_raw(split)
    aux_scaler, target_scaler, spectral_scaler = A.exobiome_scalers()
    model, ck = A.load_exobiome()
    res: dict = {"n_rows": int(len(ids)), "clean": {}, "noised": {}}
    aux, spectra = A.exobiome_inputs(spec_raw, aux_raw, aux_scaler, spectral_scaler, None)
    for s in scales:
        pred = A.exobiome_predict(model, aux, spectra, target_scaler, s)
        res["clean"][f"{s:.4f}"] = {"mrmse": A.mrmse(y, pred),
                                    "per_gas": dict(zip(A.TARGETS, A.per_gas_rmse(y, pred).tolist()))}
        print(f"  exobiome {split:10} clean            scale={s:.4f} mRMSE={res['clean'][f'{s:.4f}']['mrmse']:.6f}")
    band = []
    for sd in seeds:
        auxn, specn = A.exobiome_inputs(spec_raw, aux_raw, aux_scaler, spectral_scaler,
                                        np.random.default_rng(sd))
        pred = A.exobiome_predict(model, auxn, specn, target_scaler, max(scales))
        band.append(A.mrmse(y, pred))
        print(f"  exobiome {split:10} +N(0,sigma) seed={sd:<3} scale={max(scales):.4f} mRMSE={band[-1]:.6f}")
    res["noised"] = {"scale": max(scales), "seeds": seeds, "mrmse_per_seed": band,
                     "mrmse_mean": float(np.mean(band)), "mrmse_std": float(np.std(band))}
    res["degradation_factor"] = float(np.mean(band) / res["clean"][f"{max(scales):.4f}"]["mrmse"])
    return res


def eval_baseline(split: str, seeds: list[int], estimators: list[str]) -> dict:
    import torch
    import yaml
    sys.path.insert(0, str(A.REPO))
    from models.adc_winner_on_ariel.dataset import load_prepared_data, move_prepared_data_to_device
    from models.adc_winner_on_ariel.evaluate import evaluate_point_metric
    from models.adc_winner_on_ariel.model import IndependentNSF, ModelConfig

    run_dir = A.REPO / "models/adc_winner_on_ariel/trained_run"
    prepared = A.REPO / "data/generated-data/ariel_winner_nf_prepared"
    if not prepared.exists():
        return {"error": f"prepared data missing: {prepared}. Build it with "
                         "`python -m models.adc_winner_on_ariel.prepare_dataset "
                         "--data-root data/ariel-ml-dataset --split-source data/val_dataset "
                         f"--output {prepared}`"}
    settings = yaml.safe_load((run_dir / "settings_resolved.yaml").read_text())
    device = torch.device("cpu")
    data = move_prepared_data_to_device(load_prepared_data(prepared), device)
    model = IndependentNSF(ModelConfig(**settings["model"])).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model_by_mrmse.pt", map_location=device)["model"])
    res: dict = {"n_params": int(sum(p.numel() for p in model.parameters())),
                 "trained_with_noise_augmentation": True,
                 "posterior_sample_seed": POSTERIOR_SAMPLE_SEED,
                 "posterior_sample_seed_note":
                     "global torch RNG seeded before each evaluate_point_metric call; the flow's "
                     "sample() takes no generator (model.py:54), so row_seed/noise_seed do NOT "
                     "cover the posterior draw. Makes the number reproducible, not more accurate: "
                     "it is still one Monte Carlo draw of final_num_samples samples.",
                 "clean": {}, "noised": {}}
    for est in estimators:
        torch.manual_seed(POSTERIOR_SAMPLE_SEED)
        r = evaluate_point_metric(model, getattr(data, split), data.scalers, device=device,
                                  num_samples=int(settings["evaluation"]["final_num_samples"]),
                                  point_estimate=est,
                                  batch_size=int(settings["training"]["eval_batch_size"]),
                                  max_rows=None, row_seed=42, sample_noise=False, noise_seed=42)
        res["clean"][est] = {"mrmse": float(r["rmse_mean"]), "per_gas": r["rmse"], "n_rows": int(r["rows"])}
        print(f"  baseline {split:10} clean            est={est:6} mRMSE={r['rmse_mean']:.6f}")
        band = []
        for sd in seeds:
            # Seed per iteracje, bo inaczej n-te wywolanie zalezy od stanu RNG po (n-1) poprzednich.
            # Wiazemy go z `sd`, zeby ramie zaszumione nadal mierzylo rozrzut po seedach SZUMU
            # (sd wchodzi do noise_seed nizej), a nie po stanie globalnego RNG.
            torch.manual_seed(POSTERIOR_SAMPLE_SEED + sd)
            rn = evaluate_point_metric(model, getattr(data, split), data.scalers, device=device,
                                       num_samples=int(settings["evaluation"]["final_num_samples"]),
                                       point_estimate=est,
                                       batch_size=int(settings["training"]["eval_batch_size"]),
                                       max_rows=None, row_seed=42, sample_noise=True, noise_seed=sd)
            band.append(float(rn["rmse_mean"]))
            print(f"  baseline {split:10} +N(0,sigma) seed={sd:<3} est={est:6} mRMSE={band[-1]:.6f}")
        res["noised"][est] = {"seeds": seeds, "mrmse_per_seed": band,
                              "mrmse_mean": float(np.mean(band)), "mrmse_std": float(np.std(band))}
        res.setdefault("degradation_factor", {})[est] = float(np.mean(band) / res["clean"][est]["mrmse"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout", choices=["holdout", "validation"])
    ap.add_argument("--seeds", default="42,1,2")
    ap.add_argument("--scales", default="0.0,0.5,1.0")
    ap.add_argument("--estimators", default="mean,median")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    scales = [float(s) for s in args.scales.split(",")]

    exo = eval_exobiome(args.split, seeds, scales)
    base = {} if args.skip_baseline else eval_baseline(args.split, seeds, args.estimators.split(","))

    pub = published_estimator()
    payload = {"split": args.split, "exobiome": exo, "baseline_nsf": base,
               "published_estimator": pub,
               "published_comparison": {"exobiome": 0.2993761897087097, "baseline": 0.5522884130477905,
                                        "claimed_ratio": 0.5522884130477905 / 0.2993761897087097}}
    status = "INFO"
    if base and "clean" in base and base["clean"]:
        exo_clean = exo["clean"][f"{max(scales):.4f}"]["mrmse"]
        # Kazde policzone ramie zostaje w payloadzie — raport cytuje i median, i mean, wiec usuniecie
        # ktoregokolwiek zerwaloby mozliwosc weryfikacji cytatu. Ramie publikowane jest tylko
        # WSKAZANE, nie uprzywilejowane w liczeniu.
        by_est: dict[str, dict] = {}
        for e in base["clean"]:
            d_clean_e = exo_clean - base["clean"][e]["mrmse"]
            d_noised_e = exo["noised"]["mrmse_mean"] - base["noised"][e]["mrmse_mean"]
            by_est[e] = {
                "estimator": e,
                "is_published_arm": bool(e == pub["estimator"]),
                "delta_clean_input(exo-base)": d_clean_e,
                "delta_noised_input(exo-base)": d_noised_e,
                "ratio_clean(base/exo)": base["clean"][e]["mrmse"] / exo_clean,
                "ratio_noised(base/exo)": base["noised"][e]["mrmse_mean"] / exo["noised"]["mrmse_mean"],
                "sign_flips": bool(np.sign(d_clean_e) != np.sign(d_noised_e)),
                "degradation_factor_baseline": base.get("degradation_factor", {}).get(e),
            }
        payload["comparison_by_estimator"] = by_est
        # Werdykt zawisa na ramieniu publikowanym; jesli wywolano check bez tego ramienia, mowimy to
        # wprost, zamiast po cichu ocenic inne ramie jako "to publikowane".
        est = pub["estimator"] if pub["estimator"] in by_est else next(iter(by_est))
        payload["comparison"] = dict(by_est[est])
        payload["comparison"]["published_arm_evaluated"] = bool(pub["estimator"] in by_est)
        payload["comparison"]["published_arm_source"] = pub["source"]
        d_clean = payload["comparison"]["delta_clean_input(exo-base)"]
        d_noised = payload["comparison"]["delta_noised_input(exo-base)"]
        status = "FAIL" if payload["comparison"]["sign_flips"] else "WARN"
        print(f"\n  published point-estimate arm = {pub['estimator']} "
              f"(from {pub['source']}, field {pub['field']})")
        for e, c in by_est.items():
            print(f"  est={e:6}{' *published*' if c['is_published_arm'] else '           '} "
                  f"d_clean={c['delta_clean_input(exo-base)']:+.4f} "
                  f"d_noised={c['delta_noised_input(exo-base)']:+.4f} "
                  f"ratio_clean={c['ratio_clean(base/exo)']:.4f} "
                  f"ratio_noised={c['ratio_noised(base/exo)']:.4f} flips={c['sign_flips']}")
        print(f"\n  delta(exo-base) clean  = {d_clean:+.4f}")
        print(f"  delta(exo-base) noised = {d_noised:+.4f}")
        print(f"  ranking reverses with the input convention: {payload['comparison']['sign_flips']}")

    # Wniosek BUDOWANY z policzonych zmiennych. Poprzednio byl to staly napis, dodatkowo umieszczony
    # POZA blokiem `if base ...` — wiec przy --skip-baseline (albo przy braku prepared data, gdy
    # eval_baseline zwraca {"error": ...}) payload nadal twierdzil, ze "under the noised convention
    # it reverses", mimo ze zadnego porownania nie wykonano. Zahardkodowany wniosek nie moze byc
    # falsyfikowany przez wlasny check: gdyby naprawa a09 albo zmiana skal odwrocila znak, tekst
    # bylby ten sam. Kazda liczba w zdaniu pochodzi teraz ze zmiennej wyliczonej wyzej.
    if "comparison" in payload:
        c = payload["comparison"]
        deg_exo = float(exo["degradation_factor"])
        deg_base = c["degradation_factor_baseline"]
        claimed = payload["published_comparison"]["claimed_ratio"]
        bits = [
            f"The published {claimed:.3f}x advantage compares ExoBiome on the stored spectra against "
            f"the NSF's {c['estimator']} arm on spectra with N(0,sigma) added, i.e. two different "
            "inference problems.",
            f"Equalised on the CLEAN convention the ratio is {c['ratio_clean(base/exo)']:.4f}x "
            f"(delta = {c['delta_clean_input(exo-base)']:+.4f}); equalised on the NOISED convention it is "
            f"{c['ratio_noised(base/exo)']:.4f}x (delta = {c['delta_noised_input(exo-base)']:+.4f}).",
            ("The sign of (ExoBiome - baseline) FLIPS between the two conventions, so the published "
             "ranking is an artefact of the input convention."
             if c["sign_flips"] else
             "The sign of (ExoBiome - baseline) is the same under both conventions, so the ranking "
             "survives equalisation even where its magnitude does not."),
        ]
        if deg_base is not None:
            bits.append(
                f"Degradation under +N(0,sigma): ExoBiome {deg_exo:.3f}x vs NSF({c['estimator']}) "
                f"{deg_base:.3f}x — " + ("the point model degrades more steeply, so the 'noise "
                                         "robustness' hypothesis is not supported."
                                         if deg_exo > deg_base else
                                         "the point model degrades less steeply, which is consistent "
                                         "with the 'noise robustness' hypothesis."))
        if not c["published_arm_evaluated"]:
            bits.append(f"CAVEAT: the published arm ({pub['estimator']}, per {pub['source']}) was not "
                        f"among the evaluated estimators; the numbers above are for {c['estimator']}.")
        others = [e for e in payload["comparison_by_estimator"] if e != c["estimator"]]
        if others:
            bits.append("Other point-estimate arms evaluated (reported, not used for the verdict): "
                        + "; ".join(f"{e} ratio_clean="
                                    f"{payload['comparison_by_estimator'][e]['ratio_clean(base/exo)']:.4f}, "
                                    f"flips={payload['comparison_by_estimator'][e]['sign_flips']}"
                                    for e in others) + ".")
        payload["interpretation"] = " ".join(bits)
    else:
        why = "--skip-baseline was passed" if args.skip_baseline else (
            base.get("error", "the baseline arm produced no `clean` metrics")
            if isinstance(base, dict) else "the baseline arm produced no `clean` metrics")
        payload["interpretation"] = (
            f"NO cross-model comparison was performed ({why}), so this record says nothing about the "
            "published ranking in either direction. Reported on the ExoBiome side alone: mRMSE "
            f"{exo['clean'][f'{max(scales):.4f}']['mrmse']:.4f} on the stored spectra vs "
            f"{exo['noised']['mrmse_mean']:.4f} with +N(0,sigma) added, i.e. a "
            f"{exo['degradation_factor']:.3f}x degradation. Re-run without --skip-baseline to test the "
            "sign-invariance criterion."
        )
    CHECK.emit(status, payload, inputs=[A.EXOBIOME_ARTIFACT / "best_model.pt"], out=args.out)


if __name__ == "__main__":
    main()
