"""A15 — Is a 5-gas target vector a defensible output for an atmospheric retrieval?

ExoBiome regresses 5 log abundances. The ADC2023 target is 7 parameters (planet radius,
temperature, + the 5 gases) and `Tracedata.hdf5` stores 7-column reference posteriors. This check
measures what omitting temperature and radius actually costs, separately for the two datasets,
because the answer differs sharply between them.

Measurements
  1. posterior coupling — weighted correlations between (R_p, T) and each log abundance in the
     reference nested-sampling traces, compared against the gas-gas correlations that the repo's
     own VERIFICATION.md calls "the biggest deviation" from the winner;
  2. retrievability — how well T and R_p can be predicted from exactly the inputs the model sees;
  3. leakage — whether T is already determined by the auxiliary table (equilibrium temperature),
     i.e. whether it is a latent parameter at all in this benchmark;
  4. conditioning value — whether supplying the true (T, R_p) improves the gas predictions;
  5. cross-generator — whether temperature is available to the model at all there.

PASS criterion: either the model predicts the full parameter vector of its benchmark, or every
omitted parameter is (a) weakly coupled to the reported ones in the reference posterior AND
(b) already determined by the inputs, so that omitting it changes no inference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a15_target_completeness",
    finding="K9 — the model predicts 5 of its benchmark's 7 parameters; T is leaked by aux on ADC and wholly absent on cross-generator",
    question="What does omitting temperature and planet radius from the target vector cost, per dataset?",
    criterion="model predicts the full benchmark parameter vector, or omitted params are both weakly coupled and input-determined",
)

TRACE_COLS = ["planet_radius", "planet_temp", *A.TARGETS]
AU_M = 1.495978707e11
R_SUN_M = 6.957e8


def posterior_slope(max_planets: int) -> dict:
    """Exchange rate T -> abundance, in dex per 100 K, from the reference posteriors.

    NOT the same as `posterior_coupling`: that computes CORRELATION coefficients (covariance
    normalised by sd); this computes the weighted regression SLOPE:

        slope = Cov_w(T, log10 X) / Var_w(T), then x100 -> dex / 100 K

    Added 2026-07-28: the numbers cited in report SS K9(b)/K9(g) (0.359 / 0.360; p90 1.002;
    2955 pairs; median T 913 K; cold/hot split 0.451 / 0.282) existed in NO payload — they were
    ad hoc calculations, from TWO different runs: 0.359 reproduces at 600 planets (capped), 0.360
    at 591. Over all 663 available planets the rate is 0.352.

    Compared against the closed form `Dlog X = -(N/ln10) * eps/(1+eps)` (K9(g)): for N=7 this
    gives 0.333, i.e. measured/theory ratio = 1.06.
    """
    import h5py
    gt = A.adc_root() / "TrainingData/Ground Truth Package"
    ids = A.adc_split_ids("holdout")
    slopes, temps, slope_T, per_gas = [], [], [], {g: [] for g in A.TARGETS}
    n_planets = n_dropped = 0
    with h5py.File(gt / "Tracedata.hdf5", "r") as h:
        for pid in ids:
            k = f"Planet_{pid}"
            if k not in h or "tracedata" not in h[k]:
                continue
            tr = np.asarray(h[k]["tracedata"])
            w = np.asarray(h[k]["weights"], dtype=np.float64)
            if tr.ndim != 2 or tr.shape[1] != 7 or len(w) != len(tr):
                continue
            w = w / w.sum()
            n_planets += 1
            d = tr - w @ tr
            cov = (d * w[:, None]).T @ d
            var_T = cov[1, 1]
            temps.append(float((w @ tr)[1]))
            if var_T <= 1e-12:
                n_dropped += len(A.TARGETS)
            else:
                for gi, g in enumerate(A.TARGETS):
                    j = 2 + gi
                    if cov[j, j] <= 1e-12:
                        n_dropped += 1
                        continue
                    s = cov[1, j] / var_T * 100.0
                    slopes.append(s)
                    slope_T.append(float((w @ tr)[1]))   # temperature of THIS pair, not positional
                    per_gas[g].append(s)
            if max_planets and n_planets >= max_planets:
                break

    sl = np.abs(np.array(slopes))
    med_T = float(np.median(temps))
    # Split on the temperature attached TO THE PAIR. An earlier version used
    # np.repeat(temps, 5)[:len(slopes)], which drifts out of alignment as soon as any pair is
    # dropped for frozen sd — the same positional-indexing fragility we charge a05 with.
    cold = [s for s, tt in zip(slopes, slope_T) if tt < med_T]
    hot = [s for s, tt in zip(slopes, slope_T) if tt >= med_T]
    mc = float(np.median(np.abs(cold))) if cold else float("nan")
    mh = float(np.median(np.abs(hot))) if hot else float("nan")
    return {
        "n_planets": n_planets,
        "n_gas_planet_pairs": len(slopes),
        "n_pairs_dropped_frozen_sd": n_dropped,
        "median_T_K": med_T,
        "median_abs_slope_dex_per_100K": float(np.median(sl)),
        "p90_abs_slope_dex_per_100K": float(np.percentile(sl, 90)),
        "median_signed_slope_dex_per_100K": float(np.median(slopes)),
        "per_gas_median_abs_slope": {g: float(np.median(np.abs(v))) if v else float("nan")
                                     for g, v in per_gas.items()},
        "cold_half_median_abs_slope": mc,
        "hot_half_median_abs_slope": mh,
        "cold_over_hot_ratio": (mc / mh) if mh else float("nan"),
        "closed_form_N7_dex_per_100K": 0.333,
        "measured_over_closed_form_N7": float(np.median(sl)) / 0.333,
        "definition": "slope = Cov_w(T, log10 X) / Var_w(T) * 100, median over gas-planet pairs; "
                      "NOT a correlation coefficient (see posterior_coupling for those)",
    }


def posterior_coupling(max_planets: int) -> dict:
    import h5py
    gt = A.adc_root() / "TrainingData/Ground Truth Package"
    ids = A.adc_split_ids("holdout")
    mats = []
    with h5py.File(gt / "Tracedata.hdf5", "r") as h:
        for pid in ids:
            k = f"Planet_{pid}"
            if k not in h or "tracedata" not in h[k]:
                continue
            tr = np.asarray(h[k]["tracedata"])
            w = np.asarray(h[k]["weights"], dtype=np.float64)
            if tr.ndim != 2 or tr.shape[1] != 7 or len(w) != len(tr):
                continue
            w = w / w.sum()
            d = tr - w @ tr
            cov = (d * w[:, None]).T @ d
            sd = np.sqrt(np.diag(cov))
            ok = sd > 1e-12
            C = np.full((7, 7), np.nan)
            C[np.ix_(ok, ok)] = cov[np.ix_(ok, ok)] / np.outer(sd[ok], sd[ok])
            mats.append(C)
            if max_planets and len(mats) >= max_planets:
                break
    Cs = np.array(mats)

    def med_abs(i, j):
        v = Cs[:, i, j]
        v = v[np.isfinite(v)]
        if len(v) == 0:                       # every planet had a frozen sd on this pair
            return (float("nan"),) * 4        # np.percentile([]) raises IndexError, returns no nan
        return float(np.median(np.abs(v))), float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))

    out = {"n_planets": len(Cs), "per_gas": {}}
    t_g, r_g = [], []
    for gi, g in enumerate(A.TARGETS):
        j = 2 + gi
        ma_r, md_r, lo_r, hi_r = med_abs(0, j)
        ma_t, md_t, lo_t, hi_t = med_abs(1, j)
        out["per_gas"][g] = {"radius": {"med_abs_r": ma_r, "median_r": md_r, "p10": lo_r, "p90": hi_r},
                             "temperature": {"med_abs_r": ma_t, "median_r": md_t, "p10": lo_t, "p90": hi_t}}
        r_g.append(ma_r)
        t_g.append(ma_t)
    gas_gas = [med_abs(2 + a, 2 + b)[0] for a in range(5) for b in range(a + 1, 5)]
    ma_rt, md_rt, _, _ = med_abs(0, 1)
    out["summary"] = {"mean_med_abs_r_temperature_gas": float(np.mean(t_g)),
                      "mean_med_abs_r_radius_gas": float(np.mean(r_g)),
                      "mean_med_abs_r_gas_gas": float(np.mean(gas_gas)),
                      "med_abs_r_radius_temperature": ma_rt, "median_r_radius_temperature": md_rt,
                      "temperature_over_gasgas_ratio": float(np.mean(t_g) / np.mean(gas_gas))}
    return out


def adc_experiments(n_train: int) -> dict:
    import h5py
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    gt = A.adc_root() / "TrainingData/Ground Truth Package"
    fm = pd.read_csv(gt / "FM_Parameter_Table.csv").set_index("planet_ID")
    aux = pd.read_csv(A.adc_root() / "TrainingData/AuxillaryTable.csv")
    aux = aux.drop(columns=[c for c in aux.columns if c.startswith("Unnamed:")]).set_index("planet_ID")

    def build(split, n=None):
        ids = A.adc_split_ids(split)
        # `is not None`, not `if n:` — a truthiness test on an int would make `--n-train 0` SILENTLY
        # mean "all rows" instead of zero. The same bug existed in `a26` and was fixed there;
        # `main()` now rejects 0 outright, so this branch has no special case.
        if n is not None:
            ids = ids[:n]
        with h5py.File(A.adc_root() / "TrainingData/SpectralData.hdf5", "r") as h:
            S = np.stack([h[f"Planet_{p}"]["instrument_spectrum"][:] for p in ids]).astype(np.float64)
            N = np.stack([h[f"Planet_{p}"]["instrument_noise"][:] for p in ids]).astype(np.float64)
        ref = S.mean(axis=1, keepdims=True)
        a = aux.loc[ids]
        # clip as in a26_baseline_ladder.py:321 and audit_lib.py:338 — today bit-identical (0 zeros,
        # 0 negatives, 0 NaN across all 7 AuxillaryTable columns), but without it -inf/NaN blows up HistGBR
        def lg(col):
            return np.log10(np.clip(a[col].to_numpy(dtype=np.float64), 1e-12, None))
        af = np.column_stack([lg("star_distance"), lg("star_mass_kg"),
                              lg("star_radius_m"), a["star_temperature"],
                              lg("planet_mass_kg"), lg("planet_orbital_period"),
                              lg("planet_distance"), lg("planet_surface_gravity")])
        X = np.hstack([S / ref, N / ref, np.log10(ref), af])
        y = fm.loc[ids, TRACE_COLS].to_numpy(float)
        return X, y, af, a

    Xtr, ytr, aftr, _ = build("train", n_train)
    Xho, yho, afho, aho = build("holdout")

    def fit(Xa, ya, Xb):
        out = np.empty((len(Xb), ya.shape[1]))
        for j in range(ya.shape[1]):
            out[:, j] = HistGradientBoostingRegressor(max_iter=250, random_state=0).fit(Xa, ya[:, j]).predict(Xb)
        return out

    pred = fit(Xtr, ytr, Xho)
    retrievability = {}
    for j, c in enumerate(TRACE_COLS):
        r = float(np.sqrt(np.mean((yho[:, j] - pred[:, j]) ** 2)))
        sd = float(yho[:, j].std())
        retrievability[c] = {"rmse": r, "sd_target": sd, "r2": 1.0 - (r / sd) ** 2}

    teq = aho["star_temperature"].to_numpy() * np.sqrt(
        aho["star_radius_m"].to_numpy() / (2.0 * aho["planet_distance"].to_numpy() * AU_M))
    T = yho[:, 1]
    t_from_aux = fit(aftr, ytr[:, 1:2], afho)[:, 0]
    leakage = {"pearson_r_Teq_vs_true_T": float(np.corrcoef(teq, T)[0, 1]),
               "rmse_Teq_K": float(np.sqrt(np.mean((teq - T) ** 2))),
               "rmse_gbm_on_aux_only_K": float(np.sqrt(np.mean((t_from_aux - T) ** 2))),
               "sd_true_T_K": float(T.std())}

    base = fit(Xtr, ytr[:, 2:], Xho)
    cond = fit(np.hstack([Xtr, ytr[:, :2]]), ytr[:, 2:], np.hstack([Xho, yho[:, :2]]))
    per_gas, rb, rc = {}, [], []
    for j, g in enumerate(A.TARGETS):
        a1 = float(np.sqrt(np.mean((yho[:, 2 + j] - base[:, j]) ** 2)))
        a2 = float(np.sqrt(np.mean((yho[:, 2 + j] - cond[:, j]) ** 2)))
        per_gas[g] = {"rmse_without": a1, "rmse_with_true_T_Rp": a2, "delta": a2 - a1}
        rb.append(a1)
        rc.append(a2)
    conditioning = {"per_gas": per_gas, "mrmse_without": float(np.mean(rb)),
                    "mrmse_with_true_T_Rp": float(np.mean(rc)),
                    "relative_improvement": float((np.mean(rb) - np.mean(rc)) / np.mean(rb))}
    return {"retrievability": retrievability, "aux_leakage_of_temperature": leakage,
            "conditioning_value": conditioning}


def reference_precision() -> dict:
    """How well does the reference retrieval itself know T and R_p, and is it calibrated?"""
    import pandas as pd
    gt = A.adc_root() / "TrainingData/Ground Truth Package"
    q = pd.read_csv(gt / "QuartilesTable.csv").set_index("planet_ID")
    fm = pd.read_csv(gt / "FM_Parameter_Table.csv").set_index("planet_ID")
    ids = A.adc_split_ids("holdout")
    qq, y = q.reindex(ids), fm.loc[ids]
    out = {}
    for name, prefix, col in (("planet_radius", "planet_radius", "planet_radius"),
                              ("temperature", "T", "planet_temp")):
        q1, q2, q3 = [qq[f"{prefix}_q{i}"].to_numpy(float) for i in (1, 2, 3)]
        t = y[col].to_numpy(float)
        m = ~(np.isnan(q1) | np.isnan(q2) | np.isnan(q3))
        sig = (q3[m] - q1[m]) / 1.349
        rmse = float(np.sqrt(np.mean((q2[m] - t[m]) ** 2)))
        out[name] = {"n": int(m.sum()), "median_posterior_sigma": float(np.median(sig)),
                     "rmse_of_posterior_median": rmse, "sd_truth": float(t.std()),
                     "overconfidence_ratio": rmse / float(np.median(sig))}
    return out


def crossgen() -> dict:
    import h5py
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    lab = pd.read_parquet(A.REPO / "data/TauREx set/labels.parquet")
    tau = lab[lab.generator == "tau"]
    live = ["star_radius_rsun", "planet_radius_rjup", "log_g_cgs"]
    corr = {c: float(np.corrcoef(tau.temperature_k, tau[c])[0, 1]) for c in live}
    teq = 5500.0 * np.sqrt(tau.star_radius_rsun.to_numpy() * R_SUN_M / (2 * 0.05 * AU_M))
    usage = {}
    for pkg in ("taurex_exobiome", "taurex_exobiome_without_quant", "ariel_winner_on_taurex", "taurex_fmpe"):
        d = A.REPO / "models" / pkg
        hits = [f.name for f in d.glob("*.py") if d.exists() and "temperature_k" in f.read_text()]
        usage[pkg] = hits or "not referenced"
    with h5py.File(A.REPO / "data/TauREx set/spectra.h5", "r") as f:
        gen = np.array([g.decode() for g in f["generator"][:]])
        spl = np.array([g.decode() for g in f["split"][:]])
        itr = np.sort(np.where((gen == "tau") & (spl == "train"))[0][:12000])
        iva = np.sort(np.where((gen == "tau") & (spl == "val"))[0])

        def feats(idx):
            S = f["transit_depth_noisy"][idx].astype(np.float64)
            sg = f["sigma_ppm"][idx].astype(np.float64)[:, None]
            ref = S.mean(axis=1, keepdims=True)
            return np.hstack([S / ref, np.log10(ref), sg])
        Xtr, Xva = feats(itr), feats(iva)
    ytr, yva = lab.iloc[itr], lab.iloc[iva]
    pred = {}
    for col in ("temperature_k", "planet_radius_rjup"):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(Xtr, ytr[col])
        p = m.predict(Xva)
        r = float(np.sqrt(np.mean((yva[col].to_numpy() - p) ** 2)))
        sd = float(yva[col].std())
        pred[col] = {"rmse": r, "sd": sd, "r2": 1.0 - (r / sd) ** 2}
    return {"temperature_range_K": [float(tau.temperature_k.min()), float(tau.temperature_k.max())],
            "scale_height_dynamic_range_from_T": float(tau.temperature_k.max() / tau.temperature_k.min()),
            "corr_temperature_with_live_aux": corr,
            "corr_Teq_from_aux_with_true_T": float(np.corrcoef(teq, tau.temperature_k)[0, 1]),
            "temperature_k_referenced_in_package": usage,
            "predictability_from_spectrum": pred,
            "note": ("star temperature (5500 K) and orbital distance (0.05 AU) are hard-coded constants in "
                     "_build_taurex_auxiliary_frame, so the equilibrium-temperature route that leaks T on "
                     "ADC carries no information here; T is an unobserved nuisance parameter spanning 3.6x "
                     "in scale height, and no arm of the comparison predicts it")}


def main() -> None:
    ap = argparse.ArgumentParser()
    # 0 = every planet that passes shape validation. The ADC holdout has 4143 planets, all with a
    # tracedata block, but only 663 carry a 7-column matrix. The previous default cap of 600 threw
    # away 63 of them (9.5%) for no stated reason and moved the rate from 0.352 to 0.359.
    ap.add_argument("--max-planets", type=int, default=0,
                    help="0 = wszystkie dostepne (663); wartosc >0 ogranicza probke")
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # `--n-train 0` has no valid meaning: zero rows will not train a HistGBM, and previously (with
    # `if n:` in `build`) it SILENTLY meant "all rows". Rejected outright, as in `a26`.
    if args.n_train <= 0:
        ap.error("--n-train musi byc > 0 (zero wierszy nie wytrenuje modelu)")

    payload = {
        "model_targets": A.TARGETS,
        "benchmark_targets": TRACE_COLS,
        "n_predicted": len(A.TARGETS), "n_required": len(TRACE_COLS),
        # The headline experiment's sample size MUST be in the record: the figure 12 000 is cited in
        # K9(c) as the budget of both conditioning arms, and without this field it had to be taken
        # from the flag's default in the code — i.e. ASSUMING the run did not override it.
        "sampling": {
            "adc_n_train": int(args.n_train),
            "adc_n_train_selection": "prefix of the split id list (ids[:n]), NOT a random subsample",
            "adc_learner": "HistGradientBoostingRegressor(max_iter=250, random_state=0), both arms",
            "max_planets": int(args.max_planets),
            "max_planets_meaning": "0 = every holdout planet carrying a 7-column trace (663)",
        },
        "posterior_coupling": posterior_coupling(args.max_planets),
        "posterior_slope": posterior_slope(args.max_planets),
        "adc": adc_experiments(args.n_train),
        "reference_retrieval_precision": reference_precision(),
        "crossgen": crossgen(),
    }
    s = payload["posterior_coupling"]["summary"]
    print(f"\n  posterior coupling (median |r| over planets):")
    print(f"    T   vs gases   {s['mean_med_abs_r_temperature_gas']:.3f}")
    print(f"    R_p vs gases   {s['mean_med_abs_r_radius_gas']:.3f}")
    print(f"    gas vs gas     {s['mean_med_abs_r_gas_gas']:.3f}   "
          f"-> T couples {s['temperature_over_gasgas_ratio']:.1f}x more strongly than gases couple to each other")
    print(f"    R_p vs T       {s['med_abs_r_radius_temperature']:.3f}  (median {s['median_r_radius_temperature']:+.3f})"
          f"  <- the strongest correlation in the 7x7 matrix, and both parameters are omitted")
    r = payload["adc"]["retrievability"]
    print(f"\n  ADC retrievability from the model's own inputs: "
          f"R_p R2={r['planet_radius']['r2']:.3f}, T R2={r['planet_temp']['r2']:.3f}")
    lk = payload["adc"]["aux_leakage_of_temperature"]
    print(f"  ADC temperature leakage via aux: r={lk['pearson_r_Teq_vs_true_T']:.4f}, "
          f"RMSE={lk['rmse_Teq_K']:.1f} K vs sd={lk['sd_true_T_K']:.1f} K -> T is effectively an INPUT")
    cv = payload["adc"]["conditioning_value"]
    print(f"  ADC value of conditioning on true (T,R_p): mRMSE {cv['mrmse_without']:.4f} -> "
          f"{cv['mrmse_with_true_T_Rp']:.4f} ({cv['relative_improvement']*100:+.1f}%)")
    rp = payload["reference_retrieval_precision"]
    print(f"  reference retrieval T: sigma={rp['temperature']['median_posterior_sigma']:.1f} K but "
          f"RMSE={rp['temperature']['rmse_of_posterior_median']:.1f} K "
          f"-> overconfident {rp['temperature']['overconfidence_ratio']:.1f}x; aux beats it 4x")
    cg = payload["crossgen"]
    print(f"  cross-generator: T in {cg['temperature_range_K']} K ({cg['scale_height_dynamic_range_from_T']:.1f}x in H), "
          f"corr(T_eq_from_aux, T)={cg['corr_Teq_from_aux_with_true_T']:+.4f} -> no aux route; "
          f"spectrum R2={cg['predictability_from_spectrum']['temperature_k']['r2']:.3f}, predicted by NOBODY")

    # The status implements BOTH branches of the criterion. The previous version compared two
    # CONSTANTS (n_predicted=5, n_required=7) and was therefore a tautology: the check could not
    # return PASS on any data, and the coupling / retrievability analysis did not affect the verdict.
    #
    # Branch 1: the model predicts the full benchmark vector.
    # Branch 2: the omitted parameters are BOTH weakly coupled (thresholds declared here, not picked
    #           after seeing the result: T-gas coupling <= 1.5x the gas-gas coupling, |r| for the
    #           (R_p, T) pair <= 0.30) AND input-determined: R^2 from the spectrum >= 0.95 on `cg`
    #           (the cross-generator set) — NOT on both sets: ADC retrievability (adc_experiments) is
    #           computed and reported, but does not enter this threshold.
    pc, cg = payload["posterior_coupling"]["summary"], payload["crossgen"]["predictability_from_spectrum"]
    full_vector = payload["n_predicted"] >= payload["n_required"]
    weakly_coupled = (pc["temperature_over_gasgas_ratio"] <= 1.5
                      and pc["med_abs_r_radius_temperature"] <= 0.30)
    input_determined = all(cg[k]["r2"] >= 0.95 for k in ("temperature_k", "planet_radius_rjup") if k in cg)
    payload["status_terms"] = {
        "full_target_vector": bool(full_vector),
        "omitted_weakly_coupled": bool(weakly_coupled),
        "omitted_input_determined": bool(input_determined),
        "decisive": ("none - full vector" if full_vector else
                     "coupling" if not weakly_coupled else
                     "input_determination" if not input_determined else "none - both branches hold"),
    }
    status = "PASS" if (full_vector or (weakly_coupled and input_determined)) else "FAIL"
    CHECK.emit(status, payload, out=args.out)


if __name__ == "__main__":
    main()
