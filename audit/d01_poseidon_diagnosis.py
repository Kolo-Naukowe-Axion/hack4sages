"""D01 — Why are the 685 POSEIDON spectra flat? Tiered diagnosis.

`a01` establishes the fact (685/685 rows are wavelength-constant). This script finds the cause.

The repo's POSEIDON backend is a WRAPPER around the real library (`from POSEIDON.core import ...`),
not a reimplementation, so the defect must be in how the API is called. Three hypotheses survived
review; two earlier ones were refuted by the POSEIDON tutorial and are recorded here so nobody
re-tests them:

  REFUTED  log_X_params as a flat vector      -> the tutorial uses exactly that: np.array([-3.3, -5.0])
  REFUTED  He_fraction = 0.17647 is He/(H2+He) -> the tutorial states X_He/X_H2 = 0.17, i.e. He/H2
  REFUTED  wrong array length from the backend -> both rebin paths would raise on a shape mismatch

  LIVE  H1  pressure grid direction: the repo passes np.geomspace(1e-6, 100) = INCREASING,
            the POSEIDON tutorial uses np.logspace(2, -7, 100) = DECREASING.
  WEAK  H2  "the High-T database was never staged because bootstrap_poseidon_input_data.sh:10
            short-circuits on the STANDARD Opacity_database_v1.3.hdf5" — the premise is wrong:
            Opacity_database_v1.3.hdf5 IS the High-T database for database_version='1.3'
            (POSEIDON/absorption.py:784,788). The bootstrap guard therefore checks for exactly the
            file the backend needs. H2 survives only as "the file is absent altogether", which is
            what Stage 0 now tests by name instead of by a substring match on "high".
  LIVE  H3  R_p_ref / P_ref combination collapses the atmosphere.

STAGES — each one answers as much as the available resources allow:

  Stage 0  environment probe.        Needs nothing. Always runs.
  Stage 1  atmosphere geometry.      Needs POSEIDON installed. Does NOT need the 72 GB opacity data,
                                     because make_atmosphere does not read opacities. This is the
                                     stage that tests H1 and H3 — run this first.
  Stage 2  full spectra.             Needs the opacity database. Tests H2 and confirms Stage 1.

Nothing is written and nothing in the repo is modified; results go to the usual audit output dir.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="d01_poseidon_diagnosis",
    finding="K1 — root-cause diagnosis for the wavelength-constant POSEIDON spectra",
    question="Which API misuse makes POSEIDON return spectra with no wavelength dependence?",
    criterion="the repo's exact call reproduces a spectrum with the same relative variation as tau (~1e-2)",
)

MIN_REL_VARIATION = 1e-4  # same criterion as a01


def rel_variation(x: np.ndarray) -> float:
    """Zmiennosc wzgledna widma. mean == 0 to BRAK zmiennosci (0.0), nie inf.

    Wczesniej zwracalo float("inf"), a inf <= MIN_REL_VARIATION daje False — czyli widmo z samych
    zer (albo dokladnie antysymetryczne wokol zera) przechodzilo `flat_by_a01_criterion` jako "ma
    zmiennosc", i to w checku, ktorego jedynym zadaniem jest wykrycie plaskich widm. Konwencja
    ujednolicona z a01:61, gdzie ten sam blad byl juz naprawiony (`np.where(|mean| > 0, std/|mean|,
    0.0)`), bo dwie rozne konwencje na te sama wielkosc same sa wada.

    Dzis nieosiagalne: stage2 nie uruchamia sie bez 72 GB opacities, a a01 mierzy 0/42108 wierszy
    z mean_bins == 0 (min |mean| = 3.26e-03). To zabezpieczenie kryterium, wiec nie moze zalezec od
    tego, ze dane akurat sa dobre.
    """
    x = np.asarray(x, dtype=np.float64)
    m = float(x.mean())
    return float(x.std() / abs(m)) if m != 0.0 else 0.0


def nearest_log_pressure_index(P: np.ndarray, P_ref: float) -> int:
    """Indeks warstwy najblizszej P_ref, mierzony w log10.

    Siatka cisnien jest LOGARYTMICZNA (np.geomspace / np.logspace, 1e-6..100 bar w 100 warstwach),
    a poprzednio blizszosc byla liczona jako argmin |P - P_ref| w skali LINIOWEJ. W skali liniowej
    odleglosci sa zdominowane przez warstwy wysokocisnieniowe: krok nad P_ref jest wielokrotnie
    wiekszy niz pod nim, wiec "najblizsza" warstwa systematycznie wypada ponizej P_ref.

    Dla dzisiejszej siatki i P_ref = 10 bar oba kryteria wskazuja TEN SAM indeks (87 dla siatki
    repo, 12 dla tutorialowej), wiec zadna raportowana liczba sie nie zmienia. Poprawka jest
    strukturalna: przy innym P_ref (albo innej liczbie warstw) kryteria by sie rozjechaly, a
    wielkosci nazwane "r at P_ref" i "H at P_ref" musza dotyczyc warstwy najblizszej w tej metryce,
    w ktorej siatka jest rownomierna.
    """
    P = np.asarray(P, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(np.log10(P) - np.log10(float(P_ref)))
    return int(np.nanargmin(d))


# STAGE 0

def expected_opacity_filename(database: str, version: str) -> str | None:
    """Plik, ktory POSEIDON FAKTYCZNIE otwiera dla danej pary (opacity_database, database_version).

    Odwzorowanie przepisane z zainstalowanego POSEIDON 1.4.0, absorption.py:781-802:
        High-T    + '1.3' -> opacity/Opacity_database_v1.3.hdf5
        High-T    + '1.2' -> opacity/Opacity_database_v1.2.hdf5
        High-T    + '1.0' -> opacity/Opacity_database_v1.0.hdf5
        Temperate         -> opacity/Opacity_database_0.01cm-1_Temperate.hdf5
    """
    if database == "Temperate":
        return "Opacity_database_0.01cm-1_Temperate.hdf5"
    if database == "High-T":
        return f"Opacity_database_v{version}.hdf5"
    return None


def repo_opacity_request() -> tuple[str, str]:
    """Ktora baze i wersje zamawia backend repo. Fallback = wartosci z constants.py na 2026-07."""
    try:
        sys.path.insert(0, str(A.REPO))
        from data.crossgen_biosignatures.constants import (
            POSEIDON_DATABASE_VERSION, POSEIDON_OPACITY_DATABASE)
        return str(POSEIDON_OPACITY_DATABASE), str(POSEIDON_DATABASE_VERSION)
    except Exception:
        return "High-T", "1.3"


def stage0() -> dict:
    db, ver = repo_opacity_request()
    want = expected_opacity_filename(db, ver)
    out: dict = {"poseidon_importable": False, "poseidon_version": None, "input_data_root": None,
                 "opacity_files": [], "requested_opacity_database": db,
                 "requested_database_version": ver, "expected_opacity_filename": want,
                 "high_T_database_present": None, "blockers": []}
    root = os.environ.get("POSEIDON_input_data")
    if root is None:
        for cand in (Path.home() / "poseidon-inputs", Path.home() / "hack4sages" / "input_data"):
            if cand.exists():
                root = str(cand)
                break
    out["input_data_root"] = root
    try:
        import importlib.metadata as im
        import POSEIDON  # noqa: F401
        out["poseidon_importable"] = True
        for name in ("POSEIDON", "poseidon"):
            try:
                out["poseidon_version"] = im.version(name)
                break
            except Exception:
                continue
    except Exception as exc:
        out["blockers"].append(f"POSEIDON not importable: {type(exc).__name__}: {exc}")
    if root and Path(root).exists():
        opac = Path(root) / "opacity"
        if opac.exists():
            files = sorted(p.name for p in opac.glob("*.hdf5"))
            out["opacity_files"] = files
            # Wczesniej: any("high" in f.lower() for f in files). To byl FALSZYWY BLOKER, i to
            # odwrocony: POSEIDON dla opacity_database="High-T" otwiera plik
            # Opacity_database_v1.3.hdf5 (absorption.py:784,788), w ktorego nazwie slowa "high"
            # NIE MA w ogole. Na POPRAWNIE zainstalowanych danych check zglaszal wiec brak bazy
            # High-T, czyli podtrzymywal hipoteze H2 dokladnie w sytuacji, ktora ja obala — i
            # odwrotnie, dowolny plik z "high" w nazwie (np. Opacity_high_res_test.hdf5) zdejmowal
            # blokera bez zadnego zwiazku z tym, co backend laduje.
            out["high_T_database_present"] = bool(want and want in files)
            if want and not out["high_T_database_present"]:
                out["blockers"].append(
                    f"opacity dir has no {want!r} -> `opacity_database={db!r}`, "
                    f"`database_version={ver!r}` (data/crossgen_biosignatures/constants.py) resolves "
                    f"to opacity/{want} in POSEIDON.absorption.py and there is nothing to load. "
                    f"Present instead: {files or '(none)'}. This is hypothesis H2.")
            elif want is None:
                out["blockers"].append(
                    f"opacity_database={db!r} is not one of POSEIDON's known values "
                    "('High-T', 'Temperate'); cannot tell which file it would open")
        else:
            out["blockers"].append(f"{opac} does not exist")
    else:
        out["blockers"].append(
            "POSEIDON input data not staged. Stage 1 does NOT need it; Stage 2 does "
            "(zenodo record 16107813, inputs.zip, 72.1 GB, ~140 GB peak disk while unzipping).")
    return out


# SHARED SETUP

def build_repo_objects(sample: dict):
    """Recreate exactly what poseidon_backend.py builds, with no modifications."""
    sys.path.insert(0, str(A.REPO))
    from data.crossgen_biosignatures.constants import (
        BACKGROUND_HE_TO_H2_RATIO, FIXED_PLANET_SEMIMAJOR_AXIS_AU, FIXED_STAR_LOG_G_CGS,
        FIXED_STAR_METALLICITY, FIXED_STAR_TEMPERATURE_K, FIXED_SYSTEM_DISTANCE_PC,
        JUPITER_RADIUS_M, PARSEC_M, PRESSURE_LEVELS, PRESSURE_MAX_BAR, PRESSURE_MIN_BAR,
        REFERENCE_PRESSURE_BAR, SOLAR_RADIUS_M, TARGET_WAVELENGTH_MAX_UM,
        TARGET_WAVELENGTH_MIN_UM, TRACE_SPECIES, POSEIDON_NATIVE_RESOLUTION,
    )
    from POSEIDON.core import create_planet, create_star, define_model, wl_grid_constant_R

    wl = np.asarray(wl_grid_constant_R(TARGET_WAVELENGTH_MIN_UM, TARGET_WAVELENGTH_MAX_UM,
                                       POSEIDON_NATIVE_RESOLUTION), dtype=np.float64)
    model = define_model(model_name="crossgen_diagnosis", bulk_species=["H2", "He"],
                         param_species=[n for _, n in TRACE_SPECIES], object_type="transiting",
                         PT_profile="isotherm", X_profile="isochem", cloud_model="cloud-free",
                         gravity_setting="fixed", mass_setting="fixed", stellar_contam=None)
    star = create_star(R_s=float(sample["star_radius_rsun"]) * SOLAR_RADIUS_M,
                       T_eff=FIXED_STAR_TEMPERATURE_K, log_g=FIXED_STAR_LOG_G_CGS,
                       Met=FIXED_STAR_METALLICITY, stellar_grid="blackbody", wl=wl)
    r_p_m = float(sample["planet_radius_rjup"]) * JUPITER_RADIUS_M
    # NIE "naprawiaj" d na parseki. Recenzja zarzucila blad jednostki; zarzut jest FALSZYWY.
    # W zainstalowanym POSEIDON 1.4.0, core.py:337 docstring create_planet mowi wprost
    # "d (float): Distance to system (m)", a a_p analogicznie "(m) -- NOT in AU". Kod podaje
    # metry (PC * PARSEC_M, AU * 1.495978707e11) i jest poprawny; zamiana na parseki wprowadzilaby
    # blad rzedu 3e16.
    planet = create_planet(planet_name=str(sample["sample_id"]), R_p=r_p_m,
                           log_g=float(sample["log_g_cgs"]), T_eq=float(sample["temperature_k"]),
                           d=FIXED_SYSTEM_DISTANCE_PC * PARSEC_M,
                           a_p=FIXED_PLANET_SEMIMAJOR_AXIS_AU * 1.495978707e11)
    grids = {
        "repo_increasing": np.geomspace(PRESSURE_MIN_BAR, PRESSURE_MAX_BAR, PRESSURE_LEVELS),
        "tutorial_decreasing": np.logspace(np.log10(PRESSURE_MAX_BAR), np.log10(PRESSURE_MIN_BAR),
                                           PRESSURE_LEVELS),
    }
    cfg = {"wl": wl, "model": model, "star": star, "planet": planet, "r_p_m": r_p_m,
           "r_s_m": float(sample["star_radius_rsun"]) * SOLAR_RADIUS_M,
           "P_ref": REFERENCE_PRESSURE_BAR, "He_fraction": BACKGROUND_HE_TO_H2_RATIO,
           "log_X": np.asarray([float(sample[f"log10_vmr_{k}"]) for k, _ in TRACE_SPECIES]),
           "T": float(sample["temperature_k"]), "grids": grids,
           "species": [n for _, n in TRACE_SPECIES]}
    return cfg


# STAGE 1

def stage1(cfg: dict, sample: dict) -> dict:
    """Atmosphere geometry only. No opacities needed — this is the cheap, decisive test of H1/H3.

    Also runs the saturation test (`saturation_test` below), which is what actually narrowed K1(c):
    geometry alone comes out healthy, so the question becomes which radius the recorded depth
    corresponds to.
    """
    from POSEIDON.core import make_atmosphere
    res: dict = {"grids": {}}
    for label, P in cfg["grids"].items():
        P = np.asarray(P, dtype=np.float64)
        entry = {"P_first": float(P[0]), "P_last": float(P[-1]),
                 "monotonic": "increasing" if P[1] > P[0] else "decreasing"}
        try:
            atm = make_atmosphere(planet=cfg["planet"], model=cfg["model"], P=P,
                                  P_ref=cfg["P_ref"], R_p_ref=cfg["r_p_m"],
                                  PT_params=np.asarray([cfg["T"]], dtype=np.float64),
                                  log_X_params=cfg["log_X"], He_fraction=cfg["He_fraction"])
            r = np.asarray(atm["r"], dtype=np.float64).ravel()
            P_of_layer = np.asarray(atm.get("P", P), dtype=np.float64).ravel()
            entry.update({
                "ok": True,
                "atmosphere_keys": sorted(k for k in atm if not k.startswith("_")),
                "r_min_km": float(np.nanmin(r) / 1e3), "r_max_km": float(np.nanmax(r) / 1e3),
                "vertical_extent_km": float((np.nanmax(r) - np.nanmin(r)) / 1e3),
                "extent_over_Rp": float((np.nanmax(r) - np.nanmin(r)) / cfg["r_p_m"]),
                "r_has_nan": bool(np.isnan(r).any()),
                "T_profile_unique": int(len(np.unique(np.asarray(atm["T"]).round(6)))) if "T" in atm else None,
            })

            # H3 in its sharp form: does the layer at P_ref sit exactly at the R_p_ref we passed in?
            j = None
            if len(P_of_layer) == len(r):
                j = nearest_log_pressure_index(P_of_layer, cfg["P_ref"])
                entry["P_ref_bar"] = float(cfg["P_ref"])
                entry["P_ref_layer_index_metric"] = "min |log10(P) - log10(P_ref)| (grid is logarithmic)"
                entry["P_ref_layer_index"] = j
                entry["P_ref_layer_pressure_bar"] = float(P_of_layer[j])
                entry["r_at_P_ref_km"] = float(r[j] / 1e3)
                entry["r_at_P_ref_minus_R_p_ref_m"] = float(r[j] - cfg["r_p_m"])
                entry["r_at_P_ref_over_R_p_ref"] = float(r[j] / cfg["r_p_m"])

            if "H" in atm:
                H_full = np.asarray(atm["H"], dtype=np.float64).ravel()
                H = H_full[np.isfinite(H_full)]
                if H.size:
                    # H is NOT constant under an isotherm: g falls as 1/r^2 over 18 scale heights, so the
                    # median, the endpoints and the value at P_ref differ in the third digit. Record all
                    # four, so a quoted "H = ... km" names one of them instead of an unspecified average.
                    entry["scale_height_km"] = float(np.median(H) / 1e3)
                    entry["scale_height_min_km"] = float(np.min(H) / 1e3)
                    entry["scale_height_max_km"] = float(np.max(H) / 1e3)
                    entry["scale_height_is_constant"] = bool(np.allclose(H, H[0], rtol=1e-9))
                    entry["extent_over_H"] = float((np.nanmax(r) - np.nanmin(r)) / np.median(H))
                    if len(H_full) == len(r) and j is not None:
                        # ten sam indeks warstwy co wyzej, nie druga niezalezna kopia argmina
                        entry["scale_height_at_P_ref_km"] = float(H_full[j] / 1e3)

            if "X" in atm:
                X = np.asarray(atm["X"], dtype=np.float64)
                entry["X_shape"] = list(X.shape)
                entry["X_per_species_median_log10"] = [float(np.log10(np.median(X[i]) + 1e-300))
                                                       for i in range(min(X.shape[0], 8))]
                if X.shape[0] >= 2:
                    h2, he = float(np.median(X[0])), float(np.median(X[1]))
                    entry["background_H2_fraction"] = h2
                    entry["background_He_fraction"] = he
                    entry["background_He_over_H2"] = he / h2 if h2 else None
                    entry["background_He_over_H2_requested"] = float(cfg["He_fraction"])
        except Exception as exc:
            entry.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        res["grids"][label] = entry

    a, b = res["grids"].get("repo_increasing", {}), res["grids"].get("tutorial_decreasing", {})
    if a.get("ok") and b.get("ok"):
        ea, eb = a["vertical_extent_km"], b["vertical_extent_km"]
        res["grids_bit_identical"] = bool(
            ea == eb and a["r_min_km"] == b["r_min_km"] and a["r_max_km"] == b["r_max_km"])
        res["verdict_H1"] = (
            "CONFIRMED — the repo's increasing grid collapses the atmosphere "
            f"({ea:.1f} km vs {eb:.1f} km with the tutorial ordering)"
            if eb > 0 and ea < 0.05 * eb else
            f"NOT confirmed by geometry — extents comparable ({ea:.1f} vs {eb:.1f} km); "
            "the geometry is healthy, so the cause is radiative — see saturation_test")
    elif a.get("ok") is False and b.get("ok") is True:
        res["verdict_H1"] = f"CONFIRMED — the repo's ordering raises: {a.get('error')}"

    if a.get("ok"):
        res["saturation_test"] = saturation_test(cfg, sample, a)
    return res


SATURATION_TOL = 5e-3  # a candidate "explains" the recorded depth if the ratio is within 0.5 %


def saturation_test(cfg: dict, sample: dict, geometry: dict) -> dict:
    """Which radius does the recorded flat depth correspond to?

    The stored POSEIDON depth is wavelength-constant, so it equals (r/R_s)^2 for a single r. Comparing
    that r against the bottom of the atmosphere, the reference level and the top of the atmosphere
    separates "the opacities never loaded" (depth at or below R_p_ref) from "the opacities saturated"
    (depth at the top of the grid, i.e. tau = 1 already at the topmost layer).
    """
    import h5py
    out: dict = {"sample_id": str(sample["sample_id"]), "R_s_km": cfg["r_s_m"] / 1e3}
    with h5py.File(A.REPO / "data/TauREx set/spectra.h5", "r") as f:
        ids = np.array([s.decode() for s in f["sample_id"][:]])
        hit = np.flatnonzero(ids == str(sample["sample_id"]))
        if not hit.size:
            return {**out, "error": "sample_id not found in spectra.h5"}
        row = np.asarray(f["transit_depth_noiseless"][int(hit[0])], dtype=np.float64)

    out["recorded_depth_n_unique_bins"] = int(len(np.unique(row)))
    out["recorded_depth"] = float(np.median(row))
    out["implied_radius_km"] = float(np.sqrt(out["recorded_depth"]) * cfg["r_s_m"] / 1e3)

    candidates = {
        "atmosphere_bottom": geometry["r_min_km"],
        "R_p_ref_at_P_ref": geometry.get("r_at_P_ref_km"),
        "atmosphere_top": geometry["r_max_km"],
    }
    out["candidates"] = {}
    for name, r_km in candidates.items():
        if r_km is None:
            continue
        depth = (r_km * 1e3 / cfg["r_s_m"]) ** 2
        out["candidates"][name] = {"r_km": r_km, "depth": depth,
                                   "ratio_to_recorded": depth / out["recorded_depth"]}

    matches = [n for n, d in out["candidates"].items()
               if abs(d["ratio_to_recorded"] - 1.0) < SATURATION_TOL]
    out["tolerance"] = SATURATION_TOL
    out["matching_candidates"] = matches
    out["verdict_opacity"] = (
        "SATURATED — the recorded depth is the top-of-atmosphere radius, so extinction reached "
        "tau = 1 at the topmost layer; the opacities were loaded and are too large, NOT zero"
        if matches == ["atmosphere_top"] else
        "ZERO/UNLOADED — the recorded depth is at or below the reference level, consistent with no "
        "extinction at all" if matches and matches != ["atmosphere_top"] else
        "inconclusive — no candidate radius reproduces the recorded depth within tolerance")
    return out


# STAGE 2

def stage2(cfg: dict, sample: dict) -> dict:
    from POSEIDON.core import compute_spectrum, make_atmosphere, read_opacities
    sys.path.insert(0, str(A.REPO))
    from data.crossgen_biosignatures.constants import (
        POSEIDON_DATABASE_VERSION, POSEIDON_FINE_LOG10_PRESSURE_BAR,
        POSEIDON_FINE_TEMPERATURE_GRID_K, POSEIDON_OPACITY_DATABASE, POSEIDON_OPACITY_TREATMENT,
    )
    res: dict = {}
    try:
        opac = read_opacities(cfg["model"], cfg["wl"], opacity_treatment=POSEIDON_OPACITY_TREATMENT,
                              T_fine=np.asarray(POSEIDON_FINE_TEMPERATURE_GRID_K, dtype=np.float64),
                              log_P_fine=np.asarray(POSEIDON_FINE_LOG10_PRESSURE_BAR, dtype=np.float64),
                              opacity_database=POSEIDON_OPACITY_DATABASE, device="cpu",
                              database_version=POSEIDON_DATABASE_VERSION)
    except Exception as exc:
        res["read_opacities_error"] = f"{type(exc).__name__}: {exc}"
        res["verdict_H2"] = ("CONFIRMED — the requested opacity database cannot be loaded. "
                             "Note the generator would have failed loudly, so if the shipped "
                             "dataset was produced anyway, a DIFFERENT database was in place.")
        return res
    for label, P in cfg["grids"].items():
        try:
            atm = make_atmosphere(planet=cfg["planet"], model=cfg["model"], P=P, P_ref=cfg["P_ref"],
                                  R_p_ref=cfg["r_p_m"],
                                  PT_params=np.asarray([cfg["T"]], dtype=np.float64),
                                  log_X_params=cfg["log_X"], He_fraction=cfg["He_fraction"])
            spec = np.asarray(compute_spectrum(planet=cfg["planet"], star=cfg["star"],
                                               model=cfg["model"], atmosphere=atm, opac=opac,
                                               wl=cfg["wl"], spectrum_type="transmission",
                                               suppress_print=True), dtype=np.float64)
            res[label] = {"ok": True, "n_bins": int(spec.size), "mean": float(spec.mean()),
                          "rel_variation": rel_variation(spec),
                          "n_unique": int(len(np.unique(spec.astype(np.float32)))),
                          "flat_by_a01_criterion": bool(rel_variation(spec) <= MIN_REL_VARIATION)}
        except Exception as exc:
            res[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    a, b = res.get("repo_increasing", {}), res.get("tutorial_decreasing", {})
    if a.get("ok") and b.get("ok"):
        res["verdict"] = (
            f"repo ordering rel.var = {a['rel_variation']:.3e} "
            f"(a01 threshold {MIN_REL_VARIATION:g}); tutorial ordering = {b['rel_variation']:.3e}. "
            + ("H1 CONFIRMED: the ordering alone explains the flat spectra."
               if a["flat_by_a01_criterion"] and not b["flat_by_a01_criterion"]
               else "H1 does not explain it: both orderings behave the same. Suspect H2/H3."))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default=None, help="default: the first POSEIDON row in labels.parquet")
    ap.add_argument("--stage", default="auto", choices=["0", "1", "2", "auto"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    payload: dict = {"stage0": stage0()}
    print("STAGE 0 — environment")
    for k, v in payload["stage0"].items():
        print(f"  {k}: {v}")

    import pandas as pd
    lab = pd.read_parquet(A.REPO / "data/TauREx set/labels.parquet")
    pos = lab[lab.generator == "poseidon"]
    row = pos[pos.sample_id == args.sample_id].iloc[0] if args.sample_id else pos.iloc[0]
    sample = row.to_dict()
    payload["sample_id"] = str(sample["sample_id"])
    print(f"\n  reference row: {sample['sample_id']}  T={sample['temperature_k']:.1f} K  "
          f"R_p={sample['planet_radius_rjup']:.3f} Rjup  log_g={sample['log_g_cgs']:.3f}")

    if not payload["stage0"]["poseidon_importable"]:
        payload["stopped_at"] = "stage0"
        payload["next_step"] = (
            "pip install 'git+https://github.com/MartianColonist/POSEIDON.git' into a scratch venv, "
            "then re-run. Stage 1 needs NOTHING else — no opacity download.")
        print(f"\n  POSEIDON not importable -> stopping. {payload['next_step']}")
        CHECK.emit("INFO", payload, out=args.out)
        return

    cfg = build_repo_objects(sample)
    if args.stage in ("1", "auto"):
        print("\nSTAGE 1 — atmosphere geometry (no opacity data needed)")
        payload["stage1"] = stage1(cfg, sample)
        for label, e in payload["stage1"]["grids"].items():
            if e.get("ok"):
                print(f"  {label:22} P {e['P_first']:.1e} -> {e['P_last']:.1e} bar ({e['monotonic']})  "
                      f"extent = {e['vertical_extent_km']:.1f} km  ({e['extent_over_Rp']:.4f} R_p)")
                if "scale_height_km" in e:
                    print(f"  {'':22} H = {e['scale_height_km']:.2f} km median "
                          f"({e['scale_height_min_km']:.2f}–{e['scale_height_max_km']:.2f}, "
                          f"{e.get('scale_height_at_P_ref_km', float('nan')):.2f} at P_ref) "
                          f"-> extent = {e['extent_over_H']:.2f} H")
                if "r_at_P_ref_km" in e:
                    print(f"  {'':22} r at P_ref = {e['P_ref_layer_pressure_bar']:.3g} bar is "
                          f"{e['r_at_P_ref_km']:.1f} km, R_p_ref offset "
                          f"{e['r_at_P_ref_minus_R_p_ref_m']:+.3g} m")
                if "background_He_over_H2" in e:
                    print(f"  {'':22} background H2 {e['background_H2_fraction']:.4f} / He "
                          f"{e['background_He_fraction']:.4f} -> He/H2 = "
                          f"{e['background_He_over_H2']:.5f} (requested "
                          f"{e['background_He_over_H2_requested']:.5f})")
            else:
                print(f"  {label:22} FAILED: {e.get('error')}")
        print(f"  => {payload['stage1'].get('verdict_H1', 'inconclusive')}")

        st = payload["stage1"].get("saturation_test")
        if st and "error" not in st:
            print(f"\n  saturation test — recorded depth {st['recorded_depth']:.8f} "
                  f"({st['recorded_depth_n_unique_bins']} unique bin value(s)) implies "
                  f"r = {st['implied_radius_km']:.1f} km against R_s = {st['R_s_km']:.1f} km")
            print(f"     {'candidate':22} {'r [km]':>12} {'depth':>12} {'ratio':>9}")
            for name, d in st["candidates"].items():
                print(f"     {name:22} {d['r_km']:12.1f} {d['depth']:12.8f} "
                      f"{d['ratio_to_recorded']:9.5f}")
            print(f"  => {st['verdict_opacity']}")

    if args.stage in ("2", "auto"):
        if payload["stage0"]["input_data_root"] and Path(payload["stage0"]["input_data_root"]).exists():
            print("\nSTAGE 2 — full spectra (opacity data present)")
            payload["stage2"] = stage2(cfg, sample)
            for k, v in payload["stage2"].items():
                print(f"  {k}: {v}")
        else:
            payload["stage2_skipped"] = ("opacity data not staged; Stage 2 needs zenodo 16107813 "
                                         "inputs.zip = 72.1 GB (~140 GB peak disk)")
            print(f"\nSTAGE 2 skipped — {payload['stage2_skipped']}")

    status = "INFO"
    s1 = payload.get("stage1") or {}
    if s1.get("verdict_H1", "").startswith("CONFIRMED"):
        status = "FAIL"
    elif (s1.get("saturation_test") or {}).get("verdict_opacity", "").startswith(("SATURATED", "ZERO")):
        status = "FAIL"
    CHECK.emit(status, payload, out=args.out)


if __name__ == "__main__":
    main()
