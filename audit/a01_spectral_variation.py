"""A01 — Does every spectrum actually depend on wavelength, and above the noise it was made for?

Proves/disproves finding K1. This is the check that data/crossgen_biosignatures/validate_dataset.py
does NOT perform: if a spectrum varies across bins. A generator that returns a bare transit depth passes that
validator and silently voids every downstream cross-generator number.

TWO CRITERIA, deliberately independent — the verdict must not rest on a tuned threshold:

  (1) STRUCTURAL, threshold-free: no row may be bit-constant across the spectral axis.
      `n_unique == 1` is a combinatorial fact, not a measurement.

  (2) PHYSICAL: the feature amplitude must exceed the observational noise the set was generated
      with. `std_bins(noiseless) / sigma > 1` for the median row. A dataset whose features sit
      below its own noise floor is unusable regardless of how it was produced.

  (3) median `std_bins/|mean_bins| > 1e-4`. Transit depths differ by
      orders of magnitude between planets, so an absolute threshold on `std` would reject small
      planets and pass large ones; the ratio is comparable across rows. 1e-4 sits ~3 decades above
      the float32 representation floor (~1.2e-7) and ~2 decades below real structure (~1.3e-2), so
      it can be moved by two orders in either direction without changing any verdict.

WHAT THIS CHECK CANNOT CATCH. It only detects *absent* structure.
A generator that produces structure of the *wrong amplitude* passes: the TauREx side of this very
dataset is generated without collision-induced absorption (finding K11) and is scored OK here.
Catching that needs a physical-amplitude check against `2·R_p·H/R_star^2`; no such check exists yet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a01_spectral_variation",
    finding="K1 — all 685 POSEIDON spectra are wavelength-constant; the cross-generator axis has no signal",
    question="Does each stored spectrum vary across wavelength bins, and does that variation exceed the noise?",
    criterion="zero bit-constant rows AND median feature/sigma > 1 AND median std_bins/|mean_bins| > 1e-4",
)

MIN_REL_VARIATION = 1e-4
MIN_FEATURE_SNR = 1.0
NOISELESS_FIELD = "transit_depth_noiseless"
NOISY_FIELD = "transit_depth_noisy"


def summarise(arr: np.ndarray, sigma_abs: np.ndarray, is_noiseless: bool) -> dict:
    """arr: (N, n_bins) float64. sigma_abs: (N,) absolute noise sigma in transit-depth units."""
    mean_bins = arr.mean(axis=1)
    std_bins = arr.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(mean_bins) > 0, std_bins / np.abs(mean_bins), 0.0)
    n_unique = np.array([len(np.unique(row)) for row in arr])
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sigma_abs > 0, std_bins / np.where(sigma_abs > 0, sigma_abs, 1.0),
                       np.where(std_bins > 0, np.inf, 0.0))
    return {
        "n_rows_checked": int(len(arr)),
        "field_is_noiseless": is_noiseless,
        "rel_variation_median": float(np.median(rel)),
        "rel_variation_p01": float(np.percentile(rel, 1)),
        "rel_variation_max": float(rel.max()),
        "frac_rows_below_rel_threshold": float((rel <= MIN_REL_VARIATION).mean()),
        "bit_constant_rows": int((n_unique == 1).sum()),
        "bit_constant_fraction": float((n_unique == 1).mean()),
        "n_unique_values_median": int(np.median(n_unique)),
        # for the noiseless field this is the true feature amplitude; for the noisy field the
        # bin-to-bin scatter also contains the injected noise, so it is only an APPARENT amplitude
        ("feature_amplitude_over_sigma_median" if is_noiseless
         else "apparent_amplitude_over_sigma_median"): float(np.nanmedian(snr)),
        "frac_rows_snr_gt_1": float(np.nanmean(snr > 1.0)),
        "frac_rows_snr_gt_3": float(np.nanmean(snr > 3.0)),
        "first_row_n_unique": int(n_unique[0]),
        "first_row_head": arr[0][:6].tolist(),
    }


def main() -> None:
    import h5py
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(A.REPO / "data/TauREx set/spectra.h5"),
                    help="NOTE: this single file holds BOTH generators; rows are separated by the "
                         "`generator` field, not by directory. The directory name is misleading.")
    ap.add_argument("--max-rows-per-generator", type=int, default=0,
                    help="0 = check every row (default). A positive value takes a SEEDED RANDOM "
                         "sample, never a prefix — split assignment is positional, so a prefix "
                         "would only ever look at the train shard.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.h5)
    rng = np.random.default_rng(args.seed)
    payload: dict = {"file": str(path), "criteria": {
        "structural": "no bit-constant row (threshold-free)",
        "physical": f"median feature amplitude / sigma > {MIN_FEATURE_SNR}",
        "scale_free": f"median std_bins/|mean_bins| > {MIN_REL_VARIATION:g}"},
        "cannot_catch": ("structure of the wrong amplitude — e.g. a generator missing an opacity "
                         "term (finding K11) passes this check; no dedicated amplitude check exists yet"),
        "generators": {}}

    with h5py.File(path, "r") as f:
        gen = np.array([g.decode() for g in f["generator"][:]])
        sigma_ppm_all = f["sigma_ppm"][:].astype(np.float64)
        for name in sorted(set(gen)):
            all_idx = np.where(gen == name)[0]
            if args.max_rows_per_generator and args.max_rows_per_generator < len(all_idx):
                idx = np.sort(rng.choice(all_idx, size=args.max_rows_per_generator, replace=False))
                sampling = f"seeded random sample of {len(idx)}/{len(all_idx)}"
            else:
                idx = np.sort(all_idx)
                sampling = f"all {len(idx)} rows"
            sigma_abs = sigma_ppm_all[idx] * 1e-6
            entry = {"sampling": sampling, "n_total_rows": int(len(all_idx)), "fields": {}}
            for field in (NOISELESS_FIELD, NOISY_FIELD):
                if field not in f:
                    continue
                arr = f[field][idx].astype(np.float64)
                entry["fields"][field] = summarise(arr, sigma_abs, field == NOISELESS_FIELD)
            payload["generators"][name] = entry

    failures = []
    for g, entry in payload["generators"].items():
        ref = entry["fields"].get(NOISELESS_FIELD) or next(iter(entry["fields"].values()))
        reasons = []
        if ref["bit_constant_rows"] > 0:
            reasons.append(f"{ref['bit_constant_rows']}/{ref['n_rows_checked']} rows are bit-constant")
        snr = ref.get("feature_amplitude_over_sigma_median", ref.get("apparent_amplitude_over_sigma_median"))
        if snr is not None and not (snr > MIN_FEATURE_SNR):
            reasons.append(f"median feature/sigma = {snr:.3f} <= {MIN_FEATURE_SNR}")
        if ref["rel_variation_median"] <= MIN_REL_VARIATION:
            reasons.append(f"median rel. variation = {ref['rel_variation_median']:.3e} "
                           f"<= {MIN_REL_VARIATION:g}")
        if reasons:
            failures.append({"generator": g, "reasons": reasons})

    payload["failing_generators"] = failures
    payload["consequence"] = (
        "Any metric computed on a failing generator measures the model's response to a featureless "
        "input. Every mRMSE / gap / ranking derived from it must be withdrawn."
        if failures else "All generators carry wavelength structure above their own noise floor.")

    for g, entry in payload["generators"].items():
        print(f"  {g}  ({entry['sampling']})")
        for fld, s in entry["fields"].items():
            snr_key = ("feature_amplitude_over_sigma_median" if s["field_is_noiseless"]
                       else "apparent_amplitude_over_sigma_median")
            print(f"    {fld:26} rel.var med={s['rel_variation_median']:.3e} "
                  f"p01={s['rel_variation_p01']:.3e} | bit-const {s['bit_constant_rows']}/{s['n_rows_checked']} "
                  f"| {'feat' if s['field_is_noiseless'] else 'apparent'}/sigma med={s[snr_key]:.3f} "
                  f"| SNR>1 {s['frac_rows_snr_gt_1']:.3f} | below rel.thr {s['frac_rows_below_rel_threshold']:.3f}")
    for fl in failures:
        print(f"  FAIL {fl['generator']}: " + "; ".join(fl["reasons"]))

    CHECK.emit("FAIL" if failures else "PASS", payload, inputs=[path], out=args.out)


if __name__ == "__main__":
    main()
