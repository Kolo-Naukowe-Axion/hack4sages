"""A27 — Is the reconstructed ExoBiome pipeline faithful, ROW BY ROW? (task A0.2)

`audit_lib.exobiome_inputs` rebuilds the model's input tensors from raw HDF5 plus the artifact's own
saved scalers. So far that reconstruction was validated only on the AGGREGATE: it reproduces
`mac_holdout_metrics.json = 0.298693` to six figures. An aggregate can match while individual rows
are wrong, if the errors cancel — so three findings (K3, K4, K9) currently rest on a check that
cannot rule that out.

This check does the row-by-row comparison against the artifact's own saved predictions.

TWO REFERENCES, deliberately, because they fail for different reasons:

  mac_eval_20260312/mac_holdout_predictions.csv
      Same code path as the reconstruction: CPU, `lightning.qubit`, float32. This is the strict
      target. CAVEAT: `mac_run_summary.json` records `data_root = data/full-ariel`, a directory that
      is ABSENT from the repo, whereas the reconstruction reads `data/ariel-ml-dataset`. If this
      comparison passes, it also proves those two roots agree on these rows; if it fails, the cause
      may be the data root rather than the reconstruction, and the check says which.

  holdout_predictions.csv
      Same data root as the reconstruction, but produced on CUDA with AMP (bfloat16 autocast). Here
      a residual of order 1e-3 per row is EXPECTED and is not evidence against the reconstruction —
      it is the device/dtype noise floor that finding P8 is about.

PASS criterion: max absolute per-row per-gas difference vs the mac predictions < 1e-4, at the
`quantum_scale` that a04 identified as the one those predictions were produced at (1.0).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_lib as A  # noqa: E402

CHECK = A.Check(
    name="a27_pipeline_fidelity",
    finding="A0.2 — row-by-row fidelity of the reconstructed ExoBiome input pipeline (underpins K3, K4, K9)",
    question="Do the reconstruction's predictions match the artifact's saved predictions row by row, not just on average?",
    criterion="max |diff| per row per gas < 1e-4 vs mac_holdout_predictions.csv at quantum_scale=1.0",
)

TOL = 1e-4
REFERENCES = {
    "mac_cpu": {
        "path": "artifacts/ariel_quantum_best_v4_epoch6/mac_eval_20260312/mac_holdout_predictions.csv",
        "same_code_path": True,
        "note": "CPU / lightning.qubit / fp32 — strict target. data_root recorded as data/full-ariel (absent).",
    },
    "gpu_amp": {
        "path": "artifacts/ariel_quantum_best_v4_epoch6/holdout_predictions.csv",
        "same_code_path": False,
        "note": "CUDA + bfloat16 autocast — residual ~1e-3 expected, this is the P8 noise floor.",
    },
}


def compare(pred: np.ndarray, ref: np.ndarray, ids: np.ndarray, label: str) -> dict:
    d = np.abs(pred - ref)
    per_gas = {g: {"max": float(d[:, j].max()), "p99": float(np.percentile(d[:, j], 99)),
                   "median": float(np.median(d[:, j]))} for j, g in enumerate(A.TARGETS)}
    worst_row = int(np.argmax(d.max(axis=1)))
    n_over = int((d.max(axis=1) > TOL).sum())
    return {
        "reference": label,
        "n_rows": int(len(pred)),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "p99_abs_diff": float(np.percentile(d, 99)),
        "rows_over_tolerance": n_over,
        "fraction_rows_over_tolerance": float(n_over / len(pred)),
        "per_gas": per_gas,
        "worst_row": {"planet_ID": str(ids[worst_row]),
                      "diffs": dict(zip(A.TARGETS, d[worst_row].round(8).tolist()))},
        "within_tolerance": bool(d.max() < TOL),
        # a matching aggregate with mismatching rows is exactly the failure mode this check exists for
        "aggregate_would_have_hidden_it": bool(d.max() >= TOL and abs(pred.mean() - ref.mean()) < 1e-6),
    }


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout", choices=["holdout", "validation"])
    ap.add_argument("--scales", default="1.0,0.5",
                    help="a04 identified 1.0 as the scale the saved predictions were produced at")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    ids, aux_raw, spec_raw, y = A.load_adc_raw(args.split)
    aux_scaler, target_scaler, spectral_scaler = A.exobiome_scalers()
    model, _ = A.load_exobiome()
    aux, spectra = A.exobiome_inputs(spec_raw, aux_raw, aux_scaler, spectral_scaler, None)

    payload: dict = {"split": args.split, "n_rows": int(len(ids)), "tolerance": TOL,
                     "scales_tested": scales, "references": {}, "id_alignment": {}}

    for key, meta in REFERENCES.items():
        f = A.REPO / meta["path"]
        if not f.exists():
            payload["references"][key] = {"error": f"absent: {meta['path']}"}
            continue
        ref_df = pd.read_csv(f)
        # id alignment is a precondition: a row-by-row comparison of misaligned rows is meaningless
        if "planet_ID" in ref_df.columns:
            ref_ids = ref_df["planet_ID"].astype(str).to_numpy()
            aligned = bool(len(ref_ids) == len(ids) and (ref_ids == ids).all())
            payload["id_alignment"][key] = {"same_order": aligned, "n_ref": int(len(ref_ids))}
            if not aligned:
                ref_df = ref_df.set_index(ref_df["planet_ID"].astype(str)).reindex(ids).reset_index(drop=True)
                payload["id_alignment"][key]["action"] = "reindexed on planet_ID"
        ref = ref_df[[f"pred_{t}" for t in A.TARGETS]].to_numpy(float)
        # sanity: the reference's own truth column must equal the truth we loaded independently
        if f"true_{A.TARGETS[0]}" in ref_df.columns:
            ref_true = ref_df[[f"true_{t}" for t in A.TARGETS]].to_numpy(float)
            payload["id_alignment"].setdefault(key, {})["truth_matches_independent_load"] = \
                bool(np.abs(ref_true - y).max() < 1e-5)

        by_scale = {}
        for s in scales:
            pred = A.exobiome_predict(model, aux, spectra, target_scaler, s)
            by_scale[f"{s:.4f}"] = compare(pred, ref, ids, key)
            by_scale[f"{s:.4f}"]["mrmse_reconstruction"] = A.mrmse(y, pred)
            by_scale[f"{s:.4f}"]["mrmse_reference"] = A.mrmse(y, ref)
        best = min(by_scale, key=lambda k: by_scale[k]["max_abs_diff"])
        payload["references"][key] = {"meta": meta, "by_scale": by_scale, "best_matching_scale": best}

    strict = payload["references"].get("mac_cpu", {})
    ok = False
    if "by_scale" in strict:
        ok = any(v["within_tolerance"] for v in strict["by_scale"].values())
    payload["verdict"] = (
        "reconstruction faithful row by row — K3, K4 and K9 rest on solid ground" if ok else
        "reconstruction NOT confirmed at row level — hold K3, K4, K9 until the cause is located")
    payload["diagnostic_if_failed"] = [
        "compare mrmse_reconstruction vs mrmse_reference: if they agree but rows differ, the errors "
        "cancel and the aggregate check was indeed insufficient",
        "check id_alignment.truth_matches_independent_load: if False, the reference was computed on a "
        "different data root (mac_run_summary.json records data/full-ariel, which is absent)",
        "check whether the worst rows share a property (extreme sigma, extreme transit depth) — that "
        "points at the per-sample normalisation in _normalize_sample_spectra",
    ]

    for key, r in payload["references"].items():
        if "error" in r:
            print(f"  {key:8} {r['error']}")
            continue
        print(f"  {key:8} ({'same code path' if r['meta']['same_code_path'] else 'different device/dtype'})")
        for s, v in r["by_scale"].items():
            print(f"    scale={s}  max|diff|={v['max_abs_diff']:.3e}  p99={v['p99_abs_diff']:.3e}  "
                  f"rows>tol={v['rows_over_tolerance']}/{v['n_rows']}  "
                  f"mRMSE recon/ref = {v['mrmse_reconstruction']:.6f}/{v['mrmse_reference']:.6f}")
        print(f"    best-matching scale: {r['best_matching_scale']}")
    print(f"\n  {payload['verdict']}")

    CHECK.emit("PASS" if ok else "FAIL", payload,
               inputs=[A.EXOBIOME_ARTIFACT / "best_model.pt", A.EXOBIOME_ARTIFACT / "scalers.json"],
               out=args.out)


if __name__ == "__main__":
    main()
