"""Manual verification harness for the Osoba 2 data layer.

Exercises the full path on a real planet:
  load_by_id -> pure preprocessing pipeline -> quantum model -> denormalize
  -> Prediction -> comparison vs ground truth, plus an upload round-trip.

Run from the worktree root, with the app venv:

    PYTHONPATH=$PWD .venv-app/bin/python verify_o2.py            # or
    PYTHONPATH=$PWD ../../../.venv-app/bin/python verify_o2.py --planet train37

NOTE: without PennyLane this runs the hybrid model's *classical head* only
(the quantum correction needs PennyLane). That is enough to validate the data
pipeline; it will not reproduce the headline quantum mRMSE exactly.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from app.data import loading
from app.data.comparison import aggregate, build_comparison
from app.data.pipeline import make_quantum_input
from app.data.preprocessing import denormalize_targets, vmr_to_dict
from app.data.types import GASES, Prediction


def _check(label: str, ok: bool) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planet", default="train12")
    args = parser.parse_args()

    import torch
    from models.garnet_ariel_quantum_regression.checkpoint import (
        FrozenArielHybridBridge,
        load_checkpoint_bundle,
    )

    all_ok = True

    print("== 1. curated list ==")
    curated = loading.list_curated_planets()
    for cp in curated:
        print(f"   {cp.label}")
    all_ok &= _check("curated list non-empty", len(curated) > 0)

    print(f"\n== 2. load_by_id({args.planet}) ==")
    record = loading.load_by_id(args.planet)
    all_ok &= _check("spectrum shape (52,)", record.spectrum.flux.shape == (52,))
    all_ok &= _check("aux shape (8,)", record.aux.values.shape == (8,))
    all_ok &= _check("ground truth present", record.truth is not None)

    print("\n== 3. preprocessing + quantum inference (classical head) ==")
    bundle = load_checkpoint_bundle()
    bridge = FrozenArielHybridBridge(bundle)
    build_input = make_quantum_input(bundle.aux_scaler, bundle.spectral_scaler)
    aux_n, spectra_n = build_input(record)
    all_ok &= _check("model input shapes (1,8)/(1,4,52)", aux_n.shape == (1, 8) and spectra_n.shape == (1, 4, 52))
    with torch.inference_mode():
        enc = bridge.encode_features(aux_n, spectra_n)
        pred_norm = bridge.classical_predict(enc["head_context"]).cpu().numpy()
    pred_phys = denormalize_targets(pred_norm, bundle.target_scaler)
    quantum = Prediction("quantum (classical head)", vmr_to_dict(pred_phys[0]))
    all_ok &= _check("prediction finite", bool(np.isfinite(pred_phys).all()))

    print("\n== 4. comparison vs ground truth ==")
    # transparent placeholder for adc_winner until O1 wires it in live
    winner_stub = Prediction("adc_winner (STUB)", {g: quantum.log_vmr[g] + 0.3 for g in GASES})
    rows = build_comparison(record.truth, {quantum.model_name: quantum, winner_stub.model_name: winner_stub})
    print(f"   {'gas':<9}{'true':>9}{'quantum':>10}{'winner*':>10}")
    for row in rows:
        print(f"   {row.gas:<9}{row.true:>9.2f}{row.preds[quantum.model_name]:>10.2f}{row.preds[winner_stub.model_name]:>10.2f}")
    agg = aggregate(rows)
    for name, m in agg.items():
        print(f"   per-planet RMSE [{name}] = {m['rmse_mean']:.3f}")
    all_ok &= _check("aggregate produced rmse", len(agg) == 2)

    print("\n== 5. upload round-trip (export -> parse) ==")
    csv = loading.export_record(args.planet)
    import io
    reparsed = loading.parse_upload(io.StringIO(csv))
    all_ok &= _check("flux survives round-trip", np.allclose(reparsed.spectrum.flux, record.spectrum.flux))
    all_ok &= _check("aux survives round-trip", np.allclose(reparsed.aux.values, record.aux.values))
    all_ok &= _check("truth survives round-trip", reparsed.truth is not None)

    print("\n== 6. truth cross-check vs validation_predictions.csv ==")
    csv_path = bundle.checkpoint_dir / "validation_predictions.csv"
    if csv_path.exists() and record.truth is not None:
        import pandas as pd
        ref = pd.read_csv(csv_path).set_index("planet_ID")
        if args.planet in ref.index:
            match = all(
                np.isclose(record.truth.log_vmr[g], float(ref.loc[args.planet, f"true_{g}"]), atol=1e-3)
                for g in GASES
            )
            all_ok &= _check("truth matches reference predictions file", match)
            # how close is our pipeline's classical-head output to the reference
            # full-model prediction? (they differ only by the quantum correction)
            diffs = [abs(quantum.log_vmr[g] - float(ref.loc[args.planet, f"pred_{g}"])) for g in GASES]
            print(f"   mean |our classical-head pred - reference full-model pred| = {np.mean(diffs):.3f} dex")
        else:
            print(f"   (planet {args.planet} not in validation reference — skipped)")
    else:
        print("   (reference file or truth missing — skipped)")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    print("note: '*' winner is a STUB; quantum is classical-head only (no PennyLane).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
