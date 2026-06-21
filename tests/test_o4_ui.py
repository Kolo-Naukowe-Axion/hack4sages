"""Unit tests for the Osoba 4 Streamlit UI layer - pure logic only.

Covers the chart/table builders in ``app.ui.components`` and the inference
seam in ``app.ui.inference_adapter`` without a running Streamlit app. Every
function tested here returns a DataFrame or plain Python value, so no torch
and no HDF5 dataset are needed. The one live-model test is skipped unless torch
and the checkpoint artifact are both present.

Run: PYTHONPATH=. python -m unittest tests.test_o4_ui
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import comparison
from app.data.types import (
    GASES,
    AuxFeatures,
    GroundTruth,
    PlanetRecord,
    Prediction,
    RawSpectrum,
)
from app.ui import components, inference_adapter

CHECKPOINT = PROJECT_ROOT / "artifacts" / "ariel_quantum_best_v4_epoch6"
_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_LIVE_MODEL_READY = _TORCH_AVAILABLE and (CHECKPOINT / "best_model.pt").exists()


def _spectrum(planet_id: str = "x") -> RawSpectrum:
    rng = np.arange(52, dtype=np.float32) + 1.0
    return RawSpectrum(planet_id, flux=rng.copy(), noise=rng.copy() * 0.1,
                       width=rng.copy(), wavelength=rng.copy())


def _aux(planet_id: str = "x") -> AuxFeatures:
    return AuxFeatures(planet_id, values=np.arange(1, 9, dtype=np.float32))


def _record(with_truth: bool = True) -> PlanetRecord:
    truth = GroundTruth({g: -4.0 for g in GASES}) if with_truth else None
    return PlanetRecord(spectrum=_spectrum(), aux=_aux(), truth=truth)


def _prediction(name: str, base: float) -> Prediction:
    return Prediction(name, {g: base for g in GASES})


def _write_validation_csv(directory: Path) -> None:
    """Write a 2-row ``validation_predictions.csv`` with true_/pred_ per gas."""
    rows = []
    for i, planet in enumerate(("planet_a", "planet_b")):
        row: dict[str, object] = {"planet_ID": planet}
        for j, gas in enumerate(GASES):
            row[f"true_{gas}"] = -4.0 - i - 0.1 * j
            row[f"pred_{gas}"] = -3.0 - i - 0.1 * j
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "validation_predictions.csv", index=False)


class TestComponents(unittest.TestCase):
    def test_gas_label_strips_prefix(self):
        self.assertEqual(components.gas_label("log_H2O"), "H2O")

    def test_spectrum_dataframe_shape_and_columns(self):
        df = components.spectrum_dataframe(_record())
        self.assertEqual(df.shape, (52, 4))
        for col in ("flux", "lo", "hi"):
            self.assertIn(col, df.columns)

    def test_spectrum_dataframe_does_not_mutate_record(self):
        record = _record()
        before = np.asarray(record.spectrum.flux).copy()
        components.spectrum_dataframe(record)
        np.testing.assert_array_equal(np.asarray(record.spectrum.flux), before)

    def test_comparison_long_row_count(self):
        truth = GroundTruth({g: -4.0 for g in GASES})
        preds = {"A": _prediction("A", 1.0), "B": _prediction("B", -3.0)}
        rows = comparison.build_comparison(truth, preds)
        long = components.comparison_long(rows)
        expected = sum(1 for r in rows if r.true is not None) + len(GASES) * len(preds)
        self.assertEqual(len(long), expected)

    def test_comparison_long_row_count_without_truth(self):
        preds = {"A": _prediction("A", 1.0), "B": _prediction("B", -3.0)}
        rows = comparison.build_comparison(None, preds)
        long = components.comparison_long(rows)
        # No truth -> only the per-model rows.
        expected = len(GASES) * len(preds)
        self.assertEqual(len(long), expected)

    def test_comparison_dataframe_index_and_columns(self):
        truth = GroundTruth({g: -4.0 for g in GASES})
        preds = {"A": _prediction("A", 1.0), "B": _prediction("B", -3.0)}
        rows = comparison.build_comparison(truth, preds)
        df = components.comparison_dataframe(rows)
        expected_index = [components.gas_label(g) for g in GASES]
        self.assertEqual(list(df.index), expected_index)
        self.assertIn(components.GROUND_TRUTH, df.columns)
        for model in preds:
            self.assertIn(f"|błąd| {model}", df.columns)


class TestInferenceAdapterCompare(unittest.TestCase):
    def test_compare_returns_rows_and_per_model_rmse(self):
        truth = GroundTruth({g: 0.0 for g in GASES})
        pred1 = _prediction("model-1", 1.0)
        pred2 = _prediction("model-2", -3.0)
        rows, agg = inference_adapter.compare(truth, [pred1, pred2])
        self.assertEqual(len(rows), 5)
        for name in ("model-1", "model-2"):
            self.assertIn(name, agg)
            self.assertIn("rmse_mean", agg[name])


class TestReferenceTable(unittest.TestCase):
    def test_full_quantum_prediction_matches_pred_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_validation_csv(directory)
            pred = inference_adapter.full_quantum_prediction(directory, "planet_a")
            self.assertIsNotNone(pred)
            for j, gas in enumerate(GASES):
                self.assertAlmostEqual(pred.log_vmr[gas], -3.0 - 0.1 * j, places=6)

    def test_full_quantum_prediction_absent_planet_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_validation_csv(directory)
            self.assertIsNone(
                inference_adapter.full_quantum_prediction(directory, "no_such_planet")
            )

    def test_reference_truth_matches_true_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_validation_csv(directory)
            truth = inference_adapter.reference_truth(directory, "planet_b")
            self.assertIsNotNone(truth)
            for j, gas in enumerate(GASES):
                self.assertAlmostEqual(truth.log_vmr[gas], -4.0 - 1 - 0.1 * j, places=6)

    def test_reference_truth_absent_planet_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_validation_csv(directory)
            self.assertIsNone(
                inference_adapter.reference_truth(directory, "no_such_planet")
            )


@unittest.skipUnless(
    _LIVE_MODEL_READY,
    "requires torch and artifacts/ariel_quantum_best_v4_epoch6/best_model.pt",
)
class TestLiveInference(unittest.TestCase):
    def test_load_engine_predict(self):
        import torch  # noqa: F401  (lazy: never imported at module top)

        engine = inference_adapter.load_engine(CHECKPOINT)
        prediction = engine.predict(_record())
        self.assertIsInstance(prediction, Prediction)
        for gas in GASES:
            self.assertIn(gas, prediction.log_vmr)
            self.assertTrue(np.isfinite(prediction.log_vmr[gas]))


if __name__ == "__main__":
    unittest.main()
