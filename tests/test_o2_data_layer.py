"""Unit tests for the Osoba 2 data layer — pure pieces only (no torch, no data).

Covers the paradigm concepts the layer is built on: immutability, pure
functions, functional composition, and the comparison transforms.
Run: PYTHONPATH=. python -m unittest tests.test_o2_data_layer
"""

from __future__ import annotations

import dataclasses
import io
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data import comparison, loading, pipeline, preprocessing
from app.data.types import (
    AUX_COLS,
    GASES,
    AuxFeatures,
    DataError,
    GroundTruth,
    PlanetRecord,
    Prediction,
    RawSpectrum,
)


def _spectrum(planet_id: str = "x") -> RawSpectrum:
    rng = np.arange(52, dtype=np.float32) + 1.0
    return RawSpectrum(planet_id, flux=rng.copy(), noise=rng.copy() * 0.1,
                       width=rng.copy(), wavelength=rng.copy())


def _aux(planet_id: str = "x") -> AuxFeatures:
    return AuxFeatures(planet_id, values=np.arange(1, 9, dtype=np.float32))


def _record() -> PlanetRecord:
    truth = GroundTruth({g: -4.0 for g in GASES})
    return PlanetRecord(spectrum=_spectrum(), aux=_aux(), truth=truth)


class TestImmutability(unittest.TestCase):
    def test_record_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _record().spectrum = None  # type: ignore[misc]

    def test_bad_shape_rejected(self):
        with self.assertRaises(DataError):
            RawSpectrum("x", flux=np.zeros(10), noise=np.zeros(52),
                        width=np.zeros(52), wavelength=np.zeros(52))


class TestPureFunctions(unittest.TestCase):
    def test_log_scale_aux_does_not_mutate_input(self):
        aux = _aux()
        before = aux.values.copy()
        preprocessing.log_scale_aux(aux)
        np.testing.assert_array_equal(aux.values, before)

    def test_log_scale_applies_to_seven_of_eight(self):
        aux = AuxFeatures("x", values=np.full(8, 100.0, dtype=np.float32))
        out = preprocessing.log_scale_aux(aux)
        temp_idx = AUX_COLS.index("star_temperature")
        self.assertAlmostEqual(out[temp_idx], 100.0, places=4)          # untouched
        self.assertAlmostEqual(out[0], 2.0, places=4)                    # log10(100)

    def test_normalize_divides_by_spectrum_mean(self):
        two = np.stack([np.full(52, 4.0), np.full(52, 2.0)]).astype(np.float32)  # (2,52)
        out = preprocessing.normalize_sample_spectra(two)
        np.testing.assert_allclose(out[0], 1.0, atol=1e-6)   # 4 / mean(4)=1
        np.testing.assert_allclose(out[1], 0.5, atol=1e-6)   # 2 / 4 = 0.5


class TestComposition(unittest.TestCase):
    def test_pipe_left_to_right(self):
        f = pipeline.pipe(lambda x: x + 1, lambda x: x * 2)
        self.assertEqual(f(3), 8)  # (3+1)*2


class TestComparison(unittest.TestCase):
    def test_build_and_aggregate(self):
        truth = GroundTruth({g: 0.0 for g in GASES})
        a = Prediction("A", {g: 1.0 for g in GASES})   # err 1 everywhere
        b = Prediction("B", {g: -3.0 for g in GASES})  # err 3 everywhere
        rows = comparison.build_comparison(truth, {"A": a, "B": b})
        self.assertEqual(len(rows), len(GASES))
        agg = comparison.aggregate(rows)
        self.assertAlmostEqual(agg["A"]["rmse_mean"], 1.0)
        self.assertAlmostEqual(agg["B"]["rmse_mean"], 3.0)

    def test_no_truth_means_no_aggregate(self):
        a = Prediction("A", {g: 1.0 for g in GASES})
        rows = comparison.build_comparison(None, {"A": a})
        self.assertEqual(comparison.aggregate(rows), {})


class TestUploadRoundTrip(unittest.TestCase):

    def test_missing_aux_rejected(self):
        with self.assertRaises(DataError):
            loading.parse_upload(io.StringIO("planet_id,flux_0\nx,1.0\n"))


class TestHardening(unittest.TestCase):
    def test_spectrum_array_is_read_only(self):
        with self.assertRaises(ValueError):
            _spectrum().flux[0] = 999.0

    def test_truth_dict_is_read_only(self):
        with self.assertRaises(TypeError):
            _record().truth.log_vmr["log_H2O"] = 1.0  # type: ignore[index]

    def test_nan_spectrum_rejected(self):
        bad = np.arange(52, dtype=np.float32)
        bad[0] = np.nan
        with self.assertRaises(DataError):
            RawSpectrum("x", flux=bad, noise=np.ones(52, np.float32),
                        width=np.ones(52, np.float32), wavelength=np.ones(52, np.float32))

    def test_comparison_missing_gas_raises_dataerror(self):
        truth = GroundTruth({g: 0.0 for g in GASES})
        with self.assertRaises(DataError):
            comparison.build_comparison(truth, {"A": Prediction("A", {"log_H2O": 1.0})})


if __name__ == "__main__":
    unittest.main()
