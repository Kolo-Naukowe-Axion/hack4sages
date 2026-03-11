from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from prt_transmission_benchmark.constants import (
    BENCHMARK_RESOLUTION,
    HEAVY_SPECIES_MAX_FRACTION,
    PRIMARY_TARGET_SPECIES,
    SPLIT_COUNTS,
    WAVELENGTH_MAX_UM,
    WAVELENGTH_MIN_UM,
)
from prt_transmission_benchmark.generate_dataset import assign_splits
from prt_transmission_benchmark.grid import build_rebin_matrix, make_log_resolving_power_grid
from prt_transmission_benchmark.physics import build_mass_fraction_profile


class GridTests(unittest.TestCase):
    def test_log_r_grid_is_monotonic(self) -> None:
        centers, edges = make_log_resolving_power_grid(WAVELENGTH_MIN_UM, WAVELENGTH_MAX_UM, BENCHMARK_RESOLUTION)
        self.assertTrue(np.all(np.diff(centers) > 0.0))
        self.assertTrue(np.all(np.diff(edges) > 0.0))
        self.assertGreater(len(centers), 0)
        self.assertAlmostEqual(edges[0], WAVELENGTH_MIN_UM)
        self.assertAlmostEqual(edges[-1], WAVELENGTH_MAX_UM)

    def test_rebin_matrix_rows_sum_to_one(self) -> None:
        source_centers = np.geomspace(0.5, 5.0, 300)
        _, target_edges = make_log_resolving_power_grid(0.5, 5.0, 100)
        matrix = build_rebin_matrix(source_centers, target_edges)
        row_sums = matrix.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-12))


class PhysicsTests(unittest.TestCase):
    def test_mass_fraction_profile_sums_to_one(self) -> None:
        log_x = {species: -6.0 for species in PRIMARY_TARGET_SPECIES}
        abundances, mmw, heavy_fraction = build_mass_fraction_profile(log_x, pressure_levels=8)
        total = sum(float(profile[0]) for profile in abundances.values())
        self.assertLess(heavy_fraction, HEAVY_SPECIES_MAX_FRACTION)
        self.assertAlmostEqual(total, 1.0, places=12)
        self.assertTrue(np.all(mmw > 0.0))


class SplitTests(unittest.TestCase):
    def test_assign_splits_is_deterministic_and_hits_canonical_counts(self) -> None:
        total_rows = sum(SPLIT_COUNTS.values())
        sample_ids = np.asarray(["tx%06d" % (idx + 1) for idx in range(total_rows)], dtype=object)
        labels = pd.DataFrame({"sample_id": sample_ids})
        provenance = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "ood_candidate": np.asarray([1 if idx < 12000 else 0 for idx in range(total_rows)], dtype=np.int64),
            }
        )

        split_a = assign_splits(labels, provenance, seed=10032026)
        split_b = assign_splits(labels, provenance, seed=10032026)
        self.assertTrue(split_a.equals(split_b))

        counts = split_a.value_counts().to_dict()
        for split_name, expected in SPLIT_COUNTS.items():
            self.assertEqual(int(counts.get(split_name, 0)), int(expected))


if __name__ == "__main__":
    unittest.main()
