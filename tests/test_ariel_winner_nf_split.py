from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from models.ariel_winner_nf.prepare_dataset import load_saved_split_ids, split_indices_from_saved_ids
except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated import
    load_saved_split_ids = None
    split_indices_from_saved_ids = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


class SavedSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        if IMPORT_ERROR is not None:
            self.skipTest(f"ariel_winner_nf dependencies unavailable: {IMPORT_ERROR}")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="winner-nf-split-test-"))

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def write_ids(self, filename: str, values: list[str]) -> None:
        path = self.temp_dir / filename
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["planet_ID"])
            for value in values:
                writer.writerow([value])

    def test_saved_split_ids_preserve_membership_and_order(self) -> None:
        self.write_ids("train_planet_ids.csv", ["Planet_train4", "Planet_train1"])
        self.write_ids("validation_planet_ids.csv", ["Planet_train3"])
        self.write_ids("holdout_planet_ids.csv", ["Planet_train2"])

        saved = load_saved_split_ids(self.temp_dir)
        np.testing.assert_array_equal(saved["train"], np.array([4, 1], dtype=np.int64))
        np.testing.assert_array_equal(saved["validation"], np.array([3], dtype=np.int64))
        np.testing.assert_array_equal(saved["holdout"], np.array([2], dtype=np.int64))

        all_ids = np.array([1, 2, 3, 4], dtype=np.int64)
        train_idx, val_idx, holdout_idx = split_indices_from_saved_ids(all_ids, self.temp_dir)
        np.testing.assert_array_equal(train_idx, np.array([3, 0], dtype=np.int64))
        np.testing.assert_array_equal(val_idx, np.array([2], dtype=np.int64))
        np.testing.assert_array_equal(holdout_idx, np.array([1], dtype=np.int64))

    def test_saved_split_ids_require_full_partition(self) -> None:
        self.write_ids("train_planet_ids.csv", ["Planet_train1", "Planet_train2"])
        self.write_ids("validation_planet_ids.csv", ["Planet_train3"])
        self.write_ids("holdout_planet_ids.csv", ["Planet_train3"])

        with self.assertRaises(ValueError):
            load_saved_split_ids(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
