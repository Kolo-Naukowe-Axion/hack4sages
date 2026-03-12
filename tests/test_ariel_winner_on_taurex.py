from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HAS_H5PY = importlib.util.find_spec("h5py") is not None
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_H5PY and HAS_PYARROW, "h5py and pyarrow are required for TauREx preparation tests")
class TauRExPreparationTests(unittest.TestCase):
    def _write_synthetic_dataset(self, root: Path) -> None:
        import h5py

        wavelength = np.linspace(0.6, 5.2, 218, dtype=np.float64)
        rows = []
        spectra = []
        sigma_ppm = []

        def add_rows(generator: str, split: str, count: int, start_index: int) -> int:
            index = start_index
            for local_idx in range(count):
                sample_id = f"{generator}_{index:06d}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "generator": generator,
                        "split": split,
                        "planet_radius_rjup": 0.8 + 0.02 * local_idx,
                        "log_g_cgs": 3.0 + 0.03 * local_idx,
                        "temperature_k": 700.0 + 20.0 * local_idx,
                        "star_radius_rsun": 0.7 + 0.01 * local_idx,
                        "log10_vmr_h2o": -7.0 + 0.1 * local_idx,
                        "log10_vmr_co2": -6.8 + 0.1 * local_idx,
                        "log10_vmr_co": -6.6 + 0.1 * local_idx,
                        "log10_vmr_ch4": -6.4 + 0.1 * local_idx,
                        "log10_vmr_nh3": -6.2 + 0.1 * local_idx,
                    }
                )
                base = 0.010 + 1.0e-4 * index
                slope = 5.0e-5 + 1.0e-6 * local_idx
                spectra.append((base + slope * np.linspace(0.0, 1.0, len(wavelength))).astype(np.float32))
                sigma_ppm.append(np.float32(25.0 + local_idx))
                index += 1
            return index

        next_index = add_rows("tau", "train", 8, 1)
        next_index = add_rows("tau", "val", 4, next_index)
        add_rows("poseidon", "test", 3, 1)

        labels = pd.DataFrame(rows)
        labels.to_parquet(root / "labels.parquet", index=False)
        (root / "manifest.json").write_text(json.dumps({"dataset_name": "synthetic"}, indent=2) + "\n")

        with h5py.File(root / "spectra.h5", "w") as handle:
            handle.create_dataset("sample_id", data=np.asarray(labels["sample_id"].tolist(), dtype="S32"))
            handle.create_dataset("generator", data=np.asarray(labels["generator"].tolist(), dtype="S16"))
            handle.create_dataset("split", data=np.asarray(labels["split"].tolist(), dtype="S16"))
            handle.create_dataset("wavelength_um", data=wavelength)
            handle.create_dataset("transit_depth_noisy", data=np.stack(spectra, axis=0), compression="gzip")
            handle.create_dataset("sigma_ppm", data=np.asarray(sigma_ppm, dtype=np.float32))

    def test_prepare_dataset_builds_expected_splits(self) -> None:
        from models.ariel_winner_on_taurex.prepare_dataset import main

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            data_root = temp_root / "tau_bundle"
            output_root = temp_root / "prepared"
            data_root.mkdir(parents=True, exist_ok=True)
            self._write_synthetic_dataset(data_root)

            argv = sys.argv[:]
            try:
                sys.argv = [
                    "prepare_dataset.py",
                    "--data-root",
                    str(data_root),
                    "--output",
                    str(output_root),
                    "--overwrite",
                ]
                main()
            finally:
                sys.argv = argv

            manifest = json.loads((output_root / "manifest.json").read_text())
            self.assertEqual(manifest["split_sizes"]["train"], 8)
            self.assertEqual(manifest["split_sizes"]["validation"], 4)
            self.assertEqual(manifest["split_sizes"]["holdout"], 3)
            self.assertEqual(manifest["context_layout"]["total"], 450)
            self.assertEqual(manifest["target_columns"][0], "log10_vmr_h2o")

            train = np.load(output_root / "train.npz")
            validation = np.load(output_root / "validation.npz")
            holdout = np.load(output_root / "holdout.npz")
            testdata = np.load(output_root / "testdata.npz")

            self.assertEqual(train["aux_raw"].shape, (8, 8))
            self.assertEqual(train["spectra_raw"].shape, (8, 218))
            self.assertEqual(train["noise_raw"].shape, (8, 218))
            self.assertEqual(train["targets_raw"].shape, (8, 5))
            self.assertEqual(validation["spectra_raw"].shape, (4, 218))
            self.assertEqual(holdout["targets_raw"].shape, (3, 5))
            self.assertNotIn("targets_raw", testdata.files)
            self.assertTrue(np.allclose(train["noise_raw"][:, 0], train["noise_raw"][:, -1]))
            self.assertEqual(np.load(output_root / "wavelength_um.npy").shape, (218,))


@unittest.skipUnless(HAS_H5PY and HAS_PYARROW and HAS_TORCH, "torch, h5py, and pyarrow are required for context tests")
class TauRExContextShapeTests(unittest.TestCase):
    def test_context_shape_matches_settings(self) -> None:
        from models.ariel_winner_on_taurex.dataset import build_context_batch, load_prepared_data
        from models.ariel_winner_on_taurex.prepare_dataset import main

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            data_root = temp_root / "tau_bundle"
            output_root = temp_root / "prepared"
            data_root.mkdir(parents=True, exist_ok=True)
            TauRExPreparationTests()._write_synthetic_dataset(data_root)

            argv = sys.argv[:]
            try:
                sys.argv = [
                    "prepare_dataset.py",
                    "--data-root",
                    str(data_root),
                    "--output",
                    str(output_root),
                    "--overwrite",
                ]
                main()
            finally:
                sys.argv = argv

            prepared = load_prepared_data(output_root)

            import torch

            rows = torch.arange(0, 2, dtype=torch.long)
            context = build_context_batch(prepared.train, rows, prepared.scalers, device=torch.device("cpu"), sample_noise=False)
            self.assertEqual(tuple(context.shape), (2, 450))


if __name__ == "__main__":
    unittest.main()
