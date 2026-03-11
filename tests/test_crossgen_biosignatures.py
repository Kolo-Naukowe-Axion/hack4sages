from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from crossgen_biosignatures.baseline_smoke import run_baseline_smoke
from crossgen_biosignatures.constants import DATASET_PATHS
from crossgen_biosignatures.generate_dataset import assemble_dataset
from crossgen_biosignatures.grid import build_rebin_matrix, make_constant_resolution_grid
from crossgen_biosignatures.latents import build_latents, build_tau_extension_latents
from crossgen_biosignatures.taurex_backend import resolve_taurex_ktable_dir
from crossgen_biosignatures.validate_dataset import validate_dataset
from mpi4py import MPI
import pymultinest


def _make_small_latents() -> pd.DataFrame:
    base = build_latents(master_seed=20260310)
    tau_rows = base.loc[base["generator"] == "tau"].iloc[:12].copy()
    tau_rows.loc[:, "split"] = ["train"] * 8 + ["val"] * 4
    tau_rows.loc[:, "sample_index"] = np.arange(1, len(tau_rows) + 1, dtype=np.int64)

    poseidon_rows = base.loc[base["generator"] == "poseidon"].iloc[:6].copy()
    poseidon_rows.loc[:, "split"] = ["test"] * len(poseidon_rows)
    poseidon_rows.loc[:, "sample_index"] = np.arange(1, len(poseidon_rows) + 1, dtype=np.int64)

    combined = pd.concat([tau_rows, poseidon_rows], ignore_index=True)
    combined.loc[:, "row_index"] = np.arange(len(combined), dtype=np.int64)
    return combined


class GridTests(unittest.TestCase):
    def test_constant_resolution_grid_is_monotonic(self) -> None:
        centers, edges = make_constant_resolution_grid(0.6, 5.2, 100.0)
        self.assertTrue(np.all(np.diff(centers) > 0.0))
        self.assertTrue(np.all(np.diff(edges) > 0.0))
        self.assertGreaterEqual(edges[-1], 5.2)

    def test_rebin_matrix_rows_sum_to_one(self) -> None:
        source_centers = np.geomspace(0.6, 5.2, 500)
        _, target_edges = make_constant_resolution_grid(0.6, 5.2, 100.0)
        matrix = build_rebin_matrix(source_centers, target_edges)
        self.assertTrue(np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12))


class LatentTests(unittest.TestCase):
    def test_build_latents_hits_expected_counts(self) -> None:
        latents = build_latents(master_seed=20260310)
        counts = latents["generator"].value_counts().to_dict()
        self.assertEqual(int(counts.get("tau", 0)), 41423)
        self.assertEqual(int(counts.get("poseidon", 0)), 685)
        self.assertTrue(np.all(latents["trace_vmr_total"].to_numpy() < 0.10))

    def test_build_tau_extension_latents_continues_sample_ids(self) -> None:
        latents = build_tau_extension_latents(count=10, master_seed=20260310, start_ordinal=41423)
        self.assertTrue((latents["generator"] == "tau").all())
        self.assertEqual(latents["sample_id"].iloc[0], "tau_041424")
        self.assertEqual(latents["sample_id"].iloc[-1], "tau_041433")
        self.assertEqual(int(latents["sample_index"].iloc[0]), 41424)
        self.assertEqual(int(latents["sample_index"].iloc[-1]), 41433)


class AssemblyAndValidationTests(unittest.TestCase):
    def test_assemble_validate_and_baseline(self) -> None:
        latents = _make_small_latents()
        centers, _ = make_constant_resolution_grid(0.6, 5.2, 100.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            for generator in ("tau", "poseidon"):
                shard_dir = output_root / DATASET_PATHS.shards_dir / generator
                shard_dir.mkdir(parents=True, exist_ok=True)
                subset = latents.loc[latents["generator"] == generator].reset_index(drop=True)
                noiseless = np.empty((len(subset), len(centers)), dtype=np.float32)
                noisy = np.empty_like(noiseless)
                for row_idx, row in subset.iterrows():
                    base_level = 0.01 + 1.0e-4 * row_idx
                    slope = 1.0e-5 * (row_idx + 1)
                    spectrum = base_level + slope * np.linspace(0.0, 1.0, len(centers))
                    noiseless[row_idx] = spectrum.astype(np.float32)
                    noisy[row_idx] = (spectrum + row["sigma_ppm"] * 1.0e-6).astype(np.float32)
                np.savez_compressed(
                    shard_dir / f"{generator}_000001_{len(subset):06d}.npz",
                    sample_id=subset["sample_id"].to_numpy(dtype="U32"),
                    generator=subset["generator"].to_numpy(dtype="U16"),
                    split=subset["split"].to_numpy(dtype="U16"),
                    transit_depth_noiseless=noiseless,
                    transit_depth_noisy=noisy,
                    sigma_ppm=subset["sigma_ppm"].to_numpy(dtype=np.float32),
                )
                meta_dir = output_root / DATASET_PATHS.metadata_dir
                meta_dir.mkdir(parents=True, exist_ok=True)
                (meta_dir / f"{generator}_generation.json").write_text(
                    json.dumps({"software_versions": {generator: "test"}}, indent=2) + "\n"
                )

            assemble_dataset(output_root, latents, force=True)
            summary = validate_dataset(
                output_root,
                expected_counts={"tau": 12, "poseidon": 6},
                enforce_prevalence_bounds=False,
            )
            self.assertEqual(summary["generator_counts"]["tau"], 12)
            self.assertEqual(summary["generator_counts"]["poseidon"], 6)

            baseline = run_baseline_smoke(output_root, alpha=1.0)
            self.assertEqual(baseline["train_rows"], 8)
            self.assertEqual(baseline["val_rows"], 4)
            self.assertEqual(baseline["test_rows"], 6)

    def test_tau_only_assemble_and_validate(self) -> None:
        latents = build_tau_extension_latents(count=10, master_seed=20260310, start_ordinal=41423)
        centers, _ = make_constant_resolution_grid(0.6, 5.2, 100.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            shard_dir = output_root / DATASET_PATHS.shards_dir / "tau"
            shard_dir.mkdir(parents=True, exist_ok=True)
            noiseless = np.empty((len(latents), len(centers)), dtype=np.float32)
            noisy = np.empty_like(noiseless)
            for row_idx, row in latents.reset_index(drop=True).iterrows():
                base_level = 0.008 + 1.0e-4 * row_idx
                slope = 5.0e-6 * (row_idx + 1)
                spectrum = base_level + slope * np.linspace(0.0, 1.0, len(centers))
                noiseless[row_idx] = spectrum.astype(np.float32)
                noisy[row_idx] = (spectrum + row["sigma_ppm"] * 1.0e-6).astype(np.float32)
            np.savez_compressed(
                shard_dir / "tau_041424_041433.npz",
                sample_id=latents["sample_id"].to_numpy(dtype="U32"),
                generator=latents["generator"].to_numpy(dtype="U16"),
                split=latents["split"].to_numpy(dtype="U16"),
                transit_depth_noiseless=noiseless,
                transit_depth_noisy=noisy,
                sigma_ppm=latents["sigma_ppm"].to_numpy(dtype=np.float32),
            )
            meta_dir = output_root / DATASET_PATHS.metadata_dir
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "tau_generation.json").write_text(json.dumps({"software_versions": {"tau": "test"}}, indent=2) + "\n")

            assemble_dataset(output_root, latents, force=True)
            summary = validate_dataset(
                output_root,
                expected_counts={"tau": 10},
                required_generators=("tau",),
                enforce_prevalence_bounds=False,
            )
            self.assertEqual(summary["generator_counts"]["tau"], 10)


class TauRExBackendTests(unittest.TestCase):
    def test_resolve_taurex_ktable_dir_prefers_env_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ktable_dir = Path(temp_dir)
            for species_name in ("H2O", "CO2", "CO", "CH4", "NH3"):
                (ktable_dir / f"{species_name}.h5").write_bytes(b"test")

            original = os.environ.get("CROSSGEN_TAUREX_KTABLE_DIR")
            os.environ["CROSSGEN_TAUREX_KTABLE_DIR"] = str(ktable_dir)
            try:
                self.assertEqual(resolve_taurex_ktable_dir(), ktable_dir)
            finally:
                if original is None:
                    os.environ.pop("CROSSGEN_TAUREX_KTABLE_DIR", None)
                else:
                    os.environ["CROSSGEN_TAUREX_KTABLE_DIR"] = original


class MpiStubTests(unittest.TestCase):
    def test_serial_mpi_stub_behaves_like_single_rank(self) -> None:
        self.assertEqual(MPI.COMM_WORLD.Get_rank(), 0)
        self.assertEqual(MPI.COMM_WORLD.Get_size(), 1)
        self.assertEqual(MPI.DOUBLE.Get_size(), 8)
        self.assertEqual(MPI.COMM_WORLD.allgather("x"), ["x"])

    def test_pymultinest_stub_exposes_run(self) -> None:
        self.assertTrue(hasattr(pymultinest, "Analyzer"))
        self.assertTrue(hasattr(pymultinest, "run"))


if __name__ == "__main__":
    unittest.main()
