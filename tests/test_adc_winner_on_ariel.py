from __future__ import annotations

import shutil
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

try:
    import h5py
    import pandas as pd
    import torch
    import yaml

    from models.adc_winner_on_ariel.constants import AUX_COLUMNS, TARGET_COLUMNS
    from models.adc_winner_on_ariel.dataset import load_prepared_data
    from models.adc_winner_on_ariel.evaluate import evaluate_point_metric
    from models.adc_winner_on_ariel.model import IndependentNSF, ModelConfig
    from models.adc_winner_on_ariel.prepare_dataset import save_split
    from models.adc_winner_on_ariel.preprocessing import fit_scalers
except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated import
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "full-ariel"
SPLIT_ROOT = REPO_ROOT / "data" / "val_dataset"
RUN_ROOT = REPO_ROOT / "models" / "adc_winner_on_ariel" / "trained_run"


def _load_split_table(name: str, with_targets: bool) -> pd.DataFrame:
    aux = pd.read_csv(SPLIT_ROOT / f"{name}_auxiliary.csv")
    if not with_targets:
        return aux
    targets = pd.read_csv(SPLIT_ROOT / f"{name}_targets.csv")
    return aux.merge(targets[["planet_ID", *TARGET_COLUMNS]], on="planet_ID", how="inner", validate="one_to_one")


def _load_spectral_rows(planet_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    spectra = []
    noise = []
    with h5py.File(DATA_ROOT / "TrainingData" / "SpectralData.hdf5", "r") as handle:
        for planet_id in planet_ids:
            group = handle[f"Planet_{planet_id}"]
            spectra.append(group["instrument_spectrum"][:].astype(np.float32))
            noise.append(group["instrument_noise"][:].astype(np.float32))
    return np.stack(spectra, axis=0), np.stack(noise, axis=0)


class ArchivedWinnerRunTests(unittest.TestCase):
    def setUp(self) -> None:
        if IMPORT_ERROR is not None:
            self.skipTest(f"adc_winner_on_ariel dependencies unavailable: {IMPORT_ERROR}")
        if not DATA_ROOT.is_dir():
            self.skipTest(f"Missing raw dataset root: {DATA_ROOT}")
        if not SPLIT_ROOT.is_dir():
            self.skipTest(f"Missing saved split root: {SPLIT_ROOT}")
        self.temp_dir = Path(tempfile.mkdtemp(prefix="adc-winner-on-ariel-test-"))

    def tearDown(self) -> None:
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _write_tiny_prepared_root(self) -> Path:
        train_table = _load_split_table("train", with_targets=True).head(16).copy()
        validation_table = _load_split_table("validation", with_targets=True).head(8).copy()
        holdout_table = _load_split_table("holdout", with_targets=True).head(8).copy()

        train_spectra, train_noise = _load_spectral_rows(train_table["planet_ID"].tolist())
        validation_spectra, validation_noise = _load_spectral_rows(validation_table["planet_ID"].tolist())
        holdout_spectra, holdout_noise = _load_spectral_rows(holdout_table["planet_ID"].tolist())

        scalers = fit_scalers(
            train_table[AUX_COLUMNS].to_numpy(dtype=np.float32),
            train_spectra,
            train_noise,
            train_table[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        )
        scalers.save(self.temp_dir / "scalers.npz")

        save_split(
            self.temp_dir / "train.npz",
            ids=train_table["planet_ID"].str.extract(r"(\d+)$", expand=False).to_numpy(dtype=np.int64),
            aux=train_table[AUX_COLUMNS].to_numpy(dtype=np.float32),
            spectra=train_spectra,
            noise=train_noise,
            targets=train_table[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        )
        save_split(
            self.temp_dir / "validation.npz",
            ids=validation_table["planet_ID"].str.extract(r"(\d+)$", expand=False).to_numpy(dtype=np.int64),
            aux=validation_table[AUX_COLUMNS].to_numpy(dtype=np.float32),
            spectra=validation_spectra,
            noise=validation_noise,
            targets=validation_table[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        )
        save_split(
            self.temp_dir / "holdout.npz",
            ids=holdout_table["planet_ID"].str.extract(r"(\d+)$", expand=False).to_numpy(dtype=np.int64),
            aux=holdout_table[AUX_COLUMNS].to_numpy(dtype=np.float32),
            spectra=holdout_spectra,
            noise=holdout_noise,
            targets=holdout_table[TARGET_COLUMNS].to_numpy(dtype=np.float32),
        )
        save_split(
            self.temp_dir / "testdata.npz",
            ids=holdout_table["planet_ID"].str.extract(r"(\d+)$", expand=False).head(4).to_numpy(dtype=np.int64),
            aux=holdout_table[AUX_COLUMNS].head(4).to_numpy(dtype=np.float32),
            spectra=holdout_spectra[:4],
            noise=holdout_noise[:4],
            targets=None,
        )
        return self.temp_dir

    def test_archived_checkpoint_loads_and_runs_cpu_smoke_eval(self) -> None:
        prepared_root = self._write_tiny_prepared_root()
        settings = yaml.safe_load((RUN_ROOT / "settings_resolved.yaml").read_text())
        data = load_prepared_data(prepared_root)

        model = IndependentNSF(ModelConfig(**settings["model"]))
        checkpoint = torch.load(RUN_ROOT / "best_model_by_mrmse.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        metrics = evaluate_point_metric(
            model,
            data.validation,
            data.scalers,
            device=torch.device("cpu"),
            num_samples=4,
            point_estimate="median",
            batch_size=4,
            max_rows=4,
            row_seed=42,
            sample_noise=True,
            noise_seed=42,
        )

        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["point_estimate"], "median")
        self.assertTrue(np.isfinite(metrics["rmse_mean"]))
        self.assertGreater(metrics["rmse_mean"], 0.0)
        self.assertEqual(set(metrics["rmse"].keys()), set(TARGET_COLUMNS))

    def test_archived_run_artifacts_match_expected_split_contract(self) -> None:
        prepared_manifest = json.loads((RUN_ROOT / "prepared_manifest.json").read_text())
        saved_split_manifest = json.loads((RUN_ROOT / "saved_split_manifest.json").read_text())
        validation_metrics = json.loads((RUN_ROOT / "validation_metrics.json").read_text())
        holdout_metrics = json.loads((RUN_ROOT / "holdout_metrics.json").read_text())

        self.assertEqual(prepared_manifest["split_source"], str(SPLIT_ROOT.resolve()))
        self.assertEqual(prepared_manifest["split_sizes"]["train"], 33138)
        self.assertEqual(prepared_manifest["split_sizes"]["validation"], 4142)
        self.assertEqual(prepared_manifest["split_sizes"]["holdout"], 4143)
        self.assertEqual(prepared_manifest["split_sizes"]["testdata"], 685)
        self.assertEqual(saved_split_manifest["rows"]["train"], 33138)
        self.assertEqual(saved_split_manifest["rows"]["validation"], 4142)
        self.assertEqual(saved_split_manifest["rows"]["holdout"], 4143)
        self.assertEqual(validation_metrics["rows"], 4142)
        self.assertEqual(holdout_metrics["rows"], 4143)
        self.assertEqual(set(validation_metrics["rmse"].keys()), set(TARGET_COLUMNS))
        self.assertEqual(set(holdout_metrics["rmse"].keys()), set(TARGET_COLUMNS))
        self.assertTrue(np.isfinite(validation_metrics["rmse_mean"]))
        self.assertTrue(np.isfinite(holdout_metrics["rmse_mean"]))


if __name__ == "__main__":
    unittest.main()
