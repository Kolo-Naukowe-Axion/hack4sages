"""Unit tests for Iwo Smura's training orchestration.

These tests use a fake stage runner, so they do not need the ADC dataset, a GPU,
PennyLane, or a long training run.
Run: PYTHONPATH=. python -m unittest tests.test_o3_training
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.training import (
    StageSpec,
    Trainer,
    TrainingEvent,
    build_stage_result,
    collect_artifacts,
    format_metric_summary,
    has_nonfinite_metric,
    make_classical_pretrain_spec,
    make_fail_on_nan_callback,
    make_jsonl_logger,
    plan_to_json_dict,
    read_json_file,
    read_stage_metrics,
    select_best_checkpoint,
)


DATA_ROOT = PROJECT_ROOT / "data" / "ariel-ml-dataset"


def _can_import(module_name: str) -> bool:
    try:
        if importlib.util.find_spec(module_name) is None:
            return False
        __import__(module_name)
    except Exception:
        return False
    return True


HAS_TORCH = _can_import("torch")
HAS_H5PY = _can_import("h5py")
HAS_ADC_DATA = all(
    path.exists()
    for path in (
        DATA_ROOT / "TrainingData" / "AuxillaryTable.csv",
        DATA_ROOT / "TrainingData" / "Ground Truth Package" / "FM_Parameter_Table.csv",
        DATA_ROOT / "TrainingData" / "SpectralData.hdf5",
        DATA_ROOT / "TestData" / "AuxillaryTable.csv",
        DATA_ROOT / "TestData" / "SpectralData.hdf5",
    )
)


@dataclass
class TestTrainingConfig:
    project_root: str = str(PROJECT_ROOT)
    output_dir: str = "outputs/test"
    data_root: str = "data/ariel-ml-dataset"
    prepared_cache_dir: str | None = None
    init_checkpoint_path: str | None = None
    classical_only: bool = False
    quantum_warmup_epochs: int = 5

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).expanduser().resolve()

    def resolved_output_dir(self) -> Path:
        root = Path(self.output_dir).expanduser()
        if root.is_absolute():
            return root
        return self.resolved_project_root() / root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


class FakeRunner:
    def __init__(self) -> None:
        self.configs: list[TestTrainingConfig] = []

    def __call__(self, config: TestTrainingConfig) -> dict[str, Any]:
        index = len(self.configs) + 1
        self.configs.append(config)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "best_model.pt").write_bytes(f"stage-{index}".encode("utf-8"))
        (output_dir / "last_model.pt").write_bytes(f"last-{index}".encode("utf-8"))
        (output_dir / "history.csv").write_text("epoch,val_rmse_mean\n1,0.5\n")
        _write_json(output_dir / "validation_metrics.json", {"rmse_mean": 0.5 + index, "mae_mean": 0.2})
        _write_json(output_dir / "holdout_metrics.json", {"rmse_mean": 0.6 + index, "mae_mean": 0.3})
        _write_json(output_dir / "run_summary.json", {"stage": index})
        return {"summary": {"stage": index}}


class FailingRunner:
    def __init__(self) -> None:
        self.configs: list[TestTrainingConfig] = []

    def __call__(self, config: TestTrainingConfig) -> dict[str, Any]:
        self.configs.append(config)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        raise ValueError("boom")


class TrainerWorkflowTests(unittest.TestCase):
    def test_dry_run_plan_does_not_execute_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            trainer = Trainer(
                TestTrainingConfig(project_root=str(PROJECT_ROOT), output_dir=str(Path(temp_dir) / "ignored")),
                run_root=Path(temp_dir) / "run",
                stage_runner=runner,
            )

            plan = trainer.plan_two_stage()
            payload = plan_to_json_dict(plan)

            self.assertEqual(runner.configs, [])
            self.assertEqual([stage.name for stage in plan.stages], ["stage1_classical", "stage2_hybrid"])
            self.assertEqual(Path(plan.stage2.config.init_checkpoint_path), Path(plan.stage1.config.output_dir) / "best_model.pt")
            self.assertEqual(payload["stages"][0]["name"], "stage1_classical")
            self.assertTrue(payload["expected_best_checkpoint"].endswith("stage2_hybrid/best_model.pt"))

    def test_stage_spec_clones_mutable_config_input(self) -> None:
        config = TestTrainingConfig(output_dir="first")
        spec = StageSpec(name="stage", config=config)

        config.output_dir = "second"

        self.assertEqual(spec.config.output_dir, "first")
        self.assertIsNot(spec.config, config)

    def test_two_stage_order_and_checkpoint_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = FakeRunner()
            events: list[TrainingEvent] = []
            trainer = Trainer(
                TestTrainingConfig(project_root=str(PROJECT_ROOT), output_dir=str(Path(temp_dir) / "ignored")),
                run_root=Path(temp_dir) / "run",
                stage_runner=runner,
                callbacks=(events.append,),
            )

            result = trainer.run_two_stage()

            self.assertEqual([Path(cfg.output_dir).name for cfg in runner.configs], ["stage1_classical", "stage2_hybrid"])
            self.assertTrue(runner.configs[0].classical_only)
            self.assertEqual(runner.configs[0].quantum_warmup_epochs, 0)
            self.assertFalse(runner.configs[1].classical_only)
            self.assertEqual(Path(runner.configs[1].init_checkpoint_path), result.stage1.best_checkpoint)
            self.assertEqual(result.best_checkpoint, result.stage2.best_checkpoint)
            self.assertEqual([event.status for event in events], ["started", "completed", "started", "completed"])
            self.assertEqual([event.stage_name for event in events], ["stage1_classical", "stage1_classical", "stage2_hybrid", "stage2_hybrid"])
            for stage in (result.stage1, result.stage2):
                self.assertIn("artifacts_manifest.json", stage.artifacts)
                manifest = read_json_file(stage.output_dir / "artifacts_manifest.json")
                self.assertEqual(manifest["stage_name"], stage.stage_name)
                self.assertIn("best_model.pt", manifest["artifacts"])
                self.assertIn("summary", manifest["metrics"])

    def test_callbacks_can_log_deterministic_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "events.jsonl"
            runner = FakeRunner()
            trainer = Trainer(
                TestTrainingConfig(project_root=str(PROJECT_ROOT), output_dir=str(Path(temp_dir) / "ignored")),
                run_root=Path(temp_dir) / "run",
                stage_runner=runner,
                callbacks=(make_jsonl_logger(log_path),),
            )

            trainer.run_two_stage()

            rows = [json.loads(line) for line in log_path.read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["stage_name"], "stage1_classical")
            self.assertEqual(rows[0]["status"], "started")
            self.assertEqual(rows[-1]["stage_name"], "stage2_hybrid")
            self.assertEqual(rows[-1]["status"], "completed")
            self.assertNotIn("timestamp", rows[0])

    def test_failed_stage_writes_report_and_emits_failed_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events: list[TrainingEvent] = []
            trainer = Trainer(
                TestTrainingConfig(project_root=str(PROJECT_ROOT), output_dir=str(Path(temp_dir) / "ignored")),
                run_root=Path(temp_dir) / "run",
                stage_runner=FailingRunner(),
                callbacks=(events.append,),
            )
            spec = trainer.plan_two_stage().stage1

            with self.assertRaises(ValueError):
                trainer.run_stage(spec)

            report = read_json_file(Path(spec.config.output_dir) / "failed_stage.json")
            self.assertEqual(report["stage_name"], "stage1_classical")
            self.assertEqual(report["error_type"], "ValueError")
            self.assertEqual(report["message"], "boom")
            self.assertEqual([event.status for event in events], ["started", "failed"])

    def test_failure_callback_rejects_nonfinite_completed_metric(self) -> None:
        callback = make_fail_on_nan_callback()
        with self.assertRaises(RuntimeError):
            callback(
                TrainingEvent(
                    stage_name="stage",
                    status="completed",
                    output_dir=Path("."),
                    metrics={"validation_rmse_mean": math.nan},
                )
            )


class HelperFunctionTests(unittest.TestCase):
    def test_collects_metrics_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "best_model.pt").write_bytes(b"best")
            _write_json(root / "validation_metrics.json", {"rmse_mean": 0.1, "mae_mean": 0.2})
            _write_json(root / "holdout_metrics.json", {"rmse_mean": 0.3, "mae_mean": 0.4})
            _write_json(root / "run_summary.json", {"ok": True})

            validation, holdout = read_stage_metrics(root)
            self.assertEqual(validation["rmse_mean"], 0.1)
            self.assertEqual(holdout["rmse_mean"], 0.3)
            self.assertEqual(select_best_checkpoint(root), root / "best_model.pt")

            result = build_stage_result("demo", root, TestTrainingConfig())
            self.assertIn("validation_metrics.json", collect_artifacts(root))
            self.assertEqual(result.validation_metrics["mae_mean"], 0.2)
            self.assertIn("validation_rmse_mean=0.1000", format_metric_summary(result))

    def test_missing_metrics_are_empty_and_checkpoint_can_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "last_model.pt").write_bytes(b"last")

            validation, holdout = read_stage_metrics(root)
            self.assertEqual(validation, {})
            self.assertEqual(holdout, {})
            self.assertEqual(select_best_checkpoint(root), root / "last_model.pt")

    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                select_best_checkpoint(temp_dir)

    def test_nan_metric_detection(self) -> None:
        self.assertTrue(has_nonfinite_metric({"rmse_mean": math.inf}))
        self.assertFalse(has_nonfinite_metric({"rmse_mean": 0.5, "label": "ok"}))


class CommandLineTests(unittest.TestCase):
    def test_cli_dry_run_prints_plan_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_two_stage_training.py"),
                    "--dry-run",
                    "--run-root",
                    str(Path(temp_dir) / "run"),
                    "--max-epochs",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["stages"][0]["name"], "stage1_classical")
            self.assertEqual(payload["stages"][1]["name"], "stage2_hybrid")
            self.assertFalse((Path(temp_dir) / "run" / "stage1_classical" / "best_model.pt").exists())


@unittest.skipUnless(HAS_TORCH and HAS_H5PY and HAS_ADC_DATA, "tiny integration test needs torch, h5py, and ADC data")
class TinyIntegrationTests(unittest.TestCase):
    def test_one_stage_smoke_can_be_run_through_trainer(self) -> None:
        from models.ariel_quantum_regression.training import TrainingConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = TrainingConfig(
                project_root=str(PROJECT_ROOT),
                data_root=str(DATA_ROOT),
                output_dir=str(Path(temp_dir) / "base"),
                max_epochs=1,
                early_stop_patience=1,
                scheduler_patience=1,
                batch_size=4,
                eval_batch_size=8,
                classical_only=True,
                quantum_warmup_epochs=0,
                use_amp=False,
                train_limit=16,
                val_limit=8,
                holdout_limit=8,
                test_limit=6,
                log_every_batches=0,
            )
            trainer = Trainer(cfg, run_root=Path(temp_dir) / "run")
            spec = make_classical_pretrain_spec(cfg, Path(temp_dir) / "run" / "stage1_classical", prepared_cache_dir=None)
            result = trainer.run_stage(spec)
            self.assertTrue(result.best_checkpoint.exists())
            self.assertIn("rmse_mean", result.validation_metrics)


if __name__ == "__main__":
    unittest.main()
