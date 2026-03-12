from __future__ import annotations

import importlib
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PENNYLANE = importlib.util.find_spec("pennylane") is not None


PACKAGE_NAME = "models.12q_taurex_codex"
MODEL_NAME = f"{PACKAGE_NAME}.model"
TRAINING_NAME = f"{PACKAGE_NAME}.training"
RUN_NAME = f"{PACKAGE_NAME}.run_12q_taurex_codex"
RUN_CV_NAME = f"{PACKAGE_NAME}.run_12q_taurex_codex_cross_validation"


class PackageContractTests(unittest.TestCase):
    def test_default_configs_target_12_qubits(self) -> None:
        constants_module = importlib.import_module(f"{PACKAGE_NAME}.constants")

        self.assertEqual(str(constants_module.DEFAULT_OUTPUT_DIR), "outputs/12q_taurex_codex")

        model_source = (PROJECT_ROOT / "models" / "12q_taurex_codex" / "model.py").read_text(encoding="utf-8")
        training_source = (PROJECT_ROOT / "models" / "12q_taurex_codex" / "training.py").read_text(encoding="utf-8")
        self.assertIn("qnn_qubits: int = 12", model_source)
        self.assertIn("qnn_qubits: int = 12", training_source)

        if HAS_TORCH:
            model_module = importlib.import_module(MODEL_NAME)
            training_module = importlib.import_module(TRAINING_NAME)
            self.assertEqual(model_module.ModelConfig().qnn_qubits, 12)
            self.assertEqual(training_module.TrainingConfig().qnn_qubits, 12)

    def test_cli_parsers_default_to_new_paths_and_12_qubits(self) -> None:
        run_module = importlib.import_module(RUN_NAME)
        run_cv_module = importlib.import_module(RUN_CV_NAME)

        run_args = run_module.build_parser().parse_args([])
        self.assertEqual(run_args.output_dir, "outputs/12q_taurex_codex")
        self.assertEqual(run_args.qnn_qubits, 12)

        run_cv_args = run_cv_module.build_parser().parse_args([])
        self.assertEqual(run_cv_args.output_dir, "outputs/12q_taurex_codex_cv")
        self.assertEqual(run_cv_args.qnn_qubits, 12)

    def test_shell_helpers_reference_new_package(self) -> None:
        package_dir = PROJECT_ROOT / "models" / "12q_taurex_codex"

        taurex_script = (package_dir / "run_12q_taurex_codex_taurex_ubuntu_gpu.sh").read_text(encoding="utf-8")
        self.assertIn("models/12q_taurex_codex/run_12q_taurex_codex.py", taurex_script)
        self.assertIn("--qnn-qubits 12", taurex_script)
        self.assertIn("outputs/12q_taurex_codex_taurex_", taurex_script)

        sync_script = (package_dir / "sync_12q_taurex_codex_remote.sh").read_text(encoding="utf-8")
        self.assertIn("models/12q_taurex_codex/", sync_script)

        launch_script = (package_dir / "launch_12q_taurex_codex_remote.sh").read_text(encoding="utf-8")
        self.assertIn("run_12q_taurex_codex_taurex_ubuntu_gpu.sh", launch_script)


@unittest.skipUnless(HAS_TORCH and HAS_PENNYLANE, "torch and pennylane are required for hybrid model tests")
class HybridModelTests(unittest.TestCase):
    def test_hybrid_model_builds_with_12_qubits(self) -> None:
        import torch

        model_module = importlib.import_module(MODEL_NAME)

        model = model_module.build_model(
            model_module.ModelConfig(
                classical_only=False,
                spectral_input_channels=4,
                qnn_qubits=12,
                qnn_depth=2,
                use_amp=False,
            ),
            torch.device("cpu"),
        )

        self.assertIsNotNone(model.quantum_block)
        self.assertEqual(model.quantum_block.n_qubits, 12)
        self.assertIsNotNone(model.quantum_head)
        self.assertEqual(model.quantum_head.net[0].in_features, 128 + 96 + 32 + 12)


if __name__ == "__main__":
    unittest.main()
