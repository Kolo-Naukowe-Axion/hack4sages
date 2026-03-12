"""Constants for the IQM Garnet Ariel quantum regression port."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parents[3]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "ariel_quantum_best_v4_epoch6"
DEFAULT_DATA_ROOT = REPO_ROOT / "ariel"
DEFAULT_OUTPUT_DIR = NOTEBOOKS_ROOT / "outputs" / "garnet_ariel_quantum_regression"
DEFAULT_NOTEBOOK_PATH = PACKAGE_ROOT / "garnet_port_tutorial.ipynb"
SUPPORTED_QUBITS = (8, 12)
DEFAULT_BACKEND_URL = "https://resonance.meetiqm.com"
DEFAULT_BACKEND_ALIAS = "garnet"
