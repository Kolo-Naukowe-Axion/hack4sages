"""Shared setup for quantum advantage test notebooks.

Usage in any notebook:
    from setup_helpers import *
"""

import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent so we can import ariel_quantum_regression
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ariel_quantum_regression.model import ModelConfig, build_model, HybridArielRegressor
from ariel_quantum_regression.dataset import prepare_data, PreparedData, LabeledSplit
from ariel_quantum_regression.training import evaluate_labeled_split
from ariel_quantum_regression.constants import TARGET_COLUMNS

# ── Paths ──
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "trained_weights" / "best_model.pt"
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "datasets" / "ariel-ml-dataset"

# ── Constants ──
TARGET_NAMES = ["H\u2082O", "CO\u2082", "CO", "CH\u2084", "NH\u2083"]
QUANTUM_SCALE_AT_BEST = 0.5
BATCH_SIZE = 128
LOSS_FN = nn.MSELoss()


def load_checkpoint():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    return ckpt


def build_trained_model(ckpt=None, quantum_device="lightning.qubit"):
    if ckpt is None:
        ckpt = load_checkpoint()
    cfg = ckpt["config"]
    model_cfg = ModelConfig(
        spectral_input_channels=4,
        dropout=cfg["dropout"],
        qnn_qubits=cfg["qnn_qubits"],
        qnn_depth=cfg["qnn_depth"],
        qnn_init_scale=cfg["qnn_init_scale"],
        quantum_device=quantum_device,
        classical_only=False,
    )
    device = torch.device("cpu")
    model = build_model(model_cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg, device


def load_data():
    data = prepare_data(
        data_root=DATA_ROOT,
        output_dir=Path(__file__).resolve().parent / "outputs",
        dataset_format="adc",
        seed=42,
    )
    assert data.train.rows == 33138, f"Train rows mismatch: {data.train.rows}"
    assert data.val.rows == 4142, f"Val rows mismatch: {data.val.rows}"
    assert data.holdout.rows == 4143, f"Holdout rows mismatch: {data.holdout.rows}"
    return data


def evaluate(model, split, target_scaler, enable_quantum=True, quantum_scale=QUANTUM_SCALE_AT_BEST):
    return evaluate_labeled_split(
        model, split, target_scaler,
        batch_size=BATCH_SIZE, loss_fn=LOSS_FN,
        enable_quantum=enable_quantum, quantum_scale=quantum_scale,
    )


def compare_table(results_dict):
    rows = []
    for label, m in results_dict.items():
        row = {"Condition": label, "mRMSE": f"{m['rmse_mean']:.4f}", "mMAE": f"{m['mae_mean']:.4f}"}
        for i, name in enumerate(TARGET_NAMES):
            row[f"RMSE {name}"] = f"{m['rmse_orig'][i]:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Condition")


def plot_per_target(results_dict, title="Per-target RMSE Comparison"):
    labels = list(results_dict.keys())
    n_groups = len(TARGET_NAMES)
    n_bars = len(labels)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        vals = results_dict[label]["rmse_orig"]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(TARGET_NAMES)
    ax.legend()
    plt.tight_layout()
    return fig
