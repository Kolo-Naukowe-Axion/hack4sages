from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "ariel_codex_architecture.png"


COLORS = {
    "notebook": "#E3F2FD",
    "data": "#E8F5E9",
    "model": "#F3E5F5",
    "training": "#FFF3E0",
    "artifacts": "#ECEFF1",
    "stage1": "#BBDEFB",
    "stage2": "#D1C4E9",
    "text": "#102027",
    "border": "#455A64",
    "arrow": "#546E7A",
}


def add_box(ax, x, y, w, h, title, body, color, fontsize=10, title_size=12):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=COLORS["border"],
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012 * w,
        y + h - 0.06 * h,
        title,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )
    ax.text(
        x + 0.012 * w,
        y + h - 0.18 * h,
        body,
        fontsize=fontsize,
        va="top",
        ha="left",
        color=COLORS["text"],
        linespacing=1.35,
        family="DejaVu Sans",
    )


def add_arrow(ax, start, end, text="", rad=0.0, fontsize=9):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.7,
        color=COLORS["arrow"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.012,
            text,
            fontsize=fontsize,
            ha="center",
            va="bottom",
            color=COLORS["text"],
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.9},
        )


def render_diagram(output_path: Path = OUTPUT_PATH) -> Path:
    fig = plt.figure(figsize=(22, 14), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.03,
        0.975,
        "Ariel Codex Notebook Architecture",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["text"],
    )
    ax.text(
        0.03,
        0.945,
        "Source: ariel/models/notebooks/ariel_codex.ipynb and models/ariel_quantum_regression/",
        fontsize=12,
        ha="left",
        va="top",
        color="#37474F",
    )
    ax.text(
        0.03,
        0.922,
        "Purpose: orchestrate cached data preparation, classical stage-1 pretraining, hybrid stage-2 fine-tuning, evaluation, and artifact export.",
        fontsize=12,
        ha="left",
        va="top",
        color="#37474F",
    )

    add_box(
        ax,
        0.03,
        0.78,
        0.28,
        0.11,
        "Notebook Control Layer",
        "Cells define:\n"
        "- RUN_MODE: stage1_only | stage2_only | two_stage\n"
        "- output/data/cache paths\n"
        "- shared hyperparameters and stage-specific learning rates\n"
        "- stage-1 and stage-2 TrainingConfig builders\n"
        "- result inspection helpers",
        COLORS["notebook"],
    )

    add_box(
        ax,
        0.03,
        0.53,
        0.28,
        0.18,
        "Data Preparation",
        "TrainingData + TestData\n"
        "- Aux table CSVs -> 8 auxiliary features\n"
        "- Ground-truth CSV -> 5 gas targets\n"
        "- SpectralData.hdf5 -> 52 wavelength bins x 4 raw channels\n\n"
        "Preparation pipeline\n"
        "- log10 transform selected aux columns\n"
        "- normalize sample channels by per-sample mean\n"
        "- append fixed channels: width template + wavelength grid\n"
        "- fit scalers on train split only\n"
        "- stratified 80/10/10 split: train / val / holdout\n"
        "- cache prepared tensors and manifests",
        COLORS["data"],
    )

    add_box(
        ax,
        0.35,
        0.78,
        0.26,
        0.11,
        "Stage 1: Classical Backbone Pretraining",
        "TrainingConfig(classical_only=True)\n"
        "- batch_size 256, eval_batch_size 512\n"
        "- max_epochs 45\n"
        "- classical_lr 1e-3\n"
        "- trains CNN+MLP backbone and classical regression head\n"
        "- saves best_model.pt for stage-2 restart/fine-tuning",
        COLORS["stage1"],
    )

    add_box(
        ax,
        0.65,
        0.78,
        0.30,
        0.11,
        "Stage 2: Hybrid Fine-Tuning",
        "TrainingConfig(classical_only=False, init_checkpoint_path=stage1 best_model.pt)\n"
        "- batch_size 16, eval_batch_size 32\n"
        "- classical_lr 5e-5, quantum_lr 2e-4\n"
        "- quantum_warmup 0, ramp 12 epochs, backbone freeze 6 epochs\n"
        "- activates projector + quantum circuit + quantum residual head",
        COLORS["stage2"],
    )

    add_box(
        ax,
        0.35,
        0.51,
        0.60,
        0.21,
        "HybridArielRegressor Model Core",
        "Inputs\n"
        "- spectra tensor: 4 x 52 (instrument_spectrum, instrument_noise, width_template, wavelength_um)\n"
        "- auxiliary tensor: 8 planet/star features\n\n"
        "Classical backbone\n"
        "- SpectralEncoder: Conv1d stem -> residual blocks 32 -> 64 -> 96 -> mean pool + attention pool -> 96-d spectral feature\n"
        "- AuxEncoder: 8 -> 32 MLP\n"
        "- FusionEncoder: concat(96, 32) -> 128 fused feature\n"
        "- Head context: concat(fused 128, spectral 96, aux 32) -> 256 dims\n"
        "- Classical head: 256 -> 192 -> 5 targets\n\n"
        "Quantum residual branch\n"
        "- QuantumProjector: 128 fused dims -> 8 angles in [-pi, pi]\n"
        "- QuantumBlock: 8-qubit PennyLane circuit, depth 2, RY embeddings + entangling blocks -> 8 expectation values\n"
        "- Quantum head: concat(head context 256, quantum 8) -> 192 -> 5 residual correction\n"
        "- Learnable tanh gate scales each target residual before adding to classical prediction",
        COLORS["model"],
    )

    add_box(
        ax,
        0.35,
        0.24,
        0.27,
        0.20,
        "Training Loop",
        "run_training_experiment()\n"
        "- set seed and runtime threading\n"
        "- choose device: CUDA/MPS/CPU for classical path, CPU or lightning.gpu for hybrid\n"
        "- prepare_data() then move tensors to device\n"
        "- build AdamW optimizer with separate LR groups:\n"
        "  backbone / quantum adapters / quantum circuit\n"
        "- ReduceLROnPlateau on validation RMSE\n"
        "- gradient clipping on all parameter groups\n"
        "- early stopping on best validation RMSE",
        COLORS["training"],
    )

    add_box(
        ax,
        0.65,
        0.24,
        0.30,
        0.20,
        "Epoch Schedule Logic",
        "For each epoch\n"
        "- resolve_quantum_scale(): 0 during warmup, then ramps to 1.0\n"
        "- optional backbone freeze during early quantum epochs\n"
        "- forward(aux, spectra, enable_quantum, quantum_scale)\n"
        "- evaluate validation split every epoch\n"
        "- save history.csv, training_state.json, best_model.pt, last_model.pt\n\n"
        "Stage behavior\n"
        "- Stage 1 returns classical prediction only\n"
        "- Stage 2 adds gated quantum correction residual",
        COLORS["training"],
    )

    add_box(
        ax,
        0.03,
        0.19,
        0.28,
        0.20,
        "Prepared Data Outputs",
        "PreparedData object\n"
        "- train / val / holdout: labeled tensors\n"
        "- testdata: inference tensors\n"
        "- aux_scaler, target_scaler, spectral_scaler\n"
        "- wavelength grid and split manifests\n\n"
        "Cache contents\n"
        "- manifest.json\n"
        "- scalers.json\n"
        "- *_aux.npy, *_spectra.npy, *_targets.npy\n"
        "- wavelength_um.npy",
        COLORS["artifacts"],
    )

    add_box(
        ax,
        0.35,
        0.03,
        0.60,
        0.15,
        "Notebook Result Inspection and Artifacts",
        "Per stage output directory contains:\n"
        "- config.json, split_manifest.json, prepared_manifest.json, scalers.json\n"
        "- history.csv and run_summary.json\n"
        "- validation_metrics.json, holdout_metrics.json\n"
        "- validation_predictions.csv, holdout_predictions.csv, testdata_predictions.csv\n"
        "- best_model.pt and last_model.pt\n\n"
        "Final notebook cells print summaries, plot RMSE history, and confirm the key artifacts exist.",
        COLORS["artifacts"],
    )

    add_arrow(ax, (0.31, 0.835), (0.35, 0.835), "build stage-1 config")
    add_arrow(ax, (0.61, 0.835), (0.65, 0.835), "checkpoint -> stage-2 config")
    add_arrow(ax, (0.17, 0.78), (0.17, 0.71), "prepare_data()")
    add_arrow(ax, (0.31, 0.62), (0.35, 0.62), "scaled tensors")
    add_arrow(ax, (0.48, 0.78), (0.48, 0.72), "classical_only=True")
    add_arrow(ax, (0.80, 0.78), (0.80, 0.72), "classical_only=False")
    add_arrow(ax, (0.65, 0.615), (0.62, 0.34), "forward/backprop", rad=0.0)
    add_arrow(ax, (0.48, 0.51), (0.48, 0.44), "parameter groups")
    add_arrow(ax, (0.80, 0.51), (0.80, 0.44), "quantum scale + freeze")
    add_arrow(ax, (0.17, 0.53), (0.17, 0.39), "PreparedData")
    add_arrow(ax, (0.48, 0.24), (0.48, 0.18), "save metrics")
    add_arrow(ax, (0.80, 0.24), (0.80, 0.18), "save checkpoints")

    ax.text(
        0.03,
        0.125,
        "Model output formula in stage 2:",
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="center",
        color=COLORS["text"],
    )
    ax.text(
        0.03,
        0.098,
        "prediction = classical_head(head_context) + quantum_scale * tanh(quantum_gate) * quantum_head([head_context, qnode(projector(fused))])",
        fontsize=11,
        ha="left",
        va="center",
        color="#263238",
        family="DejaVu Sans Mono",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    saved = render_diagram()
    print(saved)
