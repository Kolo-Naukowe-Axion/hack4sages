from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "ariel_exobiome_model_architecture.png"


COLORS = {
    "input": "#E3F2FD",
    "classical": "#E8F5E9",
    "quantum": "#F3E5F5",
    "merge": "#FFF3E0",
    "note": "#ECEFF1",
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
        x + 0.014 * w,
        y + h - 0.07 * h,
        title,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["text"],
    )
    ax.text(
        x + 0.014 * w,
        y + h - 0.20 * h,
        body,
        fontsize=fontsize,
        va="top",
        ha="left",
        color=COLORS["text"],
        linespacing=1.30,
        family="DejaVu Sans",
    )


def add_arrow(ax, start, end, text="", rad=0.0, fontsize=9):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.8,
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
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.95},
        )


def render_diagram(output_path: Path = OUTPUT_PATH) -> Path:
    fig = plt.figure(figsize=(22, 12), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.03,
        0.97,
        "Ariel ExoBiome Model Architecture",
        fontsize=26,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["text"],
    )
    ax.text(
        0.03,
        0.94,
        "HybridArielRegressor from models/ariel_quantum_regression/model.py",
        fontsize=12,
        ha="left",
        va="top",
        color="#37474F",
    )

    add_box(
        ax,
        0.03,
        0.68,
        0.17,
        0.16,
        "Spectra Input",
        "Tensor shape:\n"
        "4 x 52\n\n"
        "Channels:\n"
        "- instrument_spectrum\n"
        "- instrument_noise\n"
        "- width_template\n"
        "- wavelength_um",
        COLORS["input"],
    )

    add_box(
        ax,
        0.03,
        0.38,
        0.17,
        0.14,
        "Aux Input",
        "Tensor shape:\n"
        "8 features\n\n"
        "Planet + star metadata",
        COLORS["input"],
    )

    add_box(
        ax,
        0.24,
        0.62,
        0.20,
        0.26,
        "SpectralEncoder",
        "Conv1d stem\n"
        "4 -> 32\n\n"
        "Residual blocks\n"
        "32 -> 32 -> 64 -> 96\n\n"
        "Pooling\n"
        "- mean pool\n"
        "- attention pool\n\n"
        "Projection\n"
        "concat(96, 96) -> 96",
        COLORS["classical"],
    )

    add_box(
        ax,
        0.24,
        0.36,
        0.20,
        0.16,
        "AuxEncoder",
        "MLP\n"
        "8 -> 32 -> 32\n"
        "GELU + dropout",
        COLORS["classical"],
    )

    add_box(
        ax,
        0.48,
        0.48,
        0.19,
        0.20,
        "FusionEncoder",
        "Concatenate features\n"
        "spectral 96 + aux 32 = 128\n\n"
        "MLP\n"
        "128 -> 128 -> 128\n"
        "LayerNorm + GELU",
        COLORS["classical"],
    )

    add_box(
        ax,
        0.70,
        0.58,
        0.20,
        0.16,
        "Head Context",
        "Concatenate\n"
        "fused 128\n"
        "+ spectral 96\n"
        "+ aux 32\n\n"
        "= 256 dims",
        COLORS["merge"],
    )

    add_box(
        ax,
        0.70,
        0.78,
        0.20,
        0.13,
        "Classical Head",
        "RegressionHead\n"
        "256 -> 192 -> 5\n\n"
        "Outputs classical prediction",
        COLORS["classical"],
    )

    add_box(
        ax,
        0.48,
        0.18,
        0.19,
        0.18,
        "QuantumProjector",
        "Uses fused feature only\n\n"
        "128 -> 128 -> 8\n"
        "LayerNorm\n"
        "tanh(.) * pi\n\n"
        "Produces 8 rotation angles",
        COLORS["quantum"],
    )

    add_box(
        ax,
        0.70,
        0.16,
        0.20,
        0.22,
        "QuantumBlock",
        "8-qubit variational circuit\n"
        "depth = 2\n\n"
        "Per block:\n"
        "- input RY embedding\n"
        "- trainable RY rotations\n"
        "- ring CNOT entanglement\n"
        "- trainable RZ rotations\n"
        "- trainable CRX entanglers\n\n"
        "Output: 8 Pauli-Z expectations",
        COLORS["quantum"],
        fontsize=9.5,
    )

    add_box(
        ax,
        0.70,
        0.40,
        0.20,
        0.13,
        "Quantum Head",
        "RegressionHead\n"
        "concat(256, 8) = 264\n"
        "264 -> 192 -> 5\n\n"
        "Outputs residual correction",
        COLORS["quantum"],
    )

    add_box(
        ax,
        0.92,
        0.40,
        0.06,
        0.13,
        "Gate",
        "Learnable\n"
        "5-d vector\n\n"
        "tanh(g)",
        COLORS["quantum"],
        fontsize=9,
        title_size=11,
    )

    add_box(
        ax,
        0.92,
        0.68,
        0.06,
        0.13,
        "Sum",
        "classical\n"
        "+ scaled\n"
        "quantum\n"
        "residual",
        COLORS["merge"],
        fontsize=9,
        title_size=11,
    )

    add_box(
        ax,
        0.92,
        0.83,
        0.06,
        0.10,
        "Output",
        "5 targets\n"
        "log_H2O\n"
        "log_CO2\n"
        "log_CO\n"
        "log_CH4\n"
        "log_NH3",
        COLORS["merge"],
        fontsize=8.5,
        title_size=11,
    )

    add_box(
        ax,
        0.03,
        0.08,
        0.38,
        0.18,
        "Classical-Only vs Hybrid Behavior",
        "If `classical_only=True`, or `enable_quantum=False`, or `quantum_scale <= 0`,\n"
        "the forward pass returns only the classical head output.\n\n"
        "In hybrid mode the final prediction is:\n"
        "classical_pred + quantum_scale * tanh(quantum_gate) * quantum_correction",
        COLORS["note"],
    )

    add_box(
        ax,
        0.43,
        0.08,
        0.34,
        0.08,
        "Key Dimensions",
        "spectral_feat = 96, aux_feat = 32, fused = 128, head_context = 256, quantum_features = 8, outputs = 5",
        COLORS["note"],
    )

    add_box(
        ax,
        0.43,
        0.17,
        0.34,
        0.09,
        "Interpretation",
        "The quantum path is not a full replacement head. It is a residual branch that refines the classical prediction.",
        COLORS["note"],
    )

    add_arrow(ax, (0.20, 0.76), (0.24, 0.76), "4 x 52")
    add_arrow(ax, (0.20, 0.45), (0.24, 0.45), "8")
    add_arrow(ax, (0.44, 0.73), (0.48, 0.58), "96", rad=-0.05)
    add_arrow(ax, (0.44, 0.44), (0.48, 0.58), "32", rad=0.05)
    add_arrow(ax, (0.67, 0.58), (0.70, 0.66), "128")
    add_arrow(ax, (0.67, 0.58), (0.48, 0.27), "to projector", rad=0.15)
    add_arrow(ax, (0.80, 0.74), (0.80, 0.78), "256")
    add_arrow(ax, (0.67, 0.27), (0.70, 0.27), "8 angles")
    add_arrow(ax, (0.80, 0.38), (0.80, 0.40), "8 expvals")
    add_arrow(ax, (0.80, 0.58), (0.80, 0.53), "head context")
    add_arrow(ax, (0.90, 0.47), (0.92, 0.47), "x")
    add_arrow(ax, (0.90, 0.845), (0.92, 0.745), "classical", rad=-0.05)
    add_arrow(ax, (0.98, 0.47), (0.98, 0.745), "scaled residual")
    add_arrow(ax, (0.95, 0.81), (0.95, 0.83))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    saved = render_diagram()
    print(saved)
