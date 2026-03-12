"""Generate model architecture diagram for the Hybrid Quantum-Classical model."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 28))
ax.set_xlim(0, 14)
ax.set_ylim(0, 28)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

# Colors
C_INPUT = "#58a6ff"
C_CLASSICAL = "#3fb950"
C_FUSION = "#f47067"
C_QUANTUM = "#bc8cff"
C_HEAD = "#d2a038"
C_OUTPUT = "#f0883e"
C_TEXT = "#e6edf3"
C_DIM = "#7d8590"
C_DETAIL = "#636c76"
C_BOX_BG = "#161b22"
C_ARROW = "#6e7681"


def box(x, y, w, h, color, lw=2.0):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.2",
        facecolor=C_BOX_BG, edgecolor=color, linewidth=lw, zorder=3,
    )
    ax.add_patch(p)


def txt(x, y, s, **kw):
    defaults = dict(ha="center", va="center", zorder=4, fontsize=10, color=C_TEXT)
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.8, style="-"):
    ls = "--" if style == "--" else "-"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14, linestyle=ls),
        zorder=2,
    )


def curved_arrow(x1, y1, x2, y2, color, rad=0.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3,
                        connectionstyle=f"arc3,rad={rad}", linestyle="--"),
        zorder=2,
    )


CX = 7.0  # center x

# ═══════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════
txt(CX, 27.3, "EXOBIOME", fontsize=22, fontweight="bold", color=C_TEXT,
    fontfamily="monospace")
txt(CX, 26.75, "Hybrid Quantum-Classical Architecture", fontsize=13,
    fontweight="bold", color=C_DIM)
txt(CX, 26.3, "Biosignature Detection in Exoplanet Transmission Spectra",
    fontsize=9, color=C_DETAIL)

# ═══════════════════════════════════════════════════
# INPUTS (y ~ 25)
# ═══════════════════════════════════════════════════
txt(1.0, 25.6, "INPUTS", fontsize=8, color=C_INPUT, fontweight="bold", ha="left", alpha=0.6)

# Spectrum input
box(1.2, 24.5, 5.0, 0.9, C_INPUT)
txt(3.7, 25.1, "Transmission Spectrum", fontsize=10, fontweight="bold", color=C_INPUT)
txt(3.7, 24.75, "1 x 44 bins  (Ariel grid)", fontsize=8, color=C_DIM)

# Aux input
box(7.8, 24.5, 5.0, 0.9, C_INPUT)
txt(10.3, 25.1, "Auxiliary Features", fontsize=10, fontweight="bold", color=C_INPUT)
txt(10.3, 24.75, "Rp, log g, T, R*, log10(sigma)", fontsize=8, color=C_DIM)

# Arrows down
arrow(3.7, 24.5, 3.7, 23.6)
arrow(10.3, 24.5, 10.3, 23.6)

# ═══════════════════════════════════════════════════
# CLASSICAL ENCODERS (y ~ 21-23)
# ═══════════════════════════════════════════════════
txt(1.0, 23.6, "CLASSICAL ENCODERS", fontsize=8, color=C_CLASSICAL,
    fontweight="bold", ha="left", alpha=0.6)

# SpectralEncoder
box(1.2, 21.0, 5.0, 2.3, C_CLASSICAL)
txt(3.7, 22.85, "SpectralEncoder", fontsize=11, fontweight="bold", color=C_CLASSICAL)
txt(3.7, 22.35, "Conv1d(1 -> 32, k=7)", fontsize=8, color=C_DIM)
txt(3.7, 22.0, "Conv1d(32 -> 64, k=5, s=2)", fontsize=8, color=C_DIM)
txt(3.7, 21.65, "Conv1d(64 -> 64, k=3)", fontsize=8, color=C_DIM)
txt(3.7, 21.3, "AdaptiveAvgPool1d(1)", fontsize=8, color=C_DIM)
txt(3.7, 20.95, "-> Flatten -> Linear(64, 32)", fontsize=8, color=C_DETAIL)
# all GELU activations implied

# AuxEncoder
box(7.8, 21.0, 5.0, 2.3, C_CLASSICAL)
txt(10.3, 22.85, "AuxEncoder", fontsize=11, fontweight="bold", color=C_CLASSICAL)
txt(10.3, 22.35, "Linear(5, 64) + GELU", fontsize=8, color=C_DIM)
txt(10.3, 22.0, "Dropout(0.05)", fontsize=8, color=C_DETAIL)
txt(10.3, 21.65, "Linear(64, 64) + GELU", fontsize=8, color=C_DIM)
txt(10.3, 21.3, "Linear(64, 32) + GELU", fontsize=8, color=C_DIM)

# Dim labels
txt(4.7, 20.5, "32-d", fontsize=8, color=C_CLASSICAL, fontweight="bold")
txt(9.0, 20.5, "32-d", fontsize=8, color=C_CLASSICAL, fontweight="bold")

# Arrows to fusion
arrow(3.7, 21.0, CX - 0.8, 20.0)
arrow(10.3, 21.0, CX + 0.8, 20.0)

# ═══════════════════════════════════════════════════
# FUSION ENCODER (y ~ 18.5-20)
# ═══════════════════════════════════════════════════
txt(1.0, 20.05, "FUSION", fontsize=8, color=C_FUSION, fontweight="bold",
    ha="left", alpha=0.6)

box(3.5, 18.3, 7.0, 1.5, C_FUSION)
txt(CX, 19.4, "FusionEncoder", fontsize=11, fontweight="bold", color=C_FUSION)
txt(CX, 18.95, "Concat(aux, spectral) -> Linear(64, 48) + GELU",
    fontsize=8, color=C_DIM)
txt(CX, 18.6, "Linear(48, 12) + LayerNorm -> tanh * pi",
    fontsize=8, color=C_DIM)

txt(CX + 0.3, 17.85, "12 rotation angles", fontsize=8, color=C_FUSION, fontweight="bold")
arrow(CX, 18.3, CX, 17.5)

# ═══════════════════════════════════════════════════
# QUANTUM BLOCK (y ~ 14.5-17.5)
# ═══════════════════════════════════════════════════
txt(1.0, 17.5, "QUANTUM LAYER", fontsize=8, color=C_QUANTUM,
    fontweight="bold", ha="left", alpha=0.6)

box(2.5, 14.3, 9.0, 2.9, C_QUANTUM, lw=2.5)
txt(CX, 16.7, "QuantumBlock", fontsize=13, fontweight="bold", color=C_QUANTUM)
txt(CX, 16.15, "12 qubits  |  depth = 2  (1 ansatz block)", fontsize=9,
    color=C_DIM, fontweight="bold")

# Circuit visualization
txt(CX, 15.55, "Encoding:   RY(angle_i) on each qubit", fontsize=8.5,
    color=C_TEXT, fontfamily="monospace")
txt(CX, 15.1, "Ansatz:     RY(w) + CNOT(ring) + RZ(w) + CRX(w)",
    fontsize=8.5, color=C_TEXT, fontfamily="monospace")
txt(CX, 14.65, "Readout:    <Z_0>, <Z_1>, ..., <Z_11>", fontsize=8.5,
    color=C_TEXT, fontfamily="monospace")

# Small detail below circuit
txt(CX, 14.2, "36 trainable params  |  adjoint differentiation  |  PennyLane",
    fontsize=7.5, color=C_DETAIL, fontstyle="italic")

txt(CX + 0.3, 13.75, "12-d expectations", fontsize=8, color=C_QUANTUM, fontweight="bold")
arrow(CX, 14.3, CX, 13.3)

# ═══════════════════════════════════════════════════
# FEATURE AGGREGATION (y ~ 11.8-13)
# ═══════════════════════════════════════════════════
txt(1.0, 13.3, "AGGREGATION", fontsize=8, color=C_HEAD, fontweight="bold",
    ha="left", alpha=0.6)

box(2.5, 11.8, 9.0, 1.2, C_HEAD)
txt(CX, 12.65, "Concatenate", fontsize=11, fontweight="bold", color=C_HEAD)
txt(CX, 12.15, "quantum(12) + latent(12) + aux(32) + spectral(32) = 88-d",
    fontsize=8.5, color=C_DIM)

# ─── Skip connections ───
# spectral skip (left side)
curved_arrow(1.2, 22.0, 2.5, 12.4, C_CLASSICAL, rad=-0.25)
txt(0.8, 17.0, "skip", fontsize=7.5, color=C_CLASSICAL, rotation=90, ha="center")

# aux skip (right side)
curved_arrow(12.8, 22.0, 11.5, 12.4, C_CLASSICAL, rad=0.25)
txt(13.2, 17.0, "skip", fontsize=7.5, color=C_CLASSICAL, rotation=90, ha="center")

# latent skip (from fusion)
curved_arrow(3.5, 18.8, 3.0, 12.4, C_FUSION, rad=-0.15)
txt(2.6, 15.6, "skip", fontsize=7.5, color=C_FUSION, rotation=90, ha="center")

# quantum residual (right side, to head)
curved_arrow(11.8, 15.5, 11.0, 9.8, C_QUANTUM, rad=0.25)
txt(12.3, 12.6, "residual", fontsize=7.5, color=C_QUANTUM, rotation=80, ha="center")

txt(CX, 11.35, "88-d", fontsize=8, color=C_HEAD, fontweight="bold")
arrow(CX, 11.8, CX, 11.0)

# ═══════════════════════════════════════════════════
# ATMOSPHERE HEAD (y ~ 8.5-10.5)
# ═══════════════════════════════════════════════════
txt(1.0, 11.0, "PREDICTION HEAD", fontsize=8, color=C_HEAD, fontweight="bold",
    ha="left", alpha=0.6)

box(3.0, 8.3, 8.0, 2.4, C_HEAD)
txt(CX, 10.2, "AtmosphereHead", fontsize=11, fontweight="bold", color=C_HEAD)

txt(CX, 9.65, "Linear(88, 96) + GELU + Dropout(0.05)", fontsize=8.5, color=C_DIM)
txt(CX, 9.25, "Linear(96, 96) + GELU", fontsize=8.5, color=C_DIM)
txt(CX, 8.85, "Linear(96, 5)", fontsize=8.5, color=C_DIM)

txt(CX, 8.4, "output = MLP(concat) + Linear(12, 5)(quantum)",
    fontsize=8.5, color=C_TEXT, fontstyle="italic")

arrow(CX, 8.3, CX, 7.5)

# ═══════════════════════════════════════════════════
# OUTPUTS (y ~ 6.2-7.5)
# ═══════════════════════════════════════════════════
txt(1.0, 7.5, "OUTPUTS", fontsize=8, color=C_OUTPUT, fontweight="bold",
    ha="left", alpha=0.6)

box(3.0, 6.2, 8.0, 1.0, C_OUTPUT)
txt(CX, 6.9, "log10 VMR Predictions", fontsize=11, fontweight="bold", color=C_OUTPUT)
txt(CX, 6.5, "H2O   CO2   CO   CH4   NH3", fontsize=9, color=C_DIM,
    fontfamily="monospace")

# ═══════════════════════════════════════════════════
# MODEL SUMMARY BOX (y ~ 1-5)
# ═══════════════════════════════════════════════════
box(1.5, 1.0, 11.0, 4.5, "#30363d", lw=1.5)
txt(CX, 5.05, "Model Summary", fontsize=11, fontweight="bold", color=C_TEXT)

items = [
    ("Training",  "37,281 samples (TauREx) + 4,142 val (TauREx)"),
    ("Holdout",   "685 samples (Poseidon cross-generator)"),
    ("Spectra",   "218 wavelengths rebinned to 44 Ariel bins"),
    ("Quantum",   "12 qubits, depth 2, 36 trainable parameters"),
    ("Classical",  "~15k trainable parameters"),
    ("Optimizer", "AdamW  (classical lr=2e-3 / quantum lr=6e-4)"),
    ("Loss",      "MSE on log10 VMR, early stopping patience=6"),
]

for i, (k, v) in enumerate(items):
    yy = 4.5 - i * 0.5
    txt(3.0, yy, f"{k}:", fontsize=9, fontweight="bold", color=C_TEXT, ha="left")
    txt(5.5, yy, v, fontsize=9, color=C_DIM, ha="left")

plt.tight_layout(pad=0.3)
plt.savefig("outputs/model_crossgen_rebinned/model_architecture.png", dpi=200,
            facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print("Saved: outputs/model_crossgen_rebinned/model_architecture.png")
