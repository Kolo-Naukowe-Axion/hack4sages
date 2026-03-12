"""Generate architecture diagram for HybridAtmosphereModel — dark theme."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.set_xlim(0, 14)
ax.set_ylim(0, 20)
ax.axis("off")

BG = "#0f1123"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

C_GREEN = "#1b8a2e"
C_GREEN_BORDER = "#2ecc40"
C_BLUE = "#1a5276"
C_BLUE_BORDER = "#2e86c1"
C_PURPLE = "#6c3483"
C_PURPLE_BORDER = "#a569bd"
C_RED = "#c0392b"
C_RED_BORDER = "#e74c3c"
C_GRAY = "#4a4a5a"
C_GRAY_BORDER = "#7f8c8d"
C_ORANGE = "#e67e22"
WHITE = "#ffffff"
LIGHT = "#b0b8c8"
DIM = "#7a8494"


def rbox(x, y, w, h, fc, ec, lw=2.0, radius=0.08):
    r = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(r)


def txt(x, y, s, fs=11, c=WHITE, fw="bold", **kw):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight=fw,
            ha="center", va="center", zorder=5, **kw)


def sub(x, y, s, fs=9, c=LIGHT):
    ax.text(x, y, s, fontsize=fs, color=c, fontweight="normal",
            ha="center", va="center", zorder=5, style="italic")


def arrow(x1, y1, x2, y2, c=WHITE, lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw), zorder=3)


def skip(x1, y1, x2, y2, rad=0.3, label=None, lx=None, ly=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.5,
                                ls="--", connectionstyle=f"arc3,rad={rad}"),
                zorder=3)
    if label and lx is not None:
        ax.text(lx, ly, label, fontsize=8, color=C_ORANGE, ha="center", va="center",
                fontweight="bold", zorder=5)


# ─── TITLE ───
txt(7, 19.3, "Hybrid Quantum-Classical Architecture", 22, WHITE)
sub(7, 18.85, "ADC2023 Biosignature Detection", 12, DIM)

# ─── INPUTS ───
yi = 17.6
bw_in = 4.0
bh_in = 0.95

rbox(1.0, yi, bw_in, bh_in, C_GREEN, C_GREEN_BORDER)
txt(3.0, yi + 0.58, "Auxiliary Input", 14)
sub(3.0, yi + 0.25, "8 features (star/planet props)", 9)

rbox(9.0, yi, bw_in, bh_in, C_GREEN, C_GREEN_BORDER)
txt(11.0, yi + 0.58, "Spectral Input", 14)
sub(11.0, yi + 0.25, "52 wavelength bins", 9)

# ─── ENCODERS ───
ye = 15.6
bw_e = 4.6
bh_e = 1.1

rbox(0.7, ye, bw_e, bh_e, C_BLUE, C_BLUE_BORDER)
txt(3.0, ye + 0.68, "AuxEncoder", 14)
sub(3.0, ye + 0.3, "FFN: 8 → 64 → 64 → 32", 9.5)

rbox(8.7, ye, bw_e, bh_e, C_BLUE, C_BLUE_BORDER)
txt(11.0, ye + 0.68, "SpectralEncoder", 14)
sub(11.0, ye + 0.3, "Conv1d: (1, 52) → pool → 32", 9.5)

arrow(3.0, yi, 3.0, ye + bh_e)
arrow(11.0, yi, 11.0, ye + bh_e)

# ─── FUSION ENCODER ───
yf = 13.2
bw_f = 8.0
bh_f = 1.3

rbox(3.0, yf, bw_f, bh_f, C_PURPLE, C_PURPLE_BORDER)
txt(7.0, yf + 0.82, "FusionEncoder", 15)
sub(7.0, yf + 0.38, "Cat(32 + 32) → FFN → LayerNorm → tanh·π → 8-dim", 9.5)

arrow(3.0, ye, 5.5, yf + bh_f)
arrow(11.0, ye, 8.5, yf + bh_f)

# ─── QUANTUM BLOCK ───
yqc = 10.6
bw_q = 7.0
bh_q = 1.6

rbox(3.5, yqc, bw_q, bh_q, C_RED, C_RED_BORDER)
txt(7.0, yqc + 1.2, "Quantum Circuit (VQC)", 15)
sub(7.0, yqc + 0.75, "8 qubits · depth 2 · RY/CNOT/RZ/CRX", 10)
sub(7.0, yqc + 0.35, "24 trainable parameters · PennyLane adjoint diff", 8.5, DIM)

arrow(7.0, yf, 7.0, yqc + bh_q)

# ─── SKIP CONNECTIONS ───
# aux skip → concat
skip(1.5, ye, 1.5, 8.55 + 0.95, rad=-0.15,
     label="skip (32)", lx=2.2, ly=12.3)

# spectral skip → concat
skip(12.5, ye, 12.5, 8.55 + 0.95, rad=0.15,
     label="skip (32)", lx=11.8, ly=12.3)

# fusion/latent skip → concat
skip(4.2, yf, 3.2, 8.55 + 0.95, rad=-0.15,
     label="skip (8)", lx=2.8, ly=10.8)

# ─── CONCATENATE ───
ycat = 8.55
bw_cat = 11.4
bh_cat = 0.95

rbox(1.3, ycat, bw_cat, bh_cat, C_GRAY, C_GRAY_BORDER)
txt(7.0, ycat + 0.6, "Concatenate", 14)
sub(7.0, ycat + 0.22, "[quantum_out(8) + fusion(8) + aux(32) + spectral(32)] = 80-dim", 9)

arrow(7.0, yqc, 7.0, ycat + bh_cat)

# ─── PREDICTION HEAD ───
yph = 6.5
bw_ph = 7.5
bh_ph = 1.2

rbox(3.25, yph, bw_ph, bh_ph, C_BLUE, C_BLUE_BORDER)
txt(7.0, yph + 0.78, "PredictionHead", 15)
sub(7.0, yph + 0.32, "FFN: 80 → 96 → 96 → 5  (+ residual from quantum)", 9.5)

arrow(7.0, ycat, 7.0, yph + bh_ph)

# ─── OUTPUT ───
yo = 4.5
bw_o = 7.5
bh_o = 1.1

rbox(3.25, yo, bw_o, bh_o, C_GREEN, C_GREEN_BORDER)
txt(7.0, yo + 0.68, "Output", 15)
sub(7.0, yo + 0.28, "5 predictions: log_H2O, log_CO2, log_CO, log_CH4, log_NH3", 9.5)

arrow(7.0, yph, 7.0, yo + bh_o)

# ─── LEGEND ───
yl = 3.0
txt(1.5, yl, "Legend:", 10, LIGHT, fw="bold")

items = [
    (C_GREEN, C_GREEN_BORDER, "Input / Output"),
    (C_BLUE, C_BLUE_BORDER, "Classical (FFN / Conv1d)"),
    (C_PURPLE, C_PURPLE_BORDER, "Fusion Layer"),
    (C_RED, C_RED_BORDER, "Quantum Circuit"),
]
xs = [1.0, 4.2, 7.6, 10.5]
for (fc, ec, label), xl in zip(items, xs):
    rbox(xl, yl - 0.65, 0.45, 0.3, fc, ec, 1.5)
    ax.text(xl + 0.65, yl - 0.5, label, fontsize=8.5, color=LIGHT,
            ha="left", va="center", zorder=5)

# orange skip legend
from matplotlib.lines import Line2D
ax.plot([12.8, 13.3], [yl - 0.5, yl - 0.5], color=C_ORANGE, ls="--", lw=1.5, zorder=5)
ax.text(13.5, yl - 0.5, "Skip", fontsize=8.5, color=LIGHT, ha="left", va="center", zorder=5)

# ─── FOOTER ───
sub(7.0, 1.65, "Training: AdamW (classical lr=5e-5, quantum lr=2e-4) · MSE Loss · Early stopping (patience=8)", 8.5, DIM)
sub(7.0, 1.25, "Data: 33,138 train / 4,142 val / 4,143 holdout / 685 test · Batch size 16 · Max 30 epochs", 8.5, DIM)

plt.tight_layout(pad=0.3)
out = "/Users/michalszczesny/projects/hack4sages/quantum_model/model.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out}")
