"""ExoBiome architecture — horizontal, minimal, light, readable at presentation scale."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(36, 16))
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 16)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    BG = {"in": "#f8fafc", "enc": "#eff6ff", "fus": "#ecfdf5",
          "qnt": "#f5f3ff", "hd": "#fff1f2", "out": "#fffbeb"}
    BD = {"in": "#94a3b8", "enc": "#3b82f6", "fus": "#10b981",
          "qnt": "#8b5cf6", "hd": "#f43f5e", "out": "#d97706"}
    TXT, SUB, SKIP = "#1e293b", "#64748b", "#bfc5cc"

    BW, BH = 4.6, 1.4

    def box(cx, cy, w, h, title, sub, bg, bd, ts=18, ss=13):
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.12", fc=bg, ec=bd, lw=5.0, zorder=3))
        if sub:
            ax.text(cx, cy + 0.22, title, ha="center", va="center",
                    fontsize=ts, fontweight="bold", color=TXT, zorder=4)
            ax.text(cx, cy - 0.3, sub, ha="center", va="center",
                    fontsize=ss, color=SUB, zorder=4)
        else:
            ax.text(cx, cy, title, ha="center", va="center",
                    fontsize=ts, fontweight="bold", color=TXT, zorder=4)

    def polyarrow(pts, color, lw=6.0, ls="-"):
        xs, ys = zip(*pts)
        if len(pts) > 2:
            ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls,
                    solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.annotate("", xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    linestyle=ls, mutation_scale=35), zorder=2)

    # ═══ GRID ═══
    Y_TOP, Y_MID, Y_BOT = 12.5, 8.0, 3.5

    XI  = 2.8
    XE  = 8.2
    XF  = 14.0
    XC  = 19.5
    XCH = 25.0
    XG  = 31.5

    XQP  = XF
    XVQC = XC
    XQH  = XCH

    hw, hh = BW/2, BH/2

    # ═══ TITLE ═══
    ax.text(18, 15.3, "ExoBiome", ha="center", fontsize=32,
            fontweight="bold", color="#0f172a")
    ax.text(18, 14.7, "Classical-Quantum Hybrid  ·  Five-Gas Atmospheric Retrieval",
            ha="center", fontsize=16, color=SUB)

    # ═══ TOP ROW — Spectrum path ═══
    box(XI, Y_TOP, 3.8, BH, "Spectrum", "4 ch × 52 bins", BG["in"], BD["in"])
    box(XE, Y_TOP, BW, BH, "Spectral Encoder",
        "Stem → ResBlock×3 (residual) → DualPool → 96d",
        BG["enc"], BD["enc"], ss=12)

    # ═══ MID ROW — main pipeline ═══
    box(XF, Y_MID, BW, BH, "Fusion Encoder",
        "[s₉₆ ‖ a₃₂] → MLP(128) → LN → 128d", BG["fus"], BD["fus"])
    box(XC, Y_MID, BW, BH, "Head Context z",
        "[f₁₂₈ ‖ s₉₆ ‖ a₃₂] = 256d", BG["fus"], BD["fus"])
    box(XCH, Y_MID, BW, BH, "Classical Head",
        "Linear(256 → 192 → 5)", BG["hd"], BD["hd"])
    box(XG, Y_MID, 4.8, 2.0, "Gated Output  ŷ ∈ ℝ⁵",
        "cls + α · tanh(g) ⊙ qnt", BG["out"], BD["out"], ts=18, ss=13)
    ax.text(XG, Y_MID - 0.7, "H₂O   CO₂   CO   CH₄   NH₃",
            ha="center", fontsize=12, color=SUB, zorder=4)

    # ═══ BOT ROW — Aux path + Quantum branch ═══
    box(XI, Y_BOT, 3.8, BH, "Auxiliary", "8 features", BG["in"], BD["in"])
    box(XE, Y_BOT, BW, BH, "Aux Encoder",
        "Linear(8 → 32 → 32) → 32d", BG["enc"], BD["enc"])
    box(XQP, Y_BOT, BW, BH, "Quantum Projector",
        "MLP(128 → 128 → n_q) → LN → π·tanh", BG["qnt"], BD["qnt"], ss=12)
    box(XVQC, Y_BOT, BW, BH, "Variational QC",
        "RY enc → [RY·CNOT·RZ·CRX]×B → ⟨Z⟩", BG["qnt"], BD["qnt"], ss=12)
    box(XQH, Y_BOT, BW, BH, "Quantum Head",
        "Linear(256+n_q → 192 → 5)", BG["hd"], BD["hd"])

    # ═══════════════════════════════════════════════════
    # ARROWS
    # ═══════════════════════════════════════════════════

    # Input → Encoder (horizontal)
    polyarrow([(XI+1.9, Y_TOP), (XE-hw, Y_TOP)], BD["in"])
    polyarrow([(XI+1.9, Y_BOT), (XE-hw, Y_BOT)], BD["in"])

    # Spectral Enc → Fusion (L: right, down, right)
    mx1 = XE + hw + 0.5
    polyarrow([(XE+hw, Y_TOP), (mx1, Y_TOP),
               (mx1, Y_MID+0.18), (XF-hw, Y_MID+0.18)], BD["fus"])

    # Aux Enc → Fusion (L: right, up, right)
    mx2 = XE + hw + 1.0
    polyarrow([(XE+hw, Y_BOT), (mx2, Y_BOT),
               (mx2, Y_MID-0.18), (XF-hw, Y_MID-0.18)], BD["fus"])

    # Fusion → Context (horizontal)
    polyarrow([(XF+hw, Y_MID), (XC-hw, Y_MID)], BD["fus"])

    # Context → Classical Head (horizontal)
    polyarrow([(XC+hw, Y_MID), (XCH-hw, Y_MID)], BD["hd"])

    # Classical Head → Gate (horizontal)
    polyarrow([(XCH+hw, Y_MID), (XG-4.8/2, Y_MID)], BD["out"])

    # Fusion → Q.Proj (straight DOWN)
    polyarrow([(XF, Y_MID-hh), (XF, Y_BOT+hh)], BD["qnt"])

    # Q.Proj → VQC (horizontal)
    polyarrow([(XQP+hw, Y_BOT), (XVQC-hw, Y_BOT)], BD["qnt"])

    # VQC → Q.Head (horizontal)
    polyarrow([(XVQC+hw, Y_BOT), (XQH-hw, Y_BOT)], BD["qnt"])

    # Q.Head → Gate (L: right, up)
    polyarrow([(XQH+hw, Y_BOT), (XG, Y_BOT),
               (XG, Y_MID-2.0/2)], BD["hd"])

    # Skip: Spectral Enc → Context (dashed, above main row)
    sy = Y_TOP - 1.1
    polyarrow([(XE+hw, Y_TOP-0.3), (XE+hw+0.2, Y_TOP-0.3),
               (XE+hw+0.2, sy), (XC, sy),
               (XC, Y_MID+hh)],
              SKIP, lw=4.0, ls="--")

    # Skip: Aux Enc → Context (dashed, between rows)
    ay = Y_BOT + 1.2 + 0.5
    polyarrow([(XE+hw, Y_BOT+0.3), (XE+hw+0.2, Y_BOT+0.3),
               (XE+hw+0.2, ay), (XC+0.4, ay),
               (XC+0.4, Y_MID-hh)],
              SKIP, lw=4.0, ls="--")

    # Skip: Context → Q.Head (dashed)
    polyarrow([(XC+hw, Y_MID-0.4),
               (XC+hw+0.4, Y_MID-0.4),
               (XC+hw+0.4, Y_BOT+hh+0.2),
               (XQH-hw, Y_BOT+hh+0.2)],
              SKIP, lw=4.0, ls="--")

    # Annotation labels
    ax.text(XF-0.2, (Y_MID+Y_BOT)/2, "fused\n128d", ha="right",
            fontsize=13, color=BD["qnt"], fontstyle="italic", zorder=4)
    ax.text((XC+XCH)/2, Y_MID+0.5, "z  256d", ha="center",
            fontsize=13, color=BD["hd"], fontstyle="italic", zorder=4)

    # ═══ LEGEND ═══
    items = [("Encoder", BG["enc"], BD["enc"]),
             ("Fusion / Context", BG["fus"], BD["fus"]),
             ("Quantum", BG["qnt"], BD["qnt"]),
             ("Regression Head", BG["hd"], BD["hd"]),
             ("Output", BG["out"], BD["out"])]
    for i, (lab, bg, bd) in enumerate(items):
        lx = 5.5 + i * 4.8
        ax.add_patch(FancyBboxPatch(
            (lx, 0.7), 0.55, 0.4,
            boxstyle="round,pad=0.04", fc=bg, ec=bd, lw=1.5, zorder=3))
        ax.text(lx + 0.85, 0.9, lab, ha="left", va="center",
                fontsize=13, color=SUB)
    ax.text(31, 0.9, "– – –  skip connection", ha="left",
            va="center", fontsize=12, color=SKIP)
    ax.text(18, 0.2,
            "Default: n_q = 8 qubits  ·  depth = 2  ·  24 variational params"
            "  ·  PennyLane Lightning  ·  adjoint differentiation",
            ha="center", fontsize=13, color="#9ca3af", fontstyle="italic")

    plt.tight_layout(pad=0.5)
    return fig


if __name__ == "__main__":
    fig = draw_architecture()
    fig.savefig("architecture_diagram.png", dpi=200, bbox_inches="tight",
                facecolor="#ffffff")
    fig.savefig("architecture_diagram.pdf", bbox_inches="tight",
                facecolor="#ffffff")
    print("Saved: architecture_diagram.png, architecture_diagram.pdf")
    plt.close(fig)
