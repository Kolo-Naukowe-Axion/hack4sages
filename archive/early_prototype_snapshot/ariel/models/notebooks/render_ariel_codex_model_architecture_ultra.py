from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "ariel_codex_model_architecture_ultra.png"

W = 3200
H = 1800

BG = "#FFFFFF"
TEXT = "#0F172A"
BORDER = "#334155"
ARROW = "#475569"
CLASSICAL = "#DCFCE7"
QUANTUM = "#F3E8FF"
INPUT = "#DBEAFE"
MERGE = "#FEF3C7"
OUTPUT = "#FCE7F3"
NOTE = "#F1F5F9"


def font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
    else:
        candidates += [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE = font(64, bold=True)
SUB = font(28)
BOX_TITLE = font(34, bold=True)
BOX_BODY = font(24)
NOTE_FONT = font(28)


def box(draw: ImageDraw.ImageDraw, xy, title: str, lines: list[str], fill: str):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=28, fill=fill, outline=BORDER, width=5)
    draw.text((x1 + 26, y1 + 18), title, font=BOX_TITLE, fill=TEXT)
    y = y1 + 76
    for line in lines:
        draw.text((x1 + 26, y), line, font=BOX_BODY, fill=TEXT)
        y += 34


def center(xy):
    x1, y1, x2, y2 = xy
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def right_mid(xy):
    x1, y1, x2, y2 = xy
    return (x2, (y1 + y2) // 2)


def left_mid(xy):
    x1, y1, x2, y2 = xy
    return (x1, (y1 + y2) // 2)


def top_mid(xy):
    x1, y1, x2, y2 = xy
    return ((x1 + x2) // 2, y1)


def bottom_mid(xy):
    x1, y1, x2, y2 = xy
    return ((x1 + x2) // 2, y2)


def arrow(draw: ImageDraw.ImageDraw, start, end, label: str | None = None):
    draw.line([start, end], fill=ARROW, width=7)
    ex, ey = end
    draw.polygon([(ex, ey), (ex - 22, ey - 12), (ex - 22, ey + 12)], fill=ARROW)
    if label:
        bbox = draw.textbbox((0, 0), label, font=SUB)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        mx = (start[0] + end[0]) // 2
        my = (start[1] + end[1]) // 2
        draw.rounded_rectangle((mx - tw // 2 - 10, my - th // 2 - 6, mx + tw // 2 + 10, my + th // 2 + 6), radius=12, fill=BG)
        draw.text((mx - tw // 2, my - th // 2), label, font=SUB, fill=TEXT)


def poly_arrow(draw: ImageDraw.ImageDraw, points, label: str | None = None):
    for a, b in zip(points, points[1:]):
        draw.line([a, b], fill=ARROW, width=7)
    ex, ey = points[-1]
    px, py = points[-2]
    if ex >= px:
        head = [(ex, ey), (ex - 22, ey - 12), (ex - 22, ey + 12)]
    elif ex < px:
        head = [(ex, ey), (ex + 22, ey - 12), (ex + 22, ey + 12)]
    elif ey >= py:
        head = [(ex, ey), (ex - 12, ey - 22), (ex + 12, ey - 22)]
    else:
        head = [(ex, ey), (ex - 12, ey + 22), (ex + 12, ey + 22)]
    draw.polygon(head, fill=ARROW)
    if label:
        mid_i = len(points) // 2
        mx, my = points[mid_i]
        bbox = draw.textbbox((0, 0), label, font=SUB)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle((mx - tw // 2 - 10, my - th // 2 - 6, mx + tw // 2 + 10, my + th // 2 + 6), radius=12, fill=BG)
        draw.text((mx - tw // 2, my - th // 2), label, font=SUB, fill=TEXT)


def diamond(draw: ImageDraw.ImageDraw, cxy, text: str):
    cx, cy = cxy
    pts = [(cx, cy - 80), (cx + 110, cy), (cx, cy + 80), (cx - 110, cy)]
    draw.polygon(pts, fill=MERGE, outline=BORDER)
    bbox = draw.textbbox((0, 0), text, font=BOX_TITLE)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2), text, font=BOX_TITLE, fill=TEXT)


def render(path: Path = OUTPUT_PATH) -> Path:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((70, 50), "Ariel Codex Hybrid Model", font=TITLE, fill=TEXT)
    draw.text((74, 130), "Readable schematic of `HybridArielRegressor` only", font=SUB, fill="#334155")

    spectra = (80, 270, 420, 470)
    aux = (80, 1130, 420, 1290)

    spectral_encoder = (560, 220, 980, 520)
    aux_encoder = (560, 1080, 980, 1320)
    fusion = (1140, 620, 1540, 840)
    head_context = (1710, 540, 2140, 760)

    classical_head = (2310, 280, 2760, 500)
    classical_pred = (2890, 290, 3140, 490)

    projector = (1710, 1030, 2140, 1250)
    quantum_block = (2310, 980, 2760, 1260)
    quantum_head = (2310, 610, 2760, 830)
    quantum_corr = (2890, 620, 3140, 820)

    add_node = (2840, 1030)
    output = (2890, 990, 3140, 1190)

    note = (80, 1480, 3140, 1710)

    box(draw, spectra, "Spectra Input", ["shape: 4 x 52", "spectrum", "noise", "width", "wavelength"], INPUT)
    box(draw, aux, "Aux Input", ["shape: 8", "planet + star features"], INPUT)

    box(draw, spectral_encoder, "SpectralEncoder", ["Conv1d stem", "Residual blocks", "Mean pool + attention pool", "output: 96"], CLASSICAL)
    box(draw, aux_encoder, "AuxEncoder", ["MLP", "8 -> 32 -> 32", "output: 32"], CLASSICAL)
    box(draw, fusion, "FusionEncoder", ["concat(96, 32)", "MLP", "output: 128"], CLASSICAL)
    box(draw, head_context, "Head Context", ["concat(fused 128,", "spectral 96,", "aux 32)", "output: 256"], MERGE)

    box(draw, classical_head, "Classical Head", ["Regression MLP", "256 -> 192 -> 5"], CLASSICAL)
    box(draw, classical_pred, "Classical", ["prediction"], CLASSICAL)

    box(draw, projector, "QuantumProjector", ["from fused 128", "128 -> 128 -> 8", "angles in [-pi, pi]"], QUANTUM)
    box(draw, quantum_block, "QuantumBlock", ["8 qubits", "depth 2", "RY + CNOT + RZ + CRX", "output: 8"], QUANTUM)
    box(draw, quantum_head, "Quantum Head", ["uses head context +", "quantum features", "264 -> 192 -> 5"], QUANTUM)
    box(draw, quantum_corr, "Quantum", ["correction"], QUANTUM)

    diamond(draw, add_node, "+")
    box(draw, output, "Final Output", ["5 targets"], OUTPUT)

    arrow(draw, right_mid(spectra), left_mid(spectral_encoder), "4x52")
    arrow(draw, right_mid(aux), left_mid(aux_encoder), "8")

    poly_arrow(draw, [right_mid(spectral_encoder), (1060, right_mid(spectral_encoder)[1]), (1060, 690), left_mid(fusion)], "96")
    poly_arrow(draw, [right_mid(aux_encoder), (1060, right_mid(aux_encoder)[1]), (1060, 770), left_mid(fusion)], "32")

    arrow(draw, right_mid(fusion), left_mid(head_context), "128")
    poly_arrow(draw, [top_mid(fusion), (1340, 560), (1925, 560), bottom_mid(head_context)], None)

    poly_arrow(draw, [right_mid(head_context), left_mid(classical_head)], "256")
    arrow(draw, right_mid(classical_head), left_mid(classical_pred), "5")
    poly_arrow(draw, [bottom_mid(classical_pred), (3015, 560), (2840, 560), (2840, 950)], None)

    poly_arrow(draw, [bottom_mid(fusion), (1340, 940), (1925, 940), top_mid(projector)], "128")
    arrow(draw, right_mid(projector), left_mid(quantum_block), "8 angles")

    poly_arrow(draw, [top_mid(quantum_block), (2535, 900), (2535, 860), bottom_mid(quantum_head)], "8")
    poly_arrow(draw, [right_mid(head_context), (2220, 650), left_mid(quantum_head)], "256")

    arrow(draw, right_mid(quantum_head), left_mid(quantum_corr), "5")
    poly_arrow(draw, [bottom_mid(quantum_corr), (3015, 900), (2840, 900), (2840, 950)], None)

    poly_arrow(draw, [(2840, 1110), left_mid(output)], None)

    draw.rounded_rectangle(note, radius=28, fill=NOTE, outline=BORDER, width=4)
    draw.text((110, 1525), "Final formula:", font=BOX_TITLE, fill=TEXT)
    draw.text(
        (380, 1528),
        "output = classical_prediction + quantum_scale * tanh(quantum_gate) * quantum_correction",
        font=NOTE_FONT,
        fill=TEXT,
    )
    draw.text(
        (110, 1600),
        "If quantum is disabled, the model returns only the classical prediction.",
        font=NOTE_FONT,
        fill=TEXT,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


if __name__ == "__main__":
    saved = render()
    print(saved)
