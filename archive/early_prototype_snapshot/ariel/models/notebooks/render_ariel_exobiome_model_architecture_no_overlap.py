from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "ariel_exobiome_model_architecture_no_overlap.png"

W = 3400
H = 1700

BG = "#FFFFFF"
TEXT = "#0F172A"
BORDER = "#334155"
LINE = "#475569"
INPUT = "#DBEAFE"
CLASSICAL = "#DCFCE7"
QUANTUM = "#F3E8FF"
MERGE = "#FEF3C7"
OUTPUT = "#FCE7F3"
NOTE = "#F1F5F9"


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE = load_font(64, bold=True)
SUBTITLE = load_font(28)
BOX_TITLE = load_font(32, bold=True)
BOX_BODY = load_font(24)
NOTE_FONT = load_font(28)


def box(draw: ImageDraw.ImageDraw, rect, title: str, lines: list[str], fill: str) -> None:
    x1, y1, x2, y2 = rect
    draw.rounded_rectangle(rect, radius=30, fill=fill, outline=BORDER, width=5)
    draw.text((x1 + 24, y1 + 18), title, font=BOX_TITLE, fill=TEXT)
    y = y1 + 74
    for line in lines:
        draw.text((x1 + 24, y), line, font=BOX_BODY, fill=TEXT)
        y += 34


def diamond(draw: ImageDraw.ImageDraw, center: tuple[int, int], label: str) -> None:
    cx, cy = center
    points = [(cx, cy - 70), (cx + 90, cy), (cx, cy + 70), (cx - 90, cy)]
    draw.polygon(points, fill=MERGE, outline=BORDER)
    bbox = draw.textbbox((0, 0), label, font=BOX_TITLE)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2), label, font=BOX_TITLE, fill=TEXT)


def left_mid(rect):
    x1, y1, x2, y2 = rect
    return (x1, (y1 + y2) // 2)


def right_mid(rect):
    x1, y1, x2, y2 = rect
    return (x2, (y1 + y2) // 2)


def top_mid(rect):
    x1, y1, x2, y2 = rect
    return ((x1 + x2) // 2, y1)


def bottom_mid(rect):
    x1, y1, x2, y2 = rect
    return ((x1 + x2) // 2, y2)


def poly_arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]]) -> None:
    for start, end in zip(points, points[1:]):
        draw.line([start, end], fill=LINE, width=8)

    ex, ey = points[-1]
    px, py = points[-2]
    if ex > px:
        head = [(ex, ey), (ex - 24, ey - 14), (ex - 24, ey + 14)]
    elif ex < px:
        head = [(ex, ey), (ex + 24, ey - 14), (ex + 24, ey + 14)]
    elif ey > py:
        head = [(ex, ey), (ex - 14, ey - 24), (ex + 14, ey - 24)]
    else:
        head = [(ex, ey), (ex - 14, ey + 24), (ex + 14, ey + 24)]
    draw.polygon(head, fill=LINE)


def render(output_path: Path = OUTPUT_PATH) -> Path:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    draw.text((70, 45), "Ariel ExoBiome Hybrid Model", font=TITLE, fill=TEXT)
    draw.text((74, 125), "No-overlap architecture diagram of `HybridArielRegressor`", font=SUBTITLE, fill="#334155")

    spectra = (90, 280, 410, 450)
    aux = (90, 650, 410, 790)

    spectral_encoder = (560, 220, 980, 500)
    aux_encoder = (560, 580, 980, 820)
    fusion = (1140, 390, 1480, 610)
    head_context = (1640, 390, 1990, 610)

    classical_head = (2150, 180, 2520, 380)
    classical_pred = (2680, 180, 3010, 380)

    projector = (1640, 820, 1990, 1020)
    quantum_block = (2150, 820, 2520, 1060)
    quantum_head = (2150, 1140, 2520, 1360)
    quantum_corr = (2680, 1140, 3010, 1360)

    sum_node = (3140, 770)
    output = (3200, 670, 3360, 870)
    note = (90, 1470, 3010, 1620)

    box(draw, spectra, "Spectra Input", ["4 x 52", "spectrum", "noise", "width", "wavelength"], INPUT)
    box(draw, aux, "Aux Input", ["8 features", "planet + star"], INPUT)

    box(draw, spectral_encoder, "SpectralEncoder", ["Conv1d stem", "Residual blocks", "Mean + attention pool", "output: 96"], CLASSICAL)
    box(draw, aux_encoder, "AuxEncoder", ["MLP", "8 -> 32 -> 32", "output: 32"], CLASSICAL)
    box(draw, fusion, "FusionEncoder", ["concat(96,32)", "MLP", "output: 128"], CLASSICAL)
    box(draw, head_context, "Head Context", ["concat(128,96,32)", "output: 256"], MERGE)

    box(draw, classical_head, "Classical Head", ["Regression MLP", "256 -> 192 -> 5"], CLASSICAL)
    box(draw, classical_pred, "Classical Pred", ["5 values"], CLASSICAL)

    box(draw, projector, "QuantumProjector", ["from fused 128", "128 -> 128 -> 8"], QUANTUM)
    box(draw, quantum_block, "QuantumBlock", ["8 qubits", "depth 2", "output: 8"], QUANTUM)
    box(draw, quantum_head, "Quantum Head", ["concat(256,8)", "264 -> 192 -> 5"], QUANTUM)
    box(draw, quantum_corr, "Quantum Corr", ["5 values"], QUANTUM)

    diamond(draw, sum_node, "+")
    box(draw, output, "Output", ["5 targets"], OUTPUT)

    poly_arrow(draw, [right_mid(spectra), left_mid(spectral_encoder)])
    poly_arrow(draw, [right_mid(aux), left_mid(aux_encoder)])

    poly_arrow(draw, [right_mid(spectral_encoder), (1060, right_mid(spectral_encoder)[1]), (1060, 460), left_mid(fusion)])
    poly_arrow(draw, [right_mid(aux_encoder), (1060, right_mid(aux_encoder)[1]), (1060, 540), left_mid(fusion)])
    poly_arrow(draw, [right_mid(fusion), left_mid(head_context)])

    poly_arrow(draw, [right_mid(head_context), left_mid(classical_head)])
    poly_arrow(draw, [right_mid(classical_head), left_mid(classical_pred)])

    poly_arrow(draw, [bottom_mid(fusion), (1310, 720), (1815, 720), top_mid(projector)])
    poly_arrow(draw, [right_mid(projector), left_mid(quantum_block)])
    poly_arrow(draw, [bottom_mid(quantum_block), (2335, 1100), top_mid(quantum_head)])
    poly_arrow(draw, [bottom_mid(head_context), (1815, 1090), (1815, 1250), left_mid(quantum_head)])
    poly_arrow(draw, [right_mid(quantum_head), left_mid(quantum_corr)])

    poly_arrow(draw, [right_mid(classical_pred), (3090, 280), (3090, 700), (3050, 700)])
    poly_arrow(draw, [right_mid(quantum_corr), (3090, 1250), (3090, 840), (3050, 840)])
    poly_arrow(draw, [(3230, 770), left_mid(output)])

    draw.rounded_rectangle(note, radius=30, fill=NOTE, outline=BORDER, width=4)
    draw.text((120, 1510), "final = classical_prediction + quantum_scale * tanh(quantum_gate) * quantum_correction", font=NOTE_FONT, fill=TEXT)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


if __name__ == "__main__":
    saved = render()
    print(saved)
