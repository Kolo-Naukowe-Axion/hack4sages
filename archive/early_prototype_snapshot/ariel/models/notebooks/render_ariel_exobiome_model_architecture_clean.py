from __future__ import annotations

from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "ariel_exobiome_model_architecture_clean.png"

WIDTH = 2600
HEIGHT = 1500
BG = "#FFFFFF"
TEXT = "#102027"
LINE = "#546E7A"
BORDER = "#455A64"

COLORS = {
    "input": "#E3F2FD",
    "classical": "#E8F5E9",
    "quantum": "#F3E5F5",
    "merge": "#FFF3E0",
    "output": "#EDE7F6",
    "note": "#ECEFF1",
}


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = load_font(52, bold=True)
SUBTITLE_FONT = load_font(24)
BOX_TITLE_FONT = load_font(28, bold=True)
BOX_BODY_FONT = load_font(22)
NOTE_FONT = load_font(24)


def draw_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, lines: list[str], fill: str) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=BORDER, width=4)
    draw.text((x1 + 24, y1 + 18), title, font=BOX_TITLE_FONT, fill=TEXT)

    y = y1 + 68
    for line in lines:
        wrapped = wrap(line, width=26) if len(line) > 26 else [line]
        for segment in wrapped:
            draw.text((x1 + 24, y), segment, font=BOX_BODY_FONT, fill=TEXT)
            y += 30
        y += 6


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], label: str | None = None) -> None:
    draw.line([start, end], fill=LINE, width=6)
    ex, ey = end
    arrow = [(ex, ey), (ex - 18, ey - 10), (ex - 18, ey + 10)]
    draw.polygon(arrow, fill=LINE)
    if label:
        mx = (start[0] + end[0]) // 2
        my = (start[1] + end[1]) // 2 - 36
        bbox = draw.textbbox((0, 0), label, font=SUBTITLE_FONT)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rounded_rectangle((mx - tw // 2 - 10, my - 6, mx + tw // 2 + 10, my + th + 6), radius=14, fill="#FFFFFF")
        draw.text((mx - tw // 2, my), label, font=SUBTITLE_FONT, fill=TEXT)


def draw_merge(draw: ImageDraw.ImageDraw, center: tuple[int, int], label: str) -> None:
    cx, cy = center
    points = [(cx, cy - 46), (cx + 60, cy), (cx, cy + 46), (cx - 60, cy)]
    draw.polygon(points, fill=COLORS["merge"], outline=BORDER)
    bbox = draw.textbbox((0, 0), label, font=BOX_TITLE_FONT)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw / 2, cy - th / 2), label, font=BOX_TITLE_FONT, fill=TEXT)


def render(output_path: Path = OUTPUT_PATH) -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((70, 48), "Ariel ExoBiome Model Architecture", font=TITLE_FONT, fill=TEXT)
    draw.text((72, 114), "Simple schematic of the HybridArielRegressor", font=SUBTITLE_FONT, fill="#37474F")

    spectra = (70, 250, 360, 470)
    aux = (70, 760, 360, 930)
    spectral_encoder = (470, 180, 860, 540)
    aux_encoder = (470, 720, 860, 970)
    fusion = (980, 395, 1310, 615)
    head_context = (1410, 280, 1720, 500)
    classical_head = (1810, 160, 2180, 380)
    projector = (1410, 760, 1720, 980)
    quantum_block = (1810, 720, 2180, 1040)
    quantum_head = (1810, 470, 2180, 650)
    gate = (2280, 470, 2470, 650)
    output = (2280, 190, 2470, 380)
    note = (70, 1160, 2470, 1400)

    draw_box(draw, spectra, "Spectra Input", ["4 x 52 tensor", "spectrum", "noise", "width template", "wavelength"], COLORS["input"])
    draw_box(draw, aux, "Aux Input", ["8 scalar features", "planet and star metadata"], COLORS["input"])
    draw_box(draw, spectral_encoder, "SpectralEncoder", ["Conv1d stem: 4 -> 32", "Residual blocks: 32 -> 64 -> 96", "Mean pool + attention pool", "Output: spectral feature (96)"], COLORS["classical"])
    draw_box(draw, aux_encoder, "AuxEncoder", ["MLP: 8 -> 32 -> 32", "GELU + dropout", "Output: aux feature (32)"], COLORS["classical"])
    draw_box(draw, fusion, "FusionEncoder", ["Concatenate spectral + aux", "MLP: 128 -> 128", "Output: fused feature (128)"], COLORS["classical"])
    draw_box(draw, head_context, "Head Context", ["Concatenate", "fused 128 + spectral 96 + aux 32", "Output: 256 dims"], COLORS["merge"])
    draw_box(draw, classical_head, "Classical Head", ["Regression MLP", "256 -> 192 -> 5", "Classical prediction"], COLORS["classical"])
    draw_box(draw, projector, "QuantumProjector", ["Uses fused feature only", "128 -> 128 -> 8", "tanh(.) * pi", "Output: 8 angles"], COLORS["quantum"])
    draw_box(draw, quantum_block, "QuantumBlock", ["8 qubits, depth 2", "RY embedding", "CNOT + RZ + CRX layers", "Output: 8 expectation values"], COLORS["quantum"])
    draw_box(draw, quantum_head, "Quantum Head", ["Concatenate head context + quantum", "264 -> 192 -> 5", "Residual correction"], COLORS["quantum"])
    draw_box(draw, gate, "Gate", ["learnable 5-d", "tanh(g)"], COLORS["quantum"])
    draw_box(draw, output, "Final Output", ["5 targets", "classical + gated", "quantum residual"], COLORS["output"])

    draw_merge(draw, (1565, 610), "+")

    draw_arrow(draw, (360, 360), (470, 360), "4 x 52")
    draw_arrow(draw, (360, 845), (470, 845), "8")
    draw_arrow(draw, (860, 360), (980, 470), "96")
    draw_arrow(draw, (860, 845), (980, 540), "32")
    draw_arrow(draw, (1310, 470), (1410, 390), "128")
    draw_arrow(draw, (1720, 330), (1810, 270), "256")
    draw_arrow(draw, (1310, 540), (1410, 870), "128")
    draw_arrow(draw, (1720, 430), (1810, 560), "256")
    draw_arrow(draw, (1720, 870), (1810, 870), "8 angles")
    draw_arrow(draw, (2180, 560), (2280, 560), "residual")
    draw_arrow(draw, (2180, 870), (1995, 650), "8 expvals")
    draw_arrow(draw, (2470, 560), (1565, 610), "gate x residual")
    draw_arrow(draw, (2180, 270), (2280, 270), "classical pred")
    draw_arrow(draw, (1565, 564), (2375, 380), "add")

    draw.rounded_rectangle(note, radius=28, fill=COLORS["note"], outline=BORDER, width=4)
    draw.text((95, 1190), "Key idea", font=BOX_TITLE_FONT, fill=TEXT)
    note_lines = [
        "The model is mostly classical. The quantum path is a residual branch, not a separate full model.",
        "In classical-only mode, the forward pass stops after the classical head.",
        "In hybrid mode: final_prediction = classical_prediction + quantum_scale * tanh(quantum_gate) * quantum_correction.",
    ]
    y = 1245
    for line in note_lines:
        draw.text((95, y), line, font=NOTE_FONT, fill=TEXT)
        y += 42

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


if __name__ == "__main__":
    saved = render()
    print(saved)
