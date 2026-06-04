#!/usr/bin/env python3
"""Assemble the current manuscript Fig. 2 PNG from regenerated panel PNGs."""

from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from senogenic_vs_robustness.paths import FIGURES_DIR


OUTPUT_DIR = FIGURES_DIR / "Figure2"
OUT_PNG = OUTPUT_DIR / "Figure2.png"

TARGET_SIZE = (7468, 5990)
ARTBOARD_WIDTH = 1805.82
ARTBOARD_HEIGHT = 1520.33
BOTTOM_CROP = 60.0
VISIBLE_HEIGHT = ARTBOARD_HEIGHT - BOTTOM_CROP

TOP_GROUP_WIDTH = 1780.0
TOP_GAP = 35.0
TOP_PANEL_WIDTH = (TOP_GROUP_WIDTH - 2 * TOP_GAP) / 3
TOP_PANEL_HEIGHT = TOP_PANEL_WIDTH * (352.8 / 482.4)
TOP_X = (ARTBOARD_WIDTH - TOP_GROUP_WIDTH) / 2
TOP_Y = ARTBOARD_HEIGHT - 80

BOTTOM_WIDTH = 1780.0
BOTTOM_HEIGHT = BOTTOM_WIDTH * (629.61835 / 1259.2680854072)
BOTTOM_X = (ARTBOARD_WIDTH - BOTTOM_WIDTH) / 2
BOTTOM_Y = TOP_Y - TOP_PANEL_HEIGHT - 40

PANEL_RECTS = [
    ("fig2b_new.png", TOP_X, TOP_Y, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT),
    ("fig2a_new.png", TOP_X + TOP_PANEL_WIDTH + TOP_GAP, TOP_Y, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT),
    ("fig2c_new.png", TOP_X + 2 * (TOP_PANEL_WIDTH + TOP_GAP), TOP_Y, TOP_PANEL_WIDTH, TOP_PANEL_HEIGHT),
    ("fig2de_new.png", BOTTOM_X, BOTTOM_Y, BOTTOM_WIDTH, BOTTOM_HEIGHT),
]

LABELS = [
    ("a", 12.0, TOP_Y + 25),
    ("b", TOP_X + TOP_PANEL_WIDTH + TOP_GAP - 42, TOP_Y + 25),
    ("c", TOP_X + 2 * (TOP_PANEL_WIDTH + TOP_GAP) - 42, TOP_Y + 25),
]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def to_pixels(left: float, top: float, width: float, height: float) -> tuple[int, int, int, int]:
    scale_x = TARGET_SIZE[0] / ARTBOARD_WIDTH
    scale_y = TARGET_SIZE[1] / VISIBLE_HEIGHT
    x = round(left * scale_x)
    y = round((ARTBOARD_HEIGHT - top) * scale_y)
    w = round(width * scale_x)
    h = round(height * scale_y)
    return x, y, w, h


def paste_fit(canvas: Image.Image, image_path: Path, rect: tuple[int, int, int, int]) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing panel PNG: {image_path}")
    x, y, w, h = rect
    image = Image.open(image_path).convert("RGB")
    scale = min(w / image.width, h / image.height)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, (x + (w - new_size[0]) // 2, y + (h - new_size[1]) // 2))


def draw_labels(canvas: Image.Image) -> None:
    scale_x = TARGET_SIZE[0] / ARTBOARD_WIDTH
    scale_y = TARGET_SIZE[1] / VISIBLE_HEIGHT
    label_font = font(round(64 * scale_y))
    draw = ImageDraw.Draw(canvas)
    for label, left, top in LABELS:
        x = round(left * scale_x)
        y = round((ARTBOARD_HEIGHT - top) * scale_y - label_font.size * 0.82)
        draw.text((x, y), label, fill=(0, 0, 0), font=label_font)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", TARGET_SIZE, "white")
    for filename, left, top, width, height in PANEL_RECTS:
        paste_fit(canvas, OUTPUT_DIR / filename, to_pixels(left, top, width, height))
    draw_labels(canvas)
    canvas.save(OUT_PNG, optimize=True)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
