#!/usr/bin/env python3
"""Assemble the current manuscript Fig. 4 PNG from regenerated panel PNGs."""

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


OUTPUT_DIR = FIGURES_DIR / "Figure4"
OUT_PNG = OUTPUT_DIR / "Figure4.png"
TARGET_SIZE = (4431, 7225)

PANELS = {
    "ab": ("fig4_ab_sweden_period_projection.png", 950.25, 695.32),
    "c": ("Fig4C.png", 477.37, 364.33),
    "d": ("Fig4D_extrap.png", 532.02, 364.33),
    "e": ("sweden_sr_age0_mean_lifespan_projection_1900_2100_n1m.png", 936.00, 547.20),
}

MARGIN = 54.0
ROW_GAP = 52.0
MIDDLE_GAP = 44.0
MIDDLE_WIDTH = PANELS["c"][1] + MIDDLE_GAP + PANELS["d"][1]
CONTENT_WIDTH = max(PANELS["ab"][1], MIDDLE_WIDTH, PANELS["e"][1])
ART_WIDTH = CONTENT_WIDTH + 2 * MARGIN
ART_HEIGHT = MARGIN + PANELS["ab"][2] + ROW_GAP + PANELS["c"][2] + ROW_GAP + PANELS["e"][2] + MARGIN


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
    scale_x = TARGET_SIZE[0] / ART_WIDTH
    scale_y = TARGET_SIZE[1] / ART_HEIGHT
    x = round(left * scale_x)
    y = round((ART_HEIGHT - top) * scale_y)
    w = round(width * scale_x)
    h = round(height * scale_y)
    return x, y, w, h


def paste_fit(canvas: Image.Image, image_path: Path, rect: tuple[int, int, int, int]) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Missing panel PNG: {image_path}")
    x, y, w, h = rect
    image = trim_white_border(Image.open(image_path).convert("RGB"))
    scale = min(w / image.width, h / image.height)
    new_size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, (x + (w - new_size[0]) // 2, y + (h - new_size[1]) // 2))


def trim_white_border(image: Image.Image, *, padding: int = 18) -> Image.Image:
    grayscale = image.convert("L")
    diff = Image.eval(grayscale, lambda pixel: 255 if pixel < 250 else 0)
    bbox = diff.getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)
    return image.crop((left, top, right, bottom))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", TARGET_SIZE, "white")

    top_y = ART_HEIGHT - MARGIN
    ab_left = (ART_WIDTH - PANELS["ab"][1]) / 2
    paste_fit(canvas, OUTPUT_DIR / PANELS["ab"][0], to_pixels(ab_left, top_y, PANELS["ab"][1], PANELS["ab"][2]))

    middle_y = top_y - PANELS["ab"][2] - ROW_GAP
    middle_left = (ART_WIDTH - MIDDLE_WIDTH) / 2
    paste_fit(canvas, OUTPUT_DIR / PANELS["c"][0], to_pixels(middle_left, middle_y, PANELS["c"][1], PANELS["c"][2]))
    paste_fit(
        canvas,
        OUTPUT_DIR / PANELS["d"][0],
        to_pixels(middle_left + PANELS["c"][1] + MIDDLE_GAP, middle_y, PANELS["d"][1], PANELS["d"][2]),
    )

    bottom_y = middle_y - PANELS["c"][2] - ROW_GAP
    bottom_left = (ART_WIDTH - PANELS["e"][1]) / 2
    paste_fit(canvas, OUTPUT_DIR / PANELS["e"][0], to_pixels(bottom_left, bottom_y, PANELS["e"][1], PANELS["e"][2]))

    scale_x = TARGET_SIZE[0] / ART_WIDTH
    scale_y = TARGET_SIZE[1] / ART_HEIGHT
    label_font = font(round(28 * scale_y))
    label_x = round((bottom_left - 42) * scale_x)
    label_y = round((ART_HEIGHT - (bottom_y + 12)) * scale_y - label_font.size * 0.82)
    ImageDraw.Draw(canvas).text((label_x, label_y), "e", fill=(0, 0, 0), font=label_font)

    canvas.save(OUT_PNG, optimize=True)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
