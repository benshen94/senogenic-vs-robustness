#!/usr/bin/env python3
"""Build an alternate Fig. 1 layout with three schematic top-row panels."""

from __future__ import annotations

from io import BytesIO
from math import sin
from pathlib import Path
import random

import fitz
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

import make_fig1_panel_a_alt_sr_potential as landscape
import make_fig1_panels_ab as panels_ab
from make_fig1_panels_ab import (
    BLACK,
    GRAY,
    GRAY_LIGHT,
    OUTPUT_DIR,
    line,
    math_text,
    path_from_points,
    rect,
    svg_footer,
    svg_header,
    text,
)


ALT_A_STEM = "fig1_alt_panel_a_trajectory"
ALT_B_STEM = "fig1_alt_panel_b_landscape"

ALT_A_SVG = OUTPUT_DIR / f"{ALT_A_STEM}.svg"
ALT_A_PDF = OUTPUT_DIR / f"{ALT_A_STEM}.pdf"
ALT_A_PNG = OUTPUT_DIR / f"{ALT_A_STEM}.png"
ALT_B_SVG = OUTPUT_DIR / f"{ALT_B_STEM}.svg"
ALT_B_PDF = OUTPUT_DIR / f"{ALT_B_STEM}.pdf"
ALT_B_PNG = OUTPUT_DIR / f"{ALT_B_STEM}.png"

OUT_PDF = OUTPUT_DIR / "Fig1_alt.pdf"
OUT_PNG = OUTPUT_DIR / "Fig1_alt.png"

PAGE_WIDTH = 2067.0
PAGE_HEIGHT = 2677.0
PANEL_LABEL_SIZE = 70
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"


def trajectory_values(steps: int = 86, crossing_index: int = 76) -> list[float]:
    """One noisy parabolic trajectory that crosses the threshold late."""
    rng = random.Random(512)
    threshold = 0.74
    values: list[float] = []
    for index in range(steps):
        progress = index / (steps - 1)
        drift = 0.050 + 0.67 * progress**2.15
        noise = rng.gauss(0.0, 0.022 + 0.020 * progress) + 0.025 * progress * sin(index * 1.25)
        value = drift + noise
        values.append(max(0.025, min(value, threshold - 0.055)))

    for index in range(crossing_index):
        values[index] = min(values[index], threshold - 0.060)
    values[crossing_index - 2] = threshold - 0.17
    values[crossing_index - 1] = threshold - 0.11
    values[crossing_index] = threshold
    return values[: crossing_index + 1]


def make_alt_panel_a_trajectory() -> None:
    """Create the new top-row trajectory schematic."""
    width, height = 1250, 1250
    plot_x, plot_y, plot_w, plot_h = 145, 195, 960, 620
    axis_bottom = plot_y + plot_h
    threshold = 0.74
    values = trajectory_values()
    points = []
    for index, value in enumerate(values):
        x = plot_x + index / (len(values) - 1) * plot_w
        y = axis_bottom - value * plot_h
        points.append((x, y))

    threshold_y = axis_bottom - threshold * plot_h
    chunks = svg_header(width, height)
    chunks.append(text(["Stochastic threshold-crossing view of aging"], width / 2, 82, size=32, weight="400", anchor="middle"))
    chunks.append(line(plot_x, axis_bottom, plot_x + plot_w, axis_bottom, width=4.0, arrow="arrow"))
    chunks.append(line(plot_x, axis_bottom, plot_x, plot_y + 20, width=4.0, arrow="arrow"))
    chunks.append(line(plot_x, threshold_y, plot_x + plot_w, threshold_y, color=GRAY_LIGHT, width=4.2, dash="14 10"))
    chunks.append(text(["failure threshold"], plot_x + 18, threshold_y - 25, size=44, fill=GRAY, weight="700", anchor="start"))
    chunks.append(path_from_points(points, color=BLACK, width=7.5))

    cross_x, cross_y = points[-1]
    chunks.append(f'<circle cx="{cross_x:.1f}" cy="{cross_y:.1f}" r="15" fill="{BLACK}" stroke="{BLACK}" stroke-width="2"/>')

    chunks.append(text(["age t"], plot_x + plot_w / 2, axis_bottom + 82, size=66, anchor="middle"))
    chunks.append(
        f'<g transform="rotate(-90 {plot_x - 92:.1f} {plot_y + plot_h / 2:.1f})">'
        f'{text(["X(t)"], plot_x - 92, plot_y + plot_h / 2, size=70, anchor="middle")}'
        "</g>"
    )
    chunks.append(rect(112, 928, 1025, 230, fill="#F7F8FA", stroke="#C9CDD2", stroke_width=3, radius=8))
    chunks.append(
        math_text(
            r"$\frac{dX}{dt}=F(X,t;\vec{\theta}_{\rm sen})+\sqrt{2\epsilon}\,\xi(t)$",
            width / 2,
            1006,
            size=50,
        )
    )
    chunks.append(math_text(r"$\mathrm{death\ when}\ X(t)>X_c$", width / 2, 1104, size=66))
    chunks.append(svg_footer())
    ALT_A_SVG.write_text("\n".join(chunks), encoding="utf-8")
    landscape.convert_outputs(ALT_A_SVG, ALT_A_PDF, ALT_A_PNG)


def make_alt_panel_b_landscape() -> None:
    """Create the titled stability-landscape panel from the new potential well schematic."""
    landscape.build_svg(
        ALT_B_SVG,
        title="Dynamics within a flattening stability landscape",
        show_landscape_equation=True,
    )
    landscape.convert_outputs(ALT_B_SVG, ALT_B_PDF, ALT_B_PNG)


def ensure_parameter_panel_pdf() -> None:
    """Regenerate the parameter-classes SVG and convert it for the alt top row."""
    panels_ab.make_panel_b()
    landscape.convert_outputs(
        panels_ab.PANEL_B_SVG,
        OUTPUT_DIR / "fig1_panel_b_parameter_classes.pdf",
        OUTPUT_DIR / "fig1_panel_b_parameter_classes.png",
    )


def page_rect(pdf_path: Path) -> tuple[float, float]:
    page = PdfReader(str(pdf_path)).pages[0]
    return float(page.mediabox.width), float(page.mediabox.height)


def add_pdf_panel(page: PageObject, pdf_path: Path, rect_: fitz.Rect) -> None:
    source_page = PdfReader(str(pdf_path)).pages[0]
    source_width, source_height = page_rect(pdf_path)
    rect_width = rect_.x1 - rect_.x0
    rect_height = rect_.y1 - rect_.y0
    scale = min(rect_width / source_width, rect_height / source_height)
    placed_width = source_width * scale
    placed_height = source_height * scale
    x = rect_.x0 + (rect_width - placed_width) / 2
    y = PAGE_HEIGHT - rect_.y1 + (rect_height - placed_height) / 2
    transform = Transformation().scale(scale).translate(x, y)
    page.merge_transformed_page(source_page, transform)


def add_live_text_overlay(page: PageObject) -> None:
    """Add panel labels and top-row titles as editable embedded Arial text."""
    pdfmetrics.registerFont(TTFont("ArialLocal", ARIAL))

    buffer = BytesIO()
    overlay = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    overlay.setFillColorRGB(1, 1, 1)
    for whiteout in (
        fitz.Rect(30, 8, 548, 84),
        fitz.Rect(548, 0, 1110, 104),
        fitz.Rect(1160, 8, 2048, 98),
        fitz.Rect(10, 620, 88, 705),
        fitz.Rect(690, 620, 775, 705),
    ):
        overlay.rect(whiteout.x0, PAGE_HEIGHT - whiteout.y1, whiteout.x1 - whiteout.x0, whiteout.y1 - whiteout.y0, stroke=0, fill=1)

    overlay.setFillColorRGB(0, 0, 0)
    overlay.setFont("ArialLocal", 32)
    for line_text, x, y in [
        ("Stochastic threshold-crossing", 292, 34),
        ("view of aging", 292, 68),
        ("Dynamics within a flattening", 835, 34),
        ("stability landscape", 835, 68),
        ("Two classes of parameters", 1568, 52),
    ]:
        overlay.drawCentredString(x, PAGE_HEIGHT - y, line_text)

    overlay.setFont("ArialLocal", PANEL_LABEL_SIZE)
    for label, x, y in [
        ("a", 8, 68),
        ("b", 560, 68),
        ("c", 1118, 68),
        ("d", 8, 690),
        ("e", 688, 690),
        ("f", 1368, 690),
        ("g", 420, 1600),
    ]:
        overlay.drawString(x, PAGE_HEIGHT - y, label)
    overlay.save()
    buffer.seek(0)
    page.merge_page(PdfReader(buffer).pages[0])


def render_png(pdf_path: Path, png_path: Path) -> None:
    document = fitz.open(pdf_path)
    page = document[0]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    pixmap.save(png_path)


def make_composite() -> None:
    writer = PdfWriter()
    page = PageObject.create_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    add_pdf_panel(page, ALT_A_PDF, fitz.Rect(0, 0, 560, 590))
    add_pdf_panel(page, ALT_B_PDF, fitz.Rect(548, 0, 1110, 590))
    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_b_parameter_classes.pdf", fitz.Rect(1090, 0, 2067, 590))

    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_c_mortality_signatures.pdf", fitz.Rect(0, 600, 720, 1475))
    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_d_sr_model.pdf", fitz.Rect(690, 620, 1380, 1455))
    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_e_survival_scaling.pdf", fitz.Rect(1370, 620, 2060, 1455))
    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_f_steepness_longevity.pdf", fitz.Rect(470, 1535, 1598, 2648))

    add_live_text_overlay(page)
    writer.add_page(page)
    with OUT_PDF.open("wb") as handle:
        writer.write(handle)
    render_png(OUT_PDF, OUT_PNG)
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_PNG}")


def main() -> None:
    make_alt_panel_a_trajectory()
    if not ALT_B_PDF.exists():
        make_alt_panel_b_landscape()
    ensure_parameter_panel_pdf()
    make_composite()


if __name__ == "__main__":
    main()
