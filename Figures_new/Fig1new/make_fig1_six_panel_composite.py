#!/usr/bin/env python3
"""Compose revised six-panel Fig. 1 from source panel PDFs.

The panel drawings are generated separately. This script only controls the
manuscript-page layout, so the composite can be rebuilt without Illustrator
copy-paste placement.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


OUTPUT_DIR = Path(__file__).resolve().parent
OUT_PDF = OUTPUT_DIR / "Fig1_new.pdf"
OUT_PNG = OUTPUT_DIR / "Fig1_new.png"

PAGE_WIDTH = 2067.0
PAGE_HEIGHT = 2677.0
PANEL_LABEL_SIZE = 70
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"


def page_rect(pdf_path: Path) -> tuple[float, float]:
    """Return the media-box width and height of the first page."""
    page = PdfReader(str(pdf_path)).pages[0]
    return float(page.mediabox.width), float(page.mediabox.height)


def add_pdf_panel(page: PageObject, pdf_path: Path, rect: fitz.Rect) -> None:
    """Merge a single-page PDF panel into a target rectangle."""
    source_page = PdfReader(str(pdf_path)).pages[0]
    source_width, source_height = page_rect(pdf_path)
    rect_width = rect.x1 - rect.x0
    rect_height = rect.y1 - rect.y0
    scale = min(rect_width / source_width, rect_height / source_height)
    placed_width = source_width * scale
    placed_height = source_height * scale
    x = rect.x0 + (rect_width - placed_width) / 2
    y = PAGE_HEIGHT - rect.y1 + (rect_height - placed_height) / 2
    transform = Transformation().scale(scale).translate(x, y)
    page.merge_transformed_page(source_page, transform)


def add_live_text_overlay(page: PageObject) -> None:
    """Add whiteouts and panel labels as editable embedded Arial text."""
    pdfmetrics.registerFont(TTFont("ArialLocal", ARIAL))

    buffer = BytesIO()
    overlay = canvas.Canvas(buffer, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    overlay.setFillColorRGB(1, 1, 1)
    for rect in (fitz.Rect(10, 620, 88, 705), fitz.Rect(690, 620, 775, 705)):
        overlay.rect(rect.x0, PAGE_HEIGHT - rect.y1, rect.x1 - rect.x0, rect.y1 - rect.y0, stroke=0, fill=1)

    overlay.setFillColorRGB(0, 0, 0)
    overlay.setFont("ArialLocal", PANEL_LABEL_SIZE)
    for label, x, y in [
        ("a", 8, 68),
        ("b", 1040, 68),
        ("c", 8, 690),
        ("d", 688, 690),
        ("e", 1368, 690),
        ("f", 420, 1600),
    ]:
        overlay.drawString(x, PAGE_HEIGHT - y, label)
    overlay.save()
    buffer.seek(0)
    overlay_page = PdfReader(buffer).pages[0]
    page.merge_page(overlay_page)


def render_png(pdf_path: Path, png_path: Path) -> None:
    """Render a high-resolution PNG preview of the composed PDF."""
    document = fitz.open(pdf_path)
    page = document[0]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
    pixmap.save(png_path)


def main() -> None:
    writer = PdfWriter()
    page = PageObject.create_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)

    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_a_stochastic_threshold.pdf", fitz.Rect(0, 0, 1038, 590))
    add_pdf_panel(page, OUTPUT_DIR / "fig1_panel_b_parameter_classes.pdf", fitz.Rect(1030, 0, 2067, 590))

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


if __name__ == "__main__":
    main()
