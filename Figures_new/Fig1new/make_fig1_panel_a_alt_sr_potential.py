#!/usr/bin/env python3
"""Build an alternate Fig. 1A SR-potential schematic.

This intentionally does not replace the existing Fig. 1A source. It creates
standalone alt-A outputs for comparison.
"""

from __future__ import annotations

import subprocess
from math import atan2, cos, pi, sin
from pathlib import Path
from shutil import which

import fitz

from make_fig1_panels_ab import (
    BLACK,
    BLUE_OLD,
    BLUE_YOUNG,
    FIGURES_DIR,
    GRAY,
    GRAY_LIGHT,
    OUTPUT_DIR,
    circle,
    line,
    math_text,
    path_from_points,
    png_image,
    svg_footer,
    svg_header,
    text,
    translucent_circle,
)


OUTPUT_STEM = "fig1_panel_a_alt_sr_potential"
SVG_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
PDF_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
PNG_PATH = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
LEGACY_PDF_PATH = FIGURES_DIR / "fig1b.pdf"

GREEN_HOMEOSTASIS = "#DCEFD9"
RED_THRESHOLD = "#FFD8D8"
LEGACY_LEFT = 63.0
LEGACY_TOP = 88.4056396484375
LEGACY_RIGHT = 621.0
LEGACY_BOTTOM = 642.8056030273438
LEGACY_THRESHOLD_X = 397.79998779296875
LEGACY_YOUNG_CENTER = (195.2383575439453, 363.8665466308594)
LEGACY_OLD_CENTER = (306.92572021484375, 460.2869567871094)
LEGACY_ARROW = (
    (382.61370849609375, 207.48019409179688),
    (458.4914855957031, 269.6542663574219),
    (475.3321533203125, 344.1938171386719),
    (433.13568115234375, 431.09881591796875),
)


def arrowhead(
    tip: tuple[float, float],
    control: tuple[float, float],
    *,
    length: float = 42,
    width: float = 38,
    fill: str = "#505A64",
    opacity: float = 0.92,
) -> str:
    angle = atan2(tip[1] - control[1], tip[0] - control[0])
    base_x = tip[0] - length * cos(angle)
    base_y = tip[1] - length * sin(angle)
    left_x = base_x + width * cos(angle + pi / 2) / 2
    left_y = base_y + width * sin(angle + pi / 2) / 2
    right_x = base_x + width * cos(angle - pi / 2) / 2
    right_y = base_y + width * sin(angle - pi / 2) / 2
    return (
        f'<path d="M {tip[0]:.1f},{tip[1]:.1f} L {left_x:.1f},{left_y:.1f} '
        f'L {right_x:.1f},{right_y:.1f} Z" fill="{fill}" opacity="{opacity}"/>'
    )


def close_color(found: tuple[float, float, float] | None, target: tuple[float, float, float]) -> bool:
    if found is None:
        return False
    return sum(abs(found[index] - target[index]) for index in range(3)) < 0.08


def legacy_curve_points(target_color: tuple[float, float, float]) -> list[tuple[float, float]]:
    """Extract the old Fig. 1B potential curve as a plain polyline."""
    page = fitz.open(LEGACY_PDF_PATH)[0]
    candidates = []
    for drawing in page.get_drawings():
        rect = drawing.get("rect")
        if rect is None or rect.width < 250 or rect.height < 250:
            continue
        if not close_color(drawing.get("color"), target_color):
            continue
        points: list[tuple[float, float]] = []
        for item in drawing["items"]:
            if item[0] == "l":
                if not points:
                    points.append((item[1].x, item[1].y))
                points.append((item[2].x, item[2].y))
        if points:
            candidates.append(points)
    if not candidates:
        raise RuntimeError(f"Could not extract legacy curve from {LEGACY_PDF_PATH}")
    return max(candidates, key=len)


def clip_to_legacy_panel(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Numerically clip the old curves to the panel bounds, avoiding SVG masks."""
    clipped: list[tuple[float, float]] = []
    previous = points[0]

    def inside(point: tuple[float, float]) -> bool:
        _, y = point
        return LEGACY_TOP <= y <= LEGACY_BOTTOM

    for current in points[1:]:
        previous_inside = inside(previous)
        current_inside = inside(current)
        if previous_inside and not clipped:
            clipped.append(previous)
        if previous_inside != current_inside:
            boundary = LEGACY_TOP if min(previous[1], current[1]) < LEGACY_TOP else LEGACY_BOTTOM
            fraction = (boundary - previous[1]) / (current[1] - previous[1])
            clipped.append((previous[0] + fraction * (current[0] - previous[0]), boundary))
        if current_inside:
            clipped.append(current)
        previous = current
    return clipped


def scale_legacy_point(
    point: tuple[float, float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    x, y = point
    return (
        left + (x - LEGACY_LEFT) / (LEGACY_RIGHT - LEGACY_LEFT) * width,
        top + (y - LEGACY_TOP) / (LEGACY_BOTTOM - LEGACY_TOP) * height,
    )


def scale_legacy_points(
    points: list[tuple[float, float]],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
) -> list[tuple[float, float]]:
    return [scale_legacy_point(point, left=left, top=top, width=width, height=height) for point in points]


def convert_outputs(
    svg_path: Path = SVG_PATH,
    pdf_path: Path = PDF_PATH,
    png_path: Path = PNG_PATH,
) -> None:
    """Convert a transient SVG source to PDF and PNG, then remove the SVG."""
    converter = which("rsvg-convert") or ("/opt/homebrew/bin/rsvg-convert" if Path("/opt/homebrew/bin/rsvg-convert").exists() else None)
    if converter is None:
        svg_path.unlink(missing_ok=True)
        print(f"Removed {svg_path}; rsvg-convert not found, skipped PDF/PNG conversion")
        return

    subprocess.run([converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)], check=True)
    subprocess.run([converter, "-f", "png", "-d", "500", "-p", "500", "-o", str(png_path), str(svg_path)], check=True)
    svg_path.unlink(missing_ok=True)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def build_svg(
    svg_path: Path = SVG_PATH,
    *,
    title: str | None = None,
    show_landscape_equation: bool = False,
) -> None:
    width, height = 1250, 1250
    top = 150 if title else 70
    left, plot_w, plot_h = 100, 1035, 1100 - top
    panel_right = width - 42
    origin_y = top + plot_h
    threshold_px = left + (LEGACY_THRESHOLD_X - LEGACY_LEFT) / (LEGACY_RIGHT - LEGACY_LEFT) * plot_w
    if show_landscape_equation:
        threshold_px = min(threshold_px + 95, panel_right - 275)

    young_points = scale_legacy_points(
        clip_to_legacy_panel(legacy_curve_points((0.529411792755127, 0.8078431487083435, 0.9803921580314636))),
        left=left,
        top=top,
        width=plot_w,
        height=plot_h,
    )
    old_points = scale_legacy_points(
        clip_to_legacy_panel(legacy_curve_points((0.0, 0.0, 1.0))),
        left=left,
        top=top,
        width=plot_w,
        height=plot_h,
    )

    chunks = svg_header(width, height)
    if title:
        chunks.append(text([title], width / 2, 75, size=54, fill=BLACK, weight="400", anchor="middle"))
    chunks.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{threshold_px - left:.1f}" height="{plot_h:.1f}" fill="{GREEN_HOMEOSTASIS}" opacity="0.88"/>')
    chunks.append(f'<rect x="{threshold_px:.1f}" y="{top:.1f}" width="{panel_right - threshold_px:.1f}" height="{plot_h:.1f}" fill="{RED_THRESHOLD}" opacity="0.86"/>')

    origin_x = left
    chunks.append(line(origin_x, origin_y, panel_right, origin_y, width=3.2))
    chunks.append(line(origin_x, origin_y, origin_x, top, width=3.2))
    chunks.append(line(threshold_px, top, threshold_px, origin_y, color=GRAY, width=3.0, dash="14 12", opacity=0.72))

    chunks.append(path_from_points(young_points, color=BLUE_YOUNG, width=8.0, opacity=0.98))
    chunks.append(path_from_points(old_points, color=BLUE_OLD, width=8.4, opacity=0.98))

    young_min = scale_legacy_point(LEGACY_YOUNG_CENTER, left=left, top=top, width=plot_w, height=plot_h)
    old_min = scale_legacy_point(LEGACY_OLD_CENTER, left=left, top=top, width=plot_w, height=plot_h)

    for dx, dy in [(-70, 0), (-34, 12), (34, 7), (72, -2)]:
        chunks.append(translucent_circle(young_min[0] + dx, young_min[1] + dy, 28, fill="#F0A083", stroke=GRAY_LIGHT, width=4, opacity=0.30))
    for dx, dy in [(-114, -8), (-62, 13), (0, -2), (62, 12), (115, -7)]:
        chunks.append(translucent_circle(old_min[0] + dx, old_min[1] + dy, 31, fill="#F0A083", stroke=GRAY_LIGHT, width=4, opacity=0.30))
    chunks.append(circle(young_min[0], young_min[1], 29, fill="#E76F51", stroke=BLACK, width=4))
    chunks.append(circle(old_min[0], old_min[1], 32, fill="#E31A1C", stroke=BLACK, width=4))

    young_label_y = young_min[1] - 285
    young_icon_y = young_min[1] - 248
    old_label_y = old_min[1] - 300
    old_icon_y = old_min[1] - 262
    chunks.append(text(["Young"], young_min[0], young_label_y, size=58, fill=BLUE_YOUNG, weight="700", anchor="middle"))
    chunks.append(png_image(FIGURES_DIR / "young.png", young_min[0] - 47, young_icon_y, 94, 126))
    chunks.append(text(["Old"], old_min[0], old_label_y, size=58, fill=BLUE_OLD, weight="700", anchor="middle"))
    chunks.append(png_image(FIGURES_DIR / "old.png", old_min[0] - 47, old_icon_y, 94, 126))

    arrow_start, arrow_ctrl_1, arrow_ctrl_2, arrow_end = [
        scale_legacy_point(point, left=left, top=top, width=plot_w, height=plot_h)
        for point in LEGACY_ARROW
    ]
    arrow_path = (
        f"M {arrow_start[0]:.1f},{arrow_start[1]:.1f} "
        f"C {arrow_ctrl_1[0]:.1f},{arrow_ctrl_1[1]:.1f} "
        f"{arrow_ctrl_2[0]:.1f},{arrow_ctrl_2[1]:.1f} "
        f"{arrow_end[0]:.1f},{arrow_end[1]:.1f}"
    )
    chunks.append(
        f'<path d="{arrow_path}" stroke="url(#time-arrow-gradient)" stroke-width="12.5" '
        'fill="none" stroke-linecap="round" opacity="0.88"/>'
    )
    chunks.append(arrowhead(arrow_end, arrow_ctrl_2))
    chunks.append(png_image(OUTPUT_DIR / "clock.png", threshold_px + 30, arrow_start[1] - 70, 56, 56, opacity=1.0))

    chunks.append(text(["failure threshold"], panel_right - 30, origin_y - 28, size=46, fill=BLACK, weight="700", anchor="end"))
    if show_landscape_equation:
        chunks.append(math_text(r"$U(X,t)=\int F(X,t)\,dX$", left + 52, origin_y - 50, size=62, fill=BLACK, anchor="start"))
    chunks.append(text(["Damage X"], left + plot_w / 2, origin_y + 78, size=72, fill=BLACK, anchor="middle"))
    chunks.append(
        f'<g transform="rotate(-90 {left - 45:.1f} {top + plot_h / 2:.1f})">'
        f'{text(["Effective Potential U(X,t)"], left - 45, top + plot_h / 2, size=64, fill=BLACK, anchor="middle")}'
        "</g>"
    )

    chunks.append(svg_footer())
    svg_path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    build_svg()
    convert_outputs()


if __name__ == "__main__":
    main()
