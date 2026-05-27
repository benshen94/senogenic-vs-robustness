#!/usr/bin/env python3
"""Build editable SVG source panels for the revised Fig. 1 A/B concepts.

The SVGs are intentionally simple: curves, paths, circles, rectangles, and text.
They can be opened in Illustrator and saved as editable AI/PDF artwork.
"""

from __future__ import annotations

import base64
from html import escape
from math import atan2, cos, exp, pi, sin
from pathlib import Path
import random


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = OUTPUT_DIR.parents[1] / "Figures"
PANEL_A_SVG = OUTPUT_DIR / "fig1_panel_a_stochastic_threshold.svg"
PANEL_B_SVG = OUTPUT_DIR / "fig1_panel_b_parameter_classes.svg"

BLUE = "#2166AC"
BLUE_DARK = "#174A7A"
BLUE_LIGHT = "#EAF4FF"
BLUE_YOUNG = "#79B7E5"
BLUE_OLD = "#155A9C"
TEAL = "#2A9D8F"
TEAL_LIGHT = "#B8E1DC"
PURPLE = "#7B3294"
PURPLE_LIGHT = "#F7ECFA"
ORANGE = "#E76F51"
ORANGE_LIGHT = "#F7A38B"
ORANGE_DARK = "#C74A22"
ORANGE_PALE = "#FFF1E9"
RED_DANGER = "#D84A3A"
TRAJ_BLUE = "#5B82AE"
TRAJ_ORANGE = "#D3915C"
BLACK = "#111111"
GRAY = "#59616B"
GRAY_LIGHT = "#A7ADB4"


def svg_header(width: int, height: int) -> list[str]:
    """Return SVG preamble and shared definitions."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">',
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{BLACK}"/>',
        "</marker>",
        '<marker id="arrow-gray" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">',
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{GRAY}"/>',
        "</marker>",
        '<marker id="arrow-blue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">',
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{BLUE}"/>',
        "</marker>",
        '<linearGradient id="time-arrow-gradient" x1="0%" y1="0%" x2="100%" y2="0%">',
        '<stop offset="0%" stop-color="#EEF2F5"/>',
        '<stop offset="55%" stop-color="#AEB7BF"/>',
        '<stop offset="100%" stop-color="#68717A"/>',
        "</linearGradient>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
    ]


def svg_footer() -> str:
    return "</svg>"


def text(
    lines: list[str],
    x: float,
    y: float,
    *,
    size: float = 30,
    fill: str = BLACK,
    weight: str = "400",
    anchor: str = "start",
    line_height: float = 1.2,
    italic: bool = False,
    raw: bool = False,
) -> str:
    """Create SVG text with optional multiple lines."""
    style = f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{fill}" font-weight="{weight}"'
    if italic:
        style += ' font-style="italic"'
    attrs = f'{style} text-anchor="{anchor}"'
    chunks = [f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>']
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size * line_height
        content = line if raw else escape(line)
        chunks.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{content}</tspan>')
    chunks.append("</text>")
    return "".join(chunks)


def halo_text(
    lines: list[str],
    x: float,
    y: float,
    *,
    size: float,
    fill: str,
    halo: str = "#FFFFFF",
    halo_width: float = 11,
    weight: str = "700",
    anchor: str = "middle",
) -> str:
    """Draw readable text with a light outline but no filled background box."""
    base_attrs = (
        f'x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}"'
    )
    content = "".join(f'<tspan x="{x:.1f}" dy="{0 if index == 0 else size * 1.2:.1f}">{escape(line)}</tspan>' for index, line in enumerate(lines))
    return (
        f'<text {base_attrs} fill="{halo}" stroke="{halo}" stroke-width="{halo_width}" '
        f'stroke-linejoin="round" opacity="0.88">{content}</text>'
        f'<text {base_attrs} fill="{fill}" stroke="none">{content}</text>'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = BLACK,
    width: float = 3,
    dash: str | None = None,
    arrow: str | None = None,
    opacity: float = 1,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    if not arrow:
        return (
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
            f'opacity="{opacity}" fill="none"{dash_attr}/>'
        )

    angle = atan2(y2 - y1, x2 - x1)
    head_length = max(12, width * 3.3)
    head_width = max(8, width * 2.4)
    base_x = x2 - head_length * cos(angle)
    base_y = y2 - head_length * sin(angle)
    left_x = base_x + head_width * cos(angle + pi / 2) / 2
    left_y = base_y + head_width * sin(angle + pi / 2) / 2
    right_x = base_x + head_width * cos(angle - pi / 2) / 2
    right_y = base_y + head_width * sin(angle - pi / 2) / 2
    line_end_x = x2 - head_length * 0.72 * cos(angle)
    line_end_y = y2 - head_length * 0.72 * sin(angle)
    return (
        "<g>"
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{line_end_x:.1f}" y2="{line_end_y:.1f}" '
        f'stroke="{color}" stroke-width="{width}" stroke-linecap="round" '
        f'opacity="{opacity}" fill="none"{dash_attr}/>'
        f'<path d="M {x2:.1f},{y2:.1f} L {left_x:.1f},{left_y:.1f} L {right_x:.1f},{right_y:.1f} Z" '
        f'fill="{color}" opacity="{opacity}"/>'
        "</g>"
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    stroke_width: float = 1,
    radius: float = 8,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def circle(x: float, y: float, radius: float, *, fill: str, stroke: str = BLACK, width: float = 3) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def skull_icon(x: float, y: float, *, color: str, scale: float = 1.0) -> str:
    """Draw a tiny vector skull marker in one trajectory color."""
    head_r = 13 * scale
    jaw_w = 18 * scale
    jaw_h = 10 * scale
    eye_r = 2.6 * scale
    nose_w = 4.2 * scale
    nose_h = 4.8 * scale
    tooth_y = y + 15 * scale
    return (
        "<g>"
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{head_r:.1f}" fill="{color}" stroke="{color}" stroke-width="{1.2 * scale:.1f}"/>'
        f'<rect x="{x - jaw_w / 2:.1f}" y="{y + 6 * scale:.1f}" width="{jaw_w:.1f}" height="{jaw_h:.1f}" '
        f'rx="{2.5 * scale:.1f}" fill="{color}" stroke="{color}" stroke-width="{1.2 * scale:.1f}"/>'
        f'<circle cx="{x - 5 * scale:.1f}" cy="{y - 2 * scale:.1f}" r="{eye_r:.1f}" fill="white"/>'
        f'<circle cx="{x + 5 * scale:.1f}" cy="{y - 2 * scale:.1f}" r="{eye_r:.1f}" fill="white"/>'
        f'<path d="M {x:.1f},{y + 1.5 * scale:.1f} L {x - nose_w / 2:.1f},{y + 7 * scale:.1f} '
        f'L {x + nose_w / 2:.1f},{y + 7 * scale:.1f} Z" fill="white"/>'
        f'<line x1="{x - 5.5 * scale:.1f}" y1="{tooth_y:.1f}" x2="{x - 5.5 * scale:.1f}" y2="{tooth_y + 5 * scale:.1f}" stroke="white" stroke-width="{1.3 * scale:.1f}"/>'
        f'<line x1="{x:.1f}" y1="{tooth_y:.1f}" x2="{x:.1f}" y2="{tooth_y + 5 * scale:.1f}" stroke="white" stroke-width="{1.3 * scale:.1f}"/>'
        f'<line x1="{x + 5.5 * scale:.1f}" y1="{tooth_y:.1f}" x2="{x + 5.5 * scale:.1f}" y2="{tooth_y + 5 * scale:.1f}" stroke="white" stroke-width="{1.3 * scale:.1f}"/>'
        "</g>"
    )


def translucent_circle(
    x: float,
    y: float,
    radius: float,
    *,
    fill: str,
    stroke: str,
    width: float = 3,
    opacity: float = 0.28,
) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
    )


def embedded_png(path: Path) -> str:
    """Return an embedded PNG data URL to avoid Illustrator link prompts."""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def png_image(path: Path, x: float, y: float, width: float, height: float, *, opacity: float = 1) -> str:
    """Embed a PNG image inside the SVG."""
    return (
        f'<image x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" '
        f'href="{embedded_png(path)}" opacity="{opacity}"/>'
    )


def path_from_points(
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float = 5,
    fill: str = "none",
    dash: str | None = None,
    opacity: float = 1,
) -> str:
    if not points:
        return ""
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="{fill}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_attr}/>'
    )


def filled_path(d: str, *, fill: str, opacity: float = 1, stroke: str = "none") -> str:
    return f'<path d="{d}" fill="{fill}" opacity="{opacity}" stroke="{stroke}"/>'


def mpl_path_to_svg_d(path) -> str:
    """Convert a matplotlib path into a compact SVG path string."""
    from matplotlib.path import Path as MplPath

    vertices = path.vertices
    codes = path.codes
    if codes is None:
        return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in vertices)

    commands: list[str] = []
    index = 0
    while index < len(vertices):
        code = codes[index]
        x, y = vertices[index]
        if code == MplPath.MOVETO:
            commands.append(f"M {x:.2f},{y:.2f}")
            index += 1
        elif code == MplPath.LINETO:
            commands.append(f"L {x:.2f},{y:.2f}")
            index += 1
        elif code == MplPath.CURVE3 and index + 1 < len(vertices):
            x1, y1 = vertices[index]
            x2, y2 = vertices[index + 1]
            commands.append(f"Q {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f}")
            index += 2
        elif code == MplPath.CURVE4 and index + 2 < len(vertices):
            x1, y1 = vertices[index]
            x2, y2 = vertices[index + 1]
            x3, y3 = vertices[index + 2]
            commands.append(f"C {x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}")
            index += 3
        elif code == MplPath.CLOSEPOLY:
            commands.append("Z")
            index += 1
        else:
            index += 1
    return " ".join(commands)


def math_text(
    latex: str,
    x: float,
    y: float,
    *,
    size: float,
    fill: str = BLACK,
    anchor: str = "middle",
) -> str:
    """Render a LaTeX-style math string as editable vector paths."""
    from matplotlib.textpath import TextPath

    path = TextPath((0, 0), latex, size=size)
    bbox = path.get_extents()
    if anchor == "middle":
        tx = x - (bbox.x0 + bbox.x1) / 2
    elif anchor == "end":
        tx = x - bbox.x1
    else:
        tx = x - bbox.x0
    ty = y + (bbox.y0 + bbox.y1) / 2
    return (
        f'<g transform="translate({tx:.2f} {ty:.2f}) scale(1 -1)">'
        f'<path d="{mpl_path_to_svg_d(path)}" fill="{fill}"/></g>'
    )


def smooth_path(
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float = 6,
    dash: str | None = None,
    opacity: float = 1,
) -> str:
    """Return a smoothed path using quadratic segments through midpoints."""
    if len(points) < 3:
        return path_from_points(points, color=color, width=width, dash=dash)

    d = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
    for index in range(1, len(points) - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        d += f" Q {x0:.1f},{y0:.1f} {mid_x:.1f},{mid_y:.1f}"
    d += f" T {points[-1][0]:.1f},{points[-1][1]:.1f}"
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_attr}/>'
    )


def potential_value(
    x: float,
    *,
    well_x: float,
    barrier_x: float,
    depth: float,
    barrier: float,
    well_width: float,
    barrier_width: float,
    tilt: float = 0.0,
) -> float:
    """Schematic one-basin potential."""
    basin = -depth * exp(-0.5 * ((x - well_x) / well_width) ** 2)
    peak = barrier * exp(-0.5 * ((x - barrier_x) / barrier_width) ** 2)
    confining = 0.07 * (x - well_x) ** 2
    return basin + peak + confining + tilt * (x - well_x)


def asymmetric_old_potential_value(x: float) -> float:
    """Old landscape: steep left wall, shallow approach to threshold."""
    minimum_x = 8.55
    threshold_x = 10.20
    minimum = 0.34
    if x < minimum_x:
        distance = minimum_x - x
        return minimum + 0.22 * distance**2 + 0.004 * distance**3

    distance = x - minimum_x
    shallow_slope = 0.030 * distance + 0.034 * distance**2
    threshold_hump = 0.70 * exp(-0.5 * ((x - threshold_x) / 0.52) ** 2)
    return minimum + shallow_slope + threshold_hump


def asymmetric_mini_old_value(x: float, *, well_x: float, barrier_x: float, depth: float, barrier: float) -> float:
    """Small-panel version of the asymmetric old landscape."""
    minimum = -depth + 0.36
    if x < well_x:
        distance = well_x - x
        return minimum + 0.48 * distance**2 + 0.04 * distance**3

    distance = x - well_x
    shallow_slope = 0.06 * distance + 0.23 * distance**2
    threshold_hump = barrier * 0.58 * exp(-0.5 * ((x - barrier_x) / 0.13) ** 2)
    return minimum + shallow_slope + threshold_hump


def map_points(
    xs: list[float],
    values: list[float],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    """Map data coordinates into SVG coordinates."""
    mapped = []
    for x, y in zip(xs, values):
        px = left + (x - x_min) / (x_max - x_min) * width
        py = top + height - (y - y_min) / (y_max - y_min) * height
        mapped.append((px, py))
    return mapped


def sample_curve(start: float, end: float, n: int) -> list[float]:
    step = (end - start) / (n - 1)
    return [start + index * step for index in range(n)]


def wiggle_points(cx: float, cy: float, width: float, amp: float, n: int = 80) -> list[tuple[float, float]]:
    points = []
    for index in range(n):
        phase = -pi + 2 * pi * index / (n - 1)
        x = cx + width * phase / pi
        y = cy + amp * sin(3 * phase)
        points.append((x, y))
    return points


def draw_main_landscape() -> list[str]:
    """Draw panel A's main landscape cartoon."""
    left, top, width, height = 110, 240, 1160, 650
    x_min, x_max, y_min, y_max = 0, 12, -2.2, 4.1
    chunks = []
    young_params = {
        "well_x": 1.75,
        "barrier_x": 4.15,
        "depth": 2.25,
        "barrier": 1.18,
        "well_width": 0.78,
        "barrier_width": 0.58,
        "tilt": 0.01,
    }
    old_params = {
        "well_x": 8.80,
        "barrier_x": 10.05,
        "depth": 0.70,
        "barrier": 0.95,
        "well_width": 1.35,
        "barrier_width": 0.58,
        "tilt": 0.0,
    }

    origin_x = left + 35
    origin_y = top + height - 28
    chunks.append(line(origin_x, origin_y, left + width - 15, origin_y, width=4, arrow="arrow"))
    chunks.append(line(origin_x, origin_y, origin_x, top + 20, width=4, arrow="arrow"))
    chunks.append(
        f'<g transform="rotate(-90 {origin_x - 78:.1f} {top + height / 2:.1f})">'
        f'{text(["Effective potential V(X)"], origin_x - 78, top + height / 2, size=52, anchor="middle")}'
        "</g>"
    )
    chunks.append(text(["X Damage / State variable"], left + width / 2, origin_y + 76, size=50, anchor="middle"))

    young_x = sample_curve(0.45, 5.02, 140)
    young_y = [potential_value(x, **young_params) + 0.62 for x in young_x]
    old_x = sample_curve(7.00, 11.05, 140)
    old_y_raw = [asymmetric_old_potential_value(x) - 0.02 for x in old_x]
    young_barrier = 4.15
    young_barrier_y = potential_value(young_barrier, **young_params) + 0.62
    old_peak_indices = [index for index, x_value in enumerate(old_x) if x_value > 9.45]
    old_peak_index = max(old_peak_indices, key=lambda index: old_y_raw[index])
    old_y = [value + (young_barrier_y - old_y_raw[old_peak_index]) for value in old_y_raw]
    old_reference_shift = 6.42
    old_reference_x = [x + old_reference_shift for x in young_x]
    old_reference_y = young_y[:]
    young_points = map_points(
        young_x,
        young_y,
        left=left,
        top=top,
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    old_points = map_points(
        old_x,
        old_y,
        left=left,
        top=top,
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    old_reference_points = map_points(
        old_reference_x,
        old_reference_y,
        left=left,
        top=top,
        width=width,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    )
    old_barrier = old_x[old_peak_index]
    young_danger = [point for point, x in zip(young_points, young_x) if x >= young_barrier]
    old_danger = [point for point, x in zip(old_points, old_x) if x >= old_barrier]
    if young_danger:
        start_x, _ = young_danger[0]
        end_x, _ = young_danger[-1]
        danger_d = (
            f"M {start_x:.1f},{origin_y:.1f} "
            + " L ".join(f"{x:.1f},{y:.1f}" for x, y in young_danger)
            + f" L {end_x:.1f},{origin_y:.1f} Z"
        )
        chunks.append(filled_path(danger_d, fill=RED_DANGER, opacity=0.12))
    if old_danger:
        start_x, _ = old_danger[0]
        end_x, _ = old_danger[-1]
        danger_d = (
            f"M {start_x:.1f},{origin_y:.1f} "
            + " L ".join(f"{x:.1f},{y:.1f}" for x, y in old_danger)
            + f" L {end_x:.1f},{origin_y:.1f} Z"
        )
        chunks.append(filled_path(danger_d, fill=RED_DANGER, opacity=0.12))

    chunks.append(smooth_path(young_points, color=BLUE_YOUNG, width=8))
    chunks.append(smooth_path(old_reference_points, color=BLUE_YOUNG, width=7, opacity=0.12))
    chunks.append(smooth_path(old_points, color=BLUE_OLD, width=8))

    def mapped_xy(x: float, y: float) -> tuple[float, float]:
        return map_points(
            [x],
            [y],
            left=left,
            top=top,
            width=width,
            height=height,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )[0]

    old_barrier_y = old_y[old_peak_index]
    yb_x, yb_y = mapped_xy(young_barrier, young_barrier_y)
    ob_x, ob_y = mapped_xy(old_barrier, old_barrier_y)
    chunks.append(line(yb_x, origin_y, yb_x, yb_y + 4, color=GRAY_LIGHT, width=3.5, dash="12 10"))
    chunks.append(line(ob_x, origin_y, ob_x, ob_y + 4, color=GRAY_LIGHT, width=3.5, dash="12 10"))

    old_y_shift = young_barrier_y - old_y_raw[old_peak_index]
    young_ball = mapped_xy(1.7, potential_value(1.7, **young_params) + 0.82)
    old_ball = mapped_xy(8.70, asymmetric_old_potential_value(8.70) - 0.02 + old_y_shift + 0.18)

    for x_value, y_offset in [(1.42, 0.93), (1.68, 0.84), (1.95, 0.93)]:
        x_pos, y_pos = mapped_xy(x_value, potential_value(x_value, **young_params) + y_offset)
        chunks.append(translucent_circle(x_pos, y_pos, 21, fill=ORANGE, stroke=GRAY, width=3, opacity=0.24))
    for x_value, y_offset in [(8.12, 0.26), (8.45, 0.19), (9.08, 0.23), (9.40, 0.31)]:
        x_pos, y_pos = mapped_xy(x_value, asymmetric_old_potential_value(x_value) - 0.02 + old_y_shift + y_offset)
        chunks.append(translucent_circle(x_pos, y_pos, 21, fill=ORANGE, stroke=GRAY, width=3, opacity=0.24))

    young_noise_y = young_ball[1] - 58
    old_noise_y = old_ball[1] - 78
    chunks.append(line(young_ball[0], young_noise_y, young_ball[0] - 42, young_noise_y, color=GRAY, width=2.2, arrow="arrow-gray"))
    chunks.append(line(young_ball[0], young_noise_y, young_ball[0] + 42, young_noise_y, color=GRAY, width=2.2, arrow="arrow-gray"))
    chunks.append(text(["noise"], young_ball[0], young_noise_y - 15, size=24, fill=GRAY, weight="700", anchor="middle"))
    chunks.append(line(old_ball[0], old_noise_y, old_ball[0] - 66, old_noise_y, color=GRAY, width=2.2, arrow="arrow-gray"))
    chunks.append(line(old_ball[0], old_noise_y, old_ball[0] + 66, old_noise_y, color=GRAY, width=2.2, arrow="arrow-gray"))
    chunks.append(text(["noise"], old_ball[0], old_noise_y - 15, size=24, fill=GRAY, weight="700", anchor="middle"))
    chunks.append(circle(young_ball[0], young_ball[1], 23, fill=ORANGE, width=4))
    chunks.append(circle(old_ball[0], old_ball[1], 23, fill=ORANGE, width=4))

    chunks.extend(time_arrow(592, 548, width=125, height=78))
    chunks.append(png_image(FIGURES_DIR / "young.png", young_ball[0] - 48, 155, 96, 128))
    chunks.append(png_image(FIGURES_DIR / "old.png", old_ball[0] - 48, 151, 96, 128))
    chunks.append(png_image(OUTPUT_DIR / "clock.png", 617, 456, 74, 74))
    chunks.append(text(["Young"], young_ball[0], 145, size=56, fill=BLUE_YOUNG, weight="700", anchor="middle"))
    chunks.append(text(["Old"], old_ball[0], 140, size=56, fill=BLUE_OLD, weight="700", anchor="middle"))
    chunks.append(text(["death", "threshold"], yb_x + 68, yb_y - 36, size=28, anchor="start"))
    chunks.append(line(yb_x, yb_y - 38, yb_x, yb_y + 5, color=BLACK, width=2.8, arrow="arrow"))
    return chunks


def generate_wild_trajectory(seed: int, steps: int, crossing_index: int) -> list[float]:
    """Create a spiky stochastic path that crosses the threshold at a chosen time."""
    rng = random.Random(seed)
    threshold = 0.74
    values = []
    spike_carry = 0.0
    for index in range(steps):
        age_pressure = 0.13 + 0.006 * index
        spike_carry *= 0.28
        if rng.random() < 0.12 + 0.0025 * index:
            spike_carry += rng.uniform(0.12, 0.44)

        value = 0.05 + rng.random() * age_pressure + rng.gauss(0, 0.045) + spike_carry
        values.append(max(0.03, min(value, threshold - 0.075)))

    for index in range(crossing_index):
        values[index] = min(values[index], threshold - 0.08 - 0.01 * rng.random())
    values[crossing_index - 1] = min(values[crossing_index - 1], threshold - rng.uniform(0.18, 0.33))
    values[crossing_index] = threshold
    return values


def draw_trajectory_inset() -> list[str]:
    """Draw the wild stochastic-trajectory inset for panel A."""
    x, y, width, height = 1310, 285, 385, 435
    threshold_y = y + 145
    chunks = [
        text(["Example trajectories"], x + width / 2, y - 34, size=38, fill=GRAY, anchor="middle", weight="700"),
        line(x + 42, y + height - 55, x + width - 18, y + height - 55, width=3.2, arrow="arrow"),
        line(x + 42, y + height - 55, x + 42, y + 35, width=3.2, arrow="arrow"),
        line(x + 42, threshold_y, x + width - 18, threshold_y, color="#777F87", width=3.0, dash="11 9"),
        text(["threshold"], x + width / 2, threshold_y - 18, size=24, anchor="middle", weight="700", fill=GRAY),
        text(["age t"], x + width / 2, y + height - 2, size=27, anchor="middle"),
    ]

    trajectories = [
        (14, 34, TRAJ_BLUE, "circle"),
        (31, 52, TRAJ_ORANGE, "diamond"),
    ]
    steps = 58
    for seed, cross_index, color, marker in trajectories:
        values = generate_wild_trajectory(seed, steps, cross_index)
        points = []
        for index in range(cross_index + 1):
            px = x + 42 + index / (steps - 1) * (width - 70)
            py = y + height - 55 - values[index] * (height - 105)
            points.append((px, py))
        chunks.append(path_from_points(points, color=color, width=4.2))
        cross_x, cross_y = points[-1][0], threshold_y
        chunks.append(skull_icon(cross_x - 22, cross_y + 26, color=color, scale=1.05))
        if marker == "square":
            chunks.append(rect(cross_x - 10, cross_y - 10, 20, 20, fill=color, radius=0))
        elif marker == "triangle":
            chunks.append(
                f'<path d="M {cross_x:.1f},{cross_y - 14:.1f} L {cross_x + 13:.1f},{cross_y + 11:.1f} '
                f'L {cross_x - 13:.1f},{cross_y + 11:.1f} Z" fill="{color}"/>'
            )
        elif marker == "diamond":
            chunks.append(
                f'<path d="M {cross_x:.1f},{cross_y - 13:.1f} L {cross_x + 13:.1f},{cross_y:.1f} '
                f'L {cross_x:.1f},{cross_y + 13:.1f} L {cross_x - 13:.1f},{cross_y:.1f} Z" fill="{color}"/>'
            )
        else:
            chunks.append(circle(cross_x, cross_y, 10, fill=color, stroke=color, width=1))

    chunks.append(
        f'<g transform="rotate(-90 {x + 8:.1f} {y + height / 2:.1f})">'
        f'{text(["X(t)"], x + 8, y + height / 2, size=27, anchor="middle")}'
        "</g>"
    )
    return chunks


def draw_equation_box() -> list[str]:
    """Small model definition for panel A."""
    x, y, width, height = 1248, 775, 535, 140
    chunks = [
        rect(x, y, width, height, fill="#F7F8FA", stroke="#C9CDD2", stroke_width=2, radius=6),
        math_text(
            r"$\frac{dX}{dt}=F(X,t;\theta_{\rm seno})+\sqrt{2\epsilon}\,\xi(t)$",
            x + width / 2,
            y + 50,
            size=28,
        ),
        math_text(r"$\mathrm{death\ when}\ X>X_c$", x + width / 2, y + 102, size=28),
    ]
    return chunks


def make_panel_a() -> None:
    width, height = 1800, 1000
    chunks = svg_header(width, height)
    chunks.append(text(["Stochastic threshold-crossing view of aging"], 150, 96, size=70, weight="700"))
    chunks.extend(draw_main_landscape())
    chunks.extend(draw_trajectory_inset())
    chunks.extend(draw_equation_box())
    chunks.append(svg_footer())
    PANEL_A_SVG.write_text("\n".join(chunks), encoding="utf-8")


def mini_map(
    x: float,
    value: float,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    return left + x * width, top + height - (value + 0.65) / 1.45 * height


def mini_well(
    left: float,
    top: float,
    *,
    width: float = 250,
    height: float = 170,
    color: str = BLUE,
    depth: float = 0.58,
    barrier: float = 0.62,
    well_x: float = 0.28,
    barrier_x: float = 0.67,
    well_width: float = 0.16,
    barrier_width: float = 0.14,
    dashed: bool = False,
    lower_noise: bool = False,
    higher_threshold: bool = False,
    overlay_old: bool = False,
    overlay_young_reference: bool = False,
    asymmetric_old: bool = False,
) -> list[str]:
    xs = sample_curve(0.02, 0.98, 90)
    current_barrier_x = 0.80 if higher_threshold else barrier_x
    chunks = []
    if higher_threshold:
        previous_values = [
            potential_value(
                x,
                well_x=well_x,
                barrier_x=barrier_x,
                depth=depth,
                barrier=barrier * 0.82,
                well_width=well_width,
                barrier_width=barrier_width,
                tilt=0.02,
            )
            for x in xs
        ]
        previous_points = [
            mini_map(x, value, left=left, top=top, width=width, height=height)
            for x, value in zip(xs, previous_values)
        ]
        chunks.append(smooth_path(previous_points, color=ORANGE_LIGHT, width=4, dash="9 8"))

    values = [
        asymmetric_mini_old_value(
            x,
            well_x=well_x,
            barrier_x=current_barrier_x,
            depth=depth,
            barrier=barrier * 1.08 if higher_threshold else barrier,
        )
        if asymmetric_old
        else potential_value(
            x,
            well_x=well_x,
            barrier_x=current_barrier_x,
            depth=depth,
            barrier=barrier * 1.08 if higher_threshold else barrier,
            well_width=well_width,
            barrier_width=barrier_width,
            tilt=0.02,
        )
        for x in xs
    ]
    points = [mini_map(x, value, left=left, top=top, width=width, height=height) for x, value in zip(xs, values)]
    if overlay_young_reference:
        reference_values = [
            potential_value(
                x,
                well_x=0.28,
                barrier_x=barrier_x,
                depth=0.70,
                barrier=0.74,
                well_width=0.16,
                barrier_width=0.14,
                tilt=0.02,
            )
            for x in xs
        ]
        reference_points = [
            mini_map(x, value, left=left, top=top, width=width, height=height)
            for x, value in zip(xs, reference_values)
        ]
        chunks.append(smooth_path(reference_points, color=GRAY_LIGHT, width=4, dash="9 8", opacity=0.70))
    chunks.append(smooth_path(points, color=color, width=6, dash="10 10" if dashed else None))

    if overlay_old:
        old_values = [
            potential_value(
                x,
                well_x=well_x,
                barrier_x=barrier_x,
                depth=0.28,
                barrier=0.32,
                well_width=max(well_width, 0.20),
                barrier_width=0.17,
                tilt=0.02,
            )
            for x in xs
        ]
        old_points = [mini_map(x, value, left=left, top=top, width=width, height=height) for x, value in zip(xs, old_values)]
        chunks.append(smooth_path(old_points, color=GRAY_LIGHT, width=5, dash="9 8", opacity=0.72))

    threshold_x = current_barrier_x
    threshold_y_value = potential_value(
        threshold_x,
        well_x=well_x,
        barrier_x=current_barrier_x,
        depth=depth,
        barrier=barrier * 1.08 if higher_threshold else barrier,
        well_width=well_width,
        barrier_width=barrier_width,
        tilt=0.02,
    )
    if asymmetric_old:
        peak_indices = [index for index, x in enumerate(xs) if x > well_x + 0.20]
        peak_index = max(peak_indices, key=lambda index: values[index])
        threshold_x = xs[peak_index]
        threshold_y_value = values[peak_index]
    tx, ty = mini_map(threshold_x, threshold_y_value, left=left, top=top, width=width, height=height)
    _, bottom_y = mini_map(threshold_x, -0.63, left=left, top=top, width=width, height=height)
    chunks.append(line(tx, bottom_y, tx, ty, color=GRAY_LIGHT, width=2.7, dash="10 8"))

    if higher_threshold:
        control_y = potential_value(
            0.67,
            well_x=well_x,
            barrier_x=barrier_x,
            depth=depth,
            barrier=barrier * 0.82,
            well_width=well_width,
            barrier_width=barrier_width,
            tilt=0.02,
        )
        control_tx, control_ty = mini_map(barrier_x, control_y, left=left, top=top, width=width, height=height)
        chunks.append(line(control_tx, bottom_y, control_tx, control_ty, color=GRAY_LIGHT, width=2, dash="4 7", opacity=0.75))

    ball_x, ball_y = mini_map(well_x, -depth + 0.07, left=left, top=top, width=width, height=height)
    chunks.append(path_from_points(wiggle_points(ball_x, ball_y, 34 if not lower_noise else 18, 9), color=ORANGE, width=4))
    if lower_noise:
        chunks.append(path_from_points(wiggle_points(ball_x, ball_y, 42, 11), color=ORANGE_LIGHT, width=3, opacity=0.45))
    chunks.append(circle(ball_x, ball_y, 16, fill=ORANGE, width=3))
    return chunks


def mini_noise_trace(
    left: float,
    top: float,
    *,
    width: float,
    height: float,
    seed: int,
    amplitude: float,
    color: str,
    label: str,
) -> list[str]:
    """Tiny stochastic trace used to clarify high versus low noise."""
    rng = random.Random(seed)
    values: list[float] = []
    for index in range(62):
        progress = index / 61
        trend = 0.28 + 0.38 * progress
        jitter = rng.gauss(0, amplitude)
        if amplitude > 0.06 and rng.random() < 0.12:
            jitter += rng.choice([-1, 1]) * rng.uniform(0.10, 0.22)
        values.append(max(0.11, min(0.88, trend + jitter)))

    points = []
    for index, value in enumerate(values):
        px = left + index / (len(values) - 1) * width
        py = top + height - value * height
        points.append((px, py))
    return [
        line(left, top + height, left + width, top + height, color=GRAY_LIGHT, width=2.2),
        line(left, top + height, left, top + 5, color=GRAY_LIGHT, width=2.2),
        line(left, top + height * 0.24, left + width, top + height * 0.24, color=GRAY_LIGHT, width=2.1, dash="7 7"),
        path_from_points(points, color=color, width=3.6),
        text([label], left + width / 2, top + height + 28, size=18, fill=GRAY, anchor="middle", weight="700"),
    ]


def noise_comparison(left: float, top: float) -> list[str]:
    """Two small traces that explicitly show noisy versus reduced-noise motion."""
    chunks: list[str] = []
    chunks.extend(mini_noise_trace(left, top, width=130, height=76, seed=12, amplitude=0.115, color=TRAJ_BLUE, label="higher noise"))
    chunks.extend(mini_noise_trace(left + 168, top, width=130, height=76, seed=12, amplitude=0.026, color=TRAJ_ORANGE, label="lower noise"))
    return chunks


def hourglass_icon(cx: float, top: float, *, scale: float = 1.0) -> list[str]:
    """Draw an editable hourglass icon."""
    w = 58 * scale
    h = 78 * scale
    stroke_width = 5 * scale
    chunks = [
        line(cx - w / 2, top, cx + w / 2, top, color=GRAY, width=stroke_width),
        line(cx - w / 2, top + h, cx + w / 2, top + h, color=GRAY, width=stroke_width),
        path_from_points(
            [(cx - w / 2 + 7 * scale, top + 7 * scale), (cx - 5 * scale, top + h / 2), (cx - w / 2 + 7 * scale, top + h - 7 * scale)],
            color=GRAY,
            width=4 * scale,
        ),
        path_from_points(
            [(cx + w / 2 - 7 * scale, top + 7 * scale), (cx + 5 * scale, top + h / 2), (cx + w / 2 - 7 * scale, top + h - 7 * scale)],
            color=GRAY,
            width=4 * scale,
        ),
        filled_path(
            f"M {cx - 15 * scale:.1f},{top + 16 * scale:.1f} "
            f"L {cx + 15 * scale:.1f},{top + 16 * scale:.1f} "
            f"L {cx:.1f},{top + 36 * scale:.1f} Z",
            fill="#8F98A1",
            opacity=0.72,
        ),
        filled_path(
            f"M {cx:.1f},{top + 45 * scale:.1f} "
            f"L {cx - 17 * scale:.1f},{top + h - 16 * scale:.1f} "
            f"L {cx + 17 * scale:.1f},{top + h - 16 * scale:.1f} Z",
            fill="#8F98A1",
            opacity=0.72,
        ),
        line(cx, top + 37 * scale, cx, top + 44 * scale, color="#8F98A1", width=3 * scale, opacity=0.7),
    ]
    return chunks


def time_arrow(left: float, center_y: float, *, width: float = 140, height: float = 92) -> list[str]:
    """Draw a large gradient arrow to show time passing."""
    head = 40
    top = center_y - height / 2
    bottom = center_y + height / 2
    mid = center_y
    arrow_d = (
        f"M {left:.1f},{top + 24:.1f} "
        f"L {left + width - head:.1f},{top + 24:.1f} "
        f"L {left + width - head:.1f},{top:.1f} "
        f"L {left + width:.1f},{mid:.1f} "
        f"L {left + width - head:.1f},{bottom:.1f} "
        f"L {left + width - head:.1f},{bottom - 24:.1f} "
        f"L {left:.1f},{bottom - 24:.1f} Z"
    )
    highlight_d = (
        f"M {left + 10:.1f},{top + 34:.1f} "
        f"L {left + width - head - 4:.1f},{top + 34:.1f} "
        f"L {left + width - 24:.1f},{mid:.1f} "
        f"L {left + width - head - 4:.1f},{bottom - 34:.1f} "
        f"L {left + 10:.1f},{bottom - 34:.1f} Z"
    )
    return [
        f'<path d="{arrow_d}" fill="url(#time-arrow-gradient)" opacity="0.86"/>',
        filled_path(highlight_d, fill="#FFFFFF", opacity=0.18),
    ]


def mini_trajectory_plot(
    left: float,
    top: float,
    *,
    width: float,
    height: float,
    values: list[float],
    color: str,
    label: str | None = None,
    threshold: float = 0.64,
    show_y_label: bool = True,
    show_x_label: bool = True,
) -> list[str]:
    """Draw a compact trajectory plot with one dashed threshold."""
    chunks = [
        line(left, top + height, left + width, top + height, color=GRAY_LIGHT, width=2.8),
        line(left, top + height, left, top, color=GRAY_LIGHT, width=2.8),
        line(left, top + height * (1 - threshold), left + width, top + height * (1 - threshold), color=GRAY_LIGHT, width=2.8, dash="9 8"),
        path_from_points(
            [
                (left + index / (len(values) - 1) * width, top + height - value * height)
                for index, value in enumerate(values)
            ],
            color=color,
            width=5.2,
        ),
    ]
    if show_x_label:
        chunks.append(text(["Age, t"], left + width / 2, top + height + 42, size=30, fill=GRAY, anchor="middle", weight="700"))
    if show_y_label:
        chunks.append(
            f'<g transform="rotate(-90 {left - 25:.1f} {top + height / 2:.1f})">'
            f'{text(["X(t)"], left - 25, top + height / 2, size=27, fill=GRAY, anchor="middle", weight="700")}'
            "</g>"
        )
    if label:
        chunks.append(text([label], left + width / 2, top - 28, size=26, fill=GRAY, anchor="middle", weight="700"))
    return chunks


def convex_aging_values(n: int, *, noise: float, seed: int, exponent: float = 2.25) -> list[float]:
    """Return a noisy convex trajectory that starts near zero and curls upward."""
    rng = random.Random(seed)
    values: list[float] = []
    for index in range(n):
        progress = index / (n - 1)
        jitter = rng.gauss(0.0, noise) + 0.012 * sin(index * 1.7)
        value = 0.05 + 0.80 * progress**exponent + jitter
        values.append(max(0.02, min(0.90, value)))
    values[0] = 0.04
    return values


def noisy_parabola_values(
    xs: list[float],
    *,
    intercept: float,
    coefficient: float,
    noise: float,
    seed: int,
) -> list[float]:
    """Return a noisy convex-up parabola for schematic aging dynamics."""
    rng = random.Random(seed)
    values: list[float] = []
    for index, x in enumerate(xs):
        jitter = rng.gauss(0.0, noise) + 0.45 * noise * sin(index * 1.45)
        values.append(max(0.02, intercept + coefficient * x**2 + jitter))
    values[0] = intercept
    return values


def visible_until_axis_top(xs: list[float], values: list[float], upper: float) -> tuple[list[float], list[float]]:
    """Stop a schematic curve at the top of its plotting area without a clip mask."""
    clipped_xs: list[float] = []
    clipped_values: list[float] = []
    previous_x = xs[0]
    previous_value = values[0]
    if previous_value <= upper:
        clipped_xs.append(previous_x)
        clipped_values.append(previous_value)

    for current_x, current_value in zip(xs[1:], values[1:]):
        previous_inside = previous_value <= upper
        current_inside = current_value <= upper
        if previous_inside != current_inside:
            fraction = (upper - previous_value) / (current_value - previous_value)
            clipped_xs.append(previous_x + fraction * (current_x - previous_x))
            clipped_values.append(upper)
        if current_inside:
            clipped_xs.append(current_x)
            clipped_values.append(current_value)
        previous_x = current_x
        previous_value = current_value
    return clipped_xs, clipped_values


def senogenic_dynamics_diagram(left: float, top: float) -> list[str]:
    """Show senogenic change as altered deterministic aging dynamics."""
    chunks: list[str] = []
    plot_x, plot_y, plot_w, plot_h = left + 70, top + 72, 630, 382
    chunks.append(line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, color=GRAY_LIGHT, width=3.4))
    chunks.append(line(plot_x, plot_y + plot_h, plot_x, plot_y + 10, color=GRAY_LIGHT, width=3.4))
    y_axis_max = 12.90
    threshold = 7.55
    threshold_y = plot_y + plot_h * (1 - threshold / y_axis_max)
    chunks.append(line(plot_x, threshold_y, plot_x + plot_w, threshold_y, color=GRAY_LIGHT, width=2.8, dash="10 9"))
    chunks.append(text(["threshold"], plot_x + 14, threshold_y - 14, size=27, fill=GRAY, weight="700", anchor="start"))

    x_max = 1.00
    xs = sample_curve(0.0, x_max, 95)
    start_value = 0.060
    early_coefficient = 24.50
    delayed_coefficient = 8.25
    early = noisy_parabola_values(xs, intercept=start_value, coefficient=early_coefficient, noise=0.205, seed=101)
    delayed = noisy_parabola_values(xs, intercept=start_value, coefficient=delayed_coefficient, noise=0.165, seed=102)
    early_xs, early = visible_until_axis_top(xs, early, y_axis_max)
    delayed_xs, delayed = visible_until_axis_top(xs, delayed, y_axis_max)
    early_points = [
        (plot_x + x / x_max * plot_w, plot_y + plot_h - y / y_axis_max * plot_h)
        for x, y in zip(early_xs, early)
    ]
    delayed_points = [
        (plot_x + x / x_max * plot_w, plot_y + plot_h - y / y_axis_max * plot_h)
        for x, y in zip(delayed_xs, delayed)
    ]
    chunks.append(path_from_points(early_points, color=TEAL, width=7.2, opacity=0.50))
    chunks.append(path_from_points(delayed_points, color=TEAL, width=7.2))

    arrow_y_value = 4.25
    arrow_y = plot_y + plot_h * (1 - arrow_y_value / y_axis_max)
    inner_x = ((arrow_y_value - start_value) / early_coefficient) ** 0.5 / x_max
    outer_x = ((arrow_y_value - start_value) / delayed_coefficient) ** 0.5 / x_max
    chunks.append(
        line(
            plot_x + plot_w * (inner_x + 0.03),
            arrow_y,
            plot_x + plot_w * (outer_x - 0.035),
            arrow_y,
            color=RED_DANGER,
            width=8.5,
            arrow="arrow",
            opacity=0.90,
        )
    )
    label_x = (plot_x + plot_w * (inner_x + 0.03) + plot_x + plot_w * (outer_x - 0.035)) / 2
    chunks.append(halo_text(["senogenic change"], label_x, arrow_y - 28, size=30, fill=RED_DANGER, halo_width=12))
    chunks.append(text(["Age, t"], plot_x + plot_w / 2, plot_y + plot_h + 52, size=31, fill=GRAY, weight="700", anchor="middle"))
    chunks.append(
        f'<g transform="rotate(-90 {plot_x - 42:.1f} {plot_y + plot_h / 2:.1f})">'
        f'{text(["X(t)"], plot_x - 42, plot_y + plot_h / 2, size=31, fill=GRAY, anchor="middle", weight="700")}'
        "</g>"
    )
    return chunks


def robustness_trajectory_diagrams(left: float, top: float) -> list[str]:
    """Show robustness as threshold or noise changes, without potential wells."""
    chunks: list[str] = []
    threshold_values = convex_aging_values(34, noise=0.045, seed=3, exponent=2.10)
    high_noise = convex_aging_values(34, noise=0.120, seed=10, exponent=2.15)
    low_noise = convex_aging_values(34, noise=0.025, seed=11, exponent=2.25)

    center_x = left + 365

    chunks.append(text(["altered threshold"], center_x, top - 28, size=39, anchor="middle", weight="700"))
    plot_x, plot_y, plot_w, plot_h = left + 115, top + 12, 520, 205
    chunks.extend(
        mini_trajectory_plot(
            plot_x,
            plot_y,
            width=plot_w,
            height=plot_h,
            values=threshold_values,
            color=TRAJ_ORANGE,
            threshold=0.60,
            show_x_label=False,
        )
    )
    original_y = plot_y + plot_h * (1 - 0.60)
    higher_y = plot_y + plot_h * (1 - 0.82)
    chunks.append(line(plot_x, higher_y, plot_x + plot_w, higher_y, color=ORANGE_DARK, width=3.0, dash="10 8", opacity=0.68))
    chunks.append(text(["original threshold"], plot_x + 12, original_y - 19, size=23, fill=GRAY, weight="700"))
    chunks.append(text(["higher threshold"], plot_x + 12, higher_y - 19, size=23, fill=ORANGE_DARK, weight="700"))
    chunks.append(line(plot_x + plot_w + 32, original_y - 3, plot_x + plot_w + 32, higher_y + 3, color=ORANGE_DARK, width=4.5, arrow="arrow-gray"))

    chunks.append(text(["altered noise"], center_x, top + 275, size=38, anchor="middle", weight="700"))
    chunks.extend(
        mini_trajectory_plot(
            center_x - 265,
            top + 335,
            width=220,
            height=145,
            values=high_noise,
            color=ORANGE_DARK,
            label="high amplitude",
            threshold=0.64,
        )
    )
    chunks.extend(
        mini_trajectory_plot(
            center_x + 45,
            top + 335,
            width=220,
            height=145,
            values=low_noise,
            color=TRAJ_ORANGE,
            label="low amplitude",
            threshold=0.64,
        )
    )
    return chunks


def make_panel_b() -> None:
    width, height = 1800, 1000
    chunks = svg_header(width, height)
    chunks.append(text(["Two classes of parameters"], 150, 72, size=32, weight="400"))

    left_x, right_x = 90, 930
    card_y, card_w, card_h = 170, 780, 760
    chunks.append(rect(left_x, card_y, card_w, card_h, fill=BLUE_LIGHT, stroke="#9BC4EC", stroke_width=3))
    chunks.append(rect(right_x, card_y, card_w, card_h, fill=ORANGE_PALE, stroke="#F0A078", stroke_width=3))
    chunks.append(text(["I. Senogenic parameters"], left_x + 45, card_y + 76, size=48, fill=BLUE_DARK, weight="700"))
    chunks.append(text(["change deterministic aging dynamics"], left_x + 45, card_y + 126, size=31))
    chunks.append(text(["II. Robustness parameters"], right_x + 45, card_y + 76, size=48, fill=ORANGE_DARK, weight="700"))
    chunks.append(text(["change how easily the threshold is crossed"], right_x + 45, card_y + 126, size=31))

    chunks.extend(senogenic_dynamics_diagram(left_x, card_y + 190))

    # Robustness branch: threshold-crossing diagrams only. These avoid implying
    # that robustness changes the stability landscape itself.
    chunks.extend(robustness_trajectory_diagrams(right_x + 25, card_y + 230))
    chunks.append(svg_footer())
    PANEL_B_SVG.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_panel_a()
    make_panel_b()
    print(f"Saved {PANEL_A_SVG}")
    print(f"Saved {PANEL_B_SVG}")


if __name__ == "__main__":
    main()
