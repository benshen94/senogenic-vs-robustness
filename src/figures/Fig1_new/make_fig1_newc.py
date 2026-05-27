#!/usr/bin/env python3
"""Build revised Fig. 1C: SR equation and parameter-class mapping."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.thresholds.paths import FIGURES_NEW_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Fig1new"
OUTPUT_STEM = "fig1_panel_d_sr_model"

BLACK = "#111111"
GRAY = "#59616B"
GRAY_LIGHT = "#D8DCE0"
GRAY_PALE = "#F6F7F9"

ETA = "#0B7F8C"
BETA = "#173A6A"
XC = "#D77A16"
EPSILON = "#E5A100"

SENOGENIC_PALE = "#E8F4F6"
ROBUSTNESS_PALE = "#FFF3DC"


def configure_matplotlib() -> None:
    """Use manuscript-friendly defaults and real LaTeX where requested."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
            "text.latex.preamble": "\n".join(
                [
                    r"\usepackage{xcolor}",
                    rf"\definecolor{{etaColor}}{{HTML}}{{{ETA.lstrip('#')}}}",
                    rf"\definecolor{{betaColor}}{{HTML}}{{{BETA.lstrip('#')}}}",
                    rf"\definecolor{{xcColor}}{{HTML}}{{{XC.lstrip('#')}}}",
                    rf"\definecolor{{epsilonColor}}{{HTML}}{{{EPSILON.lstrip('#')}}}",
                ]
            ),
        }
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLACK,
    lw: float = 2.2,
    ls: str = "-",
) -> None:
    """Draw a clean arrow in axes coordinates."""
    arrow = patches.FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=8,
        shrinkB=8,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def add_arc_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float,
    color: str = BLACK,
    lw: float = 2.2,
) -> None:
    """Draw a curved arrow in axes coordinates."""
    arrow = patches.FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def add_squiggle_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = BLACK,
    lw: float = 1.8,
) -> None:
    """Draw a wavy cubic arrow in axes coordinates."""
    path = MplPath(
        points,
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )
    arrow = patches.FancyArrowPatch(
        path=path,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    lw: float = 1.8,
    radius: float = 0.018,
) -> patches.FancyBboxPatch:
    """Draw a rounded rectangle in axes coordinates."""
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        transform=ax.transAxes,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
    )
    ax.add_patch(box)
    return box


def add_math(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    color: str = BLACK,
    ha: str = "center",
    va: str = "center",
    weight: str = "normal",
    usetex: bool = False,
) -> None:
    """Add math or text in axes coordinates."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontweight=weight,
        usetex=usetex,
    )


def add_math_outline(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float,
    color: str = BLACK,
) -> None:
    """Add math as vector outlines so Illustrator does not need TeX fonts."""
    path = TextPath((0, 0), text, size=size, usetex=False)
    bbox = path.get_extents()
    axes_box = ax.get_position()
    axes_width_pt = ax.figure.get_figwidth() * 72 * axes_box.width
    axes_height_pt = ax.figure.get_figheight() * 72 * axes_box.height
    text_center_x = (bbox.x0 + bbox.x1) / 2
    text_center_y = (bbox.y0 + bbox.y1) / 2

    transform = (
        Affine2D()
        .scale(1 / axes_width_pt, 1 / axes_height_pt)
        .translate(x - text_center_x / axes_width_pt, y - text_center_y / axes_height_pt)
        + ax.transAxes
    )
    patch = patches.PathPatch(path, transform=transform, facecolor=color, edgecolor="none")
    ax.add_patch(patch)


def draw_circuit(ax: plt.Axes) -> None:
    """Draw a close redraw of the original SR circuit cartoon."""
    add_arrow(ax, (0.165, 0.700), (0.335, 0.700), color=BLACK, lw=2.0)
    add_arrow(ax, (0.250, 0.815), (0.250, 0.715), color=BLACK, lw=1.8)
    add_math(ax, 0.250, 0.845, "Age t", size=16.0)
    add_math(ax, 0.250, 0.660, "Production", size=14.0)
    add_math(ax, 0.250, 0.637, "rate", size=14.0)
    add_math_outline(ax, 0.250, 0.600, r"$\eta$", size=17)

    add_math(ax, 0.500, 0.700, "Damage", size=16.5)
    add_math(ax, 0.500, 0.667, "X", size=16.5)
    add_arrow(ax, (0.500, 0.640), (0.500, 0.575), color=BLACK, lw=1.8)
    add_math(ax, 0.500, 0.540, "Morbidity, Mortality", size=14.0)

    add_math(ax, 0.472, 0.875, "Noise", size=16.0)
    add_math_outline(ax, 0.548, 0.875, r"$\xi$", size=16.0)
    add_squiggle_arrow(
        ax,
        [
            (0.500, 0.850),
            (0.535, 0.820),
            (0.526, 0.790),
            (0.500, 0.770),
            (0.482, 0.754),
            (0.500, 0.748),
            (0.500, 0.730),
        ],
        color=BLACK,
        lw=1.8,
    )

    add_arrow(ax, (0.650, 0.700), (0.815, 0.700), color=BLACK, lw=2.0)
    add_math_outline(ax, 0.866, 0.700, r"$\Phi$", size=16.5)
    add_math(ax, 0.720, 0.660, "Removal", size=14.0)
    add_math(ax, 0.720, 0.637, "rate", size=14.0)
    add_math_outline(ax, 0.720, 0.600, r"$\beta$", size=17)
    add_arc_arrow(ax, (0.560, 0.748), (0.735, 0.705), rad=-0.42, color=BLACK, lw=1.6)


def draw_equation_box(ax: plt.Axes) -> None:
    """Draw the LaTeX SR equation in a central box."""
    add_box(ax, (0.070, 0.310), 0.860, 0.160, facecolor=GRAY_PALE, edgecolor=BLACK, lw=2.1, radius=0.010)
    equation = r"$\frac{dX}{dt}=\eta t-\frac{\beta X}{X+\kappa}+\sqrt{2\epsilon}\,\xi(t)$"
    death_rule = r"$\mathrm{death\ when}\ X>X_c$"
    add_math_outline(ax, 0.500, 0.420, equation, size=18.8)
    add_math_outline(ax, 0.500, 0.355, death_rule, size=17.5)


def draw_parameter_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    title: str,
    title_color: str,
    facecolor: str,
    edgecolor: str,
    rows: list[tuple[str, str, str]],
) -> None:
    """Draw one parameter-class mapping card."""
    add_box(ax, xy, width, height, facecolor=facecolor, edgecolor=edgecolor, lw=2.0)
    x0, y0 = xy
    add_math(ax, x0 + 0.040, y0 + height - 0.045, title, size=15.0, color=title_color, ha="left", weight="bold")

    for index, (symbol, color, label) in enumerate(rows):
        y = y0 + height - 0.105 - index * 0.073
        add_box(ax, (x0 + 0.040, y - 0.025), 0.082, 0.044, facecolor="white", edgecolor=color, lw=1.7, radius=0.012)
        add_math_outline(ax, x0 + 0.081, y - 0.003, symbol, size=17.5, color=color)
        add_math(ax, x0 + 0.145, y - 0.003, label, size=13.5, color=BLACK, ha="left")


def draw_parameter_mapping(ax: plt.Axes) -> None:
    """Draw the bottom mapping between model parameters and classes."""
    draw_parameter_card(
        ax,
        (0.070, 0.015),
        0.415,
        0.220,
        title=r"Senogenic parameters",
        title_color=BETA,
        facecolor=SENOGENIC_PALE,
        edgecolor=ETA,
        rows=[
            (r"$\eta$", ETA, "damage production"),
            (r"$\beta$", BETA, "damage removal"),
        ],
    )
    draw_parameter_card(
        ax,
        (0.515, 0.015),
        0.415,
        0.220,
        title=r"Robustness parameters",
        title_color=XC,
        facecolor=ROBUSTNESS_PALE,
        edgecolor=XC,
        rows=[
            (r"$X_c$", XC, "death threshold"),
            (r"$\epsilon$", EPSILON, "noise amplitude"),
        ],
    )


def build_figure() -> plt.Figure:
    """Build the complete Fig. 1C panel."""
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.7, 7.4), constrained_layout=False)
    fig.patch.set_facecolor("white")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    add_math(ax, 0.120, 0.948, "Saturating-removal model", size=25, ha="left", weight="normal")

    draw_circuit(ax)
    draw_equation_box(ax)
    draw_parameter_mapping(ax)
    return fig


def save_outputs(fig: plt.Figure) -> None:
    """Save PNG and PDF outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output_path = OUTPUT_DIR / f"{OUTPUT_STEM}.{suffix}"
        if suffix == "png":
            fig.savefig(output_path, dpi=320, bbox_inches="tight", pad_inches=0.06)
            continue
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06)


def main() -> None:
    fig = build_figure()
    save_outputs(fig)
    plt.close(fig)
    print(f"Saved outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
