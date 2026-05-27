#!/usr/bin/env python3
"""Build standalone Fig. 1C mortality-signature schematic.

The panel is intentionally model-general. It contrasts robustness-like changes,
which preserve mortality compensation, with senogenic-like changes that can
shift mortality curves without preserving one shared compensation line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig1_panel_c_mortality_signatures"

BASELINE = "#333333"
ROBUSTNESS_COLORS = ["#F7A38B", "#E76F51", "#C74A22"]
BETA_COLORS = ["#79B7E5", "#2166AC", "#174A7A"]
ETA_COLORS = ["#B8E1DC", "#2A9D8F", "#087E8B"]
ANNOTATION = "#333333"

X_LIMITS = (0.0, 1.0)
Y_LIMITS = (-0.80, 1.48)


def configure_matplotlib() -> None:
    """Set compact manuscript-friendly defaults."""
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )


def style_axis(ax: plt.Axes, show_xlabel: bool = False) -> None:
    """Apply shared schematic-axis styling."""
    ax.set_xlim(*X_LIMITS)
    ax.set_ylim(*Y_LIMITS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Age, t" if show_xlabel else "", fontsize=26, labelpad=5)
    ax.set_ylabel("log mortality", fontsize=26, labelpad=7)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


def draw_robustness_row(ax: plt.Axes) -> None:
    """Draw orange lines that share one mortality-compensation point."""
    x_cross = 0.90
    x = np.linspace(0.08, x_cross, 160)
    y_cross = 0.95
    slopes = [0.82, 1.16, 1.50]

    for slope, color in zip(slopes, ROBUSTNESS_COLORS, strict=True):
        y = y_cross + slope * (x - x_cross)
        ax.plot(x, y, color=color, lw=3.0, alpha=0.97, solid_capstyle="round")

    ax.annotate(
        "Strehler-Mildvan\ncorrelation preserved",
        xy=(x_cross, y_cross),
        xytext=(0.55, 0.06),
        fontsize=17.0,
        color=ANNOTATION,
        ha="center",
        va="top",
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.1,
            "color": ANNOTATION,
            "shrinkA": 3,
            "shrinkB": 3,
        },
    )


def draw_senogenic_row(ax: plt.Axes) -> None:
    """Draw senogenic-like parallel shifts and slope changes together."""
    x = np.linspace(0.08, 0.98, 160)
    parallel_slope = 0.64
    parallel_intercepts = [0.32, 0.54, 0.76]

    for intercept, color in zip(parallel_intercepts, BETA_COLORS, strict=True):
        y = intercept + parallel_slope * x
        ax.plot(x, y, color=color, lw=3.0, alpha=0.97, solid_capstyle="round")

    x_start = 0.08
    y_start = -0.66
    slopes = [0.44, 0.74, 1.04]

    for slope, color in zip(slopes, ETA_COLORS, strict=True):
        y = y_start + slope * (x - x_start)
        ax.plot(x, y, color=color, lw=3.0, alpha=0.97, solid_capstyle="round")


def add_row_header(
    fig: plt.Figure,
    x: float,
    y: float,
    title: str,
    color: str = BASELINE,
) -> None:
    """Add a compact row title above one plot."""
    fig.text(x, y, title, ha="center", va="bottom", fontsize=20.5, fontweight="bold", color=color)


def save_outputs(fig: plt.Figure) -> None:
    """Save PDF and a 600-dpi PNG preview."""
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf", transparent=False)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.png", dpi=600, transparent=False)


def main() -> None:
    configure_matplotlib()

    fig = plt.figure(figsize=(6.0875, 5.818), constrained_layout=False)
    fig.patch.set_facecolor("white")

    fig.text(
        0.500,
        0.985,
        "Mortality signatures\nof parameter classes",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="normal",
        color=BASELINE,
        linespacing=0.90,
    )

    add_row_header(fig, 0.520, 0.848, "Robustness-like variation", ROBUSTNESS_COLORS[2])
    add_row_header(fig, 0.520, 0.456, "Senogenic-like variation", BETA_COLORS[2])

    ax_robust = fig.add_axes([0.135, 0.558, 0.780, 0.292])
    ax_senogenic = fig.add_axes([0.135, 0.128, 0.780, 0.292])

    style_axis(ax_robust)
    style_axis(ax_senogenic, show_xlabel=True)

    draw_robustness_row(ax_robust)
    draw_senogenic_row(ax_senogenic)

    save_outputs(fig)
    plt.close(fig)

    print(f"Saved {OUTPUT_DIR / f'{OUTPUT_STEM}.pdf'}")
    print(f"Saved {OUTPUT_DIR / f'{OUTPUT_STEM}.png'}")


if __name__ == "__main__":
    main()
