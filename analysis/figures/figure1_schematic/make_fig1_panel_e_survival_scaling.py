#!/usr/bin/env python3
"""Build Fig. 1E survival scaling/steepening schematic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig1_panel_e_survival_scaling"

BLACK = "#111111"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 1.0,
        }
    )


def survival(age: np.ndarray, median: float, shape: float) -> np.ndarray:
    return np.exp(-np.log(2.0) * (age / median) ** shape)


def style_axis(ax: plt.Axes, *, xlabel: str, ylabel: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=4.2, width=1.0, color=BLACK, pad=2)
    ax.set_xlabel(xlabel, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=4)


def draw_row_arrow(fig: plt.Figure, y: float) -> None:
    arrow = matplotlib.patches.FancyArrowPatch(
        (0.445, y),
        (0.540, y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.4,
        color=BLACK,
    )
    fig.patches.append(arrow)
    fig.text(0.492, y + 0.036, "Normalise age", ha="center", va="bottom", fontsize=11.5)


def main() -> None:
    configure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.6), constrained_layout=False)
    fig.patch.set_facecolor("white")

    age = np.linspace(0.0, 115.0, 300)
    normalized_age = np.linspace(0.6, 1.5, 300)

    baseline_scaling = survival(age, median=58.0, shape=6.2)
    changed_scaling = survival(age, median=82.0, shape=6.2)
    axes[0, 0].plot(age, baseline_scaling, color=BLACK, lw=2.0, label="baseline")
    axes[0, 0].plot(age[::10], changed_scaling[::10], color=BLACK, marker="o", linestyle="none", ms=4.8, label="parameter change")
    axes[0, 0].set_xlim(0, 110)
    axes[0, 0].set_ylim(0, 1.05)
    style_axis(axes[0, 0], xlabel="Age, t", ylabel="Survival")

    axes[0, 1].plot(normalized_age, survival(normalized_age, median=1.0, shape=6.2), color=BLACK, lw=2.0)
    axes[0, 1].plot(
        normalized_age[::10],
        survival(normalized_age, median=1.0, shape=6.2)[::10],
        color=BLACK,
        marker="o",
        linestyle="none",
        ms=4.8,
    )
    axes[0, 1].set_xlim(0.55, 1.50)
    axes[0, 1].set_ylim(0, 1.05)
    style_axis(axes[0, 1], xlabel=r"Age normalised, $t/t_{50}$")
    axes[0, 1].text(0.76, 0.70, "Scaling", transform=axes[0, 1].transAxes, fontsize=17)

    baseline_steep = survival(age, median=58.0, shape=6.2)
    changed_steep = survival(age, median=74.0, shape=10.5)
    axes[1, 0].plot(age, baseline_steep, color=BLACK, lw=2.0)
    axes[1, 0].plot(age[::10], changed_steep[::10], color=BLACK, marker="o", linestyle="none", ms=4.8)
    axes[1, 0].set_xlim(0, 110)
    axes[1, 0].set_ylim(0, 1.05)
    style_axis(axes[1, 0], xlabel="Age, t", ylabel="Survival")

    axes[1, 1].plot(normalized_age, survival(normalized_age, median=1.0, shape=6.2), color=BLACK, lw=2.0)
    axes[1, 1].plot(
        normalized_age[::10],
        survival(normalized_age, median=1.0, shape=10.5)[::10],
        color=BLACK,
        marker="o",
        linestyle="none",
        ms=4.8,
    )
    axes[1, 1].set_xlim(0.55, 1.50)
    axes[1, 1].set_ylim(0, 1.05)
    style_axis(axes[1, 1], xlabel=r"Age normalised, $t/t_{50}$")
    axes[1, 1].text(0.73, 0.67, "Steepening", transform=axes[1, 1].transAxes, fontsize=17)

    draw_row_arrow(fig, 0.705)
    draw_row_arrow(fig, 0.315)

    fig.legend(
        handles=[
            matplotlib.lines.Line2D([0], [0], color=BLACK, lw=2.2, label="baseline"),
            matplotlib.lines.Line2D(
                [0],
                [0],
                color=BLACK,
                marker="o",
                linestyle="none",
                markersize=5.5,
                label="parameter change",
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.500, 0.468),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="#D9D9D9",
        fontsize=13.5,
        handlelength=2.4,
        columnspacing=1.4,
        borderpad=0.45,
    )

    fig.subplots_adjust(left=0.120, right=0.970, bottom=0.090, top=0.965, wspace=0.42, hspace=0.74)

    for suffix in ("png", "pdf", "svg"):
        output_path = OUTPUT_DIR / f"{OUTPUT_STEM}.{suffix}"
        if suffix == "png":
            fig.savefig(output_path, dpi=500, bbox_inches="tight", pad_inches=0.03)
        else:
            fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
