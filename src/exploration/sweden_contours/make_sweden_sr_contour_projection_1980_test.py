#!/usr/bin/env python3
"""Standalone Fig. 4d-style SR contour test starting at 1980."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


FROM_AGE = 20
DISPLAY_START_YEAR = 1980
DISPLAY_END_YEAR = 2100
N_SIM_LABEL = "n1m"

RESULTS_DIR = SAVED_RESULTS_DIR / "fig4_new"
FIGURE_DIR = FIGURES_NEW_DIR / "Fig4_new"

HMD_CONTOURS_CSV = RESULTS_DIR / "sweden_period_both_conditional_age20_contours_1900_2020.csv"
SUMMARY_CSV = RESULTS_DIR / f"sweden_sr_contour_projection_summary_{N_SIM_LABEL}.csv"
OUTPUT_PNG = FIGURE_DIR / f"sweden_sr_contour_projection_1980_2100_test_{N_SIM_LABEL}.png"
OUTPUT_PDF = FIGURE_DIR / f"sweden_sr_contour_projection_1980_2100_test_{N_SIM_LABEL}.pdf"

SURVIVAL_LEVELS = {
    "Median": 0.5,
    "Top 10%": 0.1,
    "Top 1%": 0.01,
    "Top 0.01%": 1e-4,
}

COLORS = {
    "Median": "#2b6cb0",
    "Top 10%": "#2f855a",
    "Top 1%": "#b7791f",
    "Top 0.01%": "#c53030",
}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not HMD_CONTOURS_CSV.exists():
        raise FileNotFoundError(f"Missing HMD contours: {HMD_CONTOURS_CSV}")
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing SR summary: {SUMMARY_CSV}")

    hmd = pd.read_csv(HMD_CONTOURS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    return hmd, summary


def plot_projection(hmd: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.labelsize": 21,
            "axes.titlesize": 22,
            "legend.fontsize": 15,
            "legend.title_fontsize": 15,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.axvspan(2020, DISPLAY_END_YEAR, color="0.88", alpha=0.48, zorder=0)
    plot_hmd_contours(ax, hmd)
    plot_historical_sr(ax, summary)
    plot_projection_sr(
        ax,
        summary,
        "linear",
        linestyle="-",
        fill_alpha=0.09,
        line_alpha=0.72,
    )
    plot_projection_sr(
        ax,
        summary,
        "exponential",
        linestyle="--",
        fill_alpha=0.055,
        line_alpha=0.52,
    )
    plot_recent_linear_continuations(ax, summary)
    add_direct_contour_labels(ax, summary)

    ax.axvline(2020, color="0.45", linestyle=":", linewidth=1.6)
    ax.text(2022.0, 122.7, "forecast", color="0.35", fontsize=11, va="top", rotation=90)

    ax.set_title("Extrapolating recent increase in robustness predicts diminishing longevity gains")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Lifespan [years]\n(conditional on survival to age {FROM_AGE})")
    ax.set_xlim(DISPLAY_START_YEAR, DISPLAY_END_YEAR)
    ax.set_ylim(72, 125)
    ax.set_xticks(np.arange(1980, 2101, 20))
    ax.tick_params(axis="both", width=1.2, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.20", alpha=0.08, linewidth=0.8)
    fitted_legend = ax.legend(
        handles=fitted_handles(),
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.035),
        borderaxespad=0,
    )
    ax.add_artist(fitted_legend)
    projection_legend = ax.legend(
        handles=projection_handles(),
        title="Model forecasts",
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.035),
        borderaxespad=0,
    )
    projection_legend.get_title().set_fontweight("bold")

    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.88)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, format="png", dpi=280)
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)


def plot_hmd_contours(ax: plt.Axes, hmd: pd.DataFrame) -> None:
    rows = hmd[hmd["year"] >= DISPLAY_START_YEAR].sort_values("year")
    for label in SURVIVAL_LEVELS:
        ax.plot(
            rows["year"],
            rows[f"{label} smoothed"],
            color=COLORS[label],
            linewidth=2.8,
            alpha=0.65,
            zorder=2,
        )


def plot_historical_sr(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = historical_rows(summary)
    for label in SURVIVAL_LEVELS:
        ax.fill_between(
            rows["year"],
            rows[f"{label} low"],
            rows[f"{label} high"],
            color=COLORS[label],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rows["year"],
            rows[label],
            color=COLORS[label],
            linewidth=1.9,
            marker="o",
            markersize=4.5,
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=4,
        )


def plot_projection_sr(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scenario: str,
    *,
    linestyle: str,
    fill_alpha: float,
    line_alpha: float,
) -> None:
    rows = summary[summary["scenario"] == scenario].sort_values("year")
    for label in SURVIVAL_LEVELS:
        ax.fill_between(
            rows["year"],
            rows[f"{label} low"],
            rows[f"{label} high"],
            color=COLORS[label],
            alpha=fill_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rows["year"],
            rows[label],
            color=COLORS[label],
            linewidth=2.0,
            linestyle=linestyle,
            alpha=line_alpha,
            zorder=3,
        )


def plot_recent_linear_continuations(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = historical_rows(summary)
    recent = rows.tail(4)
    line_years = np.arange(int(recent["year"].min()), DISPLAY_END_YEAR + 1)

    for label in SURVIVAL_LEVELS:
        slope, intercept = np.polyfit(
            recent["year"].to_numpy(dtype=float),
            recent[label].to_numpy(dtype=float),
            1,
        )
        ax.plot(
            line_years,
            slope * line_years + intercept,
            color="0.45",
            linewidth=3.0,
            linestyle=(0, (1.0, 3.0)),
            alpha=0.30,
            zorder=2.5,
        )


def add_direct_contour_labels(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = historical_rows(summary)
    label_year = 2000
    x_values = rows["year"].to_numpy(dtype=float)

    for label in SURVIVAL_LEVELS:
        y_values = rows[label].to_numpy(dtype=float)
        y_label = float(np.interp(label_year, x_values, y_values))
        text = ax.text(
            label_year,
            y_label,
            label,
            color=COLORS[label],
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=6,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3.0, foreground="white", alpha=0.85),
                path_effects.Normal(),
            ]
        )


def historical_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["scenario"] == "historical")
        & (summary["year"] >= DISPLAY_START_YEAR)
    ].sort_values("year")


def fitted_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="0.45", lw=2.8, label="HMD period contours"),
        Line2D(
            [0],
            [0],
            color="0.20",
            lw=1.9,
            marker="o",
            label=r"SR model: projected $X_c$ and fitted $m_{ex}$",
        ),
    ]


def projection_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="0.20", lw=2.0, linestyle="-", alpha=0.72, label="Linear increase in robustness"),
        Line2D([0], [0], color="0.20", lw=2.0, linestyle="--", alpha=0.52, label="Exponential increase in robustness"),
        Line2D([0], [0], color="0.45", lw=3.0, linestyle=(0, (1.0, 3.0)), alpha=0.30, label="Naive recent continuation"),
    ]


def main() -> None:
    hmd, summary = load_inputs()
    plot_projection(hmd, summary)
    print(f"saved {OUTPUT_PNG}")
    print(f"saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
