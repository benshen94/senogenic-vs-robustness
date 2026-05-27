#!/usr/bin/env python3
"""Standalone age-0 SR contour plot with mean lifespan replacing median."""

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

from src.shared.thresholds.paths import FIGURES_NEW_DIR, HMD_DATA_DIR, SAVED_RESULTS_DIR


PLOT_START_YEAR = 1900
SR_START_YEAR = 1980
DISPLAY_END_YEAR = 2100
SMOOTHING_WINDOW = 5
N_SIM_LABEL = "n1m"

RESULTS_DIR = SAVED_RESULTS_DIR / "fig4_new"
FIGURE_DIR = FIGURES_NEW_DIR / "Fig4_new"

HMD_AGE0_CONTOURS_CSV = RESULTS_DIR / "sweden_period_both_conditional_age0_contours_1900_2020.csv"
SR_SUMMARY_CSV = RESULTS_DIR / f"sweden_sr_age0_mean_lifespan_contour_summary_1980_2100_{N_SIM_LABEL}.csv"
OUTPUT_PNG = FIGURE_DIR / f"sweden_sr_age0_mean_lifespan_projection_1900_2100_{N_SIM_LABEL}.png"
OUTPUT_PDF = FIGURE_DIR / f"sweden_sr_age0_mean_lifespan_projection_1900_2100_{N_SIM_LABEL}.pdf"

LIFE_TABLE_PATH = HMD_DATA_DIR / "mortality.org_File_GetDocument_hmd.v6_SWE_STATS_bltper_1x1.txt"

CONTOURS = ("Mean lifespan", "Top 10%", "Top 1%", "Top 0.01%")

LINE_COLOR = "0.0"
NAIVE_COLOR = "0.45"


def load_hmd_contours() -> pd.DataFrame:
    contours = pd.read_csv(HMD_AGE0_CONTOURS_CSV)
    mean_lifespan = load_hmd_life_expectancy()
    hmd = contours.merge(mean_lifespan, on="year", how="left")
    hmd["Mean lifespan smoothed"] = (
        hmd["Mean lifespan"]
        .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
        .mean()
    )
    return hmd


def load_hmd_life_expectancy() -> pd.DataFrame:
    rows = []
    with LIFE_TABLE_PATH.open(errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("Year") or text.startswith("Sweden"):
                continue

            parts = text.split()
            if len(parts) < 10:
                continue

            try:
                year = int(parts[0])
                age = int(parts[1].replace("+", ""))
                ex = float(parts[9])
            except ValueError:
                continue

            if age == 0:
                rows.append({"year": year, "Mean lifespan": ex})

    return pd.DataFrame(rows)


def load_sr_summary() -> pd.DataFrame:
    if not SR_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing SR age-0 summary: {SR_SUMMARY_CSV}")

    summary = pd.read_csv(SR_SUMMARY_CSV)
    return summary.rename(
        columns={
            "mean_lifespan": "Mean lifespan",
            "mean_lifespan low": "Mean lifespan low",
            "mean_lifespan high": "Mean lifespan high",
        }
    )


def plot_projection(hmd: pd.DataFrame, summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.labelsize": 21,
            "axes.titlesize": 22,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
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
        fill_alpha=0.055,
        line_alpha=1.0,
    )
    plot_projection_sr(
        ax,
        summary,
        "exponential",
        linestyle="--",
        fill_alpha=0.035,
        line_alpha=1.0,
    )
    plot_recent_linear_continuations(ax, summary)
    add_direct_contour_labels(ax, hmd)

    ax.axvline(2020, color="0.45", linestyle=":", linewidth=1.6)
    ax.text(2022.0, 118.8, "forecast", color="0.35", fontsize=11, va="top", rotation=90)

    ax.set_title("Mean lifespan and upper-tail contours from birth")
    ax.set_xlabel("Year")
    ax.set_ylabel("Age [years]\n(from birth)")
    ax.set_xlim(PLOT_START_YEAR, DISPLAY_END_YEAR)
    ax.set_ylim(45, 120)
    ax.set_xticks(np.arange(1900, 2101, 20))
    ax.tick_params(axis="both", width=1.2, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.20", alpha=0.08, linewidth=0.8)

    fitted_legend = ax.legend(
        handles=fitted_handles(),
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
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
    rows = hmd[hmd["year"] >= PLOT_START_YEAR].sort_values("year")
    for label in CONTOURS:
        ax.plot(
            rows["year"],
            rows[f"{label} smoothed"],
            color=LINE_COLOR,
            linewidth=2.8,
            alpha=0.42,
            zorder=2,
        )


def plot_historical_sr(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = historical_rows(summary)
    for label in CONTOURS:
        ax.fill_between(
            rows["year"],
            rows[f"{label} low"],
            rows[f"{label} high"],
            color=LINE_COLOR,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rows["year"],
            rows[label],
            color=LINE_COLOR,
            linewidth=1.9,
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
    for label in CONTOURS:
        ax.fill_between(
            rows["year"],
            rows[f"{label} low"],
            rows[f"{label} high"],
            color=LINE_COLOR,
            alpha=fill_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rows["year"],
            rows[label],
            color=LINE_COLOR,
            linewidth=2.0,
            linestyle=linestyle,
            alpha=line_alpha,
            zorder=3,
        )


def plot_recent_linear_continuations(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = historical_rows(summary)
    recent = rows.tail(4)
    line_years = np.arange(2020, DISPLAY_END_YEAR + 1)
    label = "Mean lifespan"
    slope, intercept = np.polyfit(
        recent["year"].to_numpy(dtype=float),
        recent[label].to_numpy(dtype=float),
        1,
    )
    ax.plot(
        line_years,
        slope * line_years + intercept,
        color=NAIVE_COLOR,
        linewidth=4.0,
        linestyle="-",
        alpha=1.0,
        zorder=2.5,
    )


def add_direct_contour_labels(ax: plt.Axes, hmd: pd.DataFrame) -> None:
    rows = hmd[hmd["year"] >= PLOT_START_YEAR].sort_values("year")
    label_year = 1960
    x_values = rows["year"].to_numpy(dtype=float)

    for label in CONTOURS:
        y_values = rows[f"{label} smoothed"].to_numpy(dtype=float)
        y_label = float(np.interp(label_year, x_values, y_values))
        display_label = "Mean" if label == "Mean lifespan" else label
        text = ax.text(
            label_year,
            y_label,
            display_label,
            color=LINE_COLOR,
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=6,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3.2, foreground="white", alpha=0.92),
                path_effects.Normal(),
            ]
        )


def historical_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        (summary["scenario"] == "historical")
        & (summary["year"] >= SR_START_YEAR)
    ].sort_values("year")


def fitted_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=LINE_COLOR, lw=2.8, alpha=0.42, label="HMD period contours"),
        Line2D(
            [0],
            [0],
            color=LINE_COLOR,
            lw=1.9,
            label=r"SR model: projected $X_c$ and fitted $m_{ex}$",
        ),
    ]


def projection_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=LINE_COLOR, lw=2.0, linestyle="-", alpha=1.0, label="Linear increase in robustness"),
        Line2D([0], [0], color=LINE_COLOR, lw=2.0, linestyle="--", alpha=1.0, label="Exponential increase in robustness"),
    ]


def main() -> None:
    hmd = load_hmd_contours()
    summary = load_sr_summary()
    plot_projection(hmd, summary)
    print(f"saved {OUTPUT_PNG}")
    print(f"saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
