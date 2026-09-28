"""Drawing helpers for saved historical FP results; no projection or fitting."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.lines import Line2D

from matplotlib.patches import Rectangle

from analysis.figures.steepness_longevity import response_plane as fig1e

ROBUSTNESS_PANEL_COLOR = fig1e.PARAM_COLORS["Xc"]

@dataclass(frozen=True)
class CountryConfig:
    key: str
    name: str

COUNTRIES = (
    CountryConfig(
        key="sweden",
        name="Sweden",
    ),
    CountryConfig(
        key="denmark",
        name="Denmark",
    ),
)

CONDITION_LABELS = {
    "with_extrinsic": "With extrinsic mortality",
    "extrinsic_removed": "Extrinsic mortality removed",
}

CONDITION_MARKERS = {
    "with_extrinsic": "^",
    "extrinsic_removed": "^",
}

HISTORICAL_CMAP = "Greys"

NEW_AXIS_LIMITS = {
    "with_extrinsic": ((0.5, 1.1), (0.3, 1.4)),
    "extrinsic_removed": ((0.5, 1.1), (0.3, 1.4)),
}

LEGEND_SEPARATOR_COLOR = "#9C9C9C"

def apply_fig4_style() -> None:
    """Use the same publication-safe style as the new Fig1E panel."""
    fig1e.apply_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

def draw_historical_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    condition: str,
    *,
    x_column: str,
    y_column: str,
    show_colorbar: bool,
):
    """Draw yearly country period coordinates on one steepness-longevity plane."""
    rows = data[data["condition"] == condition].copy()
    scatter = ax.scatter(
        rows[x_column],
        rows[y_column],
        c=rows["year"],
        cmap=HISTORICAL_CMAP,
        s=36,
        marker=CONDITION_MARKERS[condition],
        edgecolor="#111111",
        linewidth=0.45,
        alpha=0.92,
        zorder=20,
    )
    if show_colorbar:
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Year", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    return scatter

def draw_new_baseline(ax: plt.Axes) -> None:
    """Draw the new Fig1E SR response plane underlay."""
    data = fig1e.load_normalized_metrics()
    for param in fig1e.PLOT_PARAMS:
        fig1e.draw_parameter_curve(ax=ax, data=data, param=param)
    fig1e.draw_h_ext_curve(ax=ax, data=data)
    fig1e.finish_axes(ax)

def draw_vector_year_colorbar(ax: plt.Axes, scatter) -> None:
    """Draw a grayscale year colorbar as vector rectangles, not a raster image."""
    norm = scatter.norm
    cmap = scatter.cmap
    year_min, year_max = norm.vmin, norm.vmax
    edges = np.linspace(year_min, year_max, 120)

    for start, end in zip(edges[:-1], edges[1:]):
        midpoint = 0.5 * (start + end)
        ax.add_patch(
            Rectangle(
                (0.0, start),
                1.0,
                end - start,
                facecolor=cmap(norm(midpoint)),
                edgecolor="none",
                linewidth=0,
            )
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(year_min, year_max)
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Year", fontsize=15)
    ax.tick_params(axis="y", labelsize=13, width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#555555")

def build_shared_legend(ax: plt.Axes, *, country: CountryConfig, data: pd.DataFrame) -> None:
    """Add a compact model-plus-data legend to the first panel."""
    first_year = int(data["year"].min())
    last_year = int(data["year"].max())
    handles = [
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["eta"], lw=3, label=fig1e.PARAM_LABELS["eta"]),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["beta"], lw=3, label=fig1e.PARAM_LABELS["beta"]),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["Xc"], lw=3, label=fig1e.PARAM_LABELS["Xc"]),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["epsilon"], lw=3, label=fig1e.PARAM_LABELS["epsilon"]),
        Line2D([0], [0], color=LEGEND_SEPARATOR_COLOR, lw=1.4, label=" "),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["h_ext"], lw=3, label="Extrinsic mortality"),
        Line2D([], [], color="none", label=" "),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#B5B5B5",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            markersize=9,
            label=f"{country.name} period years ({first_year}-{last_year})",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=12,
        handlelength=2.0,
        labelspacing=0.38,
        borderpad=0.35,
    )
    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_fontweight("bold")

def remove_legend(ax: plt.Axes) -> None:
    """Remove whichever legend the underlay created."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
