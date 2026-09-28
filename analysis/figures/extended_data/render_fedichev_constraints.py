#!/usr/bin/env python3
"""Render Extended Data Fig. 4 from saved Fedichev-Gruber calculations."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT/'results/fedichev_gruber'
OUTPUT = ROOT/'Figures/ExtendedDataFigure4/ExtDataFig4.png'

FEDICHEV_PARAMS = (
    "beta_prime",
    "epsilon_0_init",
    "gamma",
    "beta",
    "g",
    "D0",
)

FEDICHEV_LABELS = {
    "beta_prime": r"$\beta'$ resilience decay",
    "epsilon_0_init": r"$\epsilon_0$ initial resilience",
    "gamma": r"$\gamma$ damage accumulation",
    "beta": r"$\beta$ damage coupling",
    "g": r"$g$ nonlinearity",
    "D0": r"$D_0$ noise",
}

FEDICHEV_COLORS = {
    "beta_prime": "#173A6A",
    "epsilon_0_init": "#0097A7",
    "gamma": "#2F80ED",
    "beta": "#007F5F",
    "g": "#6C63B5",
    "D0": "#E5A100",
}

SENOGENIC_PARAMS = ("beta_prime", "epsilon_0_init", "gamma", "beta", "g")

ROBUSTNESS_PARAMS = ("D0",)

MAX_X_LIMIT = (-0.35, 20.0)

FEDICHEV_MAX_Y_LIMIT = (118.0, 150.0)

SHAPE_X_LIMIT = (0.45, 1.58)

SHAPE_Y_LIMIT = (0.4, 1.62)

def configure_matplotlib() -> None:
    """Use the manuscript figure style used by the newer figure scripts."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

def marker_sizes(values: pd.Series, size_range: tuple[float, float]) -> np.ndarray:
    """Map curve values to readable marker areas."""
    numeric = values.to_numpy(dtype=float)
    low, high = np.nanmin(numeric), np.nanmax(numeric)
    if np.isclose(low, high):
        return np.full_like(numeric, np.mean(size_range), dtype=float)
    scaled = (numeric - low) / (high - low)
    return size_range[0] + (size_range[1] - size_range[0]) * scaled

def marker_size_for_factor(factor: float) -> float:
    """Map a multiplicative factor to the shape-panel marker area."""
    min_size, max_size = 16.0, 58.0
    scaled = (factor - 0.6) / (1.4 - 0.6)
    scaled = float(np.clip(scaled, 0.0, 1.0))
    return min_size + (max_size - min_size) * scaled

def style_axes(ax: plt.Axes) -> None:
    """Apply common axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(length=5.5, width=1.2, color="#222222", pad=3)
    ax.grid(False)
    ax.set_facecolor("white")

def plot_max_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_name: str,
    param_order: tuple[str, ...],
    colors: dict[str, str],
    y_limit: tuple[float, float],
) -> None:
    """Draw one maximum-lifespan panel."""
    subset = data[data["model"] == model_name]
    for param_name in param_order:
        rows = subset[subset["parameter"] == param_name].sort_values("variation_percent")
        if rows.empty:
            continue
        alpha = 1.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_ab" else 0.78
        linewidth = 3.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_ab" else 2.2
        ax.plot(
            rows["variation_percent"],
            rows["max_lifespan_smoothed"],
            color=colors[param_name],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )

    ax.set_xlim(*MAX_X_LIMIT)
    ax.set_ylim(*y_limit)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Parameter heterogeneity (CV, %)")
    ax.set_ylabel("Top 0.01% survivors [years]")
    style_axes(ax)

def plot_shape_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_name: str,
    param_order: tuple[str, ...],
    colors: dict[str, str],
) -> None:
    """Draw one steepness-longevity plane."""
    subset = data[data["model"] == model_name]
    ax.axhline(1.0, color="#B8B8B8", linewidth=1.2, linestyle=(0, (2.2, 2.2)), zorder=0)
    ax.axvline(1.0, color="#B8B8B8", linewidth=1.2, linestyle=(0, (2.2, 2.2)), zorder=0)

    for param_name in param_order:
        rows = subset[subset["parameter"] == param_name].sort_values("factor")
        if rows.empty:
            continue
        alpha = 1.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_intercept_slope" else 0.76
        linewidth = 3.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_intercept_slope" else 2.15
        ax.plot(
            rows["x_norm"],
            rows["y_norm"],
            color=colors[param_name],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            rows["x_norm"],
            rows["y_norm"],
            s=marker_sizes(rows["factor"], (16, 58)) if param_name != "makeham_m" else 28,
            color=colors[param_name],
            edgecolor="white",
            linewidth=0.45,
            alpha=alpha,
            zorder=3,
        )

    ax.set_xlim(*SHAPE_X_LIMIT)
    ax.set_ylim(*SHAPE_Y_LIMIT)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([0.4, 0.7, 1.0, 1.3, 1.6])
    ax.set_xlabel("Median lifespan relative to baseline")
    ax.set_ylabel("Steepness relative to baseline")
    style_axes(ax)

def build_factor_legend(ax: plt.Axes) -> None:
    """Add a Fig1-style legend explaining factor marker size."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=np.sqrt(marker_size_for_factor(0.6)),
            label="0.6x",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=np.sqrt(marker_size_for_factor(1.4)),
            label="1.4x",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        title="Factor change",
        loc="lower right",
        frameon=True,
        fontsize=9.5,
        title_fontsize=10.5,
        borderpad=0.5,
        labelspacing=0.42,
        handletextpad=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E3E3E3")
    legend.get_frame().set_alpha(0.92)

def draw_legend_heading(ax: plt.Axes, y: float, text: str) -> float:
    """Draw a manual legend heading and return the next y position."""
    ax.text(0.0, y, text, fontsize=12.0, fontweight="bold", ha="left", va="top")
    return y - 0.085

def draw_legend_item(ax: plt.Axes, y: float, color: str, label: str, linewidth: float = 3.0) -> float:
    """Draw one manual legend line item."""
    ax.add_line(Line2D([0.0, 0.15], [y - 0.014, y - 0.014], color=color, linewidth=linewidth))
    ax.text(0.19, y, label, fontsize=10.2, ha="left", va="top")
    return y - 0.073

def draw_fedichev_legend(ax: plt.Axes) -> None:
    """Draw the row legend for the Fedichev-Gruber panels."""
    ax.axis("off")
    y = 0.98
    y = draw_legend_heading(ax, y, "Fedichev-Gruber")
    y = draw_legend_heading(ax, y, "Senogenic parameters")
    for param_name in SENOGENIC_PARAMS:
        y = draw_legend_item(ax, y, FEDICHEV_COLORS[param_name], FEDICHEV_LABELS[param_name], linewidth=2.6)
    y -= 0.015
    y = draw_legend_heading(ax, y, "Robustness parameter")
    draw_legend_item(ax, y, FEDICHEV_COLORS["D0"], FEDICHEV_LABELS["D0"], linewidth=3.2)


def main():
    tail = pd.read_csv(DATA/'extreme_lifespan.csv')
    shape = pd.read_csv(DATA/'shape_response.csv')
    for frame in (tail, shape):
        if set(frame.model) != {'Fedichev-Gruber'} or set(frame.parameter) != set(FEDICHEV_PARAMS):
            raise ValueError('Incomplete or mixed-model source table.')
    configure_matplotlib()
    fig = plt.figure(figsize=(13.2, 4.5))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, .52], wspace=.28)
    a, b, legend = [fig.add_subplot(grid[0, i]) for i in range(3)]
    plot_max_panel(a, tail, 'Fedichev-Gruber', FEDICHEV_PARAMS,
                   FEDICHEV_COLORS, FEDICHEV_MAX_Y_LIMIT)
    plot_shape_panel(b, shape, 'Fedichev-Gruber', FEDICHEV_PARAMS, FEDICHEV_COLORS)
    a.set_title('Fedichev-Gruber: extreme lifespan', loc='left', pad=10)
    b.set_title('Fedichev-Gruber: shape response', loc='left', pad=10)
    for ax, label in ((a, 'a'), (b, 'b')):
        ax.text(-.18, 1.14, label, transform=ax.transAxes,
                fontsize=20, fontweight='normal', va='top')
    build_factor_legend(b)
    draw_fedichev_legend(legend)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(OUTPUT)


if __name__ == '__main__':
    main()
