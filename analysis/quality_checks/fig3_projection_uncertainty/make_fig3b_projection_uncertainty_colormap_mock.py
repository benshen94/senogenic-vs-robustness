#!/usr/bin/env python3
"""Mock Fig. 3B color strategies for projection-uncertainty ribbons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "fig3_exposure_projection"
CURRENT_CURVES_PATH = RESULTS_DIR / "fig3_panel_b_xc_factor_curves_fit_ci.csv"
FULL_FACTOR_CURVES_PATH = RESULTS_DIR / "projected_xc_age_gain_curves.csv"
PROJECTIONS_PATH = RESULTS_DIR / "exposure_xc_projection.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

MOCK_GRID_PATH = PLOTS_DIR / "fig3b_projection_uncertainty_colormap_mock.png"
STANDALONE_PATH = PLOTS_DIR / "fig3b_projection_uncertainty_colormap_mock_best.png"
SOURCE_DATA_PATH = RESULTS_DIR / "fig3b_projection_uncertainty_colormap_mock_data.csv"

PANEL_FACTORS = np.round(np.arange(0.80, 1.201, 0.05), 2)
NEIGHBOR_COUNT = 5


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached central curves and exposure projection intervals."""
    current = pd.read_csv(CURRENT_CURVES_PATH)
    full_grid = pd.read_csv(FULL_FACTOR_CURVES_PATH)
    projections = pd.read_csv(PROJECTIONS_PATH)
    return current, full_grid, projections


def local_projection_factor_intervals(projections: pd.DataFrame, full_grid: pd.DataFrame) -> pd.DataFrame:
    """Estimate local projection uncertainty in Xc factor around each displayed line."""
    required = {"xc_factor", "xc_factor_low", "xc_factor_high"}
    if not required.issubset(projections.columns):
        raise RuntimeError(f"Projection table must include columns: {sorted(required)}")

    factor_min = float(full_grid["factor"].min())
    factor_max = float(full_grid["factor"].max())
    usable = projections.dropna(subset=list(required)).copy()

    rows = []
    for factor in PANEL_FACTORS:
        local = usable.assign(distance=(usable["xc_factor"] - factor).abs())
        local = local.nsmallest(NEIGHBOR_COUNT, "distance")
        low_delta = float(np.median(local["xc_factor"] - local["xc_factor_low"]))
        high_delta = float(np.median(local["xc_factor_high"] - local["xc_factor"]))
        rows.append(
            {
                "factor": float(factor),
                "factor_projection_low": max(float(factor) - low_delta, factor_min),
                "factor_projection_high": min(float(factor) + high_delta, factor_max),
                "factor_projection_low_delta": low_delta,
                "factor_projection_high_delta": high_delta,
            }
        )

    return pd.DataFrame(rows)


def interpolate_gain(full_grid: pd.DataFrame, factor: float, age: int) -> float:
    """Interpolate gain by Xc factor for one age from the full cached grid."""
    rows = full_grid[full_grid["age"] == age].sort_values("factor")
    return float(np.interp(factor, rows["factor"], rows["gain_years"]))


def build_projection_ribbons(
    current: pd.DataFrame,
    full_grid: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-factor projection-uncertainty ribbons."""
    intervals = local_projection_factor_intervals(projections, full_grid)
    rows = []
    for interval in intervals.itertuples(index=False):
        central = current[np.isclose(current["factor"], interval.factor)].sort_values("age")
        for row in central.itertuples(index=False):
            low_gain = interpolate_gain(full_grid, interval.factor_projection_low, int(row.age))
            high_gain = interpolate_gain(full_grid, interval.factor_projection_high, int(row.age))
            rows.append(
                {
                    "factor": float(interval.factor),
                    "age": int(row.age),
                    "gain_years": float(row.gain_years),
                    "gain_projection_low": min(low_gain, high_gain),
                    "gain_projection_high": max(low_gain, high_gain),
                    "factor_projection_low": float(interval.factor_projection_low),
                    "factor_projection_high": float(interval.factor_projection_high),
                }
            )

    ribbons = pd.DataFrame(rows)
    ribbons.to_csv(SOURCE_DATA_PATH, index=False)
    return ribbons


def cmap_colors(name: str, n: int) -> list:
    """Return readable colors from a named Matplotlib colormap."""
    values = np.linspace(0.08, 0.92, n)
    return [plt.get_cmap(name)(value) for value in values]


def categorical_colors(n: int) -> list:
    """Return a high-contrast categorical palette."""
    base = list(plt.get_cmap("tab10").colors)
    return [base[index % len(base)] for index in range(n)]


def variant_colors(name: str, n: int) -> list:
    """Map a variant label to line colors."""
    if name == "Viridis":
        return cmap_colors("viridis", n)
    if name == "Cividis":
        return cmap_colors("cividis", n)
    if name == "Coolwarm":
        return cmap_colors("coolwarm", n)
    if name == "Tab10":
        return categorical_colors(n)
    raise ValueError(f"Unknown variant: {name}")


def style_axis(ax: plt.Axes, *, show_ylabel: bool = True) -> None:
    """Apply shared panel-B styling."""
    ax.axhline(0, color="#8F8F8F", linestyle="--", linewidth=1.1, alpha=0.70, zorder=1)
    ax.set_xlim(60, 103)
    ax.set_ylim(-9.5, 9.5)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(np.arange(-8, 9, 4))
    ax.set_xlabel("Age [years]", fontsize=13)
    if show_ylabel:
        ax.set_ylabel("Median extra years gained", fontsize=13)
    ax.tick_params(length=5.5, width=1.1, color="#222222", labelsize=11)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)


def draw_variant(
    ax: plt.Axes,
    ribbons: pd.DataFrame,
    variant: str,
    *,
    show_ylabel: bool,
    show_legend: bool,
) -> None:
    """Draw one color strategy with matching projection-uncertainty ribbons."""
    factors = np.sort(ribbons["factor"].unique())
    colors = variant_colors(variant, len(factors))

    for factor, color in zip(factors, colors):
        group = ribbons[np.isclose(ribbons["factor"], factor)].sort_values("age")
        ax.fill_between(
            group["age"],
            group["gain_projection_low"],
            group["gain_projection_high"],
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            group["age"],
            group["gain_projection_low"],
            color=color,
            linewidth=0.85,
            alpha=0.33,
            zorder=2,
        )
        ax.plot(
            group["age"],
            group["gain_projection_high"],
            color=color,
            linewidth=0.85,
            alpha=0.33,
            zorder=2,
        )
        ax.plot(
            group["age"],
            group["gain_years"],
            color=color,
            linewidth=2.35,
            zorder=3,
        )

    ax.set_title(variant, fontsize=15, pad=8)
    style_axis(ax, show_ylabel=show_ylabel)

    if not show_legend:
        return

    line_handles = [
        Line2D([0], [0], color=color, linewidth=2.35, label=f"{factor:.2f}")
        for factor, color in zip(factors, colors)
    ]
    band_handle = Patch(facecolor="0.55", alpha=0.18, edgecolor="none", label="Projection uncertainty")
    legend = ax.legend(
        handles=[band_handle, *line_handles],
        title="Xc factor",
        frameon=False,
        fontsize=8.3,
        title_fontsize=9.5,
        ncol=2,
        loc="upper right",
        handlelength=1.7,
        columnspacing=0.9,
        labelspacing=0.25,
    )
    ax.add_artist(legend)


def save_mock_grid(ribbons: pd.DataFrame) -> None:
    """Save a multi-panel comparison of projection-uncertainty color strategies."""
    variants = ["Viridis", "Cividis", "Coolwarm", "Tab10"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.9), constrained_layout=True)
    for index, (ax, variant) in enumerate(zip(axes.flat, variants)):
        draw_variant(
            ax,
            ribbons,
            variant,
            show_ylabel=index % 2 == 0,
            show_legend=index == 3,
        )

    fig.suptitle(
        "Mock panel B: per-line projection-uncertainty ribbons",
        fontsize=18,
        y=1.03,
    )
    fig.savefig(MOCK_GRID_PATH, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_standalone_candidate(ribbons: pd.DataFrame) -> None:
    """Save the strongest-looking standalone option for closer review."""
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    draw_variant(ax, ribbons, "Viridis", show_ylabel=True, show_legend=True)
    ax.set_title("Extra years from changes\nin robustness (model)", fontsize=17, pad=12)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.82)
    fig.savefig(STANDALONE_PATH, dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Render projection-uncertainty mock outputs."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    current, full_grid, projections = load_inputs()
    ribbons = build_projection_ribbons(current, full_grid, projections)
    save_mock_grid(ribbons)
    save_standalone_candidate(ribbons)
    print(f"Saved {MOCK_GRID_PATH}")
    print(f"Saved {STANDALONE_PATH}")
    print(f"Saved {SOURCE_DATA_PATH}")


if __name__ == "__main__":
    main()
