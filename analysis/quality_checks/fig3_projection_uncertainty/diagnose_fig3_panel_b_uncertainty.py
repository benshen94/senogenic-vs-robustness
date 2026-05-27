#!/usr/bin/env python3
"""Diagnose uncertainty choices for Fig. 3 panel B.

This script only makes exploratory diagnostics. It does not modify the main
figure outputs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "fig3_exposure_projection"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

CURRENT_CURVES = RESULTS_DIR / "fig3_panel_b_xc_factor_curves_fit_ci.csv"
RAW_CURVES = RESULTS_DIR / "fig3_panel_b_xc_factor_curves_fit_ci_raw.csv"
OLD_FACTOR_CURVES = RESULTS_DIR / "projected_xc_age_gain_curves.csv"
OLD_GROUP_CURVES = RESULTS_DIR / "projected_xc_panel_b_group_curves.csv"
PROJECTIONS = RESULTS_DIR / "exposure_xc_projection.csv"


SCENARIO_COLUMNS = [
    "central",
    "Xc_ci_lower",
    "Xc_ci_upper",
    "xc_std_frac_ci_lower",
    "xc_std_frac_ci_upper",
]


def load_raw_gains() -> pd.DataFrame:
    """Load raw scenario medians and convert them to gain curves."""
    raw = pd.read_csv(RAW_CURVES)
    baseline = raw[np.isclose(raw["factor"], 1.0)].copy()
    baseline = baseline.rename(columns={"remaining_median": "baseline_remaining"})
    baseline = baseline[["scenario_id", "age", "baseline_remaining"]]

    gains = raw.merge(baseline, on=["scenario_id", "age"], how="left")
    gains["gain_years"] = gains["remaining_median"] - gains["baseline_remaining"]
    return gains


def make_endpoint_summary(raw_gains: pd.DataFrame) -> pd.DataFrame:
    """Return central curves plus endpoint-derived envelope widths."""
    table = raw_gains.pivot_table(
        index=["factor", "age"],
        columns="scenario_id",
        values="gain_years",
    )
    table = table[SCENARIO_COLUMNS].reset_index()

    endpoint_values = table[SCENARIO_COLUMNS]
    table["current_low"] = endpoint_values.min(axis=1)
    table["current_high"] = endpoint_values.max(axis=1)
    table["half_xc"] = (table["Xc_ci_upper"] - table["Xc_ci_lower"]).abs() / 2.0
    table["half_std"] = (
        table["xc_std_frac_ci_upper"] - table["xc_std_frac_ci_lower"]
    ).abs() / 2.0
    table["delta_half"] = np.sqrt(table["half_xc"] ** 2 + table["half_std"] ** 2)
    table["delta_low"] = table["central"] - table["delta_half"]
    table["delta_high"] = table["central"] + table["delta_half"]
    return table


def style_axis(ax: plt.Axes) -> None:
    """Apply a simple diagnostic style."""
    ax.axhline(0, color="0.35", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlim(60, 100)
    ax.set_xlabel("Age [years]")
    ax.set_ylabel("Median extra years gained")
    ax.tick_params(labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_current_mixed(ax: plt.Axes, current: pd.DataFrame) -> None:
    """Draw the current fit band plus projection-boundary guides."""
    cmap = plt.get_cmap("Purples")
    factors = np.sort(current["factor"].unique())
    colors = np.linspace(0.35, 0.95, len(factors))
    for factor, color_value in zip(factors, colors):
        group = current[current["factor"] == factor].sort_values("age")
        color = cmap(color_value)
        ax.fill_between(
            group["age"],
            group["gain_fit_low"],
            group["gain_fit_high"],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(group["age"], group["gain_years"], color=color, linewidth=1.8)
        ax.plot(
            group["age"],
            group["gain_projection_low"],
            color=color,
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            alpha=0.25,
        )
        ax.plot(
            group["age"],
            group["gain_projection_high"],
            color=color,
            linewidth=0.9,
            linestyle=(0, (3, 3)),
            alpha=0.25,
        )
    ax.set_title("Current mixed display", fontsize=11)
    style_axis(ax)


def draw_fit_endpoint_envelope(ax: plt.Axes, endpoint: pd.DataFrame) -> None:
    """Draw the current min/max endpoint envelope alone."""
    cmap = plt.get_cmap("Purples")
    factors = np.sort(endpoint["factor"].unique())
    colors = np.linspace(0.35, 0.95, len(factors))
    for factor, color_value in zip(factors, colors):
        group = endpoint[endpoint["factor"] == factor].sort_values("age")
        color = cmap(color_value)
        ax.fill_between(
            group["age"],
            group["current_low"],
            group["current_high"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(group["age"], group["central"], color=color, linewidth=1.8)
    ax.set_title("Endpoint min/max fit envelope", fontsize=11)
    style_axis(ax)


def draw_delta_method_envelope(ax: plt.Axes, endpoint: pd.DataFrame) -> None:
    """Draw a symmetric local interval estimated from endpoint half-differences."""
    cmap = plt.get_cmap("Purples")
    factors = np.sort(endpoint["factor"].unique())
    colors = np.linspace(0.35, 0.95, len(factors))
    for factor, color_value in zip(factors, colors):
        group = endpoint[endpoint["factor"] == factor].sort_values("age")
        color = cmap(color_value)
        ax.fill_between(
            group["age"],
            group["delta_low"],
            group["delta_high"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(group["age"], group["central"], color=color, linewidth=1.8)
    ax.set_title("Symmetric local fit interval", fontsize=11)
    style_axis(ax)


def draw_projection_groups(ax: plt.Axes, groups: pd.DataFrame) -> None:
    """Draw selected exposure-projection curves and their broad bands."""
    selected = [
        "Q1 income",
        "No Activity",
        "Poor",
        "Some Activity",
        "0-1 drink/day",
        "Q4 income",
    ]
    subset = groups[groups["label"].isin(selected)].copy()
    subset["label"] = pd.Categorical(subset["label"], selected, ordered=True)
    cmap = plt.get_cmap("viridis")
    labels = [label for label in selected if label in set(subset["label"].astype(str))]
    for index, label in enumerate(labels):
        group = subset[subset["label"].astype(str) == label].sort_values("age")
        color = cmap(index / max(len(labels) - 1, 1))
        ax.fill_between(
            group["age"],
            group["gain_low"],
            group["gain_high"],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
        ax.plot(group["age"], group["gain_years"], color=color, linewidth=1.7, label=label)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("Selected exposure-projection bands", fontsize=11)
    style_axis(ax)


def draw_projection_envelope(ax: plt.Axes, groups: pd.DataFrame, factors: pd.DataFrame) -> None:
    """Draw a broad projection envelope behind the deterministic factor family."""
    envelope = groups.groupby("age", as_index=False).agg(
        gain_low=("gain_low", "min"),
        gain_high=("gain_high", "max"),
    )
    ax.fill_between(
        envelope["age"],
        envelope["gain_low"],
        envelope["gain_high"],
        color="0.75",
        alpha=0.28,
        linewidth=0,
        label="Exposure-projection range",
    )
    cmap = plt.get_cmap("Purples")
    factor_subset = factors[factors["factor"].between(0.8, 1.2)].copy()
    factor_values = np.sort(factor_subset["factor"].unique())
    colors = np.linspace(0.35, 0.95, len(factor_values))
    for factor, color_value in zip(factor_values, colors):
        group = factor_subset[np.isclose(factor_subset["factor"], factor)].sort_values("age")
        ax.plot(group["age"], group["gain_years"], color=cmap(color_value), linewidth=1.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Broad projection range as context", fontsize=11)
    style_axis(ax)


def draw_width_diagnostics(ax: plt.Axes, endpoint: pd.DataFrame) -> None:
    """Show why the current endpoint envelope is visually one-sided."""
    summary = endpoint.copy()
    summary["upper_width"] = summary["current_high"] - summary["central"]
    summary["lower_width"] = summary["central"] - summary["current_low"]
    by_factor = summary.groupby("factor", as_index=False).agg(
        lower_width=("lower_width", "mean"),
        upper_width=("upper_width", "mean"),
        symmetric_half=("delta_half", "mean"),
    )
    ax.plot(by_factor["factor"], by_factor["lower_width"], "o-", label="Lower endpoint width")
    ax.plot(by_factor["factor"], by_factor["upper_width"], "o-", label="Upper endpoint width")
    ax.plot(by_factor["factor"], by_factor["symmetric_half"], "o-", label="Symmetric half-width")
    ax.axhline(0, color="0.5", linewidth=0.8)
    ax.set_xlabel("Xc factor")
    ax.set_ylabel("Mean width [years]")
    ax.set_title("Width asymmetry by factor", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def projection_factor_intervals(
    projections: pd.DataFrame,
    factors: np.ndarray,
    *,
    clip_to: tuple[float, float] | None,
) -> pd.DataFrame:
    """Build local projection-factor intervals with optional clipping."""
    usable = projections.dropna(subset=["xc_factor", "xc_factor_low", "xc_factor_high"]).copy()
    rows = []
    for factor in factors:
        local = usable.assign(distance=(usable["xc_factor"] - factor).abs())
        local = local.nsmallest(5, "distance")

        low_delta = float(np.median(local["xc_factor"] - local["xc_factor_low"]))
        high_delta = float(np.median(local["xc_factor_high"] - local["xc_factor"]))
        low = float(factor) - low_delta
        high = float(factor) + high_delta
        if clip_to is not None:
            low = max(low, clip_to[0])
            high = min(high, clip_to[1])

        rows.append({"factor": float(factor), "low": low, "high": high})
    return pd.DataFrame(rows)


def interpolate_factor_gain(factor_curves: pd.DataFrame, factor: float, age: int) -> float:
    """Interpolate central gain by factor for one age."""
    group = factor_curves[factor_curves["age"] == age].sort_values("factor")
    return float(np.interp(factor, group["factor"], group["gain_years"]))


def projection_boundary_curves(
    intervals: pd.DataFrame,
    factor_curves: pd.DataFrame,
) -> pd.DataFrame:
    """Convert factor intervals to gain-boundary curves using central factor runs."""
    rows = []
    for interval in intervals.itertuples(index=False):
        for age in sorted(factor_curves["age"].unique()):
            low_gain = interpolate_factor_gain(factor_curves, interval.low, int(age))
            high_gain = interpolate_factor_gain(factor_curves, interval.high, int(age))
            rows.append(
                {
                    "factor": interval.factor,
                    "age": int(age),
                    "gain_low": min(low_gain, high_gain),
                    "gain_high": max(low_gain, high_gain),
                }
            )
    return pd.DataFrame(rows)


def draw_projection_clipping_comparison(
    axes: np.ndarray,
    projections: pd.DataFrame,
    factor_curves: pd.DataFrame,
) -> None:
    """Show the visual effect of clipping projection intervals to the line grid."""
    factors = np.round(np.arange(0.8, 1.201, 0.05), 2)
    clipped = projection_factor_intervals(projections, factors, clip_to=(0.8, 1.2))
    unclipped = projection_factor_intervals(projections, factors, clip_to=(0.6, 1.4))
    clipped_curves = projection_boundary_curves(clipped, factor_curves)
    unclipped_curves = projection_boundary_curves(unclipped, factor_curves)

    for ax, intervals, boundaries, title in [
        (axes[0], clipped, clipped_curves, "Clipped to 0.80-1.20"),
        (axes[1], unclipped, unclipped_curves, "Using full 0.60-1.40 central grid"),
    ]:
        cmap = plt.get_cmap("Purples")
        colors = np.linspace(0.35, 0.95, len(factors))
        for factor, color_value in zip(factors, colors):
            central = factor_curves[np.isclose(factor_curves["factor"], factor)].sort_values("age")
            boundary = boundaries[np.isclose(boundaries["factor"], factor)].sort_values("age")
            color = cmap(color_value)
            ax.plot(central["age"], central["gain_years"], color=color, linewidth=1.7)
            ax.plot(
                boundary["age"],
                boundary["gain_low"],
                color=color,
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                alpha=0.4,
            )
            ax.plot(
                boundary["age"],
                boundary["gain_high"],
                color=color,
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                alpha=0.4,
            )
        ax.set_title(title, fontsize=11)
        style_axis(ax)

    table_ax = axes[2]
    table_ax.axis("off")
    display = unclipped.rename(columns={"low": "unclipped low", "high": "unclipped high"})
    display["clipped low"] = clipped["low"]
    display["clipped high"] = clipped["high"]
    display = display[["factor", "clipped low", "clipped high", "unclipped low", "unclipped high"]]
    table_ax.table(
        cellText=np.round(display.values, 3),
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
    )
    table_ax.set_title("Projection factor intervals", fontsize=11)


def main() -> None:
    """Make diagnostic plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    current = pd.read_csv(CURRENT_CURVES)
    raw_gains = load_raw_gains()
    endpoint = make_endpoint_summary(raw_gains)
    factor_curves = pd.read_csv(OLD_FACTOR_CURVES)
    group_curves = pd.read_csv(OLD_GROUP_CURVES)
    projections = pd.read_csv(PROJECTIONS)

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    draw_current_mixed(axes[0, 0], current)
    draw_fit_endpoint_envelope(axes[0, 1], endpoint)
    draw_delta_method_envelope(axes[0, 2], endpoint)
    draw_projection_groups(axes[1, 0], group_curves)
    draw_projection_envelope(axes[1, 1], group_curves, factor_curves)
    draw_width_diagnostics(axes[1, 2], endpoint)

    output = PLOTS_DIR / "fig3b_uncertainty_diagnostics.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5), constrained_layout=True)
    draw_projection_clipping_comparison(axes, projections, factor_curves)
    output = PLOTS_DIR / "fig3b_projection_clipping_diagnostic.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
