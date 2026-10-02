#!/usr/bin/env python3
"""Make Supplementary Fig. 4: healthspan and morbidity under threshold shifts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from decimal import Decimal, ROUND_HALF_UP

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from analysis.quality_checks.artificial_survival_time import make_threshold_schematic as schematic
from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR


OUTPUT_DIR = FIGURES_DIR / "Supplementary"
PNG_PATH = OUTPUT_DIR / "SuppFig4.png"
SOURCE_DIR = RESULTS_DIR / "supplementary4_fp"

SCENARIOS = ("baseline", "xc_only", "proportional")
SCENARIO_COLORS = {"baseline": "#3A6EA5", "xc_only": "#D77A16", "proportional": "#227C59"}
STATE_COLORS = {"healthy": "#6BAA75", "sick": "#C65D4B", "dead": "#D6D6D6"}
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "xc_only": "Increase in death threshold only",
    "proportional": "Increase in death and disease thresholds together",
}
WRAPPED_SCENARIO_LABELS = {
    "baseline": "Baseline",
    "xc_only": "Increase in death\nthreshold only",
    "proportional": "Increase in death and disease\nthresholds together",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12.5,
            "axes.titlesize": 14.5,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.8,
            "ytick.labelsize": 12.8,
            "legend.fontsize": 12.0,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_schematic_row(fig: plt.Figure, outer_grid) -> list[plt.Axes]:
    grid = outer_grid.subgridspec(1, 3, wspace=0.30)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    scenarios = [
        {
            "title": "Baseline",
            "xd": schematic.BASE_XD,
            "xc": schematic.BASE_XC,
            "disease_age": 68,
            "death_age": 84,
            "reference_xd": False,
            "reference_xc": False,
            "arrows": [],
        },
        {
            "title": "Increase in death threshold only",
            "xd": schematic.BASE_XD,
            "xc": schematic.BASE_XC * schematic.THRESHOLD_FACTOR,
            "disease_age": 68,
            "death_age": 96,
            "reference_xd": False,
            "reference_xc": True,
            "arrows": [(58, schematic.BASE_XC, schematic.BASE_XC * schematic.THRESHOLD_FACTOR, "")],
        },
        {
            "title": "Increase in death and disease\nthreshold together",
            "xd": schematic.BASE_XD * schematic.THRESHOLD_FACTOR,
            "xc": schematic.BASE_XC * schematic.THRESHOLD_FACTOR,
            "disease_age": 80,
            "death_age": 96,
            "reference_xd": True,
            "reference_xc": True,
            "arrows": [
                (50, schematic.BASE_XD, schematic.BASE_XD * schematic.THRESHOLD_FACTOR, ""),
                (62, schematic.BASE_XC, schematic.BASE_XC * schematic.THRESHOLD_FACTOR, ""),
            ],
        },
    ]

    for index, (ax, scenario) in enumerate(zip(axes, scenarios)):
        schematic.draw_bands(ax, xd=scenario["xd"], xc=scenario["xc"])
        schematic.draw_reference_thresholds(
            ax,
            show_xd=scenario["reference_xd"],
            show_xc=scenario["reference_xc"],
        )
        schematic.draw_trajectory(
            ax,
            disease_age=scenario["disease_age"],
            death_age=scenario["death_age"],
            xd=scenario["xd"],
            xc=scenario["xc"],
            seed=20260521 + index,
        )
        for x, y0, y1, label in scenario["arrows"]:
            schematic.draw_shift_arrow(ax, x=x, y0=y0, y1=y1, label=label)
        schematic.draw_threshold_labels(ax, xd=scenario["xd"], xc=scenario["xc"])
        schematic.style_axis(ax)
        ax.set_xlabel(r"Age, $t$", labelpad=6)
        ax.set_title(scenario["title"], pad=8)
        ax.title.set_fontsize(13.5)
        for text in ax.texts:
            text.set_fontsize(min(text.get_fontsize() + 1.5, 15.0))

    axes[0].set_ylabel(r"$X(t)$")
    axes[0].yaxis.label.set_size(15.0)
    schematic.add_band_labels(axes[0], xd=schematic.BASE_XD, xc=schematic.BASE_XC)
    for text in axes[0].texts:
        text.set_fontsize(min(text.get_fontsize() + 1.5, 15.0))
    return axes


def draw_state_row(fig: plt.Figure, outer_grid, states: pd.DataFrame) -> list[plt.Axes]:
    grid = outer_grid.subgridspec(1, 3, wspace=0.24)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for ax, scenario in zip(axes, SCENARIOS):
        rows = states.loc[states.scenario == scenario]
        ages = rows.age.to_numpy()
        healthy = rows.healthy.to_numpy()
        sick = rows.sick.to_numpy()
        dead = rows.dead.to_numpy()

        ax.stackplot(
            ages,
            healthy,
            sick,
            dead,
            colors=[
                STATE_COLORS["healthy"],
                STATE_COLORS["sick"],
                STATE_COLORS["dead"],
            ],
            labels=["Healthy alive", "Sick alive", "Dead"],
            linewidth=0,
        )
        ax.set_title(WRAPPED_SCENARIO_LABELS[scenario])
        ax.set_xlabel("Age [years]")
        ax.set_xlim(55, 125)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction of cohort")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return axes


def draw_bar_row(fig: plt.Figure, outer_grid, summary: pd.DataFrame) -> plt.Axes:
    ax = fig.add_subplot(outer_grid)
    x = np.arange(len(SCENARIOS))
    colors = [SCENARIO_COLORS[scenario] for scenario in SCENARIOS]
    labels = [WRAPPED_SCENARIO_LABELS[scenario] for scenario in SCENARIOS]

    for index, scenario in enumerate(SCENARIOS):
        p5, q1, median, q3, p95 = 100*summary.loc[scenario,
            ['p05','p25','p50','p75','p95']].to_numpy(dtype=float)
        ax.plot([index,index],[p5,p95],color=colors[index],lw=2.2,zorder=2)
        ax.plot([index,index],[q1,q3],color='#20252B',lw=6,
                solid_capstyle='round',zorder=3)
        ax.scatter(index,median,s=125,color='#20252B',edgecolor='white',
                   linewidth=1.2,zorder=4)
        fraction = summary.loc[scenario, 'median_sick_life_fraction']
        label = (100*Decimal(str(fraction))).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)
        ax.text(index+.13,median,f'{label}%',ha='left',va='center',fontsize=13)
    ax.set_ylabel('Sick span (% of lifespan)')
    ax.text(.98,.98,'Dot: median; thick: 25–75%; thin: 5–95%',
            transform=ax.transAxes,ha='right',va='top',fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-.4,2.6)
    ax.set_ylim(0,100*summary.p95.max()*1.24)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str) -> None:
    bbox = ax.get_position()
    fig.text(
        0.025,
        bbox.y1,
        label.lower(),
        fontsize=28,
        fontweight="normal",
        va="top",
        ha="left",
    )


def make_composite(states: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:

    fig = plt.figure(figsize=(10.8, 10.8))
    outer_grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.92, 1.0, 0.70],
        hspace=0.58,
    )

    schematic_axes = draw_schematic_row(fig, outer_grid[0])
    state_axes = draw_state_row(fig, outer_grid[1], states=states)
    bar_ax = draw_bar_row(fig, outer_grid[2], summary=summary)

    add_panel_label(fig, schematic_axes[0], "a")
    add_panel_label(fig, state_axes[0], "b")
    add_panel_label(fig, bar_ax, "c")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_sources(source: Path):
    required = [source / name for name in ("manifest.json", "states.csv", "summary.csv", "sick_fraction.csv")]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(
            "Joint-FP outputs are missing. No simulation will be started. "
            "See docs/figure_methods/healthspan_fp.md.")
    manifest = json.loads(required[0].read_text())
    if manifest.get("backend") != "tagged finite-volume first passage":
        raise ValueError("Expected joint FP source, not a legacy trajectory cache")
    states = pd.read_csv(required[1])
    summary = pd.read_csv(required[2]).set_index("scenario")
    fractions = pd.read_csv(required[3])
    if (set(states.scenario) != set(SCENARIOS) or set(summary.index) != set(SCENARIOS)
            or set(fractions.scenario) != set(SCENARIOS)):
        raise ValueError("Incomplete scenario inventory")
    if not summary.index.is_unique or len(summary) != 3:
        raise ValueError("Duplicate summary scenario")
    for scenario in SCENARIOS:
        rows = states.loc[states.scenario == scenario]
        values = rows[["healthy", "sick", "dead"]].to_numpy()
        if (not np.isfinite(values).all() or (values < -1e-12).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-9, rtol=0)
                or not (np.diff(rows.age) > 0).all()
                or rows.age.iloc[0] != 0 or rows.age.iloc[-1] != manifest["horizon"]):
            raise ValueError(f"Invalid probability or age grid for {scenario}")
        median = summary.loc[scenario, "median_sick_life_fraction"]
        if not np.isfinite(median) or not 0 <= median <= 1:
            raise ValueError(f"Invalid median for {scenario}")
        distribution = fractions.loc[fractions.scenario == scenario]
        mass = distribution.probability.to_numpy()
        ratios = distribution.fraction.to_numpy()
        if (not np.isfinite(mass).all() or (mass < 0).any()
                or not np.isfinite(ratios).all() or not (np.diff(ratios) > 0).all()
                or ratios[0] != 0 or ratios[-1] != 1
                or not np.isclose(mass.sum(), 1, atol=1e-9, rtol=0)):
            raise ValueError(f"Invalid sick-life distribution for {scenario}")
        quantiles = ratios[np.searchsorted(np.cumsum(mass), [.05,.25,.5,.75,.95])]
        summary.loc[scenario,['p05','p25','p50','p75','p95']] = quantiles
        reconstructed = quantiles[2]
        if not np.isclose(median, reconstructed, atol=1e-12, rtol=0):
            raise ValueError(f"Median disagrees with joint distribution for {scenario}")
    return states, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=PNG_PATH)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png":
        raise ValueError("Output must be PNG")
    states, summary = load_sources(args.source_dir)
    configure_matplotlib()
    make_composite(states, summary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
