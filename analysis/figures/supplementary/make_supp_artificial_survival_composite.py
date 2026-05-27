#!/usr/bin/env python3
"""Make the artificial-survival-time supplementary composite figure."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from analysis.quality_checks.artificial_survival_time import make_artificial_survival_time_exploration as survival
from analysis.quality_checks.artificial_survival_time import make_threshold_schematic as schematic
from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR


OUTPUT_DIR = FIGURES_DIR / "Supplementary"
PNG_PATH = OUTPUT_DIR / "supp_artificial_survival_composite.png"
PDF_PATH = OUTPUT_DIR / "supp_artificial_survival_composite.pdf"
INDEX_PATH = RESULTS_DIR / "index" / "outputs.csv"

DATE = "2026-05-21"
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


def draw_state_row(fig: plt.Figure, outer_grid, events: dict[str, dict[str, np.ndarray]]) -> list[plt.Axes]:
    grid = outer_grid.subgridspec(1, 3, wspace=0.24)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    state_rows = survival.state_composition(events)
    by_scenario = {scenario.scenario_id: [] for scenario in survival.SCENARIOS}

    for row in state_rows:
        by_scenario[row["scenario_id"]].append(row)

    for ax, scenario in zip(axes, survival.SCENARIOS):
        rows = by_scenario[scenario.scenario_id]
        ages = np.array([row["age"] for row in rows], dtype=float)
        healthy = np.array([row["healthy_alive_fraction"] for row in rows], dtype=float)
        sick = np.array([row["sick_alive_fraction"] for row in rows], dtype=float)
        dead = np.array([row["dead_fraction"] for row in rows], dtype=float)

        ax.stackplot(
            ages,
            healthy,
            sick,
            dead,
            colors=[
                survival.STATE_COLORS["healthy"],
                survival.STATE_COLORS["sick"],
                survival.STATE_COLORS["dead"],
            ],
            labels=["Healthy alive", "Sick alive", "Dead"],
            linewidth=0,
        )
        ax.set_title(WRAPPED_SCENARIO_LABELS[scenario.scenario_id])
        ax.set_xlabel("Age [years]")
        ax.set_xlim(55, 125)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction of cohort")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[2].legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    return axes


def sick_life_percent(scenario_events: dict[str, np.ndarray]) -> np.ndarray:
    lifespan = np.where(
        np.isfinite(scenario_events["death_time"]),
        scenario_events["death_time"],
        survival.TMAX,
    )
    valid = lifespan > 0
    return 100.0 * scenario_events["sickspan"][valid] / lifespan[valid]


def draw_bar_row(fig: plt.Figure, outer_grid, events: dict[str, dict[str, np.ndarray]]) -> plt.Axes:
    ax = fig.add_subplot(outer_grid)
    medians = []

    for scenario in survival.SCENARIOS:
        scenario_events = events[scenario.scenario_id]
        medians.append(float(np.median(sick_life_percent(scenario_events))))

    x = np.arange(len(survival.SCENARIOS))
    colors = [survival.SCENARIO_COLORS[scenario.scenario_id] for scenario in survival.SCENARIOS]
    labels = [WRAPPED_SCENARIO_LABELS[scenario.scenario_id] for scenario in survival.SCENARIOS]

    ax.bar(
        x,
        medians,
        color=colors,
        width=0.52,
    )
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.04,
        "Median sick span (% of lifespan)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(medians) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for index, value in enumerate(medians):
        ax.text(index, value + 0.55, f"{value:.1f}%", ha="center", va="bottom", fontsize=13.0)
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


def make_composite() -> None:
    events = survival.run_or_load_events(n=survival.DEFAULT_N, parallel=True, force=False)

    fig = plt.figure(figsize=(10.8, 9.8))
    outer_grid = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.92, 1.0, 0.70],
        hspace=0.40,
    )

    schematic_axes = draw_schematic_row(fig, outer_grid[0])
    state_axes = draw_state_row(fig, outer_grid[1], events=events)
    bar_ax = draw_bar_row(fig, outer_grid[2], events=events)

    add_panel_label(fig, schematic_axes[0], "a")
    add_panel_label(fig, state_axes[0], "b")
    add_panel_label(fig, bar_ax, "c")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def update_output_index() -> None:
    if not INDEX_PATH.exists():
        return

    rows = [
        {
            "date": DATE,
            "task": "supp_artificial_survival_composite",
            "artifact_type": "figure",
            "path": str(PNG_PATH.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": str(survival.EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "description": "PNG preview of artificial-survival-time supplementary composite.",
            "notes": "Rows show schematic thresholds, age-specific state composition, and median sick span as a percentage of lifespan.",
        },
        {
            "date": DATE,
            "task": "supp_artificial_survival_composite",
            "artifact_type": "figure",
            "path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": str(survival.EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "description": "Vector PDF of artificial-survival-time supplementary composite.",
            "notes": "Rows show schematic thresholds, age-specific state composition, and median sick span as a percentage of lifespan.",
        },
    ]

    with INDEX_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        existing_rows = list(reader)

    if fieldnames is None:
        return

    rows_by_path = {row["path"]: row for row in rows}
    seen_paths = set()
    updated_rows = []

    for existing_row in existing_rows:
        replacement = rows_by_path.get(existing_row["path"])
        if replacement is None:
            updated_rows.append(existing_row)
            continue

        updated_rows.append(replacement)
        seen_paths.add(existing_row["path"])

    for row in rows:
        if row["path"] not in seen_paths:
            updated_rows.append(row)

    with INDEX_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)


def main() -> None:
    configure_matplotlib()
    make_composite()
    update_output_index()
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
