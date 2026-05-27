#!/usr/bin/env python3
"""Explore disease-threshold and critical-threshold SR scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR


EXPLORATION_DIR = Path(__file__).resolve().parent
PLOTS_DIR = EXPLORATION_DIR / "plots"
REPORT_PATH = EXPLORATION_DIR / "README.md"

CACHE_DIR = RESULTS_DIR / "cache" / "simulations" / "artificial_survival_time"
EVENTS_CACHE_PATH = CACHE_DIR / "matched_sweden2019_event_times.npz"
METADATA_PATH = CACHE_DIR / "matched_sweden2019_metadata.json"
SUMMARY_PATH = RESULTS_DIR / "tables" / "artificial_survival_time_summary.csv"
STATE_PATH = RESULTS_DIR / "tables" / "artificial_survival_time_state_composition.csv"
INDEX_PATH = RESULTS_DIR / "index" / "outputs.csv"
SUPP_OUTPUT_DIR = FIGURES_DIR / "Supplementary"
SUPP_FIGURE_PATH = SUPP_OUTPUT_DIR / "supp_artificial_survival_time_xdisease_075.png"

RUN_LABEL = "sweden2019_artificial_survival_time_xdisease"
DATE = "2026-05-20"

BASELINE = {
    "eta": 0.5868368257640714,
    "beta": 57.87173772073557,
    "kappa": 0.5,
    "epsilon": 49.718659304628446,
    "Xc": 21.74056340066893,
    "xc_std_frac": 0.14142135623730953,
}

DISEASE_FRACTION = 0.75
XC_FACTOR = 1.2
TMAX = 160.0
DT = 0.05
SAVE_TIMES = TMAX
BASE_SEED = 20260520
DEFAULT_N = 80_000
AGE_GRID = np.arange(0.0, 141.0, 2.0)
SICKSPAN_GRID = np.arange(0.0, 61.0, 1.0)

SCENARIO_COLORS = {
    "baseline": "#3A6EA5",
    "xc_only": "#D77A16",
    "proportional": "#227C59",
}

STATE_COLORS = {
    "healthy": "#6BAA75",
    "sick": "#C65D4B",
    "dead": "#D6D6D6",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    label: str
    short_label: str
    xc_factor: float
    xdisease_factor: float


SCENARIOS = (
    Scenario(
        scenario_id="baseline",
        label=r"Baseline: $X_D = 0.75X_c$",
        short_label=r"Baseline: $X_D = 0.75X_c$",
        xc_factor=1.0,
        xdisease_factor=1.0,
    ),
    Scenario(
        scenario_id="xc_only",
        label=r"Increase in death threshold (constant disease threshold)",
        short_label=r"Increase in death threshold (constant disease threshold)",
        xc_factor=XC_FACTOR,
        xdisease_factor=1.0,
    ),
    Scenario(
        scenario_id="proportional",
        label=r"Proportional increase in both disease and death threshold",
        short_label=r"Proportional increase in both disease and death threshold",
        xc_factor=XC_FACTOR,
        xdisease_factor=XC_FACTOR,
    ),
)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sample_positive_gaussian(mean: float, rel_std: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = mean * rel_std
    values = rng.normal(loc=mean, scale=std, size=n)

    while True:
        bad = values <= 0
        if not np.any(bad):
            return values
        values[bad] = rng.normal(loc=mean, scale=std, size=int(bad.sum()))


def build_baseline_xc(n: int) -> np.ndarray:
    return sample_positive_gaussian(
        mean=BASELINE["Xc"],
        rel_std=BASELINE["xc_std_frac"],
        n=n,
        seed=stable_seed(BASE_SEED, "baseline_xc", n),
    )


def build_params(xc: np.ndarray) -> dict[str, np.ndarray]:
    n = xc.size
    return {
        "eta": np.full(n, BASELINE["eta"], dtype=float),
        "beta": np.full(n, BASELINE["beta"], dtype=float),
        "kappa": np.full(n, BASELINE["kappa"], dtype=float),
        "epsilon": np.full(n, BASELINE["epsilon"], dtype=float),
        "Xc": xc.astype(float, copy=True),
    }


def metadata(n: int, parallel: bool) -> dict[str, object]:
    return {
        "run_label": RUN_LABEL,
        "baseline": BASELINE,
        "disease_fraction": DISEASE_FRACTION,
        "xc_factor": XC_FACTOR,
        "scenario_ids": [scenario.scenario_id for scenario in SCENARIOS],
        "n": int(n),
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "h_ext": 0.0,
        "parallel": bool(parallel),
        "base_seed": BASE_SEED,
    }


def cache_matches(current: dict[str, object]) -> bool:
    if not EVENTS_CACHE_PATH.exists() or not METADATA_PATH.exists():
        return False

    cached = json.loads(METADATA_PATH.read_text())
    return cached == current


def run_or_load_events(n: int, parallel: bool, force: bool) -> dict[str, dict[str, np.ndarray]]:
    current = metadata(n=n, parallel=parallel)
    if not force and cache_matches(current):
        return load_events()

    events = run_scenarios(n=n, parallel=parallel)
    save_events(events=events, current=current)
    return events


def run_scenarios(n: int, parallel: bool) -> dict[str, dict[str, np.ndarray]]:
    baseline_xc = build_baseline_xc(n)
    baseline_xdisease = DISEASE_FRACTION * baseline_xc
    events = {}

    for scenario in SCENARIOS:
        print(f"Running {scenario.scenario_id} with n={n:,}", flush=True)
        xc = scenario.xc_factor * baseline_xc
        xdisease = scenario.xdisease_factor * baseline_xdisease
        sim = utils.create_sr_simulation(
            params_dict=build_params(xc),
            n=n,
            h_ext=0.0,
            Xdisease=xdisease,
            tmax=TMAX,
            dt=DT,
            save_times=SAVE_TIMES,
            parallel=parallel,
            break_early=True,
            random_seed=stable_seed(BASE_SEED, scenario.scenario_id, "simulation", n),
        )
        events[scenario.scenario_id] = event_arrays(sim)

    return events


def event_arrays(sim) -> dict[str, np.ndarray]:
    death_time = np.asarray(sim.death_times, dtype=float).copy()
    alive_at_tmax = np.asarray(sim.alive_mask, dtype=bool).copy()
    death_time[alive_at_tmax] = np.inf

    disease_time = np.asarray(sim.disease_times, dtype=float).copy()
    end_time = np.where(np.isfinite(death_time), death_time, TMAX)
    became_sick = np.isfinite(disease_time) & (disease_time <= end_time)

    healthspan = np.where(became_sick, disease_time, end_time)
    sickspan = np.where(became_sick, np.maximum(0.0, end_time - disease_time), 0.0)

    return {
        "death_time": death_time,
        "disease_time": disease_time,
        "healthspan": healthspan,
        "sickspan": sickspan,
        "became_sick": became_sick,
        "alive_at_tmax": alive_at_tmax,
    }


def save_events(events: dict[str, dict[str, np.ndarray]], current: dict[str, object]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for scenario_id, scenario_events in events.items():
        for key, values in scenario_events.items():
            arrays[f"{scenario_id}__{key}"] = values
    np.savez_compressed(EVENTS_CACHE_PATH, **arrays)
    METADATA_PATH.write_text(json.dumps(current, indent=2) + "\n")


def load_events() -> dict[str, dict[str, np.ndarray]]:
    loaded = np.load(EVENTS_CACHE_PATH)
    events = {}
    for scenario in SCENARIOS:
        scenario_events = {}
        for key in ("death_time", "disease_time", "healthspan", "sickspan", "became_sick", "alive_at_tmax"):
            scenario_events[key] = loaded[f"{scenario.scenario_id}__{key}"]
        events[scenario.scenario_id] = scenario_events
    return events


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finite_deaths(values: dict[str, np.ndarray]) -> np.ndarray:
    death_time = values["death_time"]
    return death_time[np.isfinite(death_time)]


def summarize_events(events: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for scenario in SCENARIOS:
        values = events[scenario.scenario_id]
        deaths = finite_deaths(values)
        became_sick = values["became_sick"]
        sickspan = values["sickspan"]
        healthspan = values["healthspan"]
        death_time = values["death_time"]
        finite_death_time = np.where(np.isfinite(death_time), death_time, np.nan)
        sick_fraction = sickspan / np.maximum(np.where(np.isfinite(death_time), death_time, TMAX), 1e-12)

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "label": scenario.short_label,
                "n": int(death_time.size),
                "deaths_observed": int(deaths.size),
                "censored_at_tmax": int(values["alive_at_tmax"].sum()),
                "became_sick_fraction": float(np.mean(became_sick)),
                "mean_lifespan": nanmean(finite_death_time),
                "median_lifespan": nanmedian(finite_death_time),
                "mean_healthspan": float(np.mean(healthspan)),
                "median_healthspan": float(np.median(healthspan)),
                "mean_sickspan": float(np.mean(sickspan)),
                "median_sickspan": float(np.median(sickspan)),
                "mean_sickspan_among_sick": nanmean(sickspan[became_sick]),
                "median_sickspan_among_sick": nanmedian(sickspan[became_sick]),
                "mean_sick_life_fraction": float(np.mean(sick_fraction)),
                "median_sick_life_fraction": float(np.median(sick_fraction)),
            }
        )

    return rows


def nanmean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def nanmedian(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def save_summary(rows: list[dict[str, object]]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def survival_curve(event_times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([np.mean(event_times > t) for t in grid], dtype=float)


def sickspan_survival(values: dict[str, np.ndarray], grid: np.ndarray) -> np.ndarray:
    sickspan = values["sickspan"][values["became_sick"]]
    if sickspan.size == 0:
        return np.zeros(grid.size, dtype=float)
    return survival_curve(sickspan, grid)


def state_composition(events: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for scenario in SCENARIOS:
        values = events[scenario.scenario_id]
        death_time = values["death_time"]
        disease_time = values["disease_time"]
        n = death_time.size

        for age in AGE_GRID:
            dead = np.isfinite(death_time) & (death_time <= age)
            sick_alive = np.isfinite(disease_time) & (disease_time <= age) & ~dead
            healthy_alive = ~dead & ~sick_alive
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "age": float(age),
                    "healthy_alive_fraction": float(np.sum(healthy_alive) / n),
                    "sick_alive_fraction": float(np.sum(sick_alive) / n),
                    "dead_fraction": float(np.sum(dead) / n),
                }
            )

    return rows


def save_state_rows(rows: list[dict[str, object]]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATE_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_lifespan_healthspan_survival(events: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    for scenario in SCENARIOS:
        values = events[scenario.scenario_id]
        color = SCENARIO_COLORS[scenario.scenario_id]
        death_curve = survival_curve(values["death_time"], AGE_GRID)
        health_curve = survival_curve(values["healthspan"], AGE_GRID)
        axes[0].plot(AGE_GRID, death_curve, color=color, lw=2.8, label=scenario.short_label)
        axes[1].plot(AGE_GRID, health_curve, color=color, lw=2.8, label=scenario.short_label)

    axes[0].set_title("Alive")
    axes[0].set_ylabel("Fraction of cohort")
    axes[1].set_title("Alive and not yet sick")
    for ax in axes:
        ax.set_xlabel("Age")
        ax.set_xlim(40, 125)
        ax.set_ylim(0, 1.02)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(frameon=False, loc="upper right")

    path = PLOTS_DIR / "01_lifespan_healthspan_survival.png"
    save_figure(fig, path)
    return path


def plot_sickspan_survival(events: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    for scenario in SCENARIOS:
        values = events[scenario.scenario_id]
        color = SCENARIO_COLORS[scenario.scenario_id]
        ax.plot(
            SICKSPAN_GRID,
            sickspan_survival(values, SICKSPAN_GRID),
            color=color,
            lw=3.0,
            label=scenario.short_label,
        )

    ax.set_xlabel("Years lived after first crossing $X_D$")
    ax.set_ylabel("Fraction still in sick span")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    path = PLOTS_DIR / "02_sickspan_survival.png"
    save_figure(fig, path)
    return path


def plot_health_sick_bars(summary_rows: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    labels = scenario_tick_labels()
    health = np.array([row["mean_healthspan"] for row in summary_rows], dtype=float)
    sick = np.array([row["mean_sickspan"] for row in summary_rows], dtype=float)
    x = np.arange(len(labels))

    ax.bar(x, health, width=0.58, color="#6BAA75", label="Healthy years")
    ax.bar(x, sick, bottom=health, width=0.58, color="#C65D4B", label="Sick years")
    ax.set_ylabel("Mean years")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(health + sick) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper left")

    for index, total in enumerate(health + sick):
        ax.text(index, total + 1.0, f"{total:.1f}", ha="center", va="bottom", fontsize=12)

    path = PLOTS_DIR / "03_health_sick_stacked_bars.png"
    save_figure(fig, path)
    return path


def plot_state_composition(state_rows: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), sharey=True)
    by_scenario = {scenario.scenario_id: [] for scenario in SCENARIOS}
    for row in state_rows:
        by_scenario[row["scenario_id"]].append(row)

    for ax, scenario in zip(axes, SCENARIOS):
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
            colors=[STATE_COLORS["healthy"], STATE_COLORS["sick"], STATE_COLORS["dead"]],
            labels=["Healthy alive", "Sick alive", "Dead"],
            linewidth=0,
        )
        ax.set_title(scenario_panel_title(scenario))
        ax.set_xlabel("Age")
        ax.set_xlim(55, 125)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Fraction of cohort")
    axes[2].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    path = PLOTS_DIR / "04_age_specific_state_composition.png"
    save_figure(fig, path)
    return path


def plot_sick_fraction(events: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    labels = scenario_tick_labels()
    values = []
    medians = []

    for scenario in SCENARIOS:
        scenario_events = events[scenario.scenario_id]
        end_time = np.where(np.isfinite(scenario_events["death_time"]), scenario_events["death_time"], TMAX)
        fraction = scenario_events["sickspan"] / np.maximum(end_time, 1e-12)
        values.append(float(np.mean(fraction)))
        medians.append(float(np.median(fraction)))

    x = np.arange(len(labels))
    colors = [SCENARIO_COLORS[scenario.scenario_id] for scenario in SCENARIOS]
    ax.bar(x, values, color=colors, width=0.58)
    median_points = ax.scatter(
        x,
        medians,
        s=74,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
        label="Median",
    )
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.04,
        "Sick-life fraction",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(values) * 1.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[median_points], labels=["Median"], frameon=False, loc="upper right")

    for index, value in enumerate(values):
        ax.text(index, value + 0.006, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=12)

    path = PLOTS_DIR / "05_sick_fraction_of_life.png"
    save_figure(fig, path)
    return path


def plot_supplement_figure(
    events: dict[str, dict[str, np.ndarray]],
    state_rows: list[dict[str, object]],
) -> Path:
    fig = plt.figure(figsize=(11.2, 9.0))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.15, 1.0],
        hspace=0.58,
        wspace=0.25,
    )
    state_axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    bar_ax = fig.add_subplot(grid[1, :])

    by_scenario = {scenario.scenario_id: [] for scenario in SCENARIOS}
    for row in state_rows:
        by_scenario[row["scenario_id"]].append(row)

    for ax, scenario in zip(state_axes, SCENARIOS):
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
            colors=[STATE_COLORS["healthy"], STATE_COLORS["sick"], STATE_COLORS["dead"]],
            labels=["Healthy alive", "Sick alive", "Dead"],
            linewidth=0,
        )
        ax.set_title(scenario_panel_title(scenario), pad=8)
        ax.set_xlabel("Age")
        ax.set_xlim(55, 125)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    state_axes[0].set_ylabel("Fraction of cohort")
    handles, labels = state_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.99),
    )

    plot_sick_fraction_panel(events=events, ax=bar_ax)
    state_axes[0].text(
        -0.18,
        1.20,
        "A",
        transform=state_axes[0].transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
    )
    bar_ax.text(
        -0.06,
        1.08,
        "B",
        transform=bar_ax.transAxes,
        fontsize=17,
        fontweight="bold",
        va="top",
    )

    SUPP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SUPP_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return SUPP_FIGURE_PATH


def plot_sick_fraction_panel(events: dict[str, dict[str, np.ndarray]], ax: plt.Axes) -> None:
    labels = scenario_tick_labels()
    means = []
    medians = []

    for scenario in SCENARIOS:
        scenario_events = events[scenario.scenario_id]
        end_time = np.where(np.isfinite(scenario_events["death_time"]), scenario_events["death_time"], TMAX)
        fraction = scenario_events["sickspan"] / np.maximum(end_time, 1e-12)
        means.append(float(np.mean(fraction)))
        medians.append(float(np.median(fraction)))

    x = np.arange(len(labels))
    colors = [SCENARIO_COLORS[scenario.scenario_id] for scenario in SCENARIOS]
    ax.bar(x, means, color=colors, width=0.56)
    median_points = ax.scatter(
        x,
        medians,
        s=78,
        facecolor="white",
        edgecolor="black",
        linewidth=1.4,
        zorder=3,
        label="Median",
    )
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.04,
        "Sick-life fraction",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(means) * 1.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[median_points], labels=["Median"], frameon=False, loc="upper right")

    for index, value in enumerate(means):
        ax.text(index, value + 0.004, f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=12)


def plot_event_age_quantiles(events: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    for scenario in SCENARIOS:
        values = events[scenario.scenario_id]
        color = SCENARIO_COLORS[scenario.scenario_id]
        death_q = np.quantile(finite_deaths(values), quantiles)
        disease_q = np.quantile(values["disease_time"][np.isfinite(values["disease_time"])], quantiles)
        ax.plot(death_q, disease_q, "o-", color=color, lw=2.5, ms=7, label=scenario.short_label)

    ax.plot([55, 125], [55, 125], color="0.55", lw=1.2, ls="--")
    ax.set_xlabel("Death-age quantile")
    ax.set_ylabel("$X_D$ crossing-age quantile")
    ax.set_xlim(65, 125)
    ax.set_ylim(55, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    path = PLOTS_DIR / "06_event_age_quantiles.png"
    save_figure(fig, path)
    return path


def plot_disease_death_scatter(events: dict[str, dict[str, np.ndarray]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), sharex=True, sharey=True)
    rng = np.random.default_rng(BASE_SEED)

    for ax, scenario in zip(axes, SCENARIOS):
        values = events[scenario.scenario_id]
        disease = values["disease_time"]
        death = values["death_time"]
        mask = np.isfinite(disease) & np.isfinite(death)
        selected = np.flatnonzero(mask)
        if selected.size > 5_000:
            selected = rng.choice(selected, size=5_000, replace=False)

        ax.scatter(
            disease[selected],
            death[selected],
            s=5,
            alpha=0.12,
            color=SCENARIO_COLORS[scenario.scenario_id],
            linewidths=0,
        )
        ax.plot([45, 125], [45, 125], color="0.55", lw=1.0, ls="--")
        ax.set_title(scenario.short_label)
        ax.set_xlabel("$X_D$ crossing age")
        ax.set_xlim(50, 115)
        ax.set_ylim(60, 130)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Death age")

    path = PLOTS_DIR / "07_disease_vs_death_age_scatter.png"
    save_figure(fig, path)
    return path


def scenario_tick_labels() -> list[str]:
    return [
        "Baseline\n$X_D = 0.75X_c$",
        "Increase in death\nthreshold\n(constant disease\nthreshold)",
        "Proportional increase\nin both disease and\ndeath threshold",
    ]


def scenario_panel_title(scenario: Scenario) -> str:
    labels = {
        "baseline": "Baseline\n$X_D = 0.75X_c$",
        "xc_only": "Increase in death threshold\n(constant disease threshold)",
        "proportional": "Proportional increase in both\ndisease and death threshold",
    }
    return labels[scenario.scenario_id]


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    events: dict[str, dict[str, np.ndarray]],
    summary_rows: list[dict[str, object]],
    state_rows: list[dict[str, object]],
) -> list[Path]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return [
        plot_lifespan_healthspan_survival(events),
        plot_sickspan_survival(events),
        plot_health_sick_bars(summary_rows),
        plot_state_composition(state_rows),
        plot_sick_fraction(events),
        plot_event_age_quantiles(events),
        plot_disease_death_scatter(events),
    ]


def write_report(
    summary_rows: list[dict[str, object]],
    plot_paths: list[Path],
    supp_figure_path: Path,
    n: int,
    parallel: bool,
) -> None:
    row_by_id = {row["scenario_id"]: row for row in summary_rows}
    lines = [
        "# Artificial Survival Time SR Exploration",
        "",
        "This exploration responds to the reviewer concern that some late-life survival gains may not be well described as improved robustness. The operational SR proxy is a disease threshold, \(X_D\), below the critical death threshold, \(X_c\). First crossing of \(X_D\) marks disease onset; first crossing of \(X_c\) remains death.",
        "",
        "The three matched simulations use the Sweden 2019 tail-emphasis baseline and the same individual \(X_c\) heterogeneity draws:",
        "",
        f"- Baseline: \(X_D = {DISEASE_FRACTION:.2f}X_c\).",
        f"- Increase in death threshold (constant disease threshold): \(X_c\) is multiplied by \(1.2\), while each individual's original \(X_D\) is kept fixed.",
        f"- Proportional increase in both disease and death threshold: both \(X_c\) and \(X_D\) are multiplied by \(1.2\).",
        "",
        "In this setup, the constant-disease-threshold scenario is a simple model of added survival time after disease onset. The proportional scenario is closer to a robustness-like shift where the disease and death thresholds move together.",
        "",
        "## Run Details",
        "",
        f"- \(n={n:,}\) simulated individuals per scenario.",
        f"- \(t_{{max}}={TMAX:g}\), \(\Delta t={DT:g}\), \(h_{{ext}}=0\).",
        f"- Parallel simulation: `{parallel}`.",
        f"- Baseline parameters: \(\\eta={BASELINE['eta']:.6g}\), \(\\beta={BASELINE['beta']:.6g}\), \(\\kappa={BASELINE['kappa']:.6g}\), \(\\epsilon={BASELINE['epsilon']:.6g}\), \(X_c={BASELINE['Xc']:.6g}\), \(\\sigma_{{X_c}}/X_c={BASELINE['xc_std_frac']:.6g}\).",
        "",
        "## Summary",
        "",
        "| Scenario | Mean lifespan | Mean healthspan | Mean sick span | Mean sick-life fraction |",
        "|---|---:|---:|---:|---:|",
    ]

    for scenario in SCENARIOS:
        row = row_by_id[scenario.scenario_id]
        lines.append(
            f"| {scenario.short_label} | {row['mean_lifespan']:.2f} | "
            f"{row['mean_healthspan']:.2f} | {row['mean_sickspan']:.2f} | "
            f"{100 * row['mean_sick_life_fraction']:.1f}% |"
        )

    fixed = row_by_id["xc_only"]
    proportional = row_by_id["proportional"]
    baseline = row_by_id["baseline"]
    lines.extend(
        [
            "",
            "## First Read",
            "",
            f"Relative to baseline, the constant-disease-threshold scenario adds about {fixed['mean_lifespan'] - baseline['mean_lifespan']:.2f} mean years of life but also adds about {fixed['mean_sickspan'] - baseline['mean_sickspan']:.2f} mean sick years.",
            f"When \(X_D\) moves with \(X_c\), the model still extends mean lifespan by about {proportional['mean_lifespan'] - baseline['mean_lifespan']:.2f} years, while mean sick span is {baseline['mean_sickspan'] - proportional['mean_sickspan']:.2f} years lower than baseline.",
            "",
            "## Plot Gallery",
            "",
        ]
    )

    rel_supp_path = Path(os.path.relpath(supp_figure_path, EXPLORATION_DIR))
    lines.extend(
        [
            "### Supplementary Two-Panel Figure",
            "",
            "Focused version for the reviewer response: state composition over age and fraction of lifespan after crossing \(X_D\).",
            "",
            f"![supplementary artificial survival time]({rel_supp_path.as_posix()})",
            "",
        ]
    )

    captions = {
        "01_lifespan_healthspan_survival.png": "Lifespan survival and healthspan survival separate being alive from being alive without disease.",
        "02_sickspan_survival.png": "Distribution of time spent after crossing \(X_D\), conditional on crossing it.",
        "03_health_sick_stacked_bars.png": "Mean lifespan decomposed into healthy years and sick years.",
        "04_age_specific_state_composition.png": "Age-specific cohort composition: healthy alive, sick alive, or dead.",
        "05_sick_fraction_of_life.png": "Mean fraction of life spent after crossing \(X_D\).",
        "06_event_age_quantiles.png": "Matched quantiles of \(X_D\) crossing age versus death age.",
        "07_disease_vs_death_age_scatter.png": "Sampled individual disease-onset and death-age points.",
    }
    for path in plot_paths:
        rel_path = path.relative_to(EXPLORATION_DIR)
        lines.append(f"### {path.stem.replace('_', ' ').title()}")
        lines.append("")
        lines.append(captions[path.name])
        lines.append("")
        lines.append(f"![{path.stem}]({rel_path.as_posix()})")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def update_output_index(plot_paths: list[Path], supp_figure_path: Path) -> None:
    if not INDEX_PATH.exists():
        return

    existing_text = INDEX_PATH.read_text()
    entries = [
        {
            "date": DATE,
            "task": "artificial_survival_time_xdisease_exploration",
            "artifact_type": "documentation",
            "path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": "results/fits/records/joint2019_tail90_sweden_emphasis.json",
            "description": "Markdown report for disease-threshold SR exploration.",
            "notes": "Compares baseline, fixed Xdisease, and proportional Xdisease threshold shifts.",
        },
        {
            "date": DATE,
            "task": "artificial_survival_time_xdisease_exploration",
            "artifact_type": "simulation_cache",
            "path": str(EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": "results/fits/records/joint2019_tail90_sweden_emphasis.json",
            "description": "Compressed per-person disease, death, healthspan, and sickspan event arrays.",
            "notes": "Matched baseline Xc heterogeneity across all scenarios.",
        },
        {
            "date": DATE,
            "task": "artificial_survival_time_xdisease_exploration",
            "artifact_type": "csv",
            "path": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": str(EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "description": "Scenario-level lifespan, healthspan, and sickspan summaries.",
            "notes": "Healthspan is age at first Xdisease crossing or death/censoring if never sick.",
        },
        {
            "date": DATE,
            "task": "artificial_survival_time_xdisease_exploration",
            "artifact_type": "csv",
            "path": str(STATE_PATH.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": str(EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "description": "Age-specific healthy alive, sick alive, and dead fractions.",
            "notes": "Used for the stacked state-composition plots.",
        },
    ]

    for path in plot_paths:
        entries.append(
            {
                "date": DATE,
                "task": "artificial_survival_time_xdisease_exploration",
                "artifact_type": "figure",
                "path": str(path.relative_to(PROJECT_ROOT)),
                "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
                "input_paths": str(EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
                "description": f"Exploratory PNG: {path.stem.replace('_', ' ')}.",
                "notes": "PNG-only exploratory output.",
            }
        )

    entries.append(
        {
            "date": DATE,
            "task": "supp_artificial_survival_time_xdisease_075",
            "artifact_type": "figure",
            "path": str(supp_figure_path.relative_to(PROJECT_ROOT)),
            "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "input_paths": str(EVENTS_CACHE_PATH.relative_to(PROJECT_ROOT)),
            "description": "Two-panel supplementary PNG for artificial-survival-time disease-threshold analysis.",
            "notes": "Uses Xdisease = 0.75 Xc; includes age-specific state composition and sick-life fraction.",
        }
    )

    new_entries = [
        entry for entry in entries
        if f",{entry['path']}," not in existing_text and f",{entry['path']}\n" not in existing_text
    ]
    if not new_entries:
        return

    with INDEX_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(new_entries[0].keys()))
        writer.writerows(new_entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parallel = not args.serial
    configure_matplotlib()

    events = run_or_load_events(n=args.n, parallel=parallel, force=args.force)
    summary_rows = summarize_events(events)
    state_rows = state_composition(events)
    save_summary(summary_rows)
    save_state_rows(state_rows)
    plot_paths = make_plots(events=events, summary_rows=summary_rows, state_rows=state_rows)
    supp_figure_path = plot_supplement_figure(events=events, state_rows=state_rows)
    write_report(
        summary_rows=summary_rows,
        plot_paths=plot_paths,
        supp_figure_path=supp_figure_path,
        n=args.n,
        parallel=parallel,
    )
    update_output_index(plot_paths, supp_figure_path)
    print(f"Report: {REPORT_PATH}")
    print(f"Supplementary figure: {supp_figure_path}")


if __name__ == "__main__":
    main()
