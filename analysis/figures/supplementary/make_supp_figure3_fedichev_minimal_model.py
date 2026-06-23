#!/usr/bin/env python3
"""Make Supplementary Fig. 3: Fedichev-Gruber minimal aging model."""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR


OUTPUT_DIR = FIGURES_DIR / "Supplementary"
PNG_PATH = OUTPUT_DIR / "supp_figure3_fedichev_minimal_model.png"
PDF_PATH = OUTPUT_DIR / "supp_figure3_fedichev_minimal_model.pdf"
SOURCE_PATH = TABLES_DIR / "supp_figure3_fedichev_minimal_model_source.csv"
INDEX_PATH = RESULTS_DIR / "index" / "outputs.csv"

RANDOM_SEED = 20260604
N_SIM = 160_000
DT = 0.05
TMAX = 125.0
SURVIVAL_AGES = np.arange(0.0, 121.0, 1.0)
HAZARD_START = 40.0
HAZARD_END = 121.0
HAZARD_BIN_WIDTH = 1.0

BASELINE_PARAMS = {
    "epsilon_0": 4.0,
    "D0": 1.1,
    "beta": 0.015,
    "g": 0.8,
    "gamma": 1.0,
    "beta_prime": 0.013333,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-sim", action="store_true", help="Regenerate the cached source table.")
    parser.add_argument("--n-sim", type=int, default=N_SIM, help="Number of simulated individuals if regenerating.")
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 18.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def simulate_fedichev_deaths(n: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    params = BASELINE_PARAMS
    alive = np.ones(n, dtype=bool)
    z = np.zeros(n, dtype=float)
    death_times = np.full(n, np.nan, dtype=float)
    sqrt_dt = np.sqrt(DT)
    noise_strength = np.sqrt(2.0 * params["D0"])

    for step in range(int(TMAX / DT) + 1):
        if not alive.any():
            break
        t = step * DT
        idx = np.flatnonzero(alive)
        z_driver = params["gamma"] * t
        epsilon_eff = params["epsilon_0"] - params["beta_prime"] * z_driver
        discriminant = epsilon_eff**2 - 4.0 * params["g"] * (params["beta"] * z_driver)

        dead_now = np.zeros(idx.size, dtype=bool)
        if discriminant <= 0:
            dead_now[:] = True
        else:
            z_unstable = (epsilon_eff + np.sqrt(discriminant)) / (2.0 * params["g"])
            z_alive = z[idx]
            drift = params["beta"] * z_driver - epsilon_eff * z_alive + params["g"] * z_alive**2
            z_new = z_alive + drift * DT + noise_strength * rng.normal(0.0, sqrt_dt, size=idx.size)
            z[idx] = z_new
            dead_now = z_new > z_unstable

        if dead_now.any():
            deaths = idx[dead_now]
            death_times[deaths] = t
            alive[deaths] = False

    death_times[np.isnan(death_times)] = TMAX
    return death_times


def build_source_table(n: int) -> pd.DataFrame:
    death_times = simulate_fedichev_deaths(n)
    rows: list[dict[str, float | str | int]] = []

    for age in SURVIVAL_AGES:
        rows.append(
            {
                "series": "survival",
                "age": float(age),
                "value": float(np.mean(death_times > age)),
                "value_low": np.nan,
                "value_high": np.nan,
                "n_sim": int(n),
                "dt": DT,
                "random_seed": RANDOM_SEED,
            }
        )

    starts = np.arange(HAZARD_START, HAZARD_END, HAZARD_BIN_WIDTH)
    hazards = []
    lows = []
    highs = []
    for start in starts:
        end = start + HAZARD_BIN_WIDTH
        at_risk = int(np.sum(death_times >= start))
        deaths = int(np.sum((death_times >= start) & (death_times < end)))
        hazard = deaths / max(at_risk * HAZARD_BIN_WIDTH, 1)
        se = np.sqrt(max(deaths, 1)) / max(at_risk * HAZARD_BIN_WIDTH, 1)
        hazards.append(hazard)
        lows.append(max(hazard - 1.96 * se, 1e-8))
        highs.append(hazard + 1.96 * se)

    hazards = gaussian_filter1d(np.asarray(hazards), sigma=1.1)
    lows = gaussian_filter1d(np.asarray(lows), sigma=1.1)
    highs = gaussian_filter1d(np.asarray(highs), sigma=1.1)
    for age, hazard, low, high in zip(starts + 0.5 * HAZARD_BIN_WIDTH, hazards, lows, highs):
        rows.append(
            {
                "series": "mortality",
                "age": float(age),
                "value": float(max(hazard, 1e-8)),
                "value_low": float(max(low, 1e-8)),
                "value_high": float(max(high, 1e-8)),
                "n_sim": int(n),
                "dt": DT,
                "random_seed": RANDOM_SEED,
            }
        )

    table = pd.DataFrame(rows)
    SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SOURCE_PATH, index=False)
    return table


def load_or_build_source(*, force_sim: bool, n: int) -> pd.DataFrame:
    if SOURCE_PATH.exists() and not force_sim:
        return pd.read_csv(SOURCE_PATH)
    return build_source_table(n)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=19.0,
        fontweight="bold",
        ha="left",
        va="top",
    )


def plot_figure(source: pd.DataFrame) -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)

    survival = source[source["series"] == "survival"].sort_values("age")
    axes[0].plot(survival["age"], survival["value"], color="#1F78B4", lw=3.0)
    axes[0].set_xlim(0, 120)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Survival")
    axes[0].set_xlabel("Age [years]")
    axes[0].set_ylabel("Survival probability")
    panel_label(axes[0], "a")

    mortality = source[source["series"] == "mortality"].sort_values("age")
    axes[1].plot(mortality["age"], mortality["value"], color="#9E1B1B", lw=2.7)
    axes[1].fill_between(
        mortality["age"].to_numpy(dtype=float),
        mortality["value_low"].to_numpy(dtype=float),
        mortality["value_high"].to_numpy(dtype=float),
        color="#9E1B1B",
        alpha=0.20,
        linewidth=0,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlim(40, 120)
    axes[1].set_ylim(1e-4, 2.0)
    axes[1].set_title("Mortality")
    axes[1].set_xlabel("Age [years]")
    axes[1].set_ylabel(r"Mortality rate [year$^{-1}$]")
    panel_label(axes[1], "b")

    fig.savefig(PNG_PATH, dpi=350, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def update_output_index() -> None:
    fieldnames = ["date", "task", "artifact_type", "path", "source_script", "input_paths", "description", "notes"]
    existing = []
    if INDEX_PATH.exists():
        with INDEX_PATH.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    source_script = str(Path(__file__).relative_to(PROJECT_ROOT))
    rows = [
        {
            "date": date.today().isoformat(),
            "task": "supp_figure3_fedichev_minimal_model",
            "artifact_type": "figure",
            "path": str(PNG_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
            "description": "PNG preview of Supplementary Fig. 3 Fedichev-Gruber minimal model.",
            "notes": "Baseline stochastic simulation with fixed random seed; source table cached for fast reproduction.",
        },
        {
            "date": date.today().isoformat(),
            "task": "supp_figure3_fedichev_minimal_model",
            "artifact_type": "csv",
            "path": str(SOURCE_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": "",
            "description": "Source data for Supplementary Fig. 3 survival and mortality curves.",
            "notes": f"n={N_SIM}, dt={DT}, seed={RANDOM_SEED}; rerun with --force-sim to regenerate.",
        },
    ]
    replace = {row["path"] for row in rows}
    kept = [row for row in existing if row.get("path") not in replace]
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source = load_or_build_source(force_sim=args.force_sim, n=args.n_sim)
    plot_figure(source)
    update_output_index()
    print(PNG_PATH)
    print(PDF_PATH)
    print(SOURCE_PATH)


if __name__ == "__main__":
    main()
