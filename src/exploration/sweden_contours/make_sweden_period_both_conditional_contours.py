from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from src.shared.thresholds.paths import FIGURES_DIR, HMD_DATA_DIR, SAVED_RESULTS_DIR


START_YEAR = 1900
END_YEAR = 2020
FROM_AGE = 20
SMOOTHING_WINDOW = 5

FIGURE_DIR = FIGURES_DIR / "Fig_4new"
RESULTS_DIR = SAVED_RESULTS_DIR / "fig4_new"

OUTPUT_PNG = FIGURE_DIR / "sweden_period_both_conditional_age20_contours_1900_2020.png"
OUTPUT_CSV = RESULTS_DIR / "sweden_period_both_conditional_age20_contours_1900_2020.csv"

LIFE_TABLE_PATH = HMD_DATA_DIR / "mortality.org_File_GetDocument_hmd.v6_SWE_STATS_bltper_1x1.txt"

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


def load_life_table() -> pd.DataFrame:
    rows = []
    with LIFE_TABLE_PATH.open(errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("Year"):
                continue

            parts = text.split()
            if len(parts) < 10:
                continue

            try:
                year = int(parts[0])
                age = int(parts[1].replace("+", ""))
                lx = float(parts[5])
            except ValueError:
                continue

            if START_YEAR <= year <= END_YEAR:
                rows.append({"year": year, "age": age, "lx": lx})

    return pd.DataFrame(rows)


def find_threshold_age(year_data: pd.DataFrame, survival_level: float) -> float:
    data = year_data.sort_values("age")
    ages = data["age"].to_numpy(dtype=float)
    survival = data["conditional_survival"].to_numpy(dtype=float)

    if len(ages) == 0:
        return np.nan
    if survival[-1] > survival_level:
        return np.nan

    crossing_positions = np.where(survival <= survival_level)[0]
    if len(crossing_positions) == 0:
        return np.nan

    after = crossing_positions[0]
    if after == 0:
        return ages[0]

    before = after - 1
    age_before = ages[before]
    age_after = ages[after]
    survival_before = survival[before]
    survival_after = survival[after]

    if survival_before <= 0 or survival_after <= 0:
        fraction = (survival_level - survival_before) / (
            survival_after - survival_before
        )
        return age_before + fraction * (age_after - age_before)

    log_before = np.log(survival_before)
    log_after = np.log(survival_after)
    log_target = np.log(survival_level)
    fraction = (log_target - log_before) / (log_after - log_before)
    return age_before + fraction * (age_after - age_before)


def build_contours(life_table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year, year_data in life_table.groupby("year"):
        year_data = year_data.copy()
        lx_at_20 = year_data.loc[year_data["age"] == FROM_AGE, "lx"]
        if lx_at_20.empty:
            continue

        year_data["conditional_survival"] = year_data["lx"] / float(lx_at_20.iloc[0])

        record = {"year": int(year)}
        for label, survival_level in SURVIVAL_LEVELS.items():
            record[label] = find_threshold_age(year_data, survival_level)
        records.append(record)

    contours = pd.DataFrame(records).sort_values("year")
    for label in SURVIVAL_LEVELS:
        contours[f"{label} smoothed"] = (
            contours[label]
            .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
            .mean()
        )

    return contours


def plot_contours(contours: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 12,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.4))

    for label in SURVIVAL_LEVELS:
        smoothed_label = f"{label} smoothed"
        ax.plot(
            contours["year"],
            contours[smoothed_label],
            color=COLORS[label],
            linewidth=3.0,
            label=label,
        )
        last_value = contours[smoothed_label].dropna().iloc[-1]
        ax.text(
            END_YEAR - 1.5,
            last_value + 0.15,
            label,
            color=COLORS[label],
            fontsize=12,
            fontweight="bold",
            ha="right",
            va="bottom",
        )

    ax.axhline(110, color="0.35", linestyle=":", linewidth=1.8)
    ax.text(
        START_YEAR + 2,
        110.8,
        "HMD 110+ open interval",
        color="0.28",
        fontsize=12,
        va="bottom",
    )

    ax.set_title("Sweden period life table, both sexes")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Age reached | conditional on age {FROM_AGE}")
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_ylim(55, 113)
    ax.set_xticks(np.arange(1900, 2021, 20))
    ax.tick_params(axis="both", width=1.2, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=1.0)
    fig.subplots_adjust(left=0.15, right=0.955, bottom=0.14, top=0.90)
    fig.savefig(OUTPUT_PNG, dpi=260)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    life_table = load_life_table()
    contours = build_contours(life_table)
    contours.to_csv(OUTPUT_CSV, index=False)
    plot_contours(contours)

    print(f"saved {OUTPUT_CSV}")
    print(f"saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
