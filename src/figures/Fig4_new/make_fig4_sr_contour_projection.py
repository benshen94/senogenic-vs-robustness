#!/usr/bin/env python3
"""Run Sweden SR contour projections for Fig4.

The script is checkpointed at the single simulation level. Finished rows are
kept in the raw cache so plot/style iterations do not rerun the SR model.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.exploration.sweden_contours.make_sweden_sr_fig4cd_pilot_overlay import (
    conservative_envelope,
)
from src.figures.steepness_longevity.run_sweden2019_sensitivity import (
    BASELINE,
    DT,
    TMAX,
    sample_positive_gaussian,
)
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


FROM_AGE = 20
N_SIM = int(1e6)
N_SIM_LABEL = "n1m"
SAVE_TIMES = TMAX

HISTORICAL_YEARS = tuple(range(1900, 2021, 10))
PROJECTION_YEARS = tuple(range(2020, 2101, 10))
TREND_FIT_START_YEAR = 1980
TREND_FIT_END_YEAR = 2020
MODERN_SOURCE_YEAR = 2019

RESULTS_DIR = SAVED_RESULTS_DIR / "fig4_new"
CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "Fig4_new"
FIGURE_DIR = FIGURES_NEW_DIR / "Fig4_new"

FIG4C_CSV = RESULTS_DIR / "fig4c_extrinsic_mortality_projection.csv"
FIG4D_CSV = RESULTS_DIR / "fig4d_robustness_projection.csv"
HMD_CONTOURS_CSV = RESULTS_DIR / "sweden_period_both_conditional_age20_contours_1900_2020.csv"

RAW_SIM_CSV = CACHE_DIR / f"sweden_sr_contour_projection_full_{N_SIM_LABEL}.csv"
INPUTS_CSV = RESULTS_DIR / f"sweden_sr_contour_projection_inputs_{N_SIM_LABEL}.csv"
SUMMARY_CSV = RESULTS_DIR / f"sweden_sr_contour_projection_summary_{N_SIM_LABEL}.csv"
XC_ENVELOPE_CSV = RESULTS_DIR / "sweden_fig4d_conservative_xc_envelope_1900_2020.csv"
OUTPUT_PNG = FIGURE_DIR / f"sweden_sr_contour_projection_1900_2100_{N_SIM_LABEL}.png"
OUTPUT_PDF = FIGURE_DIR / f"sweden_sr_contour_projection_1900_2100_{N_SIM_LABEL}.pdf"

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

VARIANTS = ("low", "central", "high")
PROJECTION_SCENARIOS = ("linear", "exponential")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def source_year_for_target(year: int) -> int:
    if year == 2020:
        return MODERN_SOURCE_YEAR
    return year


def load_fig4_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    robustness = pd.read_csv(FIG4D_CSV)
    extrinsic = pd.read_csv(FIG4C_CSV)
    return robustness, extrinsic


def build_or_load_xc_envelope(robustness: pd.DataFrame) -> pd.DataFrame:
    historical_source_years = {source_year_for_target(year) for year in HISTORICAL_YEARS}
    trend_years = set(range(TREND_FIT_START_YEAR, TREND_FIT_END_YEAR + 1))
    required_years = historical_source_years | trend_years
    required_columns = {"year", "conservative_ci_low", "conservative_ci_high"}

    if XC_ENVELOPE_CSV.exists():
        cached = pd.read_csv(XC_ENVELOPE_CSV)
        if required_columns.issubset(cached.columns) and required_years.issubset(set(cached["year"])):
            return cached

    rows = robustness[robustness["year"].isin(required_years)].copy()
    missing = required_years - set(rows["year"].astype(int))
    if missing:
        raise ValueError(f"Missing robustness rows for Xc envelope: {sorted(missing)}")

    envelope = conservative_envelope(rows)
    envelope.to_csv(XC_ENVELOPE_CSV, index=False)
    return envelope


def build_inputs() -> pd.DataFrame:
    robustness, extrinsic = load_fig4_tables()
    xc_envelope = build_or_load_xc_envelope(robustness)
    historical = build_historical_inputs(robustness, extrinsic, xc_envelope)
    projected = build_projection_inputs(robustness, extrinsic, xc_envelope)
    inputs = pd.concat([historical, projected], ignore_index=True)
    inputs.to_csv(INPUTS_CSV, index=False)
    return inputs


def build_historical_inputs(
    robustness: pd.DataFrame,
    extrinsic: pd.DataFrame,
    xc_envelope: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for target_year in HISTORICAL_YEARS:
        source_year = source_year_for_target(target_year)
        x_row = one_year_row(robustness, source_year)
        h_row = one_year_row(extrinsic, source_year)
        envelope = one_year_row(xc_envelope, source_year)
        rows.append(
            input_row(
                target_year=target_year,
                source_year=source_year,
                scenario="historical",
                xc_central=float(x_row["estimate"]),
                xc_low=float(envelope["conservative_ci_low"]),
                xc_high=float(envelope["conservative_ci_high"]),
                h_ext_central=float(h_row["estimate"]),
                h_ext_low=float(h_row["ci_low"]),
                h_ext_high=float(h_row["ci_high"]),
            )
        )
    return pd.DataFrame(rows)


def build_projection_inputs(
    robustness: pd.DataFrame,
    extrinsic: pd.DataFrame,
    xc_envelope: pd.DataFrame,
) -> pd.DataFrame:
    fit_xc = add_envelope_to_robustness(robustness, xc_envelope)
    fit_xc = fit_xc[
        (fit_xc["year"] >= TREND_FIT_START_YEAR)
        & (fit_xc["year"] <= TREND_FIT_END_YEAR)
    ].copy()
    fit_h = extrinsic[
        (extrinsic["year"] >= TREND_FIT_START_YEAR)
        & (extrinsic["year"] <= TREND_FIT_END_YEAR)
    ].copy()

    modern_x = one_year_row(add_envelope_to_robustness(robustness, xc_envelope), MODERN_SOURCE_YEAR)
    modern_h = one_year_row(extrinsic, MODERN_SOURCE_YEAR)

    rows = []
    for scenario in PROJECTION_SCENARIOS:
        xc_trends = fit_xc_trends(fit_xc, scenario)
        h_trends = fit_log_trends(fit_h)
        for target_year in PROJECTION_YEARS:
            years_from_anchor = target_year - 2020
            rows.append(
                input_row(
                    target_year=target_year,
                    source_year=MODERN_SOURCE_YEAR,
                    scenario=scenario,
                    xc_central=project_xc(
                        float(modern_x["estimate"]),
                        xc_trends["central"],
                        years_from_anchor,
                        scenario,
                    ),
                    xc_low=project_xc(
                        float(modern_x["conservative_ci_low"]),
                        xc_trends["low"],
                        years_from_anchor,
                        scenario,
                    ),
                    xc_high=project_xc(
                        float(modern_x["conservative_ci_high"]),
                        xc_trends["high"],
                        years_from_anchor,
                        scenario,
                    ),
                    h_ext_central=project_log_value(
                        float(modern_h["estimate"]),
                        h_trends["central"],
                        years_from_anchor,
                    ),
                    h_ext_low=project_log_value(
                        float(modern_h["ci_low"]),
                        h_trends["low"],
                        years_from_anchor,
                    ),
                    h_ext_high=project_log_value(
                        float(modern_h["ci_high"]),
                        h_trends["high"],
                        years_from_anchor,
                    ),
                )
            )
    return pd.DataFrame(rows)


def add_envelope_to_robustness(robustness: pd.DataFrame, envelope: pd.DataFrame) -> pd.DataFrame:
    keep = envelope[["year", "conservative_ci_low", "conservative_ci_high"]].copy()
    return robustness.merge(keep, on="year", how="left")


def fit_xc_trends(data: pd.DataFrame, scenario: str) -> dict[str, float]:
    years = data["year"].to_numpy(dtype=float)
    if scenario == "linear":
        return {
            "central": float(np.polyfit(years, data["estimate"].to_numpy(dtype=float), 1)[0]),
            "low": float(np.polyfit(years, data["conservative_ci_low"].to_numpy(dtype=float), 1)[0]),
            "high": float(np.polyfit(years, data["conservative_ci_high"].to_numpy(dtype=float), 1)[0]),
        }

    return {
        "central": float(np.polyfit(years, np.log(data["estimate"].to_numpy(dtype=float)), 1)[0]),
        "low": float(np.polyfit(years, np.log(data["conservative_ci_low"].to_numpy(dtype=float)), 1)[0]),
        "high": float(np.polyfit(years, np.log(data["conservative_ci_high"].to_numpy(dtype=float)), 1)[0]),
    }


def fit_log_trends(data: pd.DataFrame) -> dict[str, float]:
    years = data["year"].to_numpy(dtype=float)
    return {
        "central": float(np.polyfit(years, np.log(data["estimate"].to_numpy(dtype=float)), 1)[0]),
        "low": float(np.polyfit(years, np.log(data["ci_low"].to_numpy(dtype=float)), 1)[0]),
        "high": float(np.polyfit(years, np.log(data["ci_high"].to_numpy(dtype=float)), 1)[0]),
    }


def project_xc(anchor: float, slope: float, years_from_anchor: int, scenario: str) -> float:
    if scenario == "linear":
        return float(anchor + slope * years_from_anchor)
    return float(anchor * np.exp(slope * years_from_anchor))


def project_log_value(anchor: float, log_slope: float, years_from_anchor: int) -> float:
    return float(anchor * np.exp(log_slope * years_from_anchor))


def one_year_row(data: pd.DataFrame, year: int) -> pd.Series:
    rows = data[data["year"] == year]
    if rows.empty:
        raise ValueError(f"Missing year {year} in input table.")
    return rows.iloc[0]


def input_row(
    *,
    target_year: int,
    source_year: int,
    scenario: str,
    xc_central: float,
    xc_low: float,
    xc_high: float,
    h_ext_central: float,
    h_ext_low: float,
    h_ext_high: float,
) -> dict[str, float | int | str]:
    xc_values = [float(xc_low), float(xc_central), float(xc_high)]
    h_ext_values = [float(h_ext_low), float(h_ext_central), float(h_ext_high)]
    return {
        "target_year": int(target_year),
        "source_year": int(source_year),
        "scenario": scenario,
        "xc_central": float(xc_central),
        "xc_low": float(np.nanmin(xc_values)),
        "xc_high": float(np.nanmax(xc_values)),
        "h_ext_central": float(h_ext_central),
        "h_ext_low": float(np.nanmin(h_ext_values)),
        "h_ext_high": float(np.nanmax(h_ext_values)),
    }


def completed_run_keys() -> set[tuple[str, int, str]]:
    if not RAW_SIM_CSV.exists():
        return set()

    with RAW_SIM_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (row["scenario"], int(row["target_year"]), row["variant"])
            for row in reader
        }


def append_row(row: dict[str, float | int | str]) -> None:
    fieldnames = list(row.keys())
    write_header = not RAW_SIM_CSV.exists()
    with RAW_SIM_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def variant_values(input_row, variant: str) -> tuple[float, float]:
    if variant == "low":
        return float(input_row.xc_low), float(input_row.h_ext_high)
    if variant == "high":
        return float(input_row.xc_high), float(input_row.h_ext_low)
    return float(input_row.xc_central), float(input_row.h_ext_central)


def build_params(xc_factor: float, seed: int) -> dict[str, np.ndarray]:
    params = {
        "eta": np.full(N_SIM, BASELINE["eta"], dtype=float),
        "beta": np.full(N_SIM, BASELINE["beta"], dtype=float),
        "kappa": np.full(N_SIM, BASELINE["kappa"], dtype=float),
        "epsilon": np.full(N_SIM, BASELINE["epsilon"], dtype=float),
    }
    params["Xc"] = sample_positive_gaussian(
        mean=BASELINE["Xc"] * xc_factor,
        rel_std=BASELINE["xc_std_frac"],
        n=N_SIM,
        seed=seed,
    )
    return params


def simulate_one(input_row, variant: str) -> dict[str, float | int | str]:
    xc_factor, h_ext = variant_values(input_row, variant)
    seed = stable_seed(
        "fig4_sr_contour_projection",
        input_row.scenario,
        int(input_row.target_year),
        variant,
        N_SIM,
    )
    params = build_params(xc_factor=xc_factor, seed=seed)

    sim = utils.create_sr_simulation(
        params_dict=params,
        n=N_SIM,
        h_ext=h_ext,
        parallel=True,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        break_early=True,
        random_seed=seed,
    )

    row: dict[str, float | int | str] = {
        "scenario": input_row.scenario,
        "target_year": int(input_row.target_year),
        "source_year": int(input_row.source_year),
        "variant": variant,
        "xc_factor": float(xc_factor),
        "h_ext": float(h_ext),
        "n": N_SIM,
        "from_age": FROM_AGE,
        "random_seed": int(seed),
    }
    for label, survival_level in SURVIVAL_LEVELS.items():
        row[label] = sim.find_time_at_survival(
            survival_level,
            from_t=FROM_AGE,
            relative=False,
        )
    return row


def run_missing_simulations(inputs: pd.DataFrame) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    done = completed_run_keys()

    for input_row in inputs.itertuples(index=False):
        for variant in VARIANTS:
            key = (input_row.scenario, int(input_row.target_year), variant)
            if key in done:
                continue

            xc_factor, h_ext = variant_values(input_row, variant)
            print(
                f"Running {input_row.scenario} year={int(input_row.target_year)} "
                f"variant={variant} Xc={xc_factor:.4f} h_ext={h_ext:.3e}",
                flush=True,
            )
            append_row(simulate_one(input_row, variant))

    return pd.read_csv(RAW_SIM_CSV)


def summarize_simulations(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (scenario, year), rows in raw.groupby(["scenario", "target_year"], sort=True):
        central = rows[rows["variant"] == "central"].iloc[0]
        record = {"scenario": scenario, "year": int(year)}
        for label in SURVIVAL_LEVELS:
            values = rows[label].to_numpy(dtype=float)
            record[label] = float(central[label])
            record[f"{label} low"] = float(np.nanmin(values))
            record[f"{label} high"] = float(np.nanmax(values))
        records.append(record)

    summary = pd.DataFrame(records).sort_values(["scenario", "year"])
    summary = anchor_projection_start_to_history(summary)
    summary.to_csv(SUMMARY_CSV, index=False)
    return summary


def anchor_projection_start_to_history(summary: pd.DataFrame) -> pd.DataFrame:
    history_start = summary[
        (summary["scenario"] == "historical")
        & (summary["year"] == 2020)
    ]
    if history_start.empty:
        return summary

    anchored = summary.copy()
    contour_columns = [
        column
        for column in anchored.columns
        if column not in {"scenario", "year"}
    ]
    history_values = history_start.iloc[0][contour_columns]
    mask = (
        anchored["scenario"].isin(PROJECTION_SCENARIOS)
        & (anchored["year"] == 2020)
    )
    anchored.loc[mask, contour_columns] = history_values.to_numpy()
    return anchored


def plot_projection(summary: pd.DataFrame) -> None:
    hmd = pd.read_csv(HMD_CONTOURS_CSV)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 18,
            "axes.labelsize": 21,
            "axes.titlesize": 22,
            "legend.fontsize": 15,
            "legend.title_fontsize": 15,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.axvspan(2020, 2100, color="0.88", alpha=0.48, zorder=0)
    plot_hmd_contours(ax, hmd)
    plot_historical_sr(ax, summary)
    plot_projection_sr(
        ax,
        summary,
        "linear",
        linestyle="-",
        fill_alpha=0.09,
        line_alpha=0.72,
    )
    plot_projection_sr(
        ax,
        summary,
        "exponential",
        linestyle="--",
        fill_alpha=0.055,
        line_alpha=0.52,
    )
    plot_recent_linear_continuations(ax, summary)
    add_direct_contour_labels(ax, summary)

    ax.axvline(2020, color="0.45", linestyle=":", linewidth=1.6)
    ax.text(2022.0, 122.7, "forecast", color="0.35", fontsize=11, va="top", rotation=90)

    ax.set_title("Extrapolating recent increase in robustness predicts diminishing longevity gains")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Lifespan [years]\n(conditional on survival to age {FROM_AGE})")
    ax.set_xlim(1900, 2100)
    ax.set_ylim(55, 125)
    ax.set_xticks(np.arange(1900, 2101, 20))
    ax.tick_params(axis="both", width=1.2, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.20", alpha=0.08, linewidth=0.8)
    fitted_legend = ax.legend(
        handles=fitted_handles(),
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.035),
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
    for label in SURVIVAL_LEVELS:
        ax.plot(
            hmd["year"],
            hmd[f"{label} smoothed"],
            color=COLORS[label],
            linewidth=2.8,
            alpha=0.65,
            zorder=2,
        )


def plot_historical_sr(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = summary[summary["scenario"] == "historical"].sort_values("year")
    for label in SURVIVAL_LEVELS:
        ax.fill_between(
            rows["year"],
            rows[f"{label} low"],
            rows[f"{label} high"],
            color=COLORS[label],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rows["year"],
            rows[label],
            color=COLORS[label],
            linewidth=1.9,
            marker="o",
            markersize=4.5,
            markeredgecolor="black",
            markeredgewidth=0.45,
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
    for label in SURVIVAL_LEVELS:
        ax.fill_between(
            rows["year"],
            rows[f"{label} low"],
            rows[f"{label} high"],
            color=COLORS[label],
            alpha=fill_alpha,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            rows["year"],
            rows[label],
            color=COLORS[label],
            linewidth=2.0,
            linestyle=linestyle,
            alpha=line_alpha,
            zorder=3,
        )


def add_direct_contour_labels(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = summary[summary["scenario"] == "historical"].sort_values("year")
    label_year = 2000
    x_values = rows["year"].to_numpy(dtype=float)

    for label in SURVIVAL_LEVELS:
        y_values = rows[label].to_numpy(dtype=float)
        y_label = float(np.interp(label_year, x_values, y_values))
        text = ax.text(
            label_year,
            y_label,
            label,
            color=COLORS[label],
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="left",
            zorder=6,
        )
        text.set_path_effects(
            [
                path_effects.Stroke(linewidth=3.0, foreground="white", alpha=0.85),
                path_effects.Normal(),
            ]
        )


def plot_recent_linear_continuations(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = summary[summary["scenario"] == "historical"].sort_values("year")
    recent = rows.tail(4)
    line_years = np.arange(int(recent["year"].min()), 2101)

    for label in SURVIVAL_LEVELS:
        slope, intercept = np.polyfit(
            recent["year"].to_numpy(dtype=float),
            recent[label].to_numpy(dtype=float),
            1,
        )
        ax.plot(
            line_years,
            slope * line_years + intercept,
            color="0.18",
            linewidth=1.25,
            linestyle=(0, (1.0, 3.0)),
            alpha=0.28,
            zorder=2.5,
        )


def fitted_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="0.45", lw=2.8, label="HMD period contours"),
        Line2D(
            [0],
            [0],
            color="0.20",
            lw=1.9,
            marker="o",
            label=r"SR model: projected $X_c$ and fitted $m_{ex}$",
        ),
    ]


def projection_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color="0.20", lw=2.0, linestyle="-", alpha=0.72, label="Linear increase in robustness"),
        Line2D([0], [0], color="0.20", lw=2.0, linestyle="--", alpha=0.52, label="Exponential increase in robustness"),
    ]


def contour_handles() -> list[Line2D]:
    display_order = ("Median", "Top 1%", "Top 10%", "Top 0.01%")
    return [
        Line2D([0], [0], color=COLORS[label], lw=3.0, label=label)
        for label in display_order
    ]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    inputs = build_inputs()
    raw = run_missing_simulations(inputs)
    summary = summarize_simulations(raw)
    plot_projection(summary)
    print(f"saved {RAW_SIM_CSV}")
    print(f"saved {INPUTS_CSV}")
    print(f"saved {SUMMARY_CSV}")
    print(f"saved {XC_ENVELOPE_CSV}")
    print(f"saved {OUTPUT_PNG}")
    print(f"saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
