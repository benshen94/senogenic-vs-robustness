from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis import HMD
from ageing_packages.utils import sr_utils as utils
from senogenic_vs_robustness import thresholds_functions as th
from senogenic_vs_robustness.paths import FIGURES_DIR, HMD_DATA_DIR, RESULTS_DIR


OUTPUT_CSV = RESULTS_DIR / "sweden_sr_xc_hext_contours_1900_2100_10yr_n100k.csv"
OUTPUT_PKL = RESULTS_DIR / "sweden_sr_xc_hext_contours_1900_2100_10yr_n100k.pkl"
OUTPUT_PNG = FIGURES_DIR / "sweden_sr_xc_hext_contours_overlay_1900_2100.png"

FROM_AGE = 20
REF_YEAR = 2020
START_YEAR = 1900
END_YEAR = 2020
PROJECTION_END_YEAR = 2100
YEAR_STEP = 10
N_SIM = int(1e5)
TMAX = 200

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


def load_sweden_period_table() -> pd.DataFrame:
    path = HMD_DATA_DIR / "mortality.org_File_GetDocument_hmd.v6_SWE_STATS_bltper_1x1.txt"
    rows = []
    with path.open(errors="replace") as handle:
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

            if year < START_YEAR or year > END_YEAR:
                continue

            rows.append({"Year": year, "Age": age, "lx": lx})

    return pd.DataFrame(rows)


def threshold_age(year_data: pd.DataFrame, target_survival: float) -> float:
    data = year_data.sort_values("Age")
    ages = data["Age"].to_numpy(dtype=float)
    survival = data["conditional_survival"].to_numpy(dtype=float)

    if len(ages) == 0:
        return np.nan
    if survival[0] <= target_survival:
        return ages[0]
    if survival[-1] > target_survival:
        return np.nan

    crossing_indices = np.where(survival <= target_survival)[0]
    if len(crossing_indices) == 0:
        return np.nan

    after = crossing_indices[0]
    if after == 0:
        return ages[0]

    before = after - 1
    age_before = ages[before]
    age_after = ages[after]
    survival_before = survival[before]
    survival_after = survival[after]

    if survival_before <= 0 or survival_after <= 0:
        fraction = (target_survival - survival_before) / (
            survival_after - survival_before
        )
        return age_before + fraction * (age_after - age_before)

    log_before = np.log(survival_before)
    log_after = np.log(survival_after)
    log_target = np.log(target_survival)
    fraction = (log_target - log_before) / (log_after - log_before)
    return age_before + fraction * (age_after - age_before)


def build_hmd_contours(life_table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year, year_data in life_table.groupby("Year"):
        year_data = year_data.copy()
        age_20_rows = year_data.loc[year_data["Age"] == FROM_AGE, "lx"]
        if age_20_rows.empty:
            continue

        lx_at_20 = float(age_20_rows.iloc[0])
        year_data["conditional_survival"] = year_data["lx"] / lx_at_20

        record = {"Year": int(year)}
        for label, survival_level in SURVIVAL_LEVELS.items():
            record[label] = threshold_age(year_data, survival_level)
        records.append(record)

    contours = pd.DataFrame(records).sort_values("Year")
    for label in SURVIVAL_LEVELS:
        contours[label] = contours[label].rolling(
            window=5, center=True, min_periods=1
        ).mean()
    return contours


def load_xc_and_hext_by_year() -> pd.DataFrame:
    sweden_period = HMD(country="swe", gender="both", data_type="period")
    observed_years = np.arange(START_YEAR, END_YEAR + 1, YEAR_STEP)
    _, valid_years, xc_factors = th.map_xc_factor_to_years(
        sweden_period,
        observed_years,
        ref_year=REF_YEAR,
        from_t=FROM_AGE,
        ax=None,
        plot=False,
    )

    observed = pd.DataFrame(
        {
            "year": valid_years.astype(int),
            "xc_factor": xc_factors.astype(float),
            "source": "observed_mapped",
        }
    )

    h_ext_values = []
    for year in observed["year"]:
        params = sweden_period.fit_ggm(year=int(year), age_start=20, age_end=105)
        h_ext_values.append(float(params["m"]))
    observed["h_ext"] = h_ext_values

    fit_mask = (observed["year"] >= 1980) & (observed["year"] <= 2020)
    fit_years = observed.loc[fit_mask, "year"].to_numpy(dtype=float)
    fit_xc = observed.loc[fit_mask, "xc_factor"].to_numpy(dtype=float)
    fit_hext = observed.loc[fit_mask, "h_ext"].to_numpy(dtype=float)

    xc_slope, xc_intercept = np.polyfit(fit_years, fit_xc, 1)
    hext_slope, hext_log_intercept = np.polyfit(fit_years, np.log(fit_hext), 1)

    projection_years = np.arange(END_YEAR + YEAR_STEP, PROJECTION_END_YEAR + 1, YEAR_STEP)
    projected = pd.DataFrame(
        {
            "year": projection_years.astype(int),
            "xc_factor": xc_slope * projection_years + xc_intercept,
            "h_ext": np.exp(hext_slope * projection_years + hext_log_intercept),
            "source": "extrapolated_1980_2020",
        }
    )

    return pd.concat([observed, projected], ignore_index=True)


def build_base_params() -> dict[str, np.ndarray]:
    base = utils.load_baseline_human_params_dict()
    base["Xc"] = 1.08 * base["Xc"]
    base["eta"] = 1.26 * base["eta"]
    base["beta"] = 1.17 * base["beta"]

    params = {}
    for key, value in base.items():
        if np.isscalar(value):
            params[key] = np.repeat(np.array([value]), N_SIM)
            continue

        array = np.asarray(value)
        if array.size == 1:
            params[key] = np.repeat(array, N_SIM)
            continue

        params[key] = array

    return params


def simulate_year(
    base_params: dict[str, np.ndarray],
    year: int,
    xc_factor: float,
    h_ext: float,
) -> dict[str, float | int]:
    params = {key: value.copy() for key, value in base_params.items()}
    params["Xc"] = params["Xc"] * xc_factor

    sim = utils.create_sr_simulation(
        params_dict=params,
        n=N_SIM,
        parallel=True,
        tmax=TMAX,
        h_ext=h_ext,
        random_seed=int(year),
    )

    record: dict[str, float | int] = {
        "year": int(year),
        "xc_factor": float(xc_factor),
        "h_ext": float(h_ext),
    }
    for label, survival_level in SURVIVAL_LEVELS.items():
        record[label] = sim.find_time_at_survival(
            survival_level, from_t=FROM_AGE, relative=False
        )
    return record


def run_simulations(xc_table: pd.DataFrame) -> pd.DataFrame:
    if OUTPUT_CSV.exists():
        return pd.read_csv(OUTPUT_CSV)

    base_params = build_base_params()
    records = []
    start = time.time()

    for row_index, row in enumerate(xc_table.itertuples(index=False), start=1):
        record = simulate_year(
            base_params,
            int(row.year),
            float(row.xc_factor),
            float(row.h_ext),
        )
        record["source"] = row.source
        records.append(record)
        pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)

        elapsed = time.time() - start
        print(
            f"{row_index}/{len(xc_table)} year={record['year']}: "
            f"Xc factor={record['xc_factor']:.4f}, "
            f"h_ext={record['h_ext']:.3e}, "
            f"median={record['Median']:.2f}, "
            f"top 0.01%={record['Top 0.01%']:.2f}, "
            f"elapsed={elapsed / 60:.1f} min",
            flush=True,
        )

    with OUTPUT_PKL.open("wb") as handle:
        pickle.dump(
            {
                "records": records,
                "from_age": FROM_AGE,
                "n_sim": N_SIM,
                "survival_levels": SURVIVAL_LEVELS,
                "projection": "Xc linear and h_ext log-linear fits over 1980-2020",
            },
            handle,
        )

    return pd.DataFrame(records)


def plot_overlay(sim_results: pd.DataFrame, hmd_contours: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 18,
            "axes.titlesize": 22,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(13.5, 8))
    for label in SURVIVAL_LEVELS:
        ax.plot(
            hmd_contours["Year"],
            hmd_contours[label],
            color=COLORS[label],
            linewidth=3.0,
            label=f"HMD {label}",
        )
        observed = sim_results[sim_results["source"] == "observed_mapped"]
        projected = sim_results[sim_results["source"] == "extrapolated_1980_2020"]
        ax.scatter(
            observed["year"],
            observed[label],
            s=64,
            color=COLORS[label],
            marker="o",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        ax.scatter(
            projected["year"],
            projected[label],
            s=70,
            color=COLORS[label],
            marker="^",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.axhline(110, color="0.35", linestyle=":", linewidth=2)
    ax.axvline(2020, color="0.35", linestyle="--", linewidth=1.8)
    ax.text(1901, 111.0, "HMD 110+ open interval", color="0.25", fontsize=13)
    ax.set_title("Sweden SR samples using yearly fitted $X_c$ and $h_{ext}$")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Age at conditional survivor percentile from age {FROM_AGE}")
    ax.set_xlim(START_YEAR, PROJECTION_END_YEAR)
    ax.set_ylim(45, 125)
    ax.grid(True, color="0.88", linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            label="SR fitted-$X_c$ samples",
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="^",
            linestyle="None",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="black",
            label="SR extrapolated samples",
        )
    )
    labels.extend(["SR fitted-$X_c$, fitted-$h_{ext}$", "SR extrapolated samples"])
    ax.legend(handles, labels, loc="lower right", ncol=2, frameon=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    xc_table = load_xc_and_hext_by_year()
    sim_results = run_simulations(xc_table)
    hmd_contours = build_hmd_contours(load_sweden_period_table())
    plot_overlay(sim_results, hmd_contours)

    print(f"saved {OUTPUT_CSV}")
    print(f"saved {OUTPUT_PKL}")
    print(f"saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
