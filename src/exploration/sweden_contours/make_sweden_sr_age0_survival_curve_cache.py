#!/usr/bin/env python3
"""Cache age-0 SR survival curves for Fig. 4d-style contour projections.

Each simulation writes the annual survival curve on ages 0..120. The raw cache
is long-form so future contour choices can be rebuilt without rerunning SR.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.figures.Fig4_new.make_fig4_sr_contour_projection import build_inputs
from src.figures.steepness_longevity.run_sweden2019_sensitivity import (
    BASELINE,
    DT,
    TMAX,
    sample_positive_gaussian,
)
from src.shared.thresholds.paths import SAVED_RESULTS_DIR


FROM_AGE = 0
DISPLAY_START_YEAR = 1980
N_SIM = int(os.environ.get("SR_AGE0_N_SIM", int(1e6)))
N_SIM_LABEL = os.environ.get("SR_AGE0_N_SIM_LABEL", "n1m")
MAX_RUNS = int(os.environ.get("SR_AGE0_MAX_RUNS", "0"))
SURVIVAL_AGES = tuple(range(0, 121))

RESULTS_DIR = SAVED_RESULTS_DIR / "fig4_new"
CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "Fig4_new"

INPUTS_CSV = RESULTS_DIR / f"sweden_sr_contour_projection_inputs_{N_SIM_LABEL}.csv"
RAW_CURVES_CSV = CACHE_DIR / f"sweden_sr_age0_survival_curves_1980_2100_{N_SIM_LABEL}.csv"
RAW_METRICS_CSV = CACHE_DIR / f"sweden_sr_age0_survival_metrics_1980_2100_{N_SIM_LABEL}.csv"
SUMMARY_CURVES_CSV = RESULTS_DIR / f"sweden_sr_age0_survival_curve_summary_1980_2100_{N_SIM_LABEL}.csv"
SUMMARY_METRICS_CSV = RESULTS_DIR / f"sweden_sr_age0_mean_lifespan_contour_summary_1980_2100_{N_SIM_LABEL}.csv"

SURVIVAL_LEVELS = {
    "Top 10%": 0.1,
    "Top 1%": 0.01,
    "Top 0.01%": 1e-4,
}

VARIANTS = ("low", "central", "high")


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_inputs() -> pd.DataFrame:
    if INPUTS_CSV.exists():
        inputs = pd.read_csv(INPUTS_CSV)
    else:
        inputs = build_inputs()

    return inputs[inputs["target_year"] >= DISPLAY_START_YEAR].copy()


def completed_run_keys() -> set[tuple[str, int, str]]:
    if not RAW_CURVES_CSV.exists():
        return set()

    curves = pd.read_csv(
        RAW_CURVES_CSV,
        usecols=["scenario", "target_year", "variant", "age"],
    )
    counts = curves.groupby(["scenario", "target_year", "variant"])["age"].nunique()
    return {
        (str(scenario), int(year), str(variant))
        for (scenario, year, variant), count in counts.items()
        if count == len(SURVIVAL_AGES)
    }


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


def simulate_one(input_row, variant: str) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    target_year = int(input_row.target_year)
    xc_factor, h_ext = variant_values(input_row, variant)
    seed = stable_seed(
        "fig4_sr_contour_projection",
        input_row.scenario,
        target_year,
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
        save_times=TMAX,
        break_early=True,
        random_seed=seed,
    )

    death_times = np.asarray(sim.death_times, dtype=float)
    death_times_for_mean = np.where(np.isfinite(death_times), death_times, TMAX)
    survival_values = survival_curve_from_death_times(death_times)

    base_row: dict[str, float | int | str] = {
        "scenario": input_row.scenario,
        "target_year": target_year,
        "source_year": int(input_row.source_year),
        "variant": variant,
        "xc_factor": float(xc_factor),
        "h_ext": float(h_ext),
        "n": N_SIM,
        "from_age": FROM_AGE,
        "random_seed": int(seed),
    }
    curve_rows = [
        {
            **base_row,
            "age": int(age),
            "survival": float(survival),
        }
        for age, survival in zip(SURVIVAL_AGES, survival_values)
    ]

    metric_row = {
        **base_row,
        "mean_lifespan": float(np.mean(death_times_for_mean)),
        "mean_lifespan_from_annual_curve_0_120": integrate_annual_survival(
            np.array(SURVIVAL_AGES, dtype=float),
            survival_values,
        ),
        "survival_at_120": float(survival_values[-1]),
        "n_surviving_to_tmax": int(np.sum(death_times_for_mean >= TMAX)),
    }
    for label, level in SURVIVAL_LEVELS.items():
        metric_row[label] = find_threshold_age(
            np.array(SURVIVAL_AGES, dtype=float),
            survival_values,
            level,
        )

    return curve_rows, metric_row


def survival_curve_from_death_times(death_times: np.ndarray) -> np.ndarray:
    return np.array(
        [np.mean(death_times > age) for age in SURVIVAL_AGES],
        dtype=float,
    )


def integrate_annual_survival(ages: np.ndarray, survival: np.ndarray) -> float:
    return float(np.trapezoid(survival, ages))


def find_threshold_age(ages: np.ndarray, survival: np.ndarray, survival_level: float) -> float:
    if len(ages) == 0 or survival[-1] > survival_level:
        return np.nan

    crossing_positions = np.where(survival <= survival_level)[0]
    if len(crossing_positions) == 0:
        return np.nan

    after = crossing_positions[0]
    if after == 0:
        return float(ages[0])

    before = after - 1
    age_before = ages[before]
    age_after = ages[after]
    survival_before = survival[before]
    survival_after = survival[after]

    if survival_before <= 0 or survival_after <= 0:
        fraction = (survival_level - survival_before) / (
            survival_after - survival_before
        )
        return float(age_before + fraction * (age_after - age_before))

    log_before = np.log(survival_before)
    log_after = np.log(survival_after)
    log_target = np.log(survival_level)
    fraction = (log_target - log_before) / (log_after - log_before)
    return float(age_before + fraction * (age_after - age_before))


def append_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_missing_simulations(inputs: pd.DataFrame) -> None:
    done = completed_run_keys()
    runs_completed = 0
    for input_row in inputs.sort_values(["scenario", "target_year"]).itertuples(index=False):
        for variant in VARIANTS:
            key = (str(input_row.scenario), int(input_row.target_year), variant)
            if key in done:
                continue

            xc_factor, h_ext = variant_values(input_row, variant)
            print(
                f"Running {input_row.scenario} year={int(input_row.target_year)} "
                f"variant={variant} from_age=0 Xc={xc_factor:.4f} h_ext={h_ext:.3e}",
                flush=True,
            )
            curve_rows, metric_row = simulate_one(input_row, variant)
            append_rows(RAW_CURVES_CSV, curve_rows)
            append_rows(RAW_METRICS_CSV, [metric_row])
            done.add(key)
            runs_completed += 1
            if MAX_RUNS and runs_completed >= MAX_RUNS:
                return


def write_summaries() -> None:
    curves = pd.read_csv(RAW_CURVES_CSV)
    metrics = pd.read_csv(RAW_METRICS_CSV)

    curve_summary = summarize_curves(curves)
    metric_summary = summarize_metrics(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_summary.to_csv(SUMMARY_CURVES_CSV, index=False)
    metric_summary.to_csv(SUMMARY_METRICS_CSV, index=False)


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (scenario, year, age), rows in curves.groupby(["scenario", "target_year", "age"], sort=True):
        central = rows[rows["variant"] == "central"]
        if central.empty:
            continue

        values = rows["survival"].to_numpy(dtype=float)
        records.append(
            {
                "scenario": scenario,
                "year": int(year),
                "age": int(age),
                "survival": float(central.iloc[0]["survival"]),
                "survival low": float(np.nanmin(values)),
                "survival high": float(np.nanmax(values)),
            }
        )

    columns = ["scenario", "year", "age", "survival", "survival low", "survival high"]
    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records, columns=columns).sort_values(["scenario", "year", "age"])


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mean_lifespan",
        "mean_lifespan_from_annual_curve_0_120",
        "survival_at_120",
        "Top 10%",
        "Top 1%",
        "Top 0.01%",
    ]

    records = []
    for (scenario, year), rows in metrics.groupby(["scenario", "target_year"], sort=True):
        central = rows[rows["variant"] == "central"]
        if central.empty:
            continue

        record: dict[str, float | int | str] = {
            "scenario": scenario,
            "year": int(year),
        }
        for column in metric_columns:
            values = rows[column].to_numpy(dtype=float)
            record[column] = float(central.iloc[0][column])
            record[f"{column} low"] = float(np.nanmin(values))
            record[f"{column} high"] = float(np.nanmax(values))
        records.append(record)

    columns = ["scenario", "year"]
    for column in metric_columns:
        columns.extend([column, f"{column} low", f"{column} high"])

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records, columns=columns).sort_values(["scenario", "year"])


def main() -> None:
    inputs = load_inputs()
    run_missing_simulations(inputs)
    write_summaries()
    print(f"saved {RAW_CURVES_CSV}")
    print(f"saved {RAW_METRICS_CSV}")
    print(f"saved {SUMMARY_CURVES_CSV}")
    print(f"saved {SUMMARY_METRICS_CSV}")


if __name__ == "__main__":
    main()
