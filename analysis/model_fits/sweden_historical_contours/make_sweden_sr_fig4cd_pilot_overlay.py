from __future__ import annotations

import csv
import hashlib
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import OptimizeWarning, curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis import HMD
from ageing_packages.utils import sr_utils as utils
from analysis.figures.figure4 import make_fig4_ab_sweden_period_projection as fig4
from analysis.figures.steepness_longevity.run_sweden2019_sensitivity import (
    BASELINE,
    DT,
    TMAX,
    sample_positive_gaussian,
)
from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR as PROJECT_RESULTS_DIR


PILOT_YEARS = (1900, 1950, 2000, 2019)
FROM_AGE = 20
N_SIM = int(2e5)
SAVE_TIMES = TMAX

FIGURE_DIR = FIGURES_DIR / "Fig_4new"
RESULTS_DIR = PROJECT_RESULTS_DIR / "figure4"

HMD_CONTOURS_CSV = RESULTS_DIR / "sweden_period_both_conditional_age20_contours_1900_2020.csv"
FIG4C_CSV = RESULTS_DIR / "fig4c_extrinsic_mortality_projection.csv"
FIG4D_CSV = RESULTS_DIR / "fig4d_robustness_projection.csv"

OUTPUT_CSV = RESULTS_DIR / "sweden_sr_fig4cd_pilot_conservative_xc_hext_contours_n200k.csv"
OUTPUT_PNG = FIGURE_DIR / "sweden_sr_fig4cd_pilot_conservative_xc_hext_overlay_n200k.png"
ENVELOPE_CSV = RESULTS_DIR / "sweden_fig4d_conservative_xc_envelope_pilot.csv"

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

VARIANTS = {
    "low": "ci_low",
    "central": "estimate",
    "high": "ci_high",
}

GGM_FIT_WINDOWS = (
    (20, 95),
    (20, 100),
    (20, 105),
    (30, 95),
    (30, 100),
    (30, 105),
    (40, 95),
    (40, 100),
    (40, 105),
)

GGM_COVARIANCE_DRAWS = 400
POINT_MISMATCH_DRAWS = 400


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_inputs() -> pd.DataFrame:
    robustness = pd.read_csv(FIG4D_CSV)
    extrinsic = pd.read_csv(FIG4C_CSV)

    robustness = robustness[robustness["year"].isin(PILOT_YEARS)].copy()
    extrinsic = extrinsic[extrinsic["year"].isin(PILOT_YEARS)].copy()

    extrinsic = extrinsic.rename(
        columns={
            "estimate": "estimate_h_ext",
            "ci_low": "ci_low_h_ext",
            "ci_high": "ci_high_h_ext",
        }
    )
    needed = robustness.merge(
        extrinsic[["year", "estimate_h_ext", "ci_low_h_ext", "ci_high_h_ext"]],
        on="year",
        how="inner",
    )

    missing = set(PILOT_YEARS) - set(needed["year"])
    if missing:
        raise ValueError(f"Missing Fig4C/Fig4D inputs for years: {sorted(missing)}")

    envelope = build_or_load_conservative_envelope(needed)
    needed = needed.merge(envelope, on="year", how="left")
    needed["ci_low"] = needed["conservative_ci_low"]
    needed["ci_high"] = needed["conservative_ci_high"]
    return needed.sort_values("year").reset_index(drop=True)


def build_or_load_conservative_envelope(inputs: pd.DataFrame) -> pd.DataFrame:
    required_years = set(inputs["year"].astype(int))
    required_columns = {
        "year",
        "conservative_ci_low",
        "conservative_ci_high",
        "scenario_ci_low",
        "scenario_ci_high",
        "mismatch_ci_low",
        "mismatch_ci_high",
        "ggm_cov_ci_low",
        "ggm_cov_ci_high",
        "window_ci_low",
        "window_ci_high",
    }

    if ENVELOPE_CSV.exists():
        cached = pd.read_csv(ENVELOPE_CSV)
        if required_columns.issubset(cached.columns) and required_years.issubset(set(cached["year"])):
            return cached[cached["year"].isin(required_years)].copy()

    envelope = conservative_envelope(inputs)
    envelope.to_csv(ENVELOPE_CSV, index=False)
    return envelope


def conservative_envelope(inputs: pd.DataFrame) -> pd.DataFrame:
    hmd = HMD(country="swe", gender="both", data_type="period")
    model_rows = fig4.model_rows_for_fit("Xc")
    mean_curve = fig4.mean_model_curve(model_rows)
    sr_median, sr_steepness = fig4.load_sr_baseline()

    rows = []
    for row in inputs.itertuples(index=False):
        point = np.array([float(row.point_x), float(row.point_y)])
        scenario_values = fit_point_to_all_scenarios(point, model_rows)
        mismatch_values = mismatch_envelope_values(
            point=point,
            radius=float(row.distance_to_mean_curve),
            model_rows=model_rows,
            seed=stable_seed("mismatch", int(row.year)),
        )
        ggm_values = ggm_covariance_envelope_values(
            hmd=hmd,
            year=int(row.year),
            model_rows=model_rows,
            sr_median=sr_median,
            sr_steepness=sr_steepness,
            seed=stable_seed("ggm_cov", int(row.year)),
        )
        window_values = fit_window_envelope_values(
            hmd=hmd,
            year=int(row.year),
            mean_curve=mean_curve,
            sr_median=sr_median,
            sr_steepness=sr_steepness,
        )

        low_candidates = [
            float(row.ci_low),
            quantile_or_nan(scenario_values, 0.025),
            quantile_or_nan(mismatch_values, 0.025),
            quantile_or_nan(ggm_values, 0.025),
            min_or_nan(window_values),
        ]
        high_candidates = [
            float(row.ci_high),
            quantile_or_nan(scenario_values, 0.975),
            quantile_or_nan(mismatch_values, 0.975),
            quantile_or_nan(ggm_values, 0.975),
            max_or_nan(window_values),
        ]

        rows.append(
            {
                "year": int(row.year),
                "conservative_ci_low": finite_min(low_candidates),
                "conservative_ci_high": finite_max(high_candidates),
                "scenario_ci_low": quantile_or_nan(scenario_values, 0.025),
                "scenario_ci_high": quantile_or_nan(scenario_values, 0.975),
                "mismatch_ci_low": quantile_or_nan(mismatch_values, 0.025),
                "mismatch_ci_high": quantile_or_nan(mismatch_values, 0.975),
                "ggm_cov_ci_low": quantile_or_nan(ggm_values, 0.025),
                "ggm_cov_ci_high": quantile_or_nan(ggm_values, 0.975),
                "window_ci_low": min_or_nan(window_values),
                "window_ci_high": max_or_nan(window_values),
            }
        )

    return pd.DataFrame(rows)


def fit_point_to_all_scenarios(point: np.ndarray, model_rows: pd.DataFrame) -> list[float]:
    values = []
    for _, scenario_rows in model_rows.groupby("scenario_id"):
        fit = fig4.fit_point_to_curve(point, scenario_rows, "Xc")
        if np.isfinite(fit["estimate"]):
            values.append(float(fit["estimate"]))
    return values


def mismatch_envelope_values(
    *,
    point: np.ndarray,
    radius: float,
    model_rows: pd.DataFrame,
    seed: int,
) -> list[float]:
    if not np.isfinite(radius) or radius <= 0:
        return []

    rng = np.random.default_rng(seed)
    values = []
    for _ in range(POINT_MISMATCH_DRAWS):
        angle = rng.uniform(0, 2 * np.pi)
        distance = radius * np.sqrt(rng.uniform(0, 1))
        sampled_point = point + distance * np.array([np.cos(angle), np.sin(angle)])
        values.extend(fit_point_to_all_scenarios(sampled_point, model_rows))
    return values


def ggm_covariance_envelope_values(
    *,
    hmd: HMD,
    year: int,
    model_rows: pd.DataFrame,
    sr_median: float,
    sr_steepness: float,
    seed: int,
) -> list[float]:
    params, covariance = fit_ggm_with_covariance(hmd, year)
    if params is None or covariance is None:
        return []

    rng = np.random.default_rng(seed)
    values = []
    accepted = 0
    attempts = 0
    while accepted < GGM_COVARIANCE_DRAWS and attempts < GGM_COVARIANCE_DRAWS * 20:
        attempts += 1
        sampled_params = rng.multivariate_normal(params, covariance)
        point = point_from_ggm_params(
            sampled_params,
            sr_median=sr_median,
            sr_steepness=sr_steepness,
        )
        if point is None:
            continue

        accepted += 1
        values.extend(fit_point_to_all_scenarios(point, model_rows))
    return values


def fit_ggm_with_covariance(hmd: HMD, year: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    ages, hazards = hmd.get_hazard(year)
    mask = (
        (ages >= fig4.GGM_FIT_AGE_START)
        & (ages <= fig4.GGM_FIT_AGE_END)
        & (hazards > 0)
    )

    def log_hazard_model(t, a, b, c, m):
        with np.errstate(all="ignore"):
            return np.log(hmd._ggm_hazard_model(t, a, b, c, m))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            params, covariance = curve_fit(
                log_hazard_model,
                ages[mask],
                np.log(hazards[mask]),
                p0=[5e-5, 0.1, 9, 0.005],
                absolute_sigma=False,
                maxfev=10000,
            )
    except Exception:
        return None, None

    if not np.isfinite(covariance).all():
        return None, None
    return params, covariance


def point_from_ggm_params(
    params: np.ndarray,
    *,
    sr_median: float,
    sr_steepness: float,
) -> np.ndarray | None:
    a, b, c, _ = params
    if not np.all(np.isfinite([a, b, c])) or a <= 0 or b <= 0 or c <= 0:
        return None

    metrics = fig4.ggm_iqr_metrics({"a": float(a), "b": float(b), "c": float(c), "m": 0.0})
    if not np.isfinite(metrics["median_lifespan"]) or not np.isfinite(metrics["steepness_iqr_absolute"]):
        return None

    return np.array(
        [
            metrics["median_lifespan"] / sr_median,
            metrics["steepness_iqr_absolute"] / sr_steepness,
        ]
    )


def fit_window_envelope_values(
    *,
    hmd: HMD,
    year: int,
    mean_curve: pd.DataFrame,
    sr_median: float,
    sr_steepness: float,
) -> list[float]:
    values = []
    for age_start, age_end in GGM_FIT_WINDOWS:
        params = hmd.fit_ggm(year=year, age_start=age_start, age_end=age_end)
        if any(not np.isfinite(params[name]) for name in ("a", "b", "c")):
            continue

        metrics = fig4.ggm_iqr_metrics(
            {
                "a": float(params["a"]),
                "b": float(params["b"]),
                "c": float(params["c"]),
                "m": 0.0,
            }
        )
        if not np.isfinite(metrics["median_lifespan"]) or not np.isfinite(metrics["steepness_iqr_absolute"]):
            continue

        point = np.array(
            [
                metrics["median_lifespan"] / sr_median,
                metrics["steepness_iqr_absolute"] / sr_steepness,
            ]
        )
        fit = fig4.fit_point_to_curve(point, mean_curve, "Xc")
        if np.isfinite(fit["estimate"]):
            values.append(float(fit["estimate"]))
    return values


def quantile_or_nan(values: list[float], q: float) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    return float(np.quantile(finite, q))


def min_or_nan(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    return float(np.min(finite))


def max_or_nan(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return np.nan
    return float(np.max(finite))


def finite_min(values: list[float]) -> float:
    return float(np.nanmin(np.array(values, dtype=float)))


def finite_max(values: list[float]) -> float:
    return float(np.nanmax(np.array(values, dtype=float)))


def completed_run_keys() -> set[tuple[int, str]]:
    if not OUTPUT_CSV.exists():
        return set()

    with OUTPUT_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {(int(row["year"]), row["xc_variant"]) for row in reader}


def write_rows(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    write_header = not OUTPUT_CSV.exists()
    with OUTPUT_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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


def simulate_one(year: int, variant: str, xc_factor: float, h_ext: float) -> dict[str, float | int | str]:
    seed = stable_seed("fig4cd_pilot", year, variant, N_SIM)
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
        "year": int(year),
        "xc_variant": variant,
        "h_ext_variant": h_ext_variant_for_xc_variant(variant),
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = completed_run_keys()
    new_rows = []

    for input_row in inputs.itertuples(index=False):
        year = int(input_row.year)

        for variant, column in VARIANTS.items():
            if (year, variant) in done:
                continue

            if variant == "central":
                xc_factor = float(input_row.estimate)
            else:
                xc_factor = float(getattr(input_row, column))

            h_ext = h_ext_for_variant(input_row, variant)
            print(
                f"Running year={year}, Xc={variant}, "
                f"factor={xc_factor:.4f}, h_ext={h_ext:.3e}",
                flush=True,
            )
            row = simulate_one(
                year=year,
                variant=variant,
                xc_factor=xc_factor,
                h_ext=h_ext,
            )
            new_rows.append(row)
            write_rows([row])

    if OUTPUT_CSV.exists():
        return pd.read_csv(OUTPUT_CSV)

    return pd.DataFrame(new_rows)


def h_ext_for_variant(input_row, variant: str) -> float:
    if variant == "low":
        return float(input_row.ci_high_h_ext)
    if variant == "high":
        return float(input_row.ci_low_h_ext)
    return float(input_row.estimate_h_ext)


def h_ext_variant_for_xc_variant(variant: str) -> str:
    if variant == "low":
        return "high"
    if variant == "high":
        return "low"
    return "central"


def summarize_simulation_ci(sim_results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for year, rows in sim_results.groupby("year"):
        central = rows[rows["xc_variant"] == "central"].iloc[0]
        record = {"year": int(year)}
        for label in SURVIVAL_LEVELS:
            values = rows[label].to_numpy(dtype=float)
            record[label] = float(central[label])
            record[f"{label} low"] = float(np.nanmin(values))
            record[f"{label} high"] = float(np.nanmax(values))
        records.append(record)

    return pd.DataFrame(records).sort_values("year")


def plot_overlay(sim_summary: pd.DataFrame) -> None:
    hmd = pd.read_csv(HMD_CONTOURS_CSV)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 11,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(11.0, 6.5))

    for label in SURVIVAL_LEVELS:
        smoothed = f"{label} smoothed"
        ax.plot(
            hmd["year"],
            hmd[smoothed],
            color=COLORS[label],
            linewidth=3.0,
            alpha=0.95,
        )
        ax.errorbar(
            sim_summary["year"],
            sim_summary[label],
            yerr=[
                sim_summary[label] - sim_summary[f"{label} low"],
                sim_summary[f"{label} high"] - sim_summary[label],
            ],
            fmt="o",
            markersize=7.5,
            color=COLORS[label],
            markeredgecolor="black",
            markeredgewidth=0.8,
            elinewidth=1.8,
            capsize=4,
            linestyle="none",
            zorder=5,
        )

        final_value = hmd[smoothed].dropna().iloc[-1]
        ax.text(
            2020.8,
            final_value,
            label,
            color=COLORS[label],
            fontsize=11.5,
            fontweight="bold",
            va="center",
        )

    ax.axhline(110, color="0.35", linestyle=":", linewidth=1.8)
    ax.text(1902, 110.7, "HMD 110+ open interval", color="0.28", fontsize=11.5)
    ax.set_title("Sweden HMD contours with conservative SR pilot overlay")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Age reached | conditional on age {FROM_AGE}")
    ax.set_xlim(1896, 2028)
    ax.set_ylim(55, 113)
    ax.set_xticks([1900, 1920, 1940, 1960, 1980, 2000, 2019])
    ax.tick_params(axis="both", width=1.2, length=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=1.0)

    handles = [
        Line2D([0], [0], color="0.25", lw=3, label="Colored lines: HMD period"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.25",
            markerfacecolor="white",
            markeredgecolor="black",
            lw=0,
            label="Circles: SR central estimate",
        ),
        Line2D([0], [0], color="0.25", lw=1.8, label=r"Bars: conservative SR $X_c$ + $h_{ext}$ envelope"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.14, top=0.90)
    fig.savefig(OUTPUT_PNG, dpi=260)
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    sim_results = run_missing_simulations(inputs)
    sim_summary = summarize_simulation_ci(sim_results)
    plot_overlay(sim_summary)
    print(f"saved {OUTPUT_CSV}")
    print(f"saved {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
