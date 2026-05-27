#!/usr/bin/env python3
"""Make the new Fig2d/e sibling-mortality comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import NelsonAalenFitter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Fig2_new"
PNG_PATH = OUTPUT_DIR / "fig2de_new.png"
PDF_PATH = OUTPUT_DIR / "fig2de_new.pdf"

CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "Fig2_new"
CACHE_PATH = CACHE_DIR / "fig2de_new_plot_records.pkl"
METADATA_PATH = CACHE_DIR / "fig2de_new_metadata.json"
CENTRAL_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "fig2de_new_mortality_curves.csv"
ENVELOPE_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "fig2de_new_fit_ci_envelopes.csv"
RAW_DIGITIZED_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "fig2d_raw_digitized_points.csv"

BASELINE_FIT_PATH = (
    SAVED_RESULTS_DIR
    / "fit_archive"
    / "records"
    / "joint2019_tail90_sweden_emphasis.json"
)
BASELINE_CI_PATH = (
    SAVED_RESULTS_DIR
    / "fit_archive"
    / "ci"
    / "joint2019_tail90_sweden_emphasis_ci.csv"
)

ROBUSTNESS_PARAMETERS = ("Xc", "epsilon")
SENOGENIC_PARAMETERS = ("eta", "beta")
PARAMETERS = ROBUSTNESS_PARAMETERS + SENOGENIC_PARAMETERS
CI_BASELINE_PARAMS = ("eta", "beta", "epsilon", "Xc")
CI_ROW_BY_PARAM = {
    "eta": "eta",
    "beta": "beta",
    "epsilon": "epsilon",
    "Xc": "SWE_Xc",
}

PARAMETER_STDS = {
    "Xc": 0.20,
    "epsilon": 0.30,
    "eta": 0.15,
    "beta": 0.10,
}
PARAMETER_TITLES = {
    "Xc": r"Threshold ($X_c$)",
    "epsilon": r"Noise ($\epsilon$)",
    "eta": r"Production ($\eta$)",
    "beta": r"Removal ($\beta$)",
}
PARAMETER_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
}

CENTRAL_N_SIM = 2_000_000
CI_N_SIM = 1_000_000
BOTTOM_DEATH_PERCENTILE = 10
TOP_SURVIVAL_FRACTION = 0.01
TMAX = 150.0
DT = 0.025
SAVE_TIMES = 150.0
RANDOM_SEED = 20260520

PLOT_AGE_MIN = 50
PLOT_AGE_MAX = 115
RAW_PLOT_AGE_MAX = 101
SIMULATION_PLOT_AGE_MAX = 105
PLOT_X_MIN = 48
LOG_MORTALITY_Y_MIN = -3.2
LOG_MORTALITY_Y_MAX = 1.2
RAW_LOG_MORTALITY_Y_MAX = 0.0
SIMULATION_LOG_MORTALITY_Y_MIN = -4.3
SIMULATION_LOG_MORTALITY_Y_MAX = 0.0
MIN_MORTALITY = 5e-3
MAX_MORTALITY = 1.0

BASELINE_COLOR = "#2B2B2B"
BROTHER_COLOR = "#000000"
SISTER_COLOR = "#000000"
GOOD_SURVIVOR_LINESTYLE = "-"
BAD_SURVIVOR_LINESTYLE = (0, (2.0, 1.5))
CI_BAND_ALPHA = 0.14
FULL_COHORT_LINE_ALPHA = 0.5
RAW_FIT_MIN_AGE = 60
RAW_FIT_MAX_AGE = 100
SIMULATION_MARKER_EVERY_YEARS = 1.0


def stable_seed(*parts: object) -> int:
    """Create a deterministic uint32 seed from readable metadata."""
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_sweden_baseline() -> dict[str, float]:
    """Load the Sweden 2019 baseline used by Fig2a/b/c_new."""
    record = json.loads(BASELINE_FIT_PATH.read_text())
    fitted = record["summary"]["fitted_parameters"]
    return {
        "eta": float(fitted["eta"]),
        "beta": float(fitted["beta"]),
        "kappa": 0.5,
        "epsilon": float(fitted["epsilon"]),
        "Xc": float(fitted["SWE_Xc"]),
    }


def load_sweden_ci_bounds() -> dict[str, tuple[float, float]]:
    """Load fit-local 95% CI bounds for the Sweden baseline parameters."""
    ci = pd.read_csv(BASELINE_CI_PATH).set_index("parameter")
    bounds = {}
    for param_name in CI_BASELINE_PARAMS:
        row_name = CI_ROW_BY_PARAM[param_name]
        row = ci.loc[row_name]
        bounds[param_name] = (float(row["ci95_lower"]), float(row["ci95_upper"]))
    return bounds


def build_focal_ci_scenarios(
    baseline: dict[str, float],
    focal_param: str,
) -> list[tuple[str, dict[str, float], int]]:
    """Build central plus lower/upper fit-CI endpoint baselines for one panel."""
    lower, upper = load_sweden_ci_bounds()[focal_param]

    low = dict(baseline)
    low[focal_param] = lower

    high = dict(baseline)
    high[focal_param] = upper

    return [
        ("central", dict(baseline), CENTRAL_N_SIM),
        (f"{focal_param}_ci_lower", low, CI_N_SIM),
        (f"{focal_param}_ci_upper", high, CI_N_SIM),
    ]


def build_metadata() -> dict[str, object]:
    """Return metadata that must match before reusing the plot cache."""
    return {
        "baseline_fit_path": str(BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)),
        "baseline_ci_path": str(BASELINE_CI_PATH.relative_to(PROJECT_ROOT)),
        "baseline": load_sweden_baseline(),
        "ci_method": "focal_parameter_95_ci_endpoint_envelope",
        "params_to_vary": list(PARAMETERS),
        "parameter_stds": PARAMETER_STDS,
        "central_n_sim": CENTRAL_N_SIM,
        "ci_n_sim": CI_N_SIM,
        "bottom_death_percentile": BOTTOM_DEATH_PERCENTILE,
        "top_survival_fraction": TOP_SURVIVAL_FRACTION,
        "h_ext": 0.0,
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "random_seed": RANDOM_SEED,
    }


def load_cached_plot_records(metadata: dict[str, object]) -> dict[str, dict[str, dict[str, object]]] | None:
    """Load cached plot records when metadata is current."""
    if not CACHE_PATH.exists() or not METADATA_PATH.exists():
        return None

    cached_metadata = json.loads(METADATA_PATH.read_text())
    if cached_metadata != metadata:
        return None

    with CACHE_PATH.open("rb") as handle:
        print(f"Loaded cached Fig2d/e plot data from {CACHE_PATH}")
        return pickle.load(handle)


def save_plot_records(
    plot_records: dict[str, dict[str, dict[str, object]]],
    metadata: dict[str, object],
) -> None:
    """Persist the reusable plot records."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("wb") as handle:
        pickle.dump(plot_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved cached Fig2d/e plot data to {CACHE_PATH}")


def create_dz_simulation(
    param_name: str,
    baseline: dict[str, float],
    n_sim: int,
    seed: int,
):
    """Run one DZ sibling simulation for a focal heterogeneous parameter."""
    np.random.seed(seed)
    params = utils.create_param_distribution_dict(
        params=param_name,
        std=PARAMETER_STDS[param_name],
        n=n_sim,
        dist_type="gaussian",
        params_dict=baseline,
        family="DZ",
    )
    return utils.create_sr_simulation(
        params_dict=params,
        n=n_sim,
        h_ext=0.0,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        parallel=True,
        break_early=True,
        random_seed=seed,
    )


def run_one_plot_record(
    param_name: str,
    scenario_id: str,
    baseline: dict[str, float],
    n_sim: int,
) -> dict[str, object]:
    """Run one simulation scenario and convert it to plot-ready arrays."""
    seed = stable_seed(RANDOM_SEED, param_name, scenario_id, n_sim)
    print(f"Running Fig2d/e simulation: {param_name}, {scenario_id}, n={n_sim:,}")
    sim = create_dz_simulation(param_name, baseline, n_sim, seed)
    return create_plot_record(sim, scenario_id, n_sim)


def create_plot_record(sim, scenario_id: str, n_sim: int) -> dict[str, object]:
    """Summarize one simulation into the curves needed for plotting."""
    bottom_age = get_bottom_death_percentile_age(sim)
    top_age = get_top_survival_threshold_age(sim)

    return {
        "scenario_id": scenario_id,
        "n_sim": n_sim,
        "tspan_hazard": np.asarray(sim.tspan_hazard, dtype=float),
        "full_hazard": as_1d_array(sim.hazard),
        "bottom_sibling_hazard": calc_sibling_hazard_for_bottom_percentile(sim, bottom_age),
        "top_sibling_hazard": calc_sibling_hazard_for_top_survivors(sim, top_age),
        "bottom_age": bottom_age,
        "top_age": top_age,
    }


def get_bottom_death_percentile_age(sim) -> float:
    """Return the death age defining the bad-survivor probands."""
    finite_death_times = sim.death_times[np.isfinite(sim.death_times)]
    if finite_death_times.size == 0:
        raise ValueError("Simulation has no finite death times.")
    return float(np.percentile(finite_death_times, BOTTOM_DEATH_PERCENTILE))


def get_top_survival_threshold_age(sim) -> float:
    """Return the age where survival reaches the top-survivor fraction."""
    age = sim.find_time_at_survival(TOP_SURVIVAL_FRACTION)
    if age is None:
        raise ValueError("Simulation never reached 1% survivorship.")
    return float(age)


def get_sibling_death_times_for_probands(sim, proband_indices: np.ndarray) -> np.ndarray:
    """Return each proband's paired DZ sibling death time."""
    if proband_indices.size == 0:
        return np.array([], dtype=float)

    sibling_indices = proband_indices + 1 - 2 * (proband_indices % 2)
    return sim.death_times[sibling_indices.astype(int)]


def calc_sibling_hazard_for_bottom_percentile(sim, percentile_age: float) -> np.ndarray:
    """Estimate mortality for siblings of bottom-percentile probands."""
    proband_indices = np.where(sim.death_times <= percentile_age)[0]
    return calc_sibling_hazard_from_probands(sim, proband_indices)


def calc_sibling_hazard_for_top_survivors(sim, threshold_age: float) -> np.ndarray:
    """Estimate mortality for siblings of top-survivor probands."""
    proband_indices = np.where(sim.death_times >= threshold_age)[0]
    return calc_sibling_hazard_from_probands(sim, proband_indices)


def calc_sibling_hazard_from_probands(sim, proband_indices: np.ndarray) -> np.ndarray:
    """Estimate Nelson-Aalen smoothed hazards for selected siblings."""
    death_times = get_sibling_death_times_for_probands(sim, proband_indices)
    if death_times.size == 0:
        return np.full_like(sim.tspan_hazard, np.nan, dtype=float)

    censor_time = sim.params.tmax + sim.params.dt
    event_observed = death_times < censor_time

    fitter = NelsonAalenFitter()
    fitter.fit(death_times, event_observed=event_observed, timeline=sim.tspan_hazard)
    return as_1d_array(fitter.smoothed_hazard_(bandwidth=3))


def as_1d_array(values) -> np.ndarray:
    """Return values as a one-dimensional float array."""
    if hasattr(values, "iloc"):
        return values.iloc[:, 0].to_numpy(dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def get_plot_records() -> dict[str, dict[str, dict[str, object]]]:
    """Load or create all plot records."""
    metadata = build_metadata()
    cached = load_cached_plot_records(metadata)
    if cached is not None:
        return cached

    baseline = load_sweden_baseline()
    plot_records = {}
    for param_name in PARAMETERS:
        plot_records[param_name] = {}
        for scenario_id, scenario_baseline, n_sim in build_focal_ci_scenarios(baseline, param_name):
            plot_records[param_name][scenario_id] = run_one_plot_record(
                param_name=param_name,
                scenario_id=scenario_id,
                baseline=scenario_baseline,
                n_sim=n_sim,
            )

    save_plot_records(plot_records, metadata)
    return plot_records


def plot_records_to_frame(plot_records: dict[str, dict[str, dict[str, object]]]) -> pd.DataFrame:
    """Convert plot records to long-form rows within the visible age range."""
    rows = []
    cohorts = [
        ("full", "Full cohort", "full_hazard"),
        ("good", "Good survivor siblings (top 1%)", "top_sibling_hazard"),
        ("bad", "Bad survivor siblings (bottom 10%)", "bottom_sibling_hazard"),
    ]
    for param_name, records_by_scenario in plot_records.items():
        for scenario_id, record in records_by_scenario.items():
            ages = np.asarray(record["tspan_hazard"], dtype=float)
            keep = (ages >= PLOT_AGE_MIN) & (ages <= PLOT_AGE_MAX)
            for cohort_id, cohort_label, curve_key in cohorts:
                mortality = as_1d_array(record[curve_key])
                curve_keep = keep & np.isfinite(mortality) & (mortality > 0)
                for age, value in zip(ages[curve_keep], mortality[curve_keep]):
                    rows.append(
                        {
                            "param": param_name,
                            "scenario_id": scenario_id,
                            "cohort": cohort_id,
                            "cohort_label": cohort_label,
                            "age": float(age),
                            "mortality": float(value),
                            "n_sim": int(record["n_sim"]),
                            "bottom_age": float(record["bottom_age"]),
                            "top_age": float(record["top_age"]),
                        }
                    )
    return pd.DataFrame(rows)


def save_plot_data(plot_records: dict[str, dict[str, dict[str, object]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Save central curves and fit-CI envelopes."""
    data = plot_records_to_frame(plot_records)
    central = data[data["scenario_id"] == "central"].copy()
    envelopes = (
        data.groupby(["param", "cohort", "cohort_label", "age"], as_index=False)["mortality"]
        .agg(ci_lower="min", ci_upper="max")
        .copy()
    )

    CENTRAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    central.to_csv(CENTRAL_DATA_PATH, index=False)
    envelopes.to_csv(ENVELOPE_DATA_PATH, index=False)
    return central, envelopes


def configure_matplotlib() -> None:
    """Apply publication-style defaults."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 1.25,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def raw_sibling_digitization() -> dict[str, dict[str, np.ndarray]]:
    """Return digitized empirical sibling-mortality points from the source panel."""
    return {
        "brothers_short": {
            "age": np.array([50, 51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]),
            "log_hazard": np.array([-1.95, -2.05, -1.90, -2.00, -1.80, -1.75, -1.75, -1.85, -1.75, -1.65, -1.60, -1.60, -1.55, -1.45, -1.45, -1.35, -1.40, -1.30, -1.30, -1.25, -1.20, -1.20, -1.15, -1.15, -1.05, -1.05, -1.00, -0.90, -0.95, -0.90, -0.80, -0.80, -0.90, -0.85, -0.70, -0.80, -0.75, -0.60, -0.60, -0.65, -0.55, -0.40, -0.50, -0.45, -0.35, -0.40, -0.25, -0.65, -0.20, -0.20, 0.30]),
        },
        "brothers_cent": {
            "age": np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 100, 102, 103, 104, 105, 108]),
            "log_hazard": np.array([-2.35, -2.30, -2.25, -2.20, -2.15, -2.25, -2.10, -2.15, -1.90, -1.95, -1.95, -1.90, -1.95, -1.80, -1.75, -1.75, -1.70, -1.60, -1.60, -1.50, -1.55, -1.50, -1.40, -1.40, -1.35, -1.25, -1.30, -1.20, -1.20, -1.15, -1.10, -1.05, -1.00, -1.05, -0.95, -1.00, -0.90, -0.85, -0.90, -0.80, -0.80, -0.75, -0.75, -0.65, -0.60, -0.55, -0.55, -0.50, -0.45, -0.45, -0.30, -0.55, -0.10, 0.30]),
        },
        "sisters_short": {
            "age": np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 106]),
            "log_hazard": np.array([-2.15, -2.10, -2.00, -2.00, -1.95, -2.00, -1.95, -1.85, -2.00, -1.95, -1.85, -1.90, -1.80, -1.80, -1.75, -1.70, -1.70, -1.65, -1.65, -1.50, -1.55, -1.50, -1.40, -1.45, -1.35, -1.30, -1.25, -1.20, -1.15, -1.05, -1.10, -1.00, -1.00, -0.95, -0.95, -0.85, -0.90, -0.85, -0.80, -0.80, -0.75, -0.70, -0.65, -0.60, -0.55, -0.55, -0.45, -0.45, -0.40, -0.30, -0.45, -0.30, -0.20, -0.40, -0.70, 0.10]),
        },
        "sisters_cent": {
            "age": np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 102, 103, 104, 106, 107, 108, 110, 113]),
            "log_hazard": np.array([-2.35, -2.50, -2.20, -2.25, -2.30, -2.25, -2.30, -2.20, -2.25, -2.15, -2.15, -2.10, -2.00, -2.15, -2.10, -2.05, -1.90, -1.85, -1.80, -1.75, -1.85, -1.85, -1.75, -1.70, -1.65, -1.55, -1.50, -1.50, -1.45, -1.40, -1.40, -1.30, -1.30, -1.25, -1.20, -1.15, -1.10, -1.05, -1.00, -1.00, -0.95, -0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.65, -0.60, -0.55, -0.40, -0.30, -0.25, -0.40, -0.60, -0.45, -0.65, 0.00, 0.30]),
        },
    }


def save_raw_digitized_data() -> None:
    """Save the pasted raw empirical points as an auditable CSV."""
    data = raw_sibling_digitization()
    rows = []
    for key, series in data.items():
        sibling_sex, proband_group = key.split("_", 1)
        for age, log_hazard in zip(series["age"], series["log_hazard"]):
            rows.append(
                {
                    "series": key,
                    "sibling_sex": sibling_sex,
                    "proband_group": proband_group,
                    "age": int(age),
                    "log10_hazard": float(log_hazard),
                    "plotted": int(age) <= RAW_PLOT_AGE_MAX,
                }
            )

    RAW_DIGITIZED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RAW_DIGITIZED_DATA_PATH, index=False)


def draw_raw_points(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    key: str,
    label: str,
    color: str,
    marker: str,
    filled: bool,
) -> None:
    """Draw one raw empirical scatter series."""
    series = data[key]
    keep = series["age"] <= RAW_PLOT_AGE_MAX
    facecolors = color if filled else "white"
    edgecolors = color
    linewidths = 0.75 if filled else 1.55

    ax.scatter(
        series["age"][keep],
        series["log_hazard"][keep],
        s=48,
        marker=marker,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=linewidths,
        alpha=0.88 if filled else 0.96,
        label=label,
        zorder=3 if filled else 4,
    )


def draw_raw_linear_fit(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    key: str,
    color: str,
) -> None:
    """Draw a dashed linear fit over the requested age window."""
    series = data[key]
    keep = (series["age"] >= RAW_FIT_MIN_AGE) & (series["age"] <= RAW_FIT_MAX_AGE)
    if np.sum(keep) < 2:
        return

    slope, intercept = np.polyfit(series["age"][keep], series["log_hazard"][keep], deg=1)
    ages = np.array([RAW_FIT_MIN_AGE, RAW_FIT_MAX_AGE], dtype=float)
    ax.plot(
        ages,
        slope * ages + intercept,
        color=color,
        lw=1.45,
        linestyle=(0, (3.0, 2.0)),
        alpha=0.5,
        zorder=7,
    )


def plot_raw_sex_panel(
    ax: plt.Axes,
    data: dict[str, dict[str, np.ndarray]],
    short_key: str,
    cent_key: str,
    color: str,
    title: str,
    show_xlabel: bool,
) -> None:
    """Draw one empirical brother/sister raw-data row."""
    draw_raw_linear_fit(ax, data, short_key, color)
    draw_raw_linear_fit(ax, data, cent_key, color)
    draw_raw_points(
        ax,
        data,
        short_key,
        "Short-lived probands",
        color,
        "o",
        True,
    )
    draw_raw_points(
        ax,
        data,
        cent_key,
        "Centenarian probands",
        color,
        "o",
        False,
    )

    ax.text(
        0.04,
        0.90,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="normal",
        color=color,
    )
    ax.set_xlim(PLOT_X_MIN, RAW_PLOT_AGE_MAX)
    ax.set_ylim(LOG_MORTALITY_Y_MIN, RAW_LOG_MORTALITY_Y_MAX)
    ax.set_ylabel(r"Log$_{10}$ mortality rate", fontsize=18, labelpad=6)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_yticks([-3, -2, -1, 0])
    ax.tick_params(axis="both", which="major", labelsize=14, pad=4, length=5.8, width=1.35)
    if show_xlabel:
        ax.set_xlabel("Age [years]")
    else:
        ax.tick_params(labelbottom=False)
    style_axis(ax)


def add_raw_convergence_annotation(ax: plt.Axes) -> None:
    """Add a small convergence callout to one raw empirical panel."""
    ax.annotate(
        "convergence",
        xy=(100, -0.30),
        xytext=(84, -1.55),
        fontsize=11.5,
        color="#222222",
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.18",
            "color": "#222222",
            "lw": 1.15,
        },
        zorder=5,
    )


def plot_raw_sibling_panels(top_ax: plt.Axes, bottom_ax: plt.Axes) -> None:
    """Draw the two-row raw empirical sibling-mortality panel."""
    data = raw_sibling_digitization()
    plot_raw_sex_panel(
        top_ax,
        data,
        "brothers_short",
        "brothers_cent",
        BROTHER_COLOR,
        "Brothers",
        show_xlabel=False,
    )
    plot_raw_sex_panel(
        bottom_ax,
        data,
        "sisters_short",
        "sisters_cent",
        SISTER_COLOR,
        "Sisters",
        show_xlabel=True,
    )
    top_ax.set_title(
        "Mortality converges for siblings of centenarians\nand short-lived persons",
        pad=10,
    )
    add_raw_convergence_annotation(top_ax)
    add_raw_convergence_annotation(bottom_ax)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9.5,
            markerfacecolor="black",
            markeredgecolor="black",
            label="Short-lived probands",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.7,
            label="Centenarian probands",
        ),
    ]
    top_ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, -0.20),
        ncol=2,
        fontsize=17,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def plot_sibling_panel(
    ax: plt.Axes,
    central: pd.DataFrame,
    envelopes: pd.DataFrame,
    param_name: str,
) -> None:
    """Draw one simulated sibling-mortality panel."""
    color = PARAMETER_COLORS[param_name]
    draw_ci_band(ax, envelopes, param_name, "full", BASELINE_COLOR, alpha=0.08)
    draw_ci_band(ax, envelopes, param_name, "good", color, alpha=CI_BAND_ALPHA)
    draw_ci_band(ax, envelopes, param_name, "bad", color, alpha=CI_BAND_ALPHA)

    draw_curve(ax, central, param_name, "full", BASELINE_COLOR, "-", alpha=FULL_COHORT_LINE_ALPHA)
    draw_curve(ax, central, param_name, "good", color, "None", alpha=1.0, marker="o", filled=False)
    draw_curve(ax, central, param_name, "bad", color, "None", alpha=1.0, marker="o", filled=True)

    ax.set_box_aspect(1)
    ax.set_xlim(PLOT_X_MIN, SIMULATION_PLOT_AGE_MAX)
    ax.set_ylim(LOG_MORTALITY_Y_MIN, RAW_LOG_MORTALITY_Y_MAX)
    ax.set_title(PARAMETER_TITLES[param_name], pad=8)
    ax.set_xlabel("Age [years]")
    ax.set_ylabel(r"Log$_{10}$ mortality rate", fontsize=18, labelpad=6)
    ax.set_xticks([50, 60, 70, 80, 90, 100])
    ax.set_yticks([-3, -2, -1, 0])
    ax.tick_params(axis="both", which="major", pad=4)
    style_axis(ax)


def draw_ci_band(
    ax: plt.Axes,
    envelopes: pd.DataFrame,
    param_name: str,
    cohort: str,
    color: str,
    alpha: float,
) -> None:
    """Draw the shaded fit-CI envelope for one cohort curve."""
    subset = envelopes[(envelopes["param"] == param_name) & (envelopes["cohort"] == cohort)]
    if subset.empty:
        return

    subset = subset[subset["age"] <= SIMULATION_PLOT_AGE_MAX].sort_values("age")
    upper = subset["ci_upper"].to_numpy(dtype=float)
    lower = subset["ci_lower"].to_numpy(dtype=float)
    min_visible = 10 ** SIMULATION_LOG_MORTALITY_Y_MIN
    visible = upper > 0
    ax.fill_between(
        subset["age"],
        np.where(visible, np.log10(np.maximum(lower, min_visible)), np.nan),
        np.where(visible, np.log10(np.maximum(upper, min_visible)), np.nan),
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=1,
    )


def draw_curve(
    ax: plt.Axes,
    central: pd.DataFrame,
    param_name: str,
    cohort: str,
    color: str,
    linestyle,
    alpha: float,
    marker: str | None = None,
    filled: bool = True,
) -> None:
    """Draw one central mortality curve."""
    subset = central[(central["param"] == param_name) & (central["cohort"] == cohort)]
    if subset.empty:
        return

    subset = subset[subset["age"] <= SIMULATION_PLOT_AGE_MAX].sort_values("age")
    mortality = np.maximum(
        subset["mortality"].to_numpy(dtype=float),
        10 ** SIMULATION_LOG_MORTALITY_Y_MIN,
    )
    marker_kwargs = {}
    if marker is not None:
        marker_kwargs = {
            "marker": marker,
            "markersize": 4.7,
            "markerfacecolor": color if filled else "white",
            "markeredgecolor": color,
            "markeredgewidth": 1.25,
            "markevery": max(1, int(round(SIMULATION_MARKER_EVERY_YEARS / DT))),
        }

    ax.plot(
        subset["age"],
        np.log10(mortality),
        color=color,
        lw=3.0,
        linestyle=linestyle,
        alpha=alpha,
        zorder=3,
        **marker_kwargs,
    )


def add_convergence_annotation(
    ax: plt.Axes,
    label: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
) -> None:
    """Add a small convergence/no-convergence callout."""
    ax.annotate(
        label,
        xy=xy,
        xytext=xytext,
        fontsize=10.0,
        color="#333333",
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.18",
            "color": "#333333",
            "lw": 1.05,
        },
        zorder=5,
    )


def add_panel_annotations(right_axes: list[plt.Axes]) -> None:
    """Annotate robustness as converging and senogenic panels as not converging."""
    for ax in right_axes[:2]:
        add_convergence_annotation(ax, "converges", xy=(103.5, -0.42), xytext=(86.5, -2.08))
    add_convergence_annotation(right_axes[2], "no convergence", xy=(102.5, -1.18), xytext=(91, -2.12))
    add_convergence_annotation(right_axes[3], "no convergence", xy=(101.5, -1.18), xytext=(91, -2.12))


def simplify_sibling_axis_labels(right_axes: list[plt.Axes]) -> None:
    """Keep repeated square-panel labels from crowding the figure."""
    for index, ax in enumerate(right_axes):
        is_bottom_row = index >= 2
        is_left_column = index % 2 == 0

        if not is_bottom_row:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        if not is_left_column:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)


def build_bottom_legend(legend_ax: plt.Axes) -> None:
    """Place a large cohort-style legend below the simulated 2x2 block."""
    legend_ax.axis("off")
    entries = [
        (0.05, 0.70, BASELINE_COLOR, "-", FULL_COHORT_LINE_ALPHA, "Full cohort"),
        (0.43, 0.70, "#555555", GOOD_SURVIVOR_LINESTYLE, 1.0, "Good survivor siblings (top 1%)"),
        (0.05, 0.32, "#555555", BAD_SURVIVOR_LINESTYLE, 1.0, "Bad survivor siblings (bottom 10%)"),
    ]
    for x, y, color, linestyle, alpha, label in entries:
        legend_ax.plot(
            [x, x + 0.12],
            [y, y],
            transform=legend_ax.transAxes,
            color=color,
            linestyle=linestyle,
            alpha=alpha,
            lw=4.2,
            clip_on=False,
        )
        legend_ax.text(
            x + 0.145,
            y,
            label,
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=18,
            color="black",
        )


def add_row_subtitle(fig: plt.Figure, axes: list[plt.Axes], label: str, y_offset: float) -> None:
    """Add a centered row subtitle over two square panels."""
    left_position = axes[0].get_position()
    right_position = axes[1].get_position()
    x_center = (left_position.x0 + right_position.x1) / 2
    y_top = max(left_position.y1, right_position.y1)

    fig.text(
        x_center,
        y_top + y_offset,
        label,
        ha="center",
        va="bottom",
        fontsize=18,
        fontweight="normal",
        color="#333333",
    )


def add_panel_labels(left_top_ax: plt.Axes, right_axes: list[plt.Axes]) -> None:
    """Add d/e panel labels."""
    left_top_ax.text(
        -0.12,
        1.10,
        "d",
        transform=left_top_ax.transAxes,
        fontsize=42,
        fontweight="normal",
        va="top",
        ha="right",
    )
    right_axes[0].text(
        -0.30,
        1.18,
        "e",
        transform=right_axes[0].transAxes,
        fontsize=42,
        fontweight="normal",
        va="top",
        ha="right",
    )


def style_axis(ax: plt.Axes) -> None:
    """Apply common axis cleanup."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(central: pd.DataFrame, envelopes: pd.DataFrame) -> plt.Figure:
    """Build the full d/e figure."""
    fig = plt.figure(figsize=(21.4, 10.15))
    outer_grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.88], wspace=0.055)

    left_grid = outer_grid[0, 0].subgridspec(2, 1, hspace=0.24)
    left_axes = [
        fig.add_subplot(left_grid[0, 0]),
        fig.add_subplot(left_grid[1, 0]),
    ]
    right_grid = outer_grid[0, 1].subgridspec(2, 2, wspace=0.005, hspace=0.44)
    right_axes = [
        fig.add_subplot(right_grid[0, 0]),
        fig.add_subplot(right_grid[0, 1]),
        fig.add_subplot(right_grid[1, 0]),
        fig.add_subplot(right_grid[1, 1]),
    ]

    plot_raw_sibling_panels(left_axes[0], left_axes[1])
    for ax, param_name in zip(right_axes, PARAMETERS):
        plot_sibling_panel(ax, central, envelopes, param_name)

    simplify_sibling_axis_labels(right_axes)
    add_panel_annotations(right_axes)
    add_row_subtitle(fig, right_axes[:2], "Robustness parameters", y_offset=0.018)
    add_row_subtitle(fig, right_axes[2:], "Senogenic parameters", y_offset=0.055)
    add_panel_labels(left_axes[0], right_axes)
    return fig


def update_output_index() -> None:
    """Upsert output-index rows for the Fig2d/e_new artifacts."""
    index_path = SAVED_RESULTS_DIR / "index" / "outputs.csv"
    fieldnames = [
        "date",
        "task",
        "artifact_type",
        "path",
        "source_script",
        "input_paths",
        "description",
        "notes",
    ]
    existing_rows = []
    if index_path.exists():
        with index_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)

    rows = [
        {
            "date": "2026-05-20",
            "task": "fig2de_new_sibling_mortality",
            "artifact_type": "figure",
            "path": str(PNG_PATH.relative_to(PROJECT_ROOT)),
            "source_script": "src/figures/Fig2_new/make_fig2de_new.py",
            "input_paths": "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json; saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
            "description": "PNG preview of new Fig2d/e sibling-mortality convergence panel",
            "notes": "Sweden 2019 baseline; DZ sibling simulations; shaded bands are focal-parameter fit-CI endpoint envelopes",
        },
        {
            "date": "2026-05-20",
            "task": "fig2de_new_sibling_mortality",
            "artifact_type": "figure",
            "path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "source_script": "src/figures/Fig2_new/make_fig2de_new.py",
            "input_paths": "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json; saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
            "description": "Vector PDF of new Fig2d/e sibling-mortality convergence panel",
            "notes": "Matplotlib PDF with editable text where supported",
        },
        {
            "date": "2026-05-20",
            "task": "fig2de_new_sibling_mortality",
            "artifact_type": "csv",
            "path": str(RAW_DIGITIZED_DATA_PATH.relative_to(PROJECT_ROOT)),
            "source_script": "src/figures/Fig2_new/make_fig2de_new.py",
            "input_paths": "digitized from user-supplied source points",
            "description": "Raw digitized sibling-mortality points used for Fig2d",
            "notes": "Includes all pasted points; panel d plots raw points through age 101",
        },
        {
            "date": "2026-05-20",
            "task": "fig2de_new_sibling_mortality",
            "artifact_type": "csv",
            "path": str(CENTRAL_DATA_PATH.relative_to(PROJECT_ROOT)),
            "source_script": "src/figures/Fig2_new/make_fig2de_new.py",
            "input_paths": "saved_results/cache/simulations/Fig2_new/fig2de_new_plot_records.pkl",
            "description": "Central mortality curves for Fig2d/e_new",
            "notes": "Includes full cohort, good-survivor siblings, and bad-survivor siblings for each focal heterogeneous parameter",
        },
        {
            "date": "2026-05-20",
            "task": "fig2de_new_sibling_mortality",
            "artifact_type": "csv",
            "path": str(ENVELOPE_DATA_PATH.relative_to(PROJECT_ROOT)),
            "source_script": "src/figures/Fig2_new/make_fig2de_new.py",
            "input_paths": "saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
            "description": "Fit-CI envelope source data for Fig2d/e_new shaded bands",
            "notes": "Envelope is min/max across central and lower/upper focal-parameter 95% CI endpoint perturbations",
        },
    ]

    paths_to_replace = {row["path"] for row in rows}
    kept_rows = [
        {field: row.get(field, "") for field in fieldnames}
        for row in existing_rows
        if row.get("path") not in paths_to_replace
    ]
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        writer.writerows(rows)


def main() -> None:
    """Run simulations as needed and save the figure."""
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_records = get_plot_records()
    central, envelopes = save_plot_data(plot_records)
    save_raw_digitized_data()
    fig = build_figure(central, envelopes)
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    update_output_index()

    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")
    print(f"Saved {RAW_DIGITIZED_DATA_PATH}")
    print(f"Saved {CENTRAL_DATA_PATH}")
    print(f"Saved {ENVELOPE_DATA_PATH}")


if __name__ == "__main__":
    main()
