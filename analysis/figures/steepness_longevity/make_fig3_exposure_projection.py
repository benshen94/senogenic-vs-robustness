#!/usr/bin/env python3
"""Rebuild Fig. 3 with NHANES exposure groups projected onto the Xc curve."""

from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import PchipInterpolator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, AGING_PYTHON_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ageing_packages.utils import sr_utils as utils
from analysis.figures.steepness_longevity import make_fig3_usa_steepness_longevity as fig3_base
from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR as PROJECT_RESULTS_DIR


try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - figure still renders without label repulsion.
    adjust_text = None


OUTPUT_DIR = FIGURES_DIR / "Figure3"
RESULTS_DIR = PROJECT_RESULTS_DIR / "figure3_exposure_projection"
EXPOSURE_PKL = PROJECT_RESULTS_DIR / "exposure_groups_results.pkl"
BASELINE_MANIFEST = PROJECT_RESULTS_DIR / "steepness_longevity_usa2019_sensitivity" / "manifest.json"
USAFIT_CI_PATH = PROJECT_RESULTS_DIR / "fits" / "ci" / "hybrid2019_swe_tail90_usa_refit_ci.csv"
X_AXIS_SUMMARY = (
    PROJECT_RESULTS_DIR
    / "steepness_longevity_usa2019_sensitivity"
    / "fig3_usa_steepness_longevity_point_intervals.csv"
)

PANEL_A_PATH = OUTPUT_DIR / "fig3_exposure_projection_panel_a.png"
PANEL_B_PATH = OUTPUT_DIR / "fig3_exposure_projection_panel_b.png"
PANEL_B_PDF_PATH = OUTPUT_DIR / "fig3_exposure_projection_panel_b.pdf"
PANEL_C_PATH = OUTPUT_DIR / "fig3_exposure_projection_panel_c.png"
COMPOSITE_PNG_PATH = OUTPUT_DIR / "fig3_exposure_projection.png"
COMPOSITE_PDF_PATH = OUTPUT_DIR / "fig3_exposure_projection.pdf"

PROJECTIONS_PATH = RESULTS_DIR / "exposure_xc_projection.csv"
PANEL_B_FACTOR_CURVES_PATH = RESULTS_DIR / "fig3_panel_b_xc_factor_curves_fit_ci.csv"
PANEL_B_FACTOR_CURVES_RAW_PATH = RESULTS_DIR / "fig3_panel_b_xc_factor_curves_fit_ci_raw.csv"
PANEL_B_FULL_XC_CURVES_PATH = RESULTS_DIR / "projected_xc_age_gain_curves.csv"
PANEL_B_PROJECTION_RIBBONS_PATH = RESULTS_DIR / "fig3_panel_b_projection_uncertainty_ribbons.csv"
PANEL_B_UNCERTAINTY_METHOD = "fit_ci_plus_projection_boundaries_v1"

RNG_SEED = 20260520
MONTE_CARLO_DRAWS = 3000
SIMULATION_N = 300000
SIMULATION_TMAX = 140.0
SIMULATION_DT = 0.025
AGE_GRID = np.arange(60, 105, 5)
PANEL_B_FACTOR_GRID = np.round(np.arange(0.80, 1.201, 0.05), 2)
CURVE_POINTS = 2500
COMPOSITE_DPI = 350
COMPOSITE_PIXEL_WIDTH = 4486
COMPOSITE_PIXEL_HEIGHT = 6953
COMPOSITE_WIDTH = COMPOSITE_PIXEL_WIDTH / COMPOSITE_DPI
COMPOSITE_HEIGHT = COMPOSITE_PIXEL_HEIGHT / COMPOSITE_DPI
PANEL_A_XLIM = (0.90, 1.10)
PANEL_A_YLIM = (0.60, 1.35)
LOWER_XLIM = (60, 103)
LOWER_YLIM = (-10, 10)
LOWER_YTICKS = np.arange(-10, 11, 2)

STEEPNESS_METRIC = "steepness_iqr_absolute"
LONGEVITY_METRIC = "t_median_absolute"
BASELINE_TYPE = "without_extrinsic"

TOPIC_ORDER = [
    "diet",
    "number_of_friends",
    "income",
    "alcohol",
    "physical_activity",
    "sleep_duration",
    "sleep_frailty",
    "church_frequency",
    "education_level",
]

GROUP_ORDER = {
    "diet": ["Poor", "Good"],
    "number_of_friends": ["0 friends", "1+ friends"],
    "income": ["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"],
    "alcohol": [">4 drinks/day", "0-1 drink/day"],
    "physical_activity": ["No Activity", "Some Activity"],
    "sleep_duration": ["1-<5 hours", "5-<7 hours", "7-<9 hours", ">=9 hours"],
    "sleep_frailty": ["Q4 (highest)", "Q1 (lowest)"],
    "church_frequency": ["never", "sometimes", "weekly"],
    "education_level": ["no highschool", "some college"],
}

RAW_GROUP_ALIASES = {
    ">=9 hours": ">=9 hours",
    "\u22659 hours": ">=9 hours",
}

DISPLAY_LABELS = {
    "Q1 (Lowest)": "Q1 income",
    "Q2": "Q2 income",
    "Q3": "Q3 income",
    "Q4 (Highest)": "Q4 income",
    "Q1 (lowest)": "Q1 sleep frailty",
    "Q4 (highest)": "Q4 sleep frailty",
    "0-1 drink/day": "0-1 drink/day",
    ">4 drinks/day": ">4 drinks/day",
    "1-<5 hours": "1-5 h sleep",
    "5-<7 hours": "5-7 h sleep",
    "7-<9 hours": "7-9 h sleep",
    ">=9 hours": ">=9 h sleep",
    "no highschool": "no high school",
    "some college": "some college",
}

TOPIC_MARKERS = {
    "diet": "s",
    "number_of_friends": "P",
    "income": "o",
    "alcohol": "D",
    "physical_activity": "^",
    "sleep_duration": "v",
    "sleep_frailty": "v",
    "church_frequency": "X",
    "education_level": "<",
}

PANEL_A_EXPOSURE_MARKER_SIZE = 340
PANEL_A_EXPOSURE_LABEL_SIZE = 9.2
PANEL_A_LEGEND_FONTSIZE = 15.5
PANEL_A_EXPOSURE_LEGEND_FONTSIZE = 15.0
PANEL_A_EXPOSURE_LEGEND_MARKER_SIZE = 12.5
PANEL_A_AXIS_LABEL_SIZE = 31
LOWER_AXIS_LABEL_SIZE = 27
LOWER_TITLE_SIZE = 23
LOWER_TICK_SIZE = 15
PANEL_LABEL_SIZE = 46
STANDALONE_PANEL_LABEL_SIZE = 42
PANEL_B_LEGEND_FONTSIZE = 11.2
PANEL_B_LEGEND_TITLE_SIZE = 12.3
PANEL_B_LOW_FACTOR_GREY = 0.72
PANEL_B_HIGH_FACTOR_GREY = 0.20
PANEL_B_RIBBON_ALPHA = 0.09
PANEL_B_LABEL_GREY = "0.28"
PANEL_B_EXTREME_LABEL_SIZE = 13.0
PROJECTION_UNCERTAINTY_NEIGHBORS = 5

EXPOSURE_LEGEND_ITEMS = [
    ("diet", "Diet quality"),
    ("number_of_friends", "Number of friends"),
    ("income", "Income"),
    ("alcohol", "Alcohol consumption"),
    ("physical_activity", "Physical activity"),
    ("sleep_duration", "Sleep duration"),
    ("sleep_frailty", "Sleep frailty"),
    ("church_frequency", "Church attendance"),
    ("education_level", "Education level"),
]


@dataclass(frozen=True)
class ProjectionCurve:
    factors: np.ndarray
    x_values: np.ndarray
    y_values: np.ndarray


def normalize_group_name(name: str) -> str:
    """Map raw stored group names onto ASCII display-safe names."""
    return RAW_GROUP_ALIASES.get(name, name)


def load_xc_projection_curve() -> ProjectionCurve:
    """Load the mean Xc response curve used by the new Fig. 3 background."""
    summary = pd.read_csv(X_AXIS_SUMMARY)
    rows = summary[
        (summary["curve_type"] == "parameter_factor")
        & (summary["focal_param"] == "Xc")
    ].sort_values("focal_value")

    if rows.empty:
        raise RuntimeError(f"No Xc rows found in {X_AXIS_SUMMARY}")

    factor_grid = np.linspace(rows["focal_value"].min(), rows["focal_value"].max(), CURVE_POINTS)
    x_interp = PchipInterpolator(rows["focal_value"], rows["x_mean"])
    y_interp = PchipInterpolator(rows["focal_value"], rows["y_mean"])
    return ProjectionCurve(
        factors=factor_grid,
        x_values=x_interp(factor_grid),
        y_values=y_interp(factor_grid),
    )


def project_to_curve(x_value: float, y_value: float, curve: ProjectionCurve) -> tuple[float, float, float, float]:
    """Project one x/y point to the closest point on the Xc curve."""
    distances = np.hypot(curve.x_values - x_value, curve.y_values - y_value)
    index = int(np.argmin(distances))
    return (
        float(curve.factors[index]),
        float(curve.x_values[index]),
        float(curve.y_values[index]),
        float(distances[index]),
    )


def sample_projected_factors(
    x_value: float,
    y_value: float,
    x_err: float,
    y_err: float,
    curve: ProjectionCurve,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate the factor uncertainty by sampling around the exposure point."""
    if x_err <= 0 and y_err <= 0:
        factor, _, _, _ = project_to_curve(x_value, y_value, curve)
        return np.full(MONTE_CARLO_DRAWS, factor)

    x_samples = rng.normal(x_value, max(x_err, 0.0), MONTE_CARLO_DRAWS)
    y_samples = rng.normal(y_value, max(y_err, 0.0), MONTE_CARLO_DRAWS)
    x_samples = np.clip(x_samples, *fig3_base.AX_LIMITS)
    y_samples = np.clip(y_samples, *fig3_base.AX_LIMITS)

    factors = []
    for sample_x, sample_y in zip(x_samples, y_samples):
        factor, _, _, _ = project_to_curve(float(sample_x), float(sample_y), curve)
        factors.append(factor)
    return np.array(factors)


def load_exposure_points() -> pd.DataFrame:
    """Load NHANES exposure-group points in the normalized Fig. 3 plane."""
    with EXPOSURE_PKL.open("rb") as handle:
        interventions = pickle.load(handle)

    baseline = interventions["baseline"][BASELINE_TYPE]
    longevity_base = baseline[LONGEVITY_METRIC]
    steepness_base = baseline[STEEPNESS_METRIC]

    rows = []
    for topic in TOPIC_ORDER:
        groups = interventions.get(topic, {})
        available = {normalize_group_name(name): name for name in groups}
        for group in GROUP_ORDER[topic]:
            raw_group = available.get(group)
            if raw_group is None:
                continue

            stats = groups[raw_group]
            longevity = stats.get(LONGEVITY_METRIC)
            steepness = stats.get(STEEPNESS_METRIC)
            if longevity is None or steepness is None:
                continue
            if not (np.isfinite(longevity) and np.isfinite(steepness)):
                continue

            longevity_err = stats.get(f"{LONGEVITY_METRIC}_err", 0.0) or 0.0
            steepness_err = stats.get(f"{STEEPNESS_METRIC}_err", 0.0) or 0.0
            rows.append(
                {
                    "topic": topic,
                    "group": group,
                    "label": DISPLAY_LABELS.get(group, group),
                    "x": longevity / longevity_base,
                    "y": steepness / steepness_base,
                    "x_err": longevity_err / longevity_base,
                    "y_err": steepness_err / steepness_base,
                    "n": stats.get("n"),
                    "deaths": stats.get("n_d"),
                }
            )

    return pd.DataFrame(rows)


def build_projection_table() -> pd.DataFrame:
    """Project every exposure point onto the Xc curve and save uncertainty intervals."""
    rng = np.random.default_rng(RNG_SEED)
    curve = load_xc_projection_curve()
    points = load_exposure_points()

    projected_rows = []
    for _, row in points.iterrows():
        factor, x_proj, y_proj, distance = project_to_curve(row["x"], row["y"], curve)
        factor_samples = sample_projected_factors(
            x_value=row["x"],
            y_value=row["y"],
            x_err=row["x_err"],
            y_err=row["y_err"],
            curve=curve,
            rng=rng,
        )
        projected_rows.append(
            {
                **row.to_dict(),
                "xc_factor": factor,
                "xc_factor_low": np.quantile(factor_samples, 0.025),
                "xc_factor_median_mc": np.quantile(factor_samples, 0.5),
                "xc_factor_high": np.quantile(factor_samples, 0.975),
                "xc_factor_std": np.std(factor_samples),
                "x_projected": x_proj,
                "y_projected": y_proj,
                "projection_distance": distance,
            }
        )

    table = pd.DataFrame(projected_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROJECTIONS_PATH, index=False)
    return table


def load_or_build_projection_table() -> pd.DataFrame:
    """Load exposure-to-Xc projections, rebuilding them if needed."""
    if PROJECTIONS_PATH.exists():
        return pd.read_csv(PROJECTIONS_PATH)
    return build_projection_table()


def load_baseline_params() -> dict[str, float]:
    """Load the central USA 2019 SR baseline used for the new Fig. 3 plane."""
    with BASELINE_MANIFEST.open("r") as handle:
        manifest = json.load(handle)
    baseline = manifest["baseline"]
    return {
        "eta": float(baseline["eta"]),
        "beta": float(baseline["beta"]),
        "kappa": float(baseline["kappa"]),
        "epsilon": float(baseline["epsilon"]),
        "Xc": float(baseline["Xc"]),
        "xc_std_frac": float(baseline["xc_std_frac"]),
    }


def load_usafit_ci_bounds() -> dict[str, tuple[float, float]]:
    """Load fit-local 95% CI endpoints for the USAFIT-specific parameters."""
    ci = pd.read_csv(USAFIT_CI_PATH).set_index("parameter")
    return {
        "Xc": (
            float(ci.loc["USA_Xc", "ci95_lower"]),
            float(ci.loc["USA_Xc", "ci95_upper"]),
        ),
        "xc_std_frac": (
            float(ci.loc["USA_xc_std_frac", "ci95_lower"]),
            float(ci.loc["USA_xc_std_frac", "ci95_upper"]),
        ),
    }


def build_panel_b_baseline_scenarios() -> list[tuple[str, dict[str, float]]]:
    """Build central and one-at-a-time USAFIT CI endpoint baselines."""
    baseline = load_baseline_params()
    scenarios = [("central", dict(baseline))]

    for param_name, (lower, upper) in load_usafit_ci_bounds().items():
        lower_baseline = dict(baseline)
        lower_baseline[param_name] = lower
        scenarios.append((f"{param_name}_ci_lower", lower_baseline))

        upper_baseline = dict(baseline)
        upper_baseline[param_name] = upper
        scenarios.append((f"{param_name}_ci_upper", upper_baseline))

    return scenarios


def make_params_for_xc_factor(
    factor: float,
    n: int,
    seed: int,
    baseline: dict[str, float],
) -> dict[str, np.ndarray]:
    """Create a heterogeneous-Xc parameter dictionary for one multiplicative factor."""
    np.random.seed(seed)
    base_dict = {
        "eta": baseline["eta"],
        "beta": baseline["beta"],
        "kappa": baseline["kappa"],
        "epsilon": baseline["epsilon"],
        "Xc": baseline["Xc"] * factor,
    }
    return utils.create_param_distribution_dict(
        params="Xc",
        std=baseline["xc_std_frac"],
        n=n,
        dist_type="gaussian",
        params_dict=base_dict,
        family="None",
    )


def finite_median(values: np.ndarray) -> float:
    """Return the median after dropping non-finite values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.median(values))


def remaining_medians_by_age(death_times: np.ndarray, ages: np.ndarray) -> dict[int, float]:
    """Calculate median remaining lifespan by age."""
    finite_times = np.where(np.isfinite(death_times), death_times, SIMULATION_TMAX)
    medians = {}
    for age in ages:
        alive_times = finite_times[finite_times >= age]
        remaining = alive_times - age
        medians[int(age)] = finite_median(remaining)
    return medians


def simulate_panel_b_factor_curves(*, n: int, force: bool = False) -> pd.DataFrame:
    """Simulate Fig3B Xc factor curves with a USAFIT endpoint-CI envelope."""
    if PANEL_B_FACTOR_CURVES_PATH.exists() and not force:
        cached = pd.read_csv(PANEL_B_FACTOR_CURVES_PATH)
        required = {
            "scenario_id",
            "factor",
            "age",
            "gain_years",
            "gain_low",
            "gain_high",
            "gain_fit_low",
            "gain_fit_high",
            "gain_projection_low",
            "gain_projection_high",
            "uncertainty_method",
            "n",
        }
        has_current_method = (
            "uncertainty_method" in cached.columns
            and (cached["uncertainty_method"] == PANEL_B_UNCERTAINTY_METHOD).all()
        )
        if required.issubset(cached.columns) and int(cached["n"].iloc[0]) == n and has_current_method:
            return cached

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = build_panel_b_baseline_scenarios()
    if force:
        PANEL_B_FACTOR_CURVES_PATH.unlink(missing_ok=True)
        PANEL_B_FACTOR_CURVES_RAW_PATH.unlink(missing_ok=True)

    raw = load_panel_b_raw_rows(n)
    completed = {
        (round(float(row.factor), 2), row.scenario_id)
        for row in raw[["factor", "scenario_id"]].drop_duplicates().itertuples(index=False)
    }
    total_runs = len(PANEL_B_FACTOR_GRID) * len(scenarios)

    for index, factor in enumerate(PANEL_B_FACTOR_GRID):
        seed = RNG_SEED + 5000 + index * 101
        for scenario_id, baseline in scenarios:
            run_key = (round(float(factor), 2), scenario_id)
            if run_key in completed:
                print(f"Skipping cached panel B run: factor={factor:.2f}, scenario={scenario_id}", flush=True)
                continue

            done_count = len(completed) + 1
            print(
                f"Panel B run {done_count}/{total_runs}: factor={factor:.2f}, scenario={scenario_id}, n={n}",
                flush=True,
            )
            params = make_params_for_xc_factor(float(factor), n=n, seed=seed, baseline=baseline)
            sim = utils.create_sr_simulation(
                species="human",
                n=n,
                params_dict=params,
                parallel=True,
                tmax=SIMULATION_TMAX,
                dt=SIMULATION_DT,
                save_times=SIMULATION_TMAX,
                random_seed=seed,
                break_early=True,
            )
            medians = remaining_medians_by_age(sim.death_times, AGE_GRID)
            run_rows = []
            for age, remaining_median in medians.items():
                run_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "factor": factor,
                        "age": age,
                        "remaining_median": remaining_median,
                        "n": n,
                        "random_seed": seed,
                    }
                )
            append_panel_b_raw_rows(pd.DataFrame(run_rows))
            completed.add(run_key)

    raw = load_panel_b_raw_rows(n)
    expected = {
        (round(float(factor), 2), scenario_id)
        for factor in PANEL_B_FACTOR_GRID
        for scenario_id, _ in scenarios
    }
    missing = sorted(expected - completed)
    if missing:
        raise RuntimeError(f"Panel B cache is incomplete; missing {len(missing)} runs: {missing[:5]}")

    baseline_rows = raw[np.isclose(raw["factor"], 1.0)].copy()
    if baseline_rows.empty:
        raise RuntimeError("Panel B factor grid must include baseline factor 1.0")

    baseline_lookup = {
        (row.scenario_id, row.age): row.remaining_median
        for row in baseline_rows.itertuples(index=False)
    }
    raw["baseline_remaining"] = [
        baseline_lookup[(row.scenario_id, row.age)]
        for row in raw.itertuples(index=False)
    ]
    raw["gain_years"] = raw["remaining_median"] - raw["baseline_remaining"]

    central = raw[raw["scenario_id"] == "central"].copy()
    fit_envelope = (
        raw.groupby(["factor", "age"], as_index=False)["gain_years"]
        .agg(gain_fit_low="min", gain_fit_high="max")
    )
    projection_intervals = build_projection_factor_intervals(load_or_build_projection_table())
    projection_envelope = build_projection_gain_envelope(raw, projection_intervals)

    curves = central.merge(fit_envelope, on=["factor", "age"], how="left")
    curves = curves.merge(projection_envelope, on=["factor", "age"], how="left")
    curves = curves.merge(projection_intervals, on="factor", how="left")
    curves["gain_low"] = curves["gain_fit_low"]
    curves["gain_high"] = curves["gain_fit_high"]
    curves["uncertainty_method"] = PANEL_B_UNCERTAINTY_METHOD
    curves.to_csv(PANEL_B_FACTOR_CURVES_PATH, index=False)
    return curves


def build_projection_factor_intervals(projections: pd.DataFrame) -> pd.DataFrame:
    """Estimate local factor uncertainty from nearby projected exposure groups."""
    required = {"xc_factor", "xc_factor_low", "xc_factor_high"}
    if not required.issubset(projections.columns):
        raise RuntimeError(f"Projection table must include columns: {sorted(required)}")

    usable = projections.dropna(subset=list(required)).copy()
    if usable.empty:
        raise RuntimeError("Projection table has no usable Xc-factor intervals")

    rows = []
    for factor in PANEL_B_FACTOR_GRID:
        local = usable.assign(distance=(usable["xc_factor"] - factor).abs())
        local = local.nsmallest(PROJECTION_UNCERTAINTY_NEIGHBORS, "distance")

        low_delta = float(np.median(local["xc_factor"] - local["xc_factor_low"]))
        high_delta = float(np.median(local["xc_factor_high"] - local["xc_factor"]))
        rows.append(
            {
                "factor": float(factor),
                "factor_projection_low": max(float(factor) - low_delta, float(PANEL_B_FACTOR_GRID.min())),
                "factor_projection_high": min(float(factor) + high_delta, float(PANEL_B_FACTOR_GRID.max())),
                "factor_projection_low_delta": low_delta,
                "factor_projection_high_delta": high_delta,
                "projection_neighbor_count": int(len(local)),
            }
        )

    return pd.DataFrame(rows)


def build_unclipped_projection_factor_intervals(
    projections: pd.DataFrame,
    full_curves: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate projection-factor intervals using the full cached Xc-factor grid."""
    required = {"xc_factor", "xc_factor_low", "xc_factor_high"}
    if not required.issubset(projections.columns):
        raise RuntimeError(f"Projection table must include columns: {sorted(required)}")

    usable = projections.dropna(subset=list(required)).copy()
    if usable.empty:
        raise RuntimeError("Projection table has no usable Xc-factor intervals")

    factor_min = float(full_curves["factor"].min())
    factor_max = float(full_curves["factor"].max())
    rows = []
    for factor in PANEL_B_FACTOR_GRID:
        local = usable.assign(distance=(usable["xc_factor"] - factor).abs())
        local = local.nsmallest(PROJECTION_UNCERTAINTY_NEIGHBORS, "distance")

        low_delta = float(np.median(local["xc_factor"] - local["xc_factor_low"]))
        high_delta = float(np.median(local["xc_factor_high"] - local["xc_factor"]))
        rows.append(
            {
                "factor": float(factor),
                "factor_projection_low": max(float(factor) - low_delta, factor_min),
                "factor_projection_high": min(float(factor) + high_delta, factor_max),
                "factor_projection_low_delta": low_delta,
                "factor_projection_high_delta": high_delta,
                "projection_neighbor_count": int(len(local)),
            }
        )

    return pd.DataFrame(rows)


def interpolate_gain(group: pd.DataFrame, factor: float) -> float:
    """Interpolate a scenario-specific gain curve at one Xc factor."""
    ordered = group.sort_values("factor")
    return float(np.interp(factor, ordered["factor"], ordered["gain_years"]))


def interpolate_full_factor_gain(full_curves: pd.DataFrame, factor: float, age: int) -> float:
    """Interpolate central gain by Xc factor for one age from the full cached grid."""
    rows = full_curves[full_curves["age"] == age].sort_values("factor")
    return float(np.interp(factor, rows["factor"], rows["gain_years"]))


def build_projection_gain_envelope(
    raw_gains: pd.DataFrame,
    projection_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """Propagate exposure projection uncertainty through the simulated gain curves."""
    grouped = {
        (scenario_id, age): group
        for (scenario_id, age), group in raw_gains.groupby(["scenario_id", "age"])
    }

    rows = []
    for interval in projection_intervals.itertuples(index=False):
        factor_values = [
            float(interval.factor_projection_low),
            float(interval.factor),
            float(interval.factor_projection_high),
        ]
        for age in AGE_GRID:
            gains = []
            for (scenario_id, group_age), group in grouped.items():
                if int(group_age) != int(age):
                    continue

                for factor_value in factor_values:
                    gains.append(interpolate_gain(group, factor_value))

            rows.append(
                {
                    "factor": float(interval.factor),
                    "age": int(age),
                    "gain_projection_low": float(np.min(gains)),
                    "gain_projection_high": float(np.max(gains)),
                }
            )

    return pd.DataFrame(rows)


def build_panel_b_projection_ribbons(
    factor_curves: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Build per-factor projection-uncertainty ribbons for panel B."""
    if not PANEL_B_FULL_XC_CURVES_PATH.exists():
        raise RuntimeError(f"Missing full Xc factor curves: {PANEL_B_FULL_XC_CURVES_PATH}")

    full_curves = pd.read_csv(PANEL_B_FULL_XC_CURVES_PATH)
    intervals = build_unclipped_projection_factor_intervals(projections, full_curves)

    rows = []
    for interval in intervals.itertuples(index=False):
        central = factor_curves[np.isclose(factor_curves["factor"], interval.factor)].sort_values("age")
        for row in central.itertuples(index=False):
            low_gain = interpolate_full_factor_gain(
                full_curves,
                float(interval.factor_projection_low),
                int(row.age),
            )
            high_gain = interpolate_full_factor_gain(
                full_curves,
                float(interval.factor_projection_high),
                int(row.age),
            )
            rows.append(
                {
                    "factor": float(interval.factor),
                    "age": int(row.age),
                    "gain_years": float(row.gain_years),
                    "gain_projection_low": min(low_gain, high_gain),
                    "gain_projection_high": max(low_gain, high_gain),
                    "factor_projection_low": float(interval.factor_projection_low),
                    "factor_projection_high": float(interval.factor_projection_high),
                    "projection_neighbor_count": int(interval.projection_neighbor_count),
                }
            )

    ribbons = pd.DataFrame(rows)
    ribbons.to_csv(PANEL_B_PROJECTION_RIBBONS_PATH, index=False)
    return ribbons


def load_panel_b_raw_rows(n: int) -> pd.DataFrame:
    """Load checkpointed raw panel B rows for this simulation size."""
    columns = ["scenario_id", "factor", "age", "remaining_median", "n", "random_seed"]
    if not PANEL_B_FACTOR_CURVES_RAW_PATH.exists():
        return pd.DataFrame(columns=columns)

    raw = pd.read_csv(PANEL_B_FACTOR_CURVES_RAW_PATH)
    required = set(columns)
    if not required.issubset(raw.columns):
        return pd.DataFrame(columns=columns)
    return raw[raw["n"] == n].copy()


def append_panel_b_raw_rows(rows: pd.DataFrame) -> None:
    """Append one completed panel B run to the checkpoint table."""
    write_header = not PANEL_B_FACTOR_CURVES_RAW_PATH.exists()
    rows.to_csv(PANEL_B_FACTOR_CURVES_RAW_PATH, mode="a", header=write_header, index=False)


def draw_base_plane(ax: plt.Axes, *, add_parameter_legend: bool = True) -> None:
    """Draw the new USA 2019 Fig. 3 background without factor markers."""
    data = fig3_base.load_normalized_metrics()
    for param in fig3_base.PLOT_PARAMS:
        summary = fig3_base.parameter_summary(data=data, param=param)
        color = fig3_base.PARAM_COLORS[param]
        fig3_base.draw_sensitivity_envelope(ax=ax, summary=summary, color=color)
        ax.plot(
            summary["x_mean"],
            summary["y_mean"],
            color=color,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=3,
        )

    h_ext = fig3_base.h_ext_summary(data)
    color = fig3_base.PARAM_COLORS["h_ext"]
    fig3_base.draw_sensitivity_envelope(ax=ax, summary=h_ext, color=color)
    ax.plot(
        h_ext["x_mean"],
        h_ext["y_mean"],
        color=color,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=2,
    )
    fig3_base.finish_axes(ax)
    if add_parameter_legend:
        build_parameter_legend(ax)


def build_parameter_legend(
    ax: plt.Axes,
    *,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> plt.Legend:
    """Build the grouped SR-parameter legend following the README convention."""
    handles = [
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["eta"], lw=4, label=fig3_base.PARAM_LABELS["eta"]),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["beta"], lw=4, label=fig3_base.PARAM_LABELS["beta"]),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["Xc"], lw=4, label=fig3_base.PARAM_LABELS["Xc"]),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["epsilon"], lw=4, label=fig3_base.PARAM_LABELS["epsilon"]),
        Line2D([], [], color="none", label=" "),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["h_ext"], lw=4, label="Extrinsic mortality"),
    ]
    legend_kwargs = {
        "handles": handles,
        "loc": "upper left",
        "frameon": True,
        "fontsize": PANEL_A_LEGEND_FONTSIZE,
        "handlelength": 2.7,
        "labelspacing": 0.58,
        "borderpad": 0.45,
    }
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor

    legend = ax.legend(**legend_kwargs)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_alpha(0.92)

    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_fontweight("bold")
            text.set_color("#222222")

    ax.add_artist(legend)
    add_parameter_legend_separator(ax=ax, legend=legend, row_index=6)
    return legend


def add_parameter_legend_separator(ax: plt.Axes, legend: plt.Legend, row_index: int) -> None:
    """Separate extrinsic mortality from parameter curves inside the legend."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer=renderer)
    text_box = legend.get_texts()[row_index].get_window_extent(renderer=renderer)

    y_display = (text_box.y0 + text_box.y1) / 2
    start = ax.transAxes.inverted().transform((legend_box.x0, y_display))
    end = ax.transAxes.inverted().transform((legend_box.x1, y_display))

    separator = Line2D(
        [start[0], end[0]],
        [start[1], end[1]],
        transform=ax.transAxes,
        color="#8A8A8A",
        linewidth=1.6,
        linestyle="--",
        alpha=0.5,
        solid_capstyle="butt",
        clip_on=False,
        zorder=5,
    )
    ax.add_line(separator)


def build_exposure_legend(ax: plt.Axes, *, loc: str = "lower right") -> plt.Legend:
    """Build the exposure-group legend for panel A."""
    handles = []
    for topic, label in EXPOSURE_LEGEND_ITEMS:
        marker = TOPIC_MARKERS.get(topic, "o")
        marker_facecolor = "white" if topic == "sleep_frailty" else "black"
        handle = Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=PANEL_A_EXPOSURE_LEGEND_MARKER_SIZE,
            markerfacecolor=marker_facecolor,
            markeredgecolor="black",
            markeredgewidth=1.2,
            color="black",
            label=label,
        )
        handles.append(handle)

    legend = ax.legend(
        handles=handles,
        loc=loc,
        title="Exposures",
        frameon=True,
        fontsize=PANEL_A_EXPOSURE_LEGEND_FONTSIZE,
        handlelength=1.7,
        labelspacing=0.34,
        borderpad=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D6D6")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(0.94)
    legend.get_title().set_fontsize(PANEL_A_EXPOSURE_LEGEND_FONTSIZE + 1.2)
    legend.get_title().set_fontweight("bold")
    ax.add_artist(legend)
    return legend


def draw_panel_a_exposure_legend(ax: plt.Axes, *, hide_axes: bool = False) -> None:
    """Place the exposure-group legend in the panel A lower-right corner."""
    if hide_axes:
        ax.set_axis_off()
    build_exposure_legend(ax, loc="lower right")


def apply_original_panel_a_limits(ax: plt.Axes) -> None:
    """Use the old Fig3A steepness-longevity viewport."""
    ax.set_title("", loc="left")
    ax.set_xlim(*PANEL_A_XLIM)
    ax.set_ylim(*PANEL_A_YLIM)
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(0.90, 1.101, 0.05))
    ax.set_yticks(np.arange(0.60, 1.31, 0.10))
    ax.set_xlabel("Median lifespan exposure / control", fontsize=PANEL_A_AXIS_LABEL_SIZE, labelpad=11)
    ax.set_ylabel("Steepness exposure / control", fontsize=PANEL_A_AXIS_LABEL_SIZE, labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(
        "Lifestyle exposures primarily\nimpact robustness",
        loc="center",
        fontsize=30,
        pad=17,
    )


def draw_exposure_overlay(ax: plt.Axes, projections: pd.DataFrame) -> None:
    """Overlay exposure points, error bars, labels, and projection lines."""
    texts = []
    for _, row in projections.iterrows():
        color = "#111111" if row["xc_factor"] >= 1 else "#404040"
        marker = TOPIC_MARKERS.get(row["topic"], "o")

        ax.plot(
            [row["x"], row["x_projected"]],
            [row["y"], row["y_projected"]],
            color="#666666",
            linewidth=0.7,
            alpha=0.22,
            zorder=5,
        )
        ax.errorbar(
            row["x"],
            row["y"],
            xerr=row["x_err"],
            yerr=row["y_err"],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.55,
            zorder=6,
        )
        if row["topic"] == "sleep_frailty":
            ax.scatter(
                row["x"],
                row["y"],
                marker=marker,
                s=PANEL_A_EXPOSURE_MARKER_SIZE,
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                zorder=7,
            )
        else:
            ax.scatter(
                row["x"],
                row["y"],
                marker=marker,
                s=PANEL_A_EXPOSURE_MARKER_SIZE,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=7,
            )

        text = ax.text(
            row["x"],
            row["y"],
            row["label"],
            fontsize=PANEL_A_EXPOSURE_LABEL_SIZE,
            color="#111111",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=8,
        )
        texts.append(text)

    if adjust_text is not None:
        with open(Path("/dev/null"), "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                adjust_text(
                    texts,
                    ax=ax,
                    expand=(1.7, 1.9),
                    force_text=(0.9, 1.0),
                    force_explode=(0.55, 0.8),
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45, alpha=0.55),
                )


def draw_panel_b(
    ax: plt.Axes,
    factor_curves: pd.DataFrame,
    projection_ribbons: pd.DataFrame,
) -> None:
    """Draw the Xc factor sweep with per-line projection-uncertainty ribbons."""
    ax.axhline(0, color="black", linestyle="--", linewidth=1.1, alpha=0.5, zorder=1)

    factors = np.sort(factor_curves["factor"].unique())
    colors = [panel_b_factor_color(factor, factors=factors) for factor in factors]

    for factor, color in zip(factors, colors):
        group = factor_curves[factor_curves["factor"] == factor].sort_values("age")
        ribbon = projection_ribbons[np.isclose(projection_ribbons["factor"], factor)].sort_values("age")
        ax.fill_between(
            ribbon["age"],
            ribbon["gain_projection_low"],
            ribbon["gain_projection_high"],
            color=color,
            alpha=PANEL_B_RIBBON_ALPHA,
            linewidth=0,
            edgecolor="none",
            zorder=2,
        )
        ax.plot(
            group["age"],
            group["gain_years"],
            "-",
            color=color,
            alpha=0.95,
            linewidth=2.2,
            zorder=3,
        )

    build_panel_b_projection_legend(ax, factors=factors, colors=colors)

    ax.text(
        61.2,
        4.85,
        "Largest gains in robustness",
        color=PANEL_B_LABEL_GREY,
        fontsize=PANEL_B_EXTREME_LABEL_SIZE,
        fontweight="bold",
        rotation=-20,
        ha="left",
        va="center",
        zorder=4,
    )
    ax.text(
        67.5,
        -3.85,
        "Largest reductions in robustness",
        color=PANEL_B_LABEL_GREY,
        alpha=0.72,
        fontsize=PANEL_B_EXTREME_LABEL_SIZE,
        fontweight="bold",
        rotation=24,
        ha="left",
        va="center",
        zorder=4,
    )

    ax.set_xlim(*LOWER_XLIM)
    ax.set_ylim(*LOWER_YLIM)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(LOWER_YTICKS)
    ax.set_xlabel("Age [years]", fontsize=LOWER_AXIS_LABEL_SIZE)
    ax.set_ylabel("Median extra years gained", fontsize=LOWER_AXIS_LABEL_SIZE)
    ax.set_title("Extra years from changes\nin robustness (model)", loc="center", pad=19, fontsize=LOWER_TITLE_SIZE)
    ax.tick_params(length=6.5, width=1.25, color="#222222", labelsize=LOWER_TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)


def panel_b_factor_color(factor: float, *, factors: np.ndarray) -> str:
    """Return a monotone grayscale color keyed to the Xc factor."""
    factor_min = float(np.min(factors))
    factor_max = float(np.max(factors))
    if factor_max == factor_min:
        return f"{PANEL_B_HIGH_FACTOR_GREY:.3f}"

    position = (float(factor) - factor_min) / (factor_max - factor_min)
    grey = PANEL_B_LOW_FACTOR_GREY + position * (PANEL_B_HIGH_FACTOR_GREY - PANEL_B_LOW_FACTOR_GREY)
    return f"{grey:.3f}"


def build_panel_b_projection_legend(
    ax: plt.Axes,
    *,
    factors: np.ndarray,
    colors: list,
) -> None:
    """Show the Xc factor color scale."""
    handles = [
        Line2D([0], [0], color=color, lw=2.2, label=f"{factor:.2f}")
        for factor, color in zip(factors, colors)
    ]
    legend = ax.legend(
        handles=handles,
        title="Xc factor",
        loc="upper right",
        frameon=False,
        fontsize=PANEL_B_LEGEND_FONTSIZE,
        title_fontsize=PANEL_B_LEGEND_TITLE_SIZE,
        handlelength=2.2,
        labelspacing=0.34,
        columnspacing=0.9,
        ncol=2,
    )
    ax.add_artist(legend)


def healthy_lifestyle_curves() -> dict[str, np.ndarray]:
    """Return the old Fig3C healthy-lifestyle gain curves."""
    return {
        "0-2 reference": np.array([0, 0, 0, 0, 0, 0, 0], dtype=float),
        "3": np.array([1, 0.8, 0.5, 0.2, 0.3, 0.5, 0.1], dtype=float),
        "4": np.array([3.5, 3.2, 2.6, 1.7, 0.8, 0.4, 0.2], dtype=float),
        "5": np.array([4, 3.6, 3, 2, 0.8, 0.2, 0.1], dtype=float),
        "6": np.array([4.9, 4.5, 3.8, 2.7, 1.2, 0.5, 0.7], dtype=float),
        "7-8": np.array([6.05, 5.6, 4.8, 3.5, 2.2, 1.2, 0.6], dtype=float),
    }


def draw_panel_c(ax: plt.Axes) -> None:
    """Draw the old Fig3C healthy-lifestyle data comparison."""
    x_nodes = np.array([40, 50, 60, 70, 80, 90, 100], dtype=float)
    x_smooth = np.linspace(60, 100, 220)
    curves = healthy_lifestyle_curves()
    curve_order = ["7-8", "6", "5", "4", "3", "0-2 reference"]
    alpha_values = np.linspace(1.0, 0.34, len(curve_order))

    for key, alpha in zip(curve_order, alpha_values):
        y_smooth = PchipInterpolator(x_nodes, curves[key])(x_smooth)
        ax.plot(x_smooth, y_smooth, color="black", alpha=alpha, linewidth=2.2)

    annotation_labels = [
        ("7-8", "7-8 healthy lifestyles"),
        ("6", "6"),
        ("5", "5"),
        ("4", "4"),
        ("3", "3"),
    ]
    alpha_map = dict(zip(curve_order, alpha_values))
    for key, label in annotation_labels:
        y_value = PchipInterpolator(x_nodes, curves[key])(63)
        ax.text(
            63,
            y_value + 0.35,
            label,
            fontsize=13,
            fontweight="bold",
            color="black",
            alpha=alpha_map[key],
            ha="left",
            va="bottom",
        )
    ax.text(
        63,
        -0.32,
        "0-2 reference",
        fontsize=12,
        fontweight="bold",
        color="black",
        alpha=alpha_map["0-2 reference"],
        ha="left",
        va="top",
    )

    ax.axhline(0, color="#8F8F8F", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.set_xlim(*LOWER_XLIM)
    ax.set_ylim(*LOWER_YLIM)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(LOWER_YTICKS)
    ax.set_xlabel("Age [years]", fontsize=LOWER_AXIS_LABEL_SIZE)
    ax.set_title("Extra years from\nhealthy lifestyle (data)", loc="center", pad=19, fontsize=LOWER_TITLE_SIZE)
    ax.tick_params(length=6.5, width=1.25, color="#222222", labelsize=LOWER_TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)


def save_panel_a(projections: pd.DataFrame) -> None:
    """Save standalone panel A."""
    fig3_base.apply_style()
    fig, ax = plt.subplots(figsize=(8.7, 8.2))

    draw_base_plane(ax, add_parameter_legend=False)
    apply_original_panel_a_limits(ax)
    draw_exposure_overlay(ax, projections)
    build_parameter_legend(ax)
    build_exposure_legend(ax, loc="lower right")
    ax.text(-0.14, 1.05, "a", transform=ax.transAxes, fontsize=STANDALONE_PANEL_LABEL_SIZE, fontweight="normal", va="top")
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.90)
    fig.savefig(PANEL_A_PATH, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_panel_b(factor_curves: pd.DataFrame, projection_ribbons: pd.DataFrame) -> None:
    """Save standalone panel B."""
    fig3_base.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    draw_panel_b(ax, factor_curves, projection_ribbons)
    ax.text(-0.21, 1.10, "b", transform=ax.transAxes, fontsize=STANDALONE_PANEL_LABEL_SIZE, fontweight="normal", va="top")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.80)
    fig.savefig(PANEL_B_PATH, dpi=350, bbox_inches="tight")
    fig.savefig(PANEL_B_PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def save_panel_c() -> None:
    """Save standalone panel C."""
    fig3_base.apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    draw_panel_c(ax)
    ax.text(-0.15, 1.17, "c", transform=ax.transAxes, fontsize=STANDALONE_PANEL_LABEL_SIZE, fontweight="normal", va="top")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.20, top=0.80)
    fig.savefig(PANEL_C_PATH, dpi=350, bbox_inches="tight")
    plt.close(fig)


def save_composite(
    projections: pd.DataFrame,
    factor_curves: pd.DataFrame,
    projection_ribbons: pd.DataFrame,
) -> None:
    """Save the combined two-panel figure."""
    fig3_base.apply_style()
    fig = plt.figure(figsize=(COMPOSITE_WIDTH, COMPOSITE_HEIGHT))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 0.78], hspace=0.28, wspace=0.24)
    top_grid = grid[0, :].subgridspec(1, 3, width_ratios=[0.16, 1.0, 0.30], wspace=0.0)

    ax_a = fig.add_subplot(top_grid[0, 1])
    draw_base_plane(ax_a, add_parameter_legend=False)
    apply_original_panel_a_limits(ax_a)
    draw_exposure_overlay(ax_a, projections)
    build_parameter_legend(ax_a)
    build_exposure_legend(ax_a, loc="lower right")
    ax_a.text(-0.14, 1.06, "a", transform=ax_a.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="normal", va="top")

    ax_b = fig.add_subplot(grid[1, 0])
    draw_panel_b(ax_b, factor_curves, projection_ribbons)
    ax_b.text(-0.16, 1.20, "b", transform=ax_b.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="normal", va="top")

    ax_c = fig.add_subplot(grid[1, 1], sharey=ax_b)
    draw_panel_c(ax_c)
    ax_c.set_ylabel("")
    ax_c.tick_params(labelleft=False)
    ax_c.text(-0.16, 1.20, "c", transform=ax_c.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="normal", va="top")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.075)
    fig.savefig(COMPOSITE_PNG_PATH, dpi=COMPOSITE_DPI, bbox_inches=None)
    fig.savefig(COMPOSITE_PDF_PATH, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=SIMULATION_N, help="Number of simulated people per Xc factor.")
    parser.add_argument("--force-sim", action="store_true", help="Recompute the age-curve simulation cache.")
    parser.add_argument("--panel-b-only", action="store_true", help="Only regenerate panel B and its cache.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    factor_curves = simulate_panel_b_factor_curves(n=args.n, force=args.force_sim)
    projections = build_projection_table()
    projection_ribbons = build_panel_b_projection_ribbons(factor_curves, projections)
    if args.panel_b_only:
        save_panel_b(factor_curves, projection_ribbons)
        print(f"Saved panel B: {PANEL_B_PATH}")
        print(f"Saved panel B PDF: {PANEL_B_PDF_PATH}")
        print(f"Saved panel B factor curves: {PANEL_B_FACTOR_CURVES_PATH}")
        print(f"Saved panel B projection ribbons: {PANEL_B_PROJECTION_RIBBONS_PATH}")
        return

    save_panel_a(projections)
    save_panel_b(factor_curves, projection_ribbons)
    save_panel_c()
    save_composite(projections, factor_curves, projection_ribbons)

    print(f"Saved panel A: {PANEL_A_PATH}")
    print(f"Saved panel B: {PANEL_B_PATH}")
    print(f"Saved panel C: {PANEL_C_PATH}")
    print(f"Saved composite PNG: {COMPOSITE_PNG_PATH}")
    print(f"Saved composite PDF: {COMPOSITE_PDF_PATH}")
    print(f"Saved projection table: {PROJECTIONS_PATH}")
    print(f"Saved panel B factor curves: {PANEL_B_FACTOR_CURVES_PATH}")
    print(f"Saved panel B projection ribbons: {PANEL_B_PROJECTION_RIBBONS_PATH}")


if __name__ == "__main__":
    main()
