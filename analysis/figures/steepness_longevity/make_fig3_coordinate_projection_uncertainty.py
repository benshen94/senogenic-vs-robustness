#!/usr/bin/env python3
"""Quantify coordinate-wise Fig. 3 projection uncertainty for NHANES exposures."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ageing_packages.mortality_models.gamma_gompertz import GammaGompertz
from analysis.figures.steepness_longevity import make_fig3_exposure_projection as fig3_projection
from analysis.figures.steepness_longevity import make_fig3_usa_steepness_longevity as fig3_base
from senogenic_vs_robustness.paths import NHANES_DATA_DIR, RESULTS_DIR as PROJECT_RESULTS_DIR, TABLES_DIR


RESULTS_DIR = PROJECT_RESULTS_DIR / "figure3_exposure_projection"
MEAN_CURVE_SET = "mean_model_curve"
STATISTICAL_CURVE_SET = MEAN_CURVE_SET
PARAM_ORDER = ("eta", "beta", "Xc", "epsilon", "h_ext")
PARAM_CLASS = {
    "eta": "senogenic",
    "beta": "senogenic",
    "Xc": "robustness",
    "epsilon": "robustness",
    "h_ext": "extrinsic",
}
PARAM_DISPLAY = {
    "eta": "η",
    "beta": "β",
    "Xc": "Xc",
    "epsilon": "ε",
    "h_ext": "m_ex",
}
AGES = np.arange(20, 111, 1)
FIT_AGES = np.arange(20, 101, 1)
DEFAULT_BOOTSTRAPS = 300
DEFAULT_POINT_DRAWS = 10000
RNG_SEED = 20260528
MIN_STD = 1e-4

POINT_DISTANCES_PATH = RESULTS_DIR / "exposure_coordinate_projection_distances.csv"
POINT_ASSIGNMENT_PATH = RESULTS_DIR / "exposure_coordinate_projection_point_mc.csv"
FULL_BOOTSTRAP_PATH = RESULTS_DIR / "exposure_coordinate_projection_full_bootstrap_assignments.csv"
FULL_BOOTSTRAP_SUMMARY_PATH = RESULTS_DIR / "exposure_coordinate_projection_full_bootstrap_summary.csv"
MODEL_SENSITIVITY_PATH = RESULTS_DIR / "exposure_coordinate_projection_model_sensitivity_summary.csv"
PAPER_SUMMARY_PATH = RESULTS_DIR / "exposure_coordinate_projection_paper_summary.csv"
XC_EQUIVALENT_PATH = RESULTS_DIR / "exposure_xc_equivalent_projection_full_uncertainty.csv"
EXTENDED_DATA_TABLE_PATH = RESULTS_DIR / "extended_data_table_projection_with_ranges.csv"
EXTENDED_DATA_TABLE_PUBLIC_PATH = TABLES_DIR / "extended_data_table1_fig3_projection.csv"
FAILURES_PATH = RESULTS_DIR / "exposure_coordinate_projection_full_bootstrap_failures.csv"
METADATA_PATH = RESULTS_DIR / "exposure_coordinate_projection_metadata.json"
METHODS_LOG_PATH = RESULTS_DIR / "exposure_coordinate_projection_methods_log.md"

TOPIC_DISPLAY = {
    "diet": "Diet",
    "number_of_friends": "Social Support",
    "income": "Income",
    "alcohol": "Alcohol",
    "physical_activity": "Physical Activity",
    "sleep_duration": "Sleep Duration",
    "sleep_frailty": "Sleep Frailty",
    "church_frequency": "Religious Attendance",
    "education_level": "Education",
}

TABLE_LABEL_OVERRIDES = {
    ">=9 h sleep": "≥9 h sleep",
}


@dataclass(frozen=True)
class CoordinateCurve:
    """A smooth model trajectory in log-normalized median/steepness space."""

    curve_set: str
    param: str
    factor_grid: np.ndarray
    z_x: np.ndarray
    z_y: np.ndarray


def load_nhanes_module():
    """Import nhanes_analysis while silencing its import-time pickle message."""
    with open(Path("/dev/null"), "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            return importlib.import_module("ageing_packages.hetero_analysis.nhanes_analysis")


def log_value(value: float) -> float:
    """Return log(value), guarding against non-positive numeric artifacts."""
    if not np.isfinite(value) or value <= 0:
        return np.nan
    return float(np.log(value))


def display_coordinate(coordinate: str) -> str:
    """Return the manuscript-facing coordinate name."""
    return PARAM_DISPLAY.get(coordinate, coordinate)


def project_point_to_curve(
    z: np.ndarray,
    curve: CoordinateCurve,
) -> tuple[float, float, float, float, float]:
    """Project one point onto a coordinate curve using Euclidean log-space distance."""
    deltas = np.column_stack((curve.z_x - z[0], curve.z_y - z[1]))
    distance_sq = np.einsum("ij,ij->i", deltas, deltas)
    index = int(np.nanargmin(distance_sq))
    return (
        float(np.sqrt(max(distance_sq[index], 0.0))),
        float(curve.factor_grid[index]),
        float(curve.z_x[index]),
        float(curve.z_y[index]),
        float(distance_sq[index]),
    )


def project_point_to_curves(
    z: np.ndarray,
    curves: list[CoordinateCurve],
) -> list[dict[str, float | str]]:
    """Project one point to all coordinate curves in one curve set."""
    rows = []
    for curve in curves:
        distance, factor, z_x, z_y, distance_sq = project_point_to_curve(z, curve)
        rows.append(
            {
                "curve_set": curve.curve_set,
                "coordinate": curve.param,
                "coordinate_class": PARAM_CLASS[curve.param],
                "distance": distance,
                "distance_sq": distance_sq,
                "factor": factor,
                "projected_log_median": z_x,
                "projected_log_steepness": z_y,
            }
        )
    return rows


def best_projection(
    z: np.ndarray,
    curves: list[CoordinateCurve],
) -> dict[str, float | str]:
    """Return the closest coordinate curve for one point."""
    rows = project_point_to_curves(z, curves)
    return min(rows, key=lambda row: float(row["distance_sq"]))


def make_curve(
    rows: pd.DataFrame,
    *,
    curve_set: str,
    param: str,
    factor_col: str = "focal_value",
    x_col: str = "x_mean",
    y_col: str = "y_mean",
) -> CoordinateCurve | None:
    """Interpolate one parameter trajectory in log-normalized coordinates."""
    rows = rows[[factor_col, x_col, y_col]].dropna().sort_values(factor_col)
    rows = rows[(rows[x_col] > 0) & (rows[y_col] > 0)]
    rows = rows.drop_duplicates(subset=factor_col, keep="first")
    if len(rows) < 2:
        return None
    factor_grid = np.linspace(float(rows[factor_col].min()), float(rows[factor_col].max()), 2500)
    x_interp = PchipInterpolator(rows[factor_col], np.log(rows[x_col]))
    y_interp = PchipInterpolator(rows[factor_col], np.log(rows[y_col]))
    return CoordinateCurve(
        curve_set=curve_set,
        param=param,
        factor_grid=factor_grid,
        z_x=x_interp(factor_grid),
        z_y=y_interp(factor_grid),
    )


def load_mean_curves() -> list[CoordinateCurve]:
    """Load the mean Fig. 3 coordinate curves shown in the response plane."""
    summary = pd.read_csv(fig3_base.SUMMARY_PATH)
    curves = []
    for param in PARAM_ORDER:
        rows = summary[summary["focal_param"] == param].copy()
        curve = make_curve(rows, curve_set=MEAN_CURVE_SET, param=param)
        if curve is not None:
            curves.append(curve)
    return curves


def load_scenario_curves() -> dict[str, list[CoordinateCurve]]:
    """Load one curve set per deterministic SR baseline-sensitivity scenario."""
    data = fig3_base.load_normalized_metrics()
    scenario_curves: dict[str, list[CoordinateCurve]] = {}
    for scenario_id, scenario_rows in data.groupby("scenario_id"):
        curves = []
        for param in ("eta", "beta", "Xc", "epsilon"):
            min_factor = fig3_base.MIN_FACTOR_BY_PARAM[param]
            rows = scenario_rows[
                (scenario_rows["curve_type"] == "parameter_factor")
                & (scenario_rows["focal_param"] == param)
                & (scenario_rows["focal_value"] >= min_factor)
                & (scenario_rows["focal_value"] <= fig3_base.MAX_FACTOR)
            ]
            rows = fig3_base.valid_interval_rows(rows)
            curve = make_curve(
                rows,
                curve_set=str(scenario_id),
                param=param,
                x_col="x_norm",
                y_col="y_norm",
            )
            if curve is not None:
                curves.append(curve)

        rows = scenario_rows[
            (scenario_rows["curve_type"] == "h_ext_absolute")
            & (scenario_rows["focal_param"] == "h_ext")
            & (scenario_rows["focal_value"] <= fig3_base.H_EXT_MAX_VISIBLE)
        ]
        rows = fig3_base.valid_interval_rows(rows)
        curve = make_curve(
            rows,
            curve_set=str(scenario_id),
            param="h_ext",
            x_col="x_norm",
            y_col="y_norm",
        )
        if curve is not None:
            curves.append(curve)

        if curves:
            scenario_curves[str(scenario_id)] = curves
    return scenario_curves


def topic_keep_groups(topic_name: str, valid_groups: list[str]) -> list[str]:
    """Match the topic-specific group filtering used to build Fig. 3."""
    if topic_name == "alcohol" and len(valid_groups) >= 2:
        return [valid_groups[0], valid_groups[-1]]
    if topic_name == "education_level":
        return [group for group in valid_groups if group != "high school"]
    return valid_groups


def load_group_frames(nhanes) -> dict[tuple[str, str], pd.DataFrame]:
    """Load cleaned participant frames for every exposure group shown in Fig. 3."""
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    nhanes_path = str(NHANES_DATA_DIR) + "/"
    for topic in fig3_projection.TOPIC_ORDER:
        config = nhanes.TOPIC_CONFIGS[topic]
        df_topic = nhanes.get_topic_df(topic, nhanes_path)
        df_grouped, group_col = nhanes._apply_grouping_strategy(df_topic, config)
        required_cols = [group_col, "entry_age", "exit_age", "event"]
        df_clean = df_grouped.dropna(subset=required_cols)

        expected = nhanes.get_expected_group_names(topic)
        available = {fig3_projection.normalize_group_name(str(group)): group for group in df_clean[group_col].unique()}
        valid = [group for group in expected if fig3_projection.normalize_group_name(str(group)) in available]
        keep_groups = topic_keep_groups(topic, valid)

        for group in keep_groups:
            normalized_group = fig3_projection.normalize_group_name(str(group))
            if normalized_group not in fig3_projection.GROUP_ORDER.get(topic, keep_groups):
                continue
            raw_group = available.get(normalized_group)
            if raw_group is None:
                continue
            subset = df_clean[df_clean[group_col] == raw_group].copy()
            if len(subset) >= 10:
                frames[(topic, normalized_group)] = subset
    return frames


def km_stats(df: pd.DataFrame, nhanes) -> dict[str, float]:
    """Compute median lifespan and IQR steepness from a left-truncated KM curve."""
    kmf = KaplanMeierFitter()
    kmf.fit(df["exit_age"], event_observed=df["event"], entry=df["entry_age"], timeline=AGES)
    steep = nhanes.calculate_steepness_from_kmf(kmf, float(df["entry_age"].min()), AGES)
    return {
        "t_median_absolute": float(kmf.percentile(0.50)),
        "steepness_iqr_absolute": float(steep["steepness_iqr_absolute"]),
    }


def central_death_rates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estimate one-year central death rates with left-truncated exposure time."""
    entry = df["entry_age"].to_numpy(dtype=float)
    exit_age = df["exit_age"].to_numpy(dtype=float)
    event = df["event"].to_numpy(dtype=float)
    mid_ages = []
    rates = []
    for age in FIT_AGES:
        next_age = age + 1
        exposed = np.clip(np.minimum(exit_age, next_age) - np.maximum(entry, age), 0.0, None).sum()
        deaths = ((event == 1) & (exit_age >= age) & (exit_age < next_age)).sum()
        if exposed > 0 and deaths > 0:
            mid_ages.append(age + 0.5)
            rates.append(deaths / exposed)
    return np.asarray(mid_ages, dtype=float), np.asarray(rates, dtype=float)


def survival_quantile(ages: np.ndarray, survival: np.ndarray, probability: float) -> float:
    """Return the age at which survival reaches probability."""
    if not np.any(survival <= probability):
        return np.nan
    return float(np.interp(probability, survival[::-1], ages[::-1]))


def fit_zero_makeham_baseline(
    df: pd.DataFrame,
    *,
    initial_guess: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    """Fit Makeham-Gamma-Gompertz to NHANES rates, then set m=0 for baseline."""
    model = GammaGompertz()
    ages, rates = central_death_rates(df)
    if len(ages) < 12:
        raise RuntimeError("Too few nonzero age-specific death rates for MGG fit")
    if initial_guess is None:
        initial_guess = np.array([5e-5, 0.09, 9.0, 0.001], dtype=float)

    popt, _ = curve_fit(
        model.log_hazard_function,
        ages,
        np.log(rates),
        p0=initial_guess,
        bounds=([1e-10, 1e-5, 1e-5, 0.0], [1.0, 1.0, 50.0, 0.1]),
        maxfev=20000,
    )
    hazard_no_ext = model.hazard_function(AGES, popt[0], popt[1], popt[2], 0.0)
    survival = np.exp(-cumulative_trapezoid(hazard_no_ext, AGES, initial=0.0))
    t25 = survival_quantile(AGES, survival, 0.25)
    t50 = survival_quantile(AGES, survival, 0.50)
    t75 = survival_quantile(AGES, survival, 0.75)
    steepness = abs(-t50 / (t75 - t25))
    stats = {
        "t_median_absolute": t50,
        "steepness_iqr_absolute": steepness,
        "mgg_a": float(popt[0]),
        "mgg_b": float(popt[1]),
        "mgg_c": float(popt[2]),
        "mgg_m": float(popt[3]),
    }
    return stats, popt


def sample_rows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap sample a participant frame."""
    positions = rng.integers(0, len(df), size=len(df))
    return df.iloc[positions].copy()


def make_normalized_row(
    topic: str,
    group: str,
    stats: dict[str, float],
    baseline_stats: dict[str, float],
) -> dict[str, float | str]:
    """Build one normalized log-coordinate row."""
    x = stats["t_median_absolute"] / baseline_stats["t_median_absolute"]
    y = stats["steepness_iqr_absolute"] / baseline_stats["steepness_iqr_absolute"]
    log_x = log_value(x)
    log_y = log_value(y)
    if not (np.isfinite(log_x) and np.isfinite(log_y)):
        raise RuntimeError(
            "Non-finite normalized projection coordinate "
            f"for {topic}/{group}: x={x}, y={y}"
        )
    return {
        "topic": topic,
        "group": group,
        "label": fig3_projection.DISPLAY_LABELS.get(group, group),
        "x": x,
        "y": y,
        "log_median": log_x,
        "log_steepness": log_y,
    }


def central_statistical_rows(
    nhanes,
    group_frames: dict[tuple[str, str], pd.DataFrame],
    baseline_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute central rows from the raw NHANES participant data."""
    baseline_stats, baseline_guess = fit_zero_makeham_baseline(baseline_frame)
    rows = []
    for (topic, group), frame in group_frames.items():
        rows.append(make_normalized_row(topic, group, km_stats(frame, nhanes), baseline_stats))
    return pd.DataFrame(rows), baseline_guess


def point_se_projection() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a fast diagnostic projection from existing Fig. 3 bootstrap SEs."""
    rng = np.random.default_rng(RNG_SEED)
    points = fig3_projection.load_exposure_points()
    curves = load_mean_curves()
    distance_rows = []
    assignment_rows = []

    for _, row in points.iterrows():
        z = np.array([log_value(row["x"]), log_value(row["y"])], dtype=float)
        # Delta-method diagonal covariance in log space from existing bootstrap SEs.
        x_std = max(float(row["x_err"]) / max(float(row["x"]), MIN_STD), MIN_STD)
        y_std = max(float(row["y_err"]) / max(float(row["y"]), MIN_STD), MIN_STD)
        for projection in project_point_to_curves(z, curves):
            distance_rows.append({**row.to_dict(), **projection})

        samples = rng.normal(loc=z, scale=np.array([x_std, y_std]), size=(DEFAULT_POINT_DRAWS, 2))
        for sample_index, sample in enumerate(samples):
            best = best_projection(sample, curves)
            assignment_rows.append(
                {
                    "sample_index": sample_index,
                    "topic": row["topic"],
                    "group": row["group"],
                    "label": row["label"],
                    "log_median": sample[0],
                    "log_steepness": sample[1],
                    "best_coordinate": best["coordinate"],
                    "best_coordinate_class": best["coordinate_class"],
                    "best_distance": best["distance"],
                    "best_factor": best["factor"],
                }
            )

    return pd.DataFrame(distance_rows), pd.DataFrame(assignment_rows)


def run_full_bootstrap(n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run participant-level bootstrap through KM, extrinsic correction, and projection."""
    nhanes = load_nhanes_module()
    rng = np.random.default_rng(seed)
    group_frames = load_group_frames(nhanes)
    baseline_frame = nhanes.load_core(str(NHANES_DATA_DIR) + "/").dropna(
        subset=["entry_age", "exit_age", "event"]
    )
    central_rows, baseline_guess = central_statistical_rows(nhanes, group_frames, baseline_frame)

    sample_rows_out = []
    failures = []
    current_guess = baseline_guess
    for bootstrap_index in range(n_bootstrap):
        try:
            baseline_sample = sample_rows(baseline_frame, rng)
            baseline_stats, current_guess = fit_zero_makeham_baseline(
                baseline_sample,
                initial_guess=current_guess,
            )
        except Exception as exc:
            failures.append(
                {
                    "bootstrap_index": bootstrap_index,
                    "topic": "baseline",
                    "group": "baseline",
                    "reason": str(exc),
                }
            )
            continue

        for (topic, group), frame in group_frames.items():
            try:
                stats = km_stats(sample_rows(frame, rng), nhanes)
                row = make_normalized_row(topic, group, stats, baseline_stats)
                sample_rows_out.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        **row,
                        "baseline_t_median": baseline_stats["t_median_absolute"],
                        "baseline_steepness": baseline_stats["steepness_iqr_absolute"],
                        "baseline_mgg_m": baseline_stats["mgg_m"],
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "bootstrap_index": bootstrap_index,
                        "topic": topic,
                        "group": group,
                        "reason": str(exc),
                    }
                )

    samples = pd.DataFrame(sample_rows_out)
    if samples.empty:
        raise RuntimeError("Full bootstrap produced no usable samples")
    failure_df = pd.DataFrame(failures)
    return central_rows, samples, failure_df


def assign_bootstrap_samples(
    central_rows: pd.DataFrame,
    samples: pd.DataFrame,
    scenario_curves: dict[str, list[CoordinateCurve]],
) -> pd.DataFrame:
    """Assign every bootstrap sample to the closest coordinate for each curve set."""
    assignments = []
    mean_curves = load_mean_curves()
    curve_sets = {STATISTICAL_CURVE_SET: mean_curves, **scenario_curves}

    for (topic, group), sample_group in samples.groupby(["topic", "group"], sort=False):
        central = central_rows[(central_rows["topic"] == topic) & (central_rows["group"] == group)]
        if central.empty:
            continue
        central_z = central[["log_median", "log_steepness"]].iloc[0].to_numpy(dtype=float)

        for curve_set, curves in curve_sets.items():
            central_best = best_projection(central_z, curves)
            central_distances = {
                f"central_distance_{row['coordinate']}": row["distance"]
                for row in project_point_to_curves(central_z, curves)
            }
            for row in sample_group.itertuples(index=False):
                z = np.array([row.log_median, row.log_steepness], dtype=float)
                best = best_projection(z, curves)
                assignments.append(
                    {
                        "bootstrap_index": int(row.bootstrap_index),
                        "curve_set": curve_set,
                        "topic": topic,
                        "group": group,
                        "label": row.label,
                        "x": float(row.x),
                        "y": float(row.y),
                        "log_median": float(row.log_median),
                        "log_steepness": float(row.log_steepness),
                        "best_coordinate": best["coordinate"],
                        "best_coordinate_class": best["coordinate_class"],
                        "best_distance": best["distance"],
                        "best_factor": best["factor"],
                        "central_best_coordinate": central_best["coordinate"],
                        "central_best_coordinate_class": central_best["coordinate_class"],
                        "central_best_distance": central_best["distance"],
                        **central_distances,
                    }
                )
    return pd.DataFrame(assignments)


def summarize_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    """Summarize coordinate assignment frequencies."""
    rows = []
    group_cols = ["curve_set", "topic", "group", "label"]
    for keys, group in assignments.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_bootstrap_usable"] = int(len(group))
        row["central_best_coordinate"] = group["central_best_coordinate"].iloc[0]
        row["central_best_coordinate_class"] = group["central_best_coordinate_class"].iloc[0]
        for param in PARAM_ORDER:
            row[f"p_{param}"] = float((group["best_coordinate"] == param).mean())
        for cls in ("senogenic", "robustness", "extrinsic"):
            row[f"p_{cls}"] = float((group["best_coordinate_class"] == cls).mean())
        for param in PARAM_ORDER:
            row[f"central_distance_{param}"] = float(group[f"central_distance_{param}"].iloc[0])
        row["central_best_distance"] = float(group["central_best_distance"].iloc[0])
        sorted_distances = sorted(float(row[f"central_distance_{param}"]) for param in PARAM_ORDER)
        row["central_second_distance"] = sorted_distances[1] if len(sorted_distances) > 1 else np.nan
        row["central_distance_margin"] = row["central_second_distance"] - row["central_best_distance"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_model_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize deterministic SR baseline-scenario sensitivity ranges."""
    scenario_summary = summary[summary["curve_set"] != STATISTICAL_CURVE_SET].copy()
    rows = []
    for (topic, group, label), group_df in scenario_summary.groupby(["topic", "group", "label"], sort=False):
        row = {"topic": topic, "group": group, "label": label, "n_curve_sets": int(len(group_df))}
        for param in PARAM_ORDER:
            values = group_df[f"p_{param}"]
            row[f"p_{param}_min"] = float(values.min())
            row[f"p_{param}_max"] = float(values.max())
            distances = group_df[f"central_distance_{param}"]
            row[f"central_distance_{param}_min"] = float(distances.min())
            row[f"central_distance_{param}_max"] = float(distances.max())
        for cls in ("senogenic", "robustness", "extrinsic"):
            values = group_df[f"p_{cls}"]
            row[f"p_{cls}_min"] = float(values.min())
            row[f"p_{cls}_max"] = float(values.max())
        row["central_curve_best"] = (
            summary[
                (summary["curve_set"] == STATISTICAL_CURVE_SET)
                & (summary["topic"] == topic)
                & (summary["group"] == group)
            ]["central_best_coordinate"].iloc[0]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_paper_summary(
    central_rows: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact table with central distances and uncertainty ranges."""
    central_summary = summary[summary["curve_set"] == STATISTICAL_CURVE_SET].copy()
    central_summary = central_summary.merge(
        central_rows[["topic", "group", "x", "y", "log_median", "log_steepness"]],
        on=["topic", "group"],
        how="left",
    )
    columns = [
        "topic",
        "group",
        "label",
        "x",
        "y",
        "log_median",
        "log_steepness",
        "n_bootstrap_usable",
        "central_best_coordinate",
        "central_best_coordinate_class",
        "central_best_distance",
        "central_second_distance",
        "central_distance_margin",
        *[f"central_distance_{param}" for param in PARAM_ORDER],
        *[f"p_{param}" for param in PARAM_ORDER],
        "p_senogenic",
        "p_robustness",
        "p_extrinsic",
    ]
    paper = central_summary[columns].copy()
    sensitivity_cols = [
        "topic",
        "group",
        "n_curve_sets",
        *[f"p_{param}_min" for param in PARAM_ORDER],
        *[f"p_{param}_max" for param in PARAM_ORDER],
        "p_senogenic_min",
        "p_senogenic_max",
        "p_robustness_min",
        "p_robustness_max",
        "p_extrinsic_min",
        "p_extrinsic_max",
        *[f"central_distance_{param}_min" for param in PARAM_ORDER],
        *[f"central_distance_{param}_max" for param in PARAM_ORDER],
    ]
    out = paper.merge(sensitivity[sensitivity_cols], on=["topic", "group"], how="left")
    out["central_best_coordinate_display"] = out["central_best_coordinate"].map(display_coordinate)
    out["p_m_ex"] = out["p_h_ext"]
    out["p_m_ex_min"] = out["p_h_ext_min"]
    out["p_m_ex_max"] = out["p_h_ext_max"]
    out["central_distance_m_ex"] = out["central_distance_h_ext"]
    out["central_distance_m_ex_min"] = out["central_distance_h_ext_min"]
    out["central_distance_m_ex_max"] = out["central_distance_h_ext_max"]
    return out


def relative_path(path: Path) -> str:
    """Return a repo-relative path for metadata and methods logs."""
    return str(path.relative_to(PROJECT_ROOT))


def project_many_to_curve(z_values: np.ndarray, curve: CoordinateCurve) -> np.ndarray:
    """Project many log-coordinate points to one curve and return curve factors."""
    dx = curve.z_x[None, :] - z_values[:, 0, None]
    dy = curve.z_y[None, :] - z_values[:, 1, None]
    nearest = np.nanargmin(dx * dx + dy * dy, axis=1)
    return curve.factor_grid[nearest]


def get_coordinate_curve(curves: list[CoordinateCurve], coordinate: str) -> CoordinateCurve:
    """Return the named coordinate curve from a curve set."""
    for curve in curves:
        if curve.param == coordinate:
            return curve
    raise RuntimeError(f"No {coordinate} curve found in curve set")


def quantile_interval(values: pd.Series | np.ndarray) -> tuple[float, float]:
    """Return the central 95% bootstrap interval."""
    series = pd.Series(values, dtype=float).dropna()
    if series.empty:
        return np.nan, np.nan
    return float(series.quantile(0.025)), float(series.quantile(0.975))


def build_xc_equivalent_uncertainty(
    central_rows: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    scenario_curves: dict[str, list[CoordinateCurve]],
) -> pd.DataFrame:
    """Project exposure points onto the Xc curve with statistical and model uncertainty."""
    curve_sets = {STATISTICAL_CURVE_SET: load_mean_curves(), **scenario_curves}
    rows = []
    for row in central_rows.itertuples(index=False):
        sample_group = bootstrap_samples[
            (bootstrap_samples["topic"] == row.topic)
            & (bootstrap_samples["group"] == row.group)
        ]
        z_samples = sample_group[["log_median", "log_steepness"]].to_numpy(dtype=float)
        curve_ranges = []
        boot_low = boot_high = np.nan

        for curve_set, curves in curve_sets.items():
            curve = get_coordinate_curve(curves, "Xc")
            factors = project_many_to_curve(z_samples, curve)
            low, high = quantile_interval(factors)
            curve_ranges.append((low, high))
            if curve_set == STATISTICAL_CURVE_SET:
                boot_low, boot_high = low, high

        central_curve = get_coordinate_curve(curve_sets[STATISTICAL_CURVE_SET], "Xc")
        central_factor = project_point_to_curve(
            np.array([row.log_median, row.log_steepness], dtype=float),
            central_curve,
        )[1]
        rows.append(
            {
                "topic": row.topic,
                "group": row.group,
                "label": row.label,
                "xc_factor": central_factor,
                "xc_factor_boot_low": boot_low,
                "xc_factor_boot_high": boot_high,
                "xc_factor_full_low": float(np.nanmin([low for low, _ in curve_ranges])),
                "xc_factor_full_high": float(np.nanmax([high for _, high in curve_ranges])),
                "n": int(len(sample_group)),
            }
        )
    return pd.DataFrame(rows)


def format_value_range(center: float, low: float, high: float, digits: int) -> str:
    """Format one central value with its bootstrap or full-uncertainty range."""
    if not (np.isfinite(center) and np.isfinite(low) and np.isfinite(high)):
        return "NA"
    fmt = f"{{:.{digits}f}}"
    return f"{fmt.format(center)} ({fmt.format(low)}-{fmt.format(high)})"


def format_percent_range(center: float, low: float, high: float) -> str:
    """Format assignment fractions as manuscript-facing percentages."""
    if not (np.isfinite(center) and np.isfinite(low) and np.isfinite(high)):
        return "NA"
    return f"{100 * center:.0f}% ({100 * low:.0f}%-{100 * high:.0f}%)"


def build_extended_data_table(
    paper_summary: pd.DataFrame,
    bootstrap_samples: pd.DataFrame,
    xc_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the manuscript Extended Data Table 1 projection summary."""
    xc_lookup = xc_summary.set_index(["topic", "group"])
    rows = []
    for row in paper_summary.itertuples(index=False):
        samples = bootstrap_samples[
            (bootstrap_samples["topic"] == row.topic)
            & (bootstrap_samples["group"] == row.group)
        ]
        x_low, x_high = quantile_interval(samples["x"])
        y_low, y_high = quantile_interval(samples["y"])
        xc_row = xc_lookup.loc[(row.topic, row.group)]
        rows.append(
            {
                "Topic": TOPIC_DISPLAY.get(row.topic, row.topic),
                "Group": TABLE_LABEL_OVERRIDES.get(row.label, row.label),
                "M/M₀ (bootstrap range)": format_value_range(row.x, x_low, x_high, 3),
                "S/S₀ (bootstrap range)": format_value_range(row.y, y_low, y_high, 3),
                "Robustness % (full uncertainty range)": format_percent_range(
                    row.p_robustness,
                    row.p_robustness_min,
                    row.p_robustness_max,
                ),
                "Senogenic % (full uncertainty range)": format_percent_range(
                    row.p_senogenic,
                    row.p_senogenic_min,
                    row.p_senogenic_max,
                ),
                "mₑₓ % (full uncertainty range)": format_percent_range(
                    row.p_extrinsic,
                    row.p_extrinsic_min,
                    row.p_extrinsic_max,
                ),
                "Xc factor (full uncertainty range)": format_value_range(
                    xc_row["xc_factor"],
                    xc_row["xc_factor_full_low"],
                    xc_row["xc_factor_full_high"],
                    2,
                ),
            }
        )
    return pd.DataFrame(rows)


def format_probability(value: float) -> str:
    """Format probabilities for the methods log."""
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def write_methods_log(
    args: argparse.Namespace,
    paper_summary: pd.DataFrame,
    failures: pd.DataFrame,
) -> None:
    """Write a compact prose and numeric log for manuscript drafting."""
    selected = [
        "Good",
        "Q4 income",
        "0-1 drink/day",
        "Some Activity",
        "7-9 h sleep",
        "Q1 sleep frailty",
        "weekly",
        "some college",
        "Q1 income",
        "1-5 h sleep",
        "Q4 sleep frailty",
        "no high school",
    ]
    rows = []
    for label in selected:
        match = paper_summary[paper_summary["label"] == label]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            "| {label} | {best} | {dist:.4f} | {margin:.4f} | {prob} | {prob_min}-{prob_max} | {ext} | {ext_min}-{ext_max} |".format(
                label=row["label"],
                best=display_coordinate(str(row["central_best_coordinate"])),
                dist=float(row["central_best_distance"]),
                margin=float(row["central_distance_margin"]),
                prob=format_probability(float(row["p_robustness"])),
                prob_min=format_probability(float(row["p_robustness_min"])),
                prob_max=format_probability(float(row["p_robustness_max"])),
                ext=format_probability(float(row["p_extrinsic"])),
                ext_min=format_probability(float(row["p_extrinsic_min"])),
                ext_max=format_probability(float(row["p_extrinsic_max"])),
            )
        )

    failure_text = "None"
    if not failures.empty:
        failure_text = failures.to_csv(index=False).strip()

    text = f"""# Fig. 3 NHANES Coordinate Projection Uncertainty

Generated by `{Path(__file__).name}`.

## Distance Definition

Each exposure group is represented by:

`z_i = (log(M_i / M_0), log(S_i / S_0))`

where `M_i` is median lifespan, `S_i` is steepness, and `M_0`, `S_0` are the zero-Makeham NHANES baseline values.

For each coordinate curve `C_p(q)`, including extrinsic mortality `m_ex`, the reported distance is the minimum Euclidean distance to the full model response curve:

`D_i,p = min_q || z_i - C_p(q) ||_2`

The central assignment is the coordinate with the smallest central distance. Bootstrap assignment probabilities are the fraction of bootstrap replicates assigned to each nearest coordinate.

## Uncertainty Propagation

- Statistical bootstrap: `n={args.n_bootstrap}` participant-level resamples with seed `{args.seed}`.
- In each replicate, the full NHANES baseline and each exposure group are resampled with replacement.
- For each replicate, the left-truncated Kaplan-Meier survival curve is recomputed.
- For each replicate, one-year central death rates in the full NHANES baseline are refit to a Makeham-Gamma-Gompertz model.
- The fitted Makeham term is then set to zero, generating the zero-extrinsic baseline for that replicate.
- Median lifespan and steepness are recomputed and projected onto the SR coordinate curves.
- Model-curve sensitivity is handled separately by repeating projection across all deterministic SR baseline-sensitivity curve sets used to generate the Fig. 3 shaded regions.

## Selected Results

Distances are central Euclidean log-space nearest-curve distances. Fraction ranges are the min-max bootstrap assignment frequencies across the shaded-region curve sets.

| Group | Central nearest coordinate | Distance | Margin to second | Robustness central | Robustness shaded range | Extrinsic central | Extrinsic shaded range |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Failed Bootstrap Rows

```csv
{failure_text}
```

## Output Files

- `{relative_path(PAPER_SUMMARY_PATH)}`
- `{relative_path(FULL_BOOTSTRAP_SUMMARY_PATH)}`
- `{relative_path(MODEL_SENSITIVITY_PATH)}`
- `{relative_path(XC_EQUIVALENT_PATH)}`
- `{relative_path(EXTENDED_DATA_TABLE_PATH)}`
- `{relative_path(EXTENDED_DATA_TABLE_PUBLIC_PATH)}`
- `{relative_path(FULL_BOOTSTRAP_PATH)}`
- `{relative_path(FAILURES_PATH)}`
"""
    METHODS_LOG_PATH.write_text(text, encoding="utf-8")


def write_metadata(
    args: argparse.Namespace,
    failures: pd.DataFrame,
    wrote_point_mc: bool,
    xc_range_source: str,
) -> None:
    """Write a small methods manifest for the projection analysis."""
    metadata = {
        "analysis": "Fig. 3 coordinate-wise NHANES projection uncertainty",
        "coordinate_space": "log(median_lifespan / zero-Makeham NHANES baseline), log(steepness / zero-Makeham NHANES baseline)",
        "distance_metric": "minimum Euclidean distance from each exposure point to the nearest point on each full SR coordinate response curve",
        "coordinate_names": "internal source files use h_ext for the extrinsic curve; manuscript-facing outputs label this coordinate m_ex",
        "statistical_bootstrap": {
            "n_requested": args.n_bootstrap,
            "seed": args.seed,
            "participant_resampling": "baseline and each exposure group sampled with replacement",
            "survival_estimator": "left-truncated Kaplan-Meier",
            "extrinsic_correction": "fit one-year NHANES central death rates to Makeham-Gamma-Gompertz, then set Makeham m=0",
            "projection_assignment": "nearest response curve by Euclidean distance in log-normalized median/steepness space",
            "failed_rows": int(len(failures)),
        },
        "model_curve_sensitivity": {
            "mean_curve_set": STATISTICAL_CURVE_SET,
            "scenario_curve_sets": "deterministic baseline perturbation scenarios from Fig. 3 saved metrics",
            "interpretation": "bootstrap assignment probabilities are recomputed for each scenario curve set; min-max ranges across curve sets quantify shaded-region sensitivity",
        },
        "xc_equivalent_factor_ranges": {
            "source": xc_range_source,
            "note": "The manuscript table uses the cached Fig. 3 Xc-equivalent factor-range source when present for the default 300-bootstrap run. Pass --recompute-xc-factor-ranges to regenerate the direct nearest-Xc-curve fallback.",
        },
        "outputs": {
            "point_distances": relative_path(POINT_DISTANCES_PATH) if POINT_DISTANCES_PATH.exists() else None,
            "point_mc": relative_path(POINT_ASSIGNMENT_PATH) if wrote_point_mc else None,
            "full_bootstrap_assignments": relative_path(FULL_BOOTSTRAP_PATH),
            "full_bootstrap_summary": relative_path(FULL_BOOTSTRAP_SUMMARY_PATH),
            "model_sensitivity_summary": relative_path(MODEL_SENSITIVITY_PATH),
            "paper_summary": relative_path(PAPER_SUMMARY_PATH),
            "xc_equivalent_summary": relative_path(XC_EQUIVALENT_PATH),
            "extended_data_table": relative_path(EXTENDED_DATA_TABLE_PATH),
            "extended_data_table_public_copy": relative_path(EXTENDED_DATA_TABLE_PUBLIC_PATH),
            "failures": relative_path(FAILURES_PATH),
            "methods_log": relative_path(METHODS_LOG_PATH),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--skip-point-mc", action="store_true")
    parser.add_argument(
        "--recompute-xc-factor-ranges",
        action="store_true",
        help="Regenerate Xc-equivalent factor ranges instead of reusing the cached manuscript table source.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scenario_curves = load_scenario_curves()

    wrote_point_mc = False
    if not args.skip_point_mc:
        distances, point_assignments = point_se_projection()
        distances.to_csv(POINT_DISTANCES_PATH, index=False)
        point_assignments.to_csv(POINT_ASSIGNMENT_PATH, index=False)
        wrote_point_mc = True

    central_rows, bootstrap_samples, failures = run_full_bootstrap(args.n_bootstrap, args.seed)
    assignments = assign_bootstrap_samples(central_rows, bootstrap_samples, scenario_curves)
    summary = summarize_assignments(assignments)
    sensitivity = summarize_model_sensitivity(summary)
    paper_summary = build_paper_summary(central_rows, summary, sensitivity)
    use_cached_xc_summary = (
        not args.recompute_xc_factor_ranges
        and args.n_bootstrap == DEFAULT_BOOTSTRAPS
        and args.seed == RNG_SEED
        and XC_EQUIVALENT_PATH.exists()
    )
    if use_cached_xc_summary:
        xc_summary = pd.read_csv(XC_EQUIVALENT_PATH)
        xc_range_source = "cached_current_manuscript_table"
    else:
        xc_summary = build_xc_equivalent_uncertainty(central_rows, bootstrap_samples, scenario_curves)
        xc_range_source = "recomputed_direct_nearest_xc_curve"
    extended_data_table = build_extended_data_table(paper_summary, bootstrap_samples, xc_summary)

    assignments.to_csv(FULL_BOOTSTRAP_PATH, index=False)
    summary.to_csv(FULL_BOOTSTRAP_SUMMARY_PATH, index=False)
    sensitivity.to_csv(MODEL_SENSITIVITY_PATH, index=False)
    paper_summary.to_csv(PAPER_SUMMARY_PATH, index=False)
    xc_summary.to_csv(XC_EQUIVALENT_PATH, index=False)
    extended_data_table.to_csv(EXTENDED_DATA_TABLE_PATH, index=False)
    extended_data_table.to_csv(EXTENDED_DATA_TABLE_PUBLIC_PATH, index=False)
    failures.to_csv(FAILURES_PATH, index=False)
    write_metadata(args, failures, wrote_point_mc, xc_range_source)
    write_methods_log(args, paper_summary, failures)

    if wrote_point_mc:
        print(f"Saved point distances: {POINT_DISTANCES_PATH}")
        print(f"Saved point-MC assignments: {POINT_ASSIGNMENT_PATH}")
    print(f"Saved full bootstrap assignments: {FULL_BOOTSTRAP_PATH}")
    print(f"Saved full bootstrap summary: {FULL_BOOTSTRAP_SUMMARY_PATH}")
    print(f"Saved model sensitivity summary: {MODEL_SENSITIVITY_PATH}")
    print(f"Saved paper summary: {PAPER_SUMMARY_PATH}")
    print(f"Saved Xc-equivalent uncertainty summary: {XC_EQUIVALENT_PATH}")
    print(f"Saved Extended Data Table 1: {EXTENDED_DATA_TABLE_PUBLIC_PATH}")
    print(f"Saved bootstrap failures: {FAILURES_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved methods log: {METHODS_LOG_PATH}")
    if not failures.empty:
        print(f"Warning: {len(failures)} bootstrap group rows failed; see metadata count.")


if __name__ == "__main__":
    main()
