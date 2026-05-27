#!/usr/bin/env python3
"""Explore distribution-tail constraints on senogenic heterogeneity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import SAVED_RESULTS_DIR


EXPLORATION_DIR = Path(__file__).resolve().parent
PLOTS_DIR = EXPLORATION_DIR / "plots"
REPORT_PATH = EXPLORATION_DIR / "README.md"

CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "senogenic_tail_constraints"
METADATA_PATH = CACHE_DIR / "metadata.json"
SCENARIO_CACHE_PATH = CACHE_DIR / "scenario_results.csv"
MIXTURE_COMPONENTS_PATH = CACHE_DIR / "mixture_component_survival.csv"

CSV_DIR = SAVED_RESULTS_DIR / "csv"
DISTRIBUTION_RESULTS_PATH = CSV_DIR / "senogenic_tail_distribution_scenarios.csv"
TRUNCATION_RESULTS_PATH = CSV_DIR / "senogenic_tail_truncated_eta.csv"
CORRELATION_RESULTS_PATH = CSV_DIR / "senogenic_tail_correlated_eta_beta.csv"
MIXTURE_BOUNDS_PATH = CSV_DIR / "senogenic_tail_mixture_bounds.csv"
MIXTURE_GRID_PATH = CSV_DIR / "senogenic_tail_mixture_grid.csv"

BASELINE_FIT_PATH = (
    SAVED_RESULTS_DIR
    / "fit_archive"
    / "records"
    / "joint2019_tail90_sweden_emphasis.json"
)

RANDOM_SEED = 20260525
DEFAULT_N = 200_000
DEFAULT_MIXTURE_COMPONENT_N = 500_000
TMAX = 320.0
DT = 0.1
SAVE_TIMES = TMAX
TOP_SURVIVAL = 1e-4
CONDITION_AGE = 90
HMD_BOUND_AGES = (100, 105, 110)
SURVIVAL_AGES = np.arange(CONDITION_AGE, 121, 1, dtype=float)

FAMILY_CVS = (0.025, 0.05, 0.075, 0.10, 0.125)
FAMILY_DISTS = ("gaussian", "lognormal", "uniform", "student_t4")
SENOGENIC_MODES = ("eta_only", "beta_only")

TRUNCATION_CVS = (0.05, 0.10, 0.15, 0.20)
TRUNCATION_CUTOFFS = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)

CORRELATION_CVS = (0.05, 0.10, 0.15)
CORRELATIONS = (-0.75, -0.25, 0.0, 0.5, 0.9)

MIXTURE_DELTAS = (0.02, 0.05, 0.10, 0.15, 0.20)
MIXTURE_P_GRID = np.concatenate(([0.0], np.logspace(-6, -1, 101)))

PARAM_COLORS = {
    "eta_only": "#0B7F8C",
    "beta_only": "#173A6A",
    "joint_eta_beta": "#6B4C9A",
    "truncated_eta": "#B05C2E",
}

DIST_MARKERS = {
    "gaussian": "o",
    "lognormal": "s",
    "uniform": "^",
    "student_t4": "D",
}


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    analysis: str
    mode: str
    distribution: str
    cv: float
    favorable_cutoff: float
    rho: float
    delta: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Individuals per distribution/truncation/correlation scenario.")
    parser.add_argument(
        "--mixture-component-n",
        type=int,
        default=DEFAULT_MIXTURE_COMPONENT_N,
        help="Individuals per fixed-eta component for mixture bounds.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached simulation rows.")
    parser.add_argument("--no-parallel", action="store_true", help="Run SR simulations without multiprocessing.")
    parser.add_argument("--plots-only", action="store_true", help="Reuse cached rows and rebuild plots/report only.")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_sweden_baseline() -> dict[str, float]:
    record = json.loads(BASELINE_FIT_PATH.read_text())
    fitted = record["summary"]["fitted_parameters"]
    return {
        "eta": float(fitted["eta"]),
        "beta": float(fitted["beta"]),
        "kappa": 0.5,
        "epsilon": float(fitted["epsilon"]),
        "Xc": float(fitted["SWE_Xc"]),
    }


def metadata(n: int, mixture_component_n: int, parallel: bool) -> dict[str, object]:
    return {
        "task": "senogenic_tail_constraints",
        "baseline_fit_path": str(BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)),
        "baseline": load_sweden_baseline(),
        "n": int(n),
        "mixture_component_n": int(mixture_component_n),
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "top_survival": TOP_SURVIVAL,
        "condition_age": CONDITION_AGE,
        "h_ext": 0.0,
        "family_cvs": list(FAMILY_CVS),
        "family_distributions": list(FAMILY_DISTS),
        "truncation_cvs": list(TRUNCATION_CVS),
        "truncation_cutoffs": list(TRUNCATION_CUTOFFS),
        "correlation_cvs": list(CORRELATION_CVS),
        "correlations": list(CORRELATIONS),
        "mixture_deltas": list(MIXTURE_DELTAS),
        "random_seed": RANDOM_SEED,
        "parallel": bool(parallel),
    }


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "xtick.labelsize": 14.5,
            "ytick.labelsize": 14.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sample_positive_gaussian(mean: float, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if cv == 0:
        return np.full(n, mean, dtype=float)
    values = rng.normal(mean, mean * cv, size=n)
    bad = values <= 0
    while np.any(bad):
        values[bad] = rng.normal(mean, mean * cv, size=int(bad.sum()))
        bad = values <= 0
    return values


def sample_factor(distribution: str, cv: float, n: int, rng: np.random.Generator) -> np.ndarray:
    if cv == 0:
        return np.ones(n, dtype=float)

    if distribution == "gaussian":
        return sample_positive_gaussian(1.0, cv, n, rng)

    if distribution == "lognormal":
        sigma = math.sqrt(math.log1p(cv**2))
        mu = -0.5 * sigma**2
        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    if distribution == "uniform":
        half_width = math.sqrt(3.0) * cv
        if half_width >= 1.0:
            raise ValueError("Uniform factor would cross zero.")
        return rng.uniform(1.0 - half_width, 1.0 + half_width, size=n)

    if distribution == "student_t4":
        df = 4.0
        scale = cv / math.sqrt(df / (df - 2.0))
        values = 1.0 + scale * rng.standard_t(df=df, size=n)
        bad = values <= 0
        while np.any(bad):
            values[bad] = 1.0 + scale * rng.standard_t(df=df, size=int(bad.sum()))
            bad = values <= 0
        return values

    raise ValueError(f"Unknown distribution: {distribution}")


def sample_truncated_eta_factor(cv: float, favorable_cutoff: float, n: int, rng: np.random.Generator) -> np.ndarray:
    lower = 1.0 - favorable_cutoff
    values = rng.normal(1.0, cv, size=n)
    bad = (values <= 0) | (values < lower)
    while np.any(bad):
        values[bad] = rng.normal(1.0, cv, size=int(bad.sum()))
        bad = (values <= 0) | (values < lower)
    return values


def sample_correlated_lognormal_factors(
    cv: float,
    rho: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = math.sqrt(math.log1p(cv**2))
    mu = -0.5 * sigma**2
    cov = sigma**2 * np.array([[1.0, rho], [rho, 1.0]], dtype=float)
    logs = rng.multivariate_normal(mean=np.array([mu, mu]), cov=cov, size=n)
    return np.exp(logs[:, 0]), np.exp(logs[:, 1])


def scenario_list() -> list[Scenario]:
    scenarios = [
        Scenario("baseline", "baseline", "baseline", "fixed", 0.0, np.nan, np.nan, 0.0)
    ]

    for mode in SENOGENIC_MODES:
        for distribution in FAMILY_DISTS:
            for cv in FAMILY_CVS:
                scenarios.append(
                    Scenario(
                        scenario_id=f"family__{mode}__{distribution}__cv{cv:.3f}",
                        analysis="distribution_family",
                        mode=mode,
                        distribution=distribution,
                        cv=cv,
                        favorable_cutoff=np.nan,
                        rho=np.nan,
                        delta=0.0,
                    )
                )

    for cv in TRUNCATION_CVS:
        for cutoff in TRUNCATION_CUTOFFS:
            scenarios.append(
                Scenario(
                    scenario_id=f"truncated_eta__cv{cv:.3f}__cut{cutoff:.3f}",
                    analysis="truncated_eta",
                    mode="eta_only",
                    distribution="truncated_gaussian",
                    cv=cv,
                    favorable_cutoff=cutoff,
                    rho=np.nan,
                    delta=0.0,
                )
            )

    for cv in CORRELATION_CVS:
        for rho in CORRELATIONS:
            scenarios.append(
                Scenario(
                    scenario_id=f"correlated__cv{cv:.3f}__rho{rho:.2f}",
                    analysis="correlated_eta_beta",
                    mode="joint_eta_beta",
                    distribution="correlated_lognormal",
                    cv=cv,
                    favorable_cutoff=np.nan,
                    rho=rho,
                    delta=0.0,
                )
            )

    return scenarios


def build_params_and_tau(
    scenario: Scenario,
    baseline: dict[str, float],
    n: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(stable_seed(RANDOM_SEED, scenario.scenario_id, n, "params"))
    params = {key: np.full(n, value, dtype=float) for key, value in baseline.items()}

    if scenario.analysis == "baseline":
        tau = params["beta"] / params["eta"]
        return params, tau, {}

    if scenario.analysis == "distribution_family":
        factor = sample_factor(scenario.distribution, scenario.cv, n, rng)
        if scenario.mode == "eta_only":
            params["eta"] = baseline["eta"] * factor
        elif scenario.mode == "beta_only":
            params["beta"] = baseline["beta"] * factor
        else:
            raise ValueError(f"Unsupported mode for distribution family: {scenario.mode}")
        tau = params["beta"] / params["eta"]
        return params, tau, {"factor_mean": float(np.mean(factor)), "factor_cv": coefficient_of_variation(factor)}

    if scenario.analysis == "truncated_eta":
        factor = sample_truncated_eta_factor(scenario.cv, scenario.favorable_cutoff, n, rng)
        params["eta"] = baseline["eta"] * factor
        tau = params["beta"] / params["eta"]
        return params, tau, {"factor_mean": float(np.mean(factor)), "factor_cv": coefficient_of_variation(factor)}

    if scenario.analysis == "correlated_eta_beta":
        eta_factor, beta_factor = sample_correlated_lognormal_factors(scenario.cv, scenario.rho, n, rng)
        params["eta"] = baseline["eta"] * eta_factor
        params["beta"] = baseline["beta"] * beta_factor
        tau = params["beta"] / params["eta"]
        return (
            params,
            tau,
            {
                "eta_factor_mean": float(np.mean(eta_factor)),
                "eta_factor_cv": coefficient_of_variation(eta_factor),
                "beta_factor_mean": float(np.mean(beta_factor)),
                "beta_factor_cv": coefficient_of_variation(beta_factor),
            },
        )

    raise ValueError(f"Unknown scenario analysis: {scenario.analysis}")


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if mean == 0:
        return np.nan
    return float(np.std(values, ddof=0) / mean)


def run_sr_deaths(
    params: dict[str, np.ndarray],
    n: int,
    seed: int,
    parallel: bool,
) -> np.ndarray:
    sim = utils.create_sr_simulation(
        params_dict=params,
        n=n,
        h_ext=0.0,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        parallel=parallel,
        break_early=True,
        random_seed=seed,
    )
    return np.asarray(sim.death_times, dtype=float)


def top_survival_age(death_times: np.ndarray, n: int) -> tuple[float, bool]:
    finite = np.sort(death_times[np.isfinite(death_times) & (death_times < TMAX - 0.5 * DT)])
    target_dead_count = (1.0 - TOP_SURVIVAL) * n
    if target_dead_count > finite.size:
        return TMAX, True
    return float(np.quantile(finite, target_dead_count / finite.size)), False


def survival_at_ages(death_times: np.ndarray, ages: np.ndarray) -> np.ndarray:
    return np.asarray([np.mean(death_times >= age) for age in ages], dtype=float)


def conditional_survival_at_ages(death_times: np.ndarray, ages: np.ndarray, condition_age: float) -> np.ndarray:
    at_risk = np.sum(death_times >= condition_age)
    if at_risk == 0:
        return np.full_like(ages, np.nan, dtype=float)
    return np.asarray([np.sum(death_times >= age) / at_risk for age in ages], dtype=float)


def summarize_scenario(
    scenario: Scenario,
    params: dict[str, np.ndarray],
    tau: np.ndarray,
    death_times: np.ndarray,
    n: int,
    baseline: dict[str, float],
    extra: dict[str, float],
) -> dict[str, object]:
    baseline_tau = float(baseline["beta"] / baseline["eta"])
    top_age, censored = top_survival_age(death_times, n)
    cond = conditional_survival_at_ages(death_times, np.asarray(HMD_BOUND_AGES, dtype=float), CONDITION_AGE)

    row: dict[str, object] = {
        "scenario_id": scenario.scenario_id,
        "analysis": scenario.analysis,
        "mode": scenario.mode,
        "distribution": scenario.distribution,
        "cv_requested": scenario.cv,
        "favorable_cutoff": scenario.favorable_cutoff,
        "rho": scenario.rho,
        "delta": scenario.delta,
        "n": int(n),
        "top_0_01pct_lifespan": top_age,
        "top_0_01pct_censored": bool(censored),
        "q9999_tau": float(np.quantile(tau, 0.9999)),
        "q9999_tau_factor": float(np.quantile(tau / baseline_tau, 0.9999)),
        "q999_tau_factor": float(np.quantile(tau / baseline_tau, 0.999)),
        "tau_cv": coefficient_of_variation(tau),
        "eta_cv_actual": coefficient_of_variation(params["eta"]),
        "beta_cv_actual": coefficient_of_variation(params["beta"]),
        "eta_q0001_factor": float(np.quantile(params["eta"] / np.mean(params["eta"]), 0.0001)),
        "beta_q9999_factor": float(np.quantile(params["beta"] / np.mean(params["beta"]), 0.9999)),
    }
    for age, value in zip(HMD_BOUND_AGES, cond):
        row[f"conditional_survival_{age}_from90"] = float(value)
    row.update(extra)
    return row


def load_cached_scenarios(current_metadata: dict[str, object], force: bool) -> pd.DataFrame:
    columns = list(summarize_empty_columns())
    if force or not SCENARIO_CACHE_PATH.exists() or not METADATA_PATH.exists():
        return pd.DataFrame(columns=columns)
    cached_metadata = json.loads(METADATA_PATH.read_text())
    if cached_metadata != current_metadata:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(SCENARIO_CACHE_PATH)


def summarize_empty_columns() -> list[str]:
    return [
        "scenario_id",
        "analysis",
        "mode",
        "distribution",
        "cv_requested",
        "favorable_cutoff",
        "rho",
        "delta",
        "n",
        "top_0_01pct_lifespan",
        "top_0_01pct_censored",
        "q9999_tau",
        "q9999_tau_factor",
        "q999_tau_factor",
        "tau_cv",
        "eta_cv_actual",
        "beta_cv_actual",
        "eta_q0001_factor",
        "beta_q9999_factor",
        "conditional_survival_100_from90",
        "conditional_survival_105_from90",
        "conditional_survival_110_from90",
    ]


def save_scenario_cache(rows: pd.DataFrame, current_metadata: dict[str, object]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SCENARIO_CACHE_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(current_metadata, indent=2) + "\n")


def run_missing_scenarios(
    n: int,
    parallel: bool,
    force: bool,
    plots_only: bool,
    current_metadata: dict[str, object],
) -> pd.DataFrame:
    baseline = load_sweden_baseline()
    rows = load_cached_scenarios(current_metadata, force=force)
    scenarios = scenario_list()
    required_ids = {scenario.scenario_id for scenario in scenarios}
    if not rows.empty:
        rows = rows[rows["scenario_id"].isin(required_ids)].copy()

    completed = set(rows["scenario_id"]) if not rows.empty else set()
    if plots_only:
        missing = required_ids - completed
        if missing:
            raise RuntimeError(f"Cannot use --plots-only; missing {len(missing)} scenario rows.")
        rows = refresh_parameter_summaries(rows, n)
        save_scenario_cache(rows, current_metadata)
        return rows

    for scenario in scenarios:
        if scenario.scenario_id in completed:
            continue
        print(f"Running scenario: {scenario.scenario_id}", flush=True)
        params, tau, extra = build_params_and_tau(scenario, baseline, n)
        deaths = run_sr_deaths(
            params=params,
            n=n,
            seed=stable_seed(RANDOM_SEED, scenario.scenario_id, n, "simulation"),
            parallel=parallel,
        )
        row = summarize_scenario(scenario, params, tau, deaths, n, baseline, extra)
        if rows.empty:
            rows = pd.DataFrame([row])
        else:
            rows = pd.concat([rows, pd.DataFrame([row])], ignore_index=True)
        completed.add(scenario.scenario_id)
        save_scenario_cache(rows, current_metadata)

    rows = refresh_parameter_summaries(rows, n)
    save_scenario_cache(rows, current_metadata)
    return rows


def refresh_parameter_summaries(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    """Refresh deterministic parameter summaries without rerunning SR simulations."""
    if rows.empty:
        return rows

    baseline = load_sweden_baseline()
    baseline_tau = baseline["beta"] / baseline["eta"]
    rows = rows.copy()
    row_index_by_id = {row.scenario_id: index for index, row in rows.iterrows()}

    for scenario in scenario_list():
        if scenario.scenario_id not in row_index_by_id:
            continue
        params, tau, extra = build_params_and_tau(scenario, baseline, n)
        index = row_index_by_id[scenario.scenario_id]
        rows.loc[index, "q9999_tau"] = float(np.quantile(tau, 0.9999))
        rows.loc[index, "q9999_tau_factor"] = float(np.quantile(tau / baseline_tau, 0.9999))
        rows.loc[index, "q999_tau_factor"] = float(np.quantile(tau / baseline_tau, 0.999))
        rows.loc[index, "tau_cv"] = coefficient_of_variation(tau)
        rows.loc[index, "eta_cv_actual"] = coefficient_of_variation(params["eta"])
        rows.loc[index, "beta_cv_actual"] = coefficient_of_variation(params["beta"])
        rows.loc[index, "eta_q0001_factor"] = float(np.quantile(params["eta"] / baseline["eta"], 0.0001))
        rows.loc[index, "beta_q9999_factor"] = float(np.quantile(params["beta"] / baseline["beta"], 0.9999))
        for key, value in extra.items():
            rows.loc[index, key] = value

    top_age = pd.to_numeric(rows["top_0_01pct_lifespan"], errors="coerce")
    rows.loc[top_age >= TMAX - DT, "top_0_01pct_censored"] = True
    return rows


def load_hmd_conditional_survival() -> pd.DataFrame:
    hmd = HMD("SWE", "both", "period")
    ages, survival = hmd.get_survival(2019, strict=True)
    frame = pd.DataFrame({"age": np.asarray(ages, dtype=float), "survival": np.asarray(survival, dtype=float)})
    frame = frame[(frame["age"] >= CONDITION_AGE) & (frame["age"] <= max(SURVIVAL_AGES))].copy()
    condition = float(frame.loc[np.isclose(frame["age"], CONDITION_AGE), "survival"].iloc[0])
    frame["conditional_survival"] = frame["survival"] / condition
    return frame[["age", "conditional_survival"]]


def run_mixture_components(
    mixture_component_n: int,
    parallel: bool,
    force: bool,
    plots_only: bool,
    current_metadata: dict[str, object],
) -> pd.DataFrame:
    if not force and MIXTURE_COMPONENTS_PATH.exists() and METADATA_PATH.exists():
        cached_metadata = json.loads(METADATA_PATH.read_text())
        if cached_metadata == current_metadata:
            return pd.read_csv(MIXTURE_COMPONENTS_PATH)

    if plots_only:
        raise RuntimeError("Cannot use --plots-only; mixture component cache is missing.")

    baseline = load_sweden_baseline()
    rows = []
    for delta in (0.0, *MIXTURE_DELTAS):
        label = "baseline" if delta == 0 else f"eta_improved_{delta:.3f}"
        print(f"Running mixture component: {label}", flush=True)
        n = mixture_component_n
        params = {key: np.full(n, value, dtype=float) for key, value in baseline.items()}
        params["eta"] = baseline["eta"] * (1.0 - delta)
        deaths = run_sr_deaths(
            params=params,
            n=n,
            seed=stable_seed(RANDOM_SEED, "mixture_component", delta, n),
            parallel=parallel,
        )
        unconditional = survival_at_ages(deaths, SURVIVAL_AGES)
        conditional = conditional_survival_at_ages(deaths, SURVIVAL_AGES, CONDITION_AGE)
        top_age, censored = top_survival_age(deaths, n)
        for age, uncond, cond in zip(SURVIVAL_AGES, unconditional, conditional):
            rows.append(
                {
                    "component": label,
                    "delta": float(delta),
                    "age": float(age),
                    "unconditional_survival": float(uncond),
                    "conditional_survival_from90": float(cond),
                    "top_0_01pct_lifespan": float(top_age) if np.isfinite(top_age) else np.nan,
                    "top_0_01pct_censored": bool(censored),
                    "n": int(n),
                }
            )

    frame = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MIXTURE_COMPONENTS_PATH, index=False)
    return frame


def compute_mixture_bounds(component_curves: pd.DataFrame, hmd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = component_curves[component_curves["delta"] == 0.0].copy()
    baseline_by_age = baseline.set_index("age")["conditional_survival_from90"]
    hmd_by_age = hmd.set_index("age")["conditional_survival"]

    bound_rows = []
    grid_rows = []
    for delta in MIXTURE_DELTAS:
        component = component_curves[component_curves["delta"] == delta].copy()
        comp_by_age = component.set_index("age")["conditional_survival_from90"]
        age_bounds = []
        conservative_age_bounds = []
        for age in HMD_BOUND_AGES:
            age = float(age)
            hmd_value = float(hmd_by_age.loc[age])
            baseline_value = float(baseline_by_age.loc[age])
            component_value = float(comp_by_age.loc[age])
            excess = component_value - baseline_value
            if excess <= 0:
                p_bound = np.inf
            else:
                p_bound = max(0.0, (hmd_value - baseline_value) / excess)
            conservative_bound = hmd_value / component_value if component_value > 0 else np.inf
            age_bounds.append(p_bound)
            conservative_age_bounds.append(conservative_bound)
            bound_rows.append(
                {
                    "delta": float(delta),
                    "age": age,
                    "hmd_conditional_survival_from90": hmd_value,
                    "baseline_conditional_survival_from90": baseline_value,
                    "component_conditional_survival_from90": component_value,
                    "pmax_mixture_vs_hmd": p_bound,
                    "pmax_conservative_component_only": conservative_bound,
                }
            )

        finite_bound = min(age_bounds)
        finite_conservative = min(conservative_age_bounds)
        for p in MIXTURE_P_GRID:
            mixture_by_age = (1.0 - p) * baseline_by_age + p * comp_by_age
            ratios = []
            excesses = []
            for age in HMD_BOUND_AGES:
                age = float(age)
                hmd_value = float(hmd_by_age.loc[age])
                mix_value = float(mixture_by_age.loc[age])
                ratios.append(mix_value / hmd_value if hmd_value > 0 else np.nan)
                excesses.append(mix_value - hmd_value)
            grid_rows.append(
                {
                    "delta": float(delta),
                    "p": float(p),
                    "max_ratio_to_hmd_100_105_110": float(np.nanmax(ratios)),
                    "max_excess_over_hmd_100_105_110": float(np.nanmax(excesses)),
                    "passes_hmd_bound": bool(np.nanmax(excesses) <= 0.0),
                    "pmax_mixture_vs_hmd": float(finite_bound) if np.isfinite(finite_bound) else np.inf,
                    "pmax_conservative_component_only": float(finite_conservative) if np.isfinite(finite_conservative) else np.inf,
                }
            )

    return pd.DataFrame(bound_rows), pd.DataFrame(grid_rows)


def export_csvs(rows: pd.DataFrame, mixture_bounds: pd.DataFrame, mixture_grid: pd.DataFrame) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    rows[rows["analysis"].isin(["baseline", "distribution_family"])].to_csv(DISTRIBUTION_RESULTS_PATH, index=False)
    rows[rows["analysis"].isin(["baseline", "truncated_eta"])].to_csv(TRUNCATION_RESULTS_PATH, index=False)
    rows[rows["analysis"].isin(["baseline", "correlated_eta_beta"])].to_csv(CORRELATION_RESULTS_PATH, index=False)
    mixture_bounds.to_csv(MIXTURE_BOUNDS_PATH, index=False)
    mixture_grid.to_csv(MIXTURE_GRID_PATH, index=False)


def plot_tail_collapse(rows: pd.DataFrame) -> Path:
    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    data = rows[rows["analysis"] == "distribution_family"].copy()
    baseline = rows[rows["analysis"] == "baseline"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for mode in SENOGENIC_MODES:
        for distribution in FAMILY_DISTS:
            subset = data[(data["mode"] == mode) & (data["distribution"] == distribution)].sort_values("cv_requested")
            axes[0].plot(
                100.0 * subset["cv_requested"],
                subset["top_0_01pct_lifespan"],
                color=PARAM_COLORS[mode],
                marker=DIST_MARKERS[distribution],
                lw=1.9,
                ms=5.5,
                alpha=0.88,
                label=f"{mode_label(mode)}, {distribution_label(distribution)}",
            )
            axes[1].scatter(
                subset["q9999_tau_factor"],
                subset["top_0_01pct_lifespan"],
                color=PARAM_COLORS[mode],
                marker=DIST_MARKERS[distribution],
                s=44,
                alpha=0.88,
                edgecolor="white",
                linewidth=0.35,
            )

    for ax in axes:
        ax.axhline(float(baseline["top_0_01pct_lifespan"]), color="0.25", lw=1.2, ls="--", zorder=0)
        ax.set_ylim(96, TMAX + 10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.set_ylabel("Top 0.01% lifespan [years]")

    censored_count = int(data["top_0_01pct_censored"].astype(bool).sum())
    if censored_count:
        axes[1].text(
            0.98,
            0.04,
            f"{censored_count} points censored at {TMAX:g} y",
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="0.25",
        )

    axes[0].set_xlabel("Requested one-parameter CV [%]")
    axes[0].set_title("Distribution-specific CV response")
    axes[1].set_xlabel(r"$q_{0.9999}[(\beta/\eta)/(\beta_0/\eta_0)]$")
    axes[1].set_title(r"Collapse by favorable $\beta/\eta$ quantile")
    axes[1].set_xscale("log")
    axes[1].legend(
        handles=legend_handles_for_tail_collapse(),
        loc="upper left",
        frameon=False,
        fontsize=9.5,
    )

    fig.tight_layout(pad=0.8, w_pad=2.5)
    png = PLOTS_DIR / "01_tail_collapse.png"
    pdf = PLOTS_DIR / "01_tail_collapse.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def legend_handles_for_tail_collapse() -> list[object]:
    from matplotlib.lines import Line2D

    handles: list[object] = [
        Line2D([0], [0], color=PARAM_COLORS["eta_only"], lw=2.5, label=r"$\eta$ variation"),
        Line2D([0], [0], color=PARAM_COLORS["beta_only"], lw=2.5, label=r"$\beta$ variation"),
        Line2D([0], [0], color="0.15", marker="o", lw=0, label="Gaussian"),
        Line2D([0], [0], color="0.15", marker="s", lw=0, label="Lognormal"),
        Line2D([0], [0], color="0.15", marker="^", lw=0, label="Uniform"),
        Line2D([0], [0], color="0.15", marker="D", lw=0, label=r"Student-$t_4$"),
    ]
    return handles


def mode_label(mode: str) -> str:
    if mode == "eta_only":
        return r"$\eta$"
    if mode == "beta_only":
        return r"$\beta$"
    return mode


def distribution_label(distribution: str) -> str:
    return {
        "gaussian": "Gaussian",
        "lognormal": "Lognormal",
        "uniform": "Uniform",
        "student_t4": r"Student-$t_4$",
    }.get(distribution, distribution)


def plot_truncation_heatmap(rows: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = rows[rows["analysis"] == "truncated_eta"].copy()
    pivot = data.pivot(index="favorable_cutoff", columns="cv_requested", values="top_0_01pct_lifespan")
    tau_pivot = data.pivot(index="favorable_cutoff", columns="cv_requested", values="q9999_tau_factor")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for ax, table, title, cbar_label, fmt in [
        (axes[0], pivot, "Top 0.01% lifespan", "Years", ".0f"),
        (axes[1], tau_pivot, r"Favorable $\beta/\eta$ quantile", r"$q_{0.9999}$ factor", ".2f"),
    ]:
        image = ax.imshow(table.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels([f"{100 * x:.0f}" for x in table.columns])
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels([f"{100 * y:.0f}" for y in table.index])
        ax.set_xlabel(r"Gaussian $\eta$ CV [%]")
        ax.set_ylabel("Allowed favorable cutoff [%]")
        ax.set_title(title)
        for row_idx in range(table.shape[0]):
            for col_idx in range(table.shape[1]):
                value = table.iloc[row_idx, col_idx]
                ax.text(col_idx, row_idx, format(value, fmt), ha="center", va="center", fontsize=9.3, color="white")
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label)

    fig.tight_layout(pad=0.8, w_pad=2.2)
    png = PLOTS_DIR / "02_truncated_eta_heatmap.png"
    pdf = PLOTS_DIR / "02_truncated_eta_heatmap.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_correlated_eta_beta(rows: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = rows[rows["analysis"] == "correlated_eta_beta"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))

    for cv in CORRELATION_CVS:
        subset = data[np.isclose(data["cv_requested"], cv)].sort_values("rho")
        axes[0].plot(
            subset["rho"],
            subset["q9999_tau_factor"],
            marker="o",
            lw=2.4,
            ms=5.5,
            label=f"{100 * cv:.0f}% marginal CV",
        )
        axes[1].plot(
            subset["rho"],
            subset["top_0_01pct_lifespan"],
            marker="o",
            lw=2.4,
            ms=5.5,
            label=f"{100 * cv:.0f}% marginal CV",
        )

    axes[0].set_ylabel(r"$q_{0.9999}[(\beta/\eta)/(\beta_0/\eta_0)]$")
    axes[0].set_title(r"Correlation narrows $\beta/\eta$")
    axes[1].set_ylabel("Top 0.01% lifespan [years]")
    axes[1].set_title("Tail lifespan follows ratio spread")
    for ax in axes:
        ax.set_xlabel(r"Correlation between $\log\eta$ and $\log\beta$")
        ax.axvline(0, color="0.65", lw=1.0, ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)

    fig.tight_layout(pad=0.8, w_pad=2.2)
    png = PLOTS_DIR / "03_correlated_eta_beta.png"
    pdf = PLOTS_DIR / "03_correlated_eta_beta.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def plot_mixture_bounds(bounds: pd.DataFrame, grid: pd.DataFrame, component_curves: pd.DataFrame, hmd: pd.DataFrame) -> Path:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

    for delta in (0.0, *MIXTURE_DELTAS):
        component = component_curves[component_curves["delta"] == delta].copy()
        label = "Baseline" if delta == 0 else rf"$\eta$ lower by {100 * delta:.0f}%"
        lw = 3.2 if delta == 0 else 2.0
        alpha = 1.0 if delta == 0 else 0.82
        axes[0].plot(
            component["age"],
            component["conditional_survival_from90"],
            lw=lw,
            alpha=alpha,
            label=label,
        )
    axes[0].plot(hmd["age"], hmd["conditional_survival"], color="black", lw=3.0, label="Sweden 2019 HMD")
    axes[0].set_yscale("log")
    axes[0].set_ylim(7e-5, 1.2)
    axes[0].set_xlim(CONDITION_AGE, 116)
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel(f"Conditional survival from age {CONDITION_AGE}")
    axes[0].set_title("Fixed favorable senogenic subgroups")
    axes[0].legend(frameon=False, fontsize=8.5)

    summary = (
        bounds.groupby("delta", as_index=False)
        .agg(
            pmax_mixture_vs_hmd=("pmax_mixture_vs_hmd", "min"),
            pmax_conservative_component_only=("pmax_conservative_component_only", "min"),
        )
        .copy()
    )
    axes[1].plot(
        100 * summary["delta"],
        summary["pmax_conservative_component_only"],
        marker="o",
        lw=2.5,
        color="#315B7D",
        label="Component-only bound",
    )
    axes[1].plot(
        100 * summary["delta"],
        summary["pmax_mixture_vs_hmd"].clip(lower=1e-7),
        marker="s",
        lw=2.5,
        color="#B05C2E",
        label="Full-mixture bound",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Senogenic improvement: lower $\eta$ [%]")
    axes[1].set_ylabel("Maximum subgroup frequency")
    axes[1].set_title("Bound from ages 100, 105, 110")
    axes[1].legend(frameon=False)

    heat = grid.copy()
    heat["delta_percent"] = 100 * heat["delta"]
    heat["p_log10"] = -6.2
    positive_p = heat["p"] > 0
    heat.loc[positive_p, "p_log10"] = np.log10(heat.loc[positive_p, "p"])
    pivot = heat.pivot(index="delta_percent", columns="p_log10", values="max_ratio_to_hmd_100_105_110")
    image = axes[2].imshow(
        np.log10(pivot.to_numpy()),
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-0.4,
        vmax=1.5,
    )
    x_ticks = [-6, -5, -4, -3, -2, -1]
    x_positions = [int(np.argmin(np.abs(pivot.columns.to_numpy() - tick))) for tick in x_ticks]
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels([rf"$10^{{{tick}}}$" for tick in x_ticks])
    axes[2].set_yticks(np.arange(len(pivot.index)))
    axes[2].set_yticklabels([f"{y:.0f}" for y in pivot.index])
    axes[2].set_xlabel("Subgroup frequency")
    axes[2].set_ylabel(r"Lower $\eta$ [%]")
    axes[2].set_title("Max survival ratio to HMD")
    cbar = fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.03)
    cbar.set_label(r"$\log_{10}$ max ratio")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.8, w_pad=2.2)
    png = PLOTS_DIR / "04_rare_subgroup_mixture.png"
    pdf = PLOTS_DIR / "04_rare_subgroup_mixture.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png


def safe_corr(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    return float(pearsonr(x[mask], y[mask]).statistic), float(spearmanr(x[mask], y[mask]).statistic)


def write_report(
    rows: pd.DataFrame,
    mixture_bounds: pd.DataFrame,
    mixture_grid: pd.DataFrame,
    component_curves: pd.DataFrame,
    hmd: pd.DataFrame,
    plot_paths: list[Path],
    current_metadata: dict[str, object],
) -> None:
    distribution_rows = rows[rows["analysis"] == "distribution_family"].copy()
    uncensored_distribution_rows = distribution_rows[~distribution_rows["top_0_01pct_censored"].astype(bool)].copy()
    pearson, spearman = safe_corr(
        uncensored_distribution_rows["q9999_tau_factor"],
        uncensored_distribution_rows["top_0_01pct_lifespan"],
    )
    censored_distribution_count = int(distribution_rows["top_0_01pct_censored"].astype(bool).sum())
    baseline = rows[rows["analysis"] == "baseline"].iloc[0]

    eta_gaussian_5 = first_row(
        distribution_rows,
        (distribution_rows["mode"] == "eta_only")
        & (distribution_rows["distribution"] == "gaussian")
        & np.isclose(distribution_rows["cv_requested"], 0.05),
    )
    beta_gaussian_5 = first_row(
        distribution_rows,
        (distribution_rows["mode"] == "beta_only")
        & (distribution_rows["distribution"] == "gaussian")
        & np.isclose(distribution_rows["cv_requested"], 0.05),
    )
    eta_gaussian_10 = first_row(
        distribution_rows,
        (distribution_rows["mode"] == "eta_only")
        & (distribution_rows["distribution"] == "gaussian")
        & np.isclose(distribution_rows["cv_requested"], 0.10),
    )
    trunc_20cv_2cut = first_row(
        rows,
        (rows["analysis"] == "truncated_eta")
        & np.isclose(rows["cv_requested"], 0.20)
        & np.isclose(rows["favorable_cutoff"], 0.02),
    )
    trunc_20cv_20cut = first_row(
        rows,
        (rows["analysis"] == "truncated_eta")
        & np.isclose(rows["cv_requested"], 0.20)
        & np.isclose(rows["favorable_cutoff"], 0.20),
    )

    bound_summary = (
        mixture_bounds.groupby("delta", as_index=False)
        .agg(
            pmax_mixture_vs_hmd=("pmax_mixture_vs_hmd", "min"),
            pmax_conservative_component_only=("pmax_conservative_component_only", "min"),
        )
        .sort_values("delta")
    )

    lines = [
        "# Senogenic Tail Constraint Exploration",
        "",
        "This exploration tests the reviewer-facing criticism that the current heterogeneity result is not distribution-free. The working variable is the senogenic timescale",
        "",
        r"$$",
        r"\tau_{\rm sen}=\frac{\beta}{\eta}.",
        r"$$",
        "",
        "The goal is not to edit the paper here. The goal is to see whether the simulations support a more precise claim: smooth senogenic heterogeneity is dangerous because it creates a favorable tail in \\(\\beta/\\eta\\), while broad central variation can be hidden only if that favorable tail is truncated, depleted, or compensated.",
        "",
        "## Assumptions",
        "",
        f"- Baseline: Sweden 2019 tail-emphasis fit from `{BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)}`.",
        "- Heterogeneity is isolated in senogenic parameters; \(X_c\), \(\epsilon\), and \(h_{\\rm ext}\) are fixed unless noted.",
        f"- Main scenario simulations use \(n={int(current_metadata['n']):,}\), \(t_{{max}}={TMAX:g}\), \(\Delta t={DT:g}\), and \(h_{{\\rm ext}}=0\).",
        f"- Rare-subgroup component curves use \(n={int(current_metadata['mixture_component_n']):,}\).",
        f"- The top-tail metric is the age at which unconditional model survival reaches \(10^{{-4}}\), called top 0.01% lifespan below. Rows that still have more than \(10^{{-4}}\) survival at \(t_{{max}}\) are marked as censored lower bounds.",
        f"- HMD comparisons use Sweden 2019 period survival conditional on age {CONDITION_AGE}; available local HMD tail points run through age {int(hmd['age'].max())}.",
        "- This is an exploration, not a final SI figure. Monte Carlo noise is non-negligible at \(10^{-4}\), so the qualitative collapse and boundaries matter more than the last decimal.",
        "",
        "## Result 1: distribution shape mostly enters through the favorable \\(\\beta/\\eta\\) quantile",
        "",
        f"Across uncensored Gaussian, lognormal, uniform, and Student-\(t_4\) one-parameter senogenic scenarios, the correlation between \(q_{{0.9999}}[(\\beta/\\eta)/(\\beta_0/\\eta_0)]\) and top 0.01% lifespan was Pearson \(r={pearson:.3f}\) and Spearman \(\\rho={spearman:.3f}\).",
        f"{censored_distribution_count} heavy-tailed scenario(s) reached \(t_{{max}}={TMAX:g}\) and are plotted as lower bounds.",
        f"The baseline top 0.01% lifespan was {baseline['top_0_01pct_lifespan']:.1f} years.",
        f"With 5% Gaussian heterogeneity, \\(\\eta\\) gave {eta_gaussian_5['top_0_01pct_lifespan']:.1f} years and \\(\\beta\\) gave {beta_gaussian_5['top_0_01pct_lifespan']:.1f} years.",
        f"With 10% Gaussian \\(\\eta\\) heterogeneity, the top-tail age rose to {eta_gaussian_10['top_0_01pct_lifespan']:.1f} years.",
        "",
        rel_plot(plot_paths[0]),
        "",
        "Interpretation: the previous Gaussian/lognormal statement is not distribution-free, but the simulations do support the cleaner statement that the extreme survival tail is governed by the favorable tail of \\(\\tau_{\\rm sen}=\\beta/\\eta\\).",
        "",
        "## Result 2: truncating the favorable \\(\\eta\\) tail can rescue broad central variation",
        "",
        "Here \\(\\eta\\sim N(\\eta_0,\\sigma)\\) is resampled until it is positive and above a lower cutoff \\(\\eta_0(1-\\delta_{\\max})\\). Small \\(\\delta_{\\max}\\) means the favorable low-\\(\\eta\\) tail is strongly depleted.",
        f"For requested 20% Gaussian \\(\\eta\\) CV with only a 2% favorable cutoff, the actual post-truncation \\(\\eta\\) CV was {trunc_20cv_2cut['eta_cv_actual']:.3f} and top 0.01% lifespan was {trunc_20cv_2cut['top_0_01pct_lifespan']:.1f} years.",
        f"Allowing the favorable cutoff to extend to 20% at the same requested CV gave actual \\(\\eta\\) CV {trunc_20cv_20cut['eta_cv_actual']:.3f} and top 0.01% lifespan {trunc_20cv_20cut['top_0_01pct_lifespan']:.1f} years.",
        "",
        rel_plot(plot_paths[1]),
        "",
        "Interpretation: a distribution with broad apparent central variation can evade the Gaussian-tail criticism, but only by explicitly removing the low-\\(\\eta\\), high-\\(\\beta/\\eta\\) individuals. That is the reviewer point, turned into a measurable condition.",
        "",
        "## Result 3: correlated \\(\\eta,\\beta\\) variation is allowed when it preserves the ratio",
        "",
        "For bivariate lognormal variation, the relevant variance is approximately",
        "",
        r"$$",
        r"{\rm Var}[\log(\beta/\eta)]={\rm Var}[\log\beta]+{\rm Var}[\log\eta]-2\rho\sigma_{\log\beta}\sigma_{\log\eta}.",
        r"$$",
        "",
        "Positive correlation narrows \\(\\beta/\\eta\\), so the same marginal CV in \\(\\eta\\) and \\(\\beta\\) produces a much weaker extreme-tail effect.",
        "",
        rel_plot(plot_paths[2]),
        "",
        "Interpretation: the constrained object is not arbitrary variation in \\(\\eta\\) or \\(\\beta\\) separately. It is variation in the senogenic direction that changes \\(\\beta/\\eta\\).",
        "",
        "## Result 4: rare favorable subgroups are tightly bounded by the observed tail",
        "",
        "This analysis simulates fixed favorable subgroups with \\(\\eta=\\eta_0(1-\\delta)\\). Mixtures are then computed analytically as",
        "",
        r"$$",
        r"S_{\rm mix}(t)=(1-p)S_0(t)+pS_\delta(t).",
        r"$$",
        "",
        "Two bounds are reported. The full-mixture bound asks when \(S_{\\rm mix}\) exceeds Sweden 2019 HMD conditional survival at ages 100, 105, or 110. The component-only bound is an interpretable back-of-envelope check: it ignores baseline survivors and asks only that \(pS_\\delta(t)\\le S_{\\rm HMD}(t)\). Values above one mean the chosen ages do not bound the subgroup even at \(p=1\), and are shown as not bounded.",
        "",
        "| Lower eta in subgroup | Full-mixture \(p_{\\max}\) | Component-only \(p_{\\max}\) |",
        "|---:|---:|---:|",
    ]

    for _, row in bound_summary.iterrows():
        full_value = row["pmax_mixture_vs_hmd"]
        comp_value = row["pmax_conservative_component_only"]
        lines.append(
            f"| {100 * row['delta']:.0f}% | {format_frequency(full_value)} | {format_frequency(comp_value)} |"
        )

    lines.extend(
        [
            "",
            rel_plot(plot_paths[3]),
            "",
            "Interpretation: rare favorable senogenic subgroups are the distribution-free version of the problem. If such a subgroup is common enough, it inflates the observed old-age survival tail regardless of whether the rest of the distribution is Gaussian.",
            "",
            "## Bottom line",
            "",
            "The analyses support a sharper claim than \"senogenic parameters cannot vary.\" The defensible claim is: human late-life survival constrains the favorable tail of the senogenic timescale \\(\\beta/\\eta\\). Smooth, polygenic-like senogenic variation violates this quickly; broader central variation is possible only with tail truncation, tail depletion, or correlated compensation that preserves \\(\\beta/\\eta\\).",
            "",
            "## Outputs",
            "",
            f"- Distribution source rows: `{DISTRIBUTION_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Truncation source rows: `{TRUNCATION_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Correlation source rows: `{CORRELATION_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Mixture bounds: `{MIXTURE_BOUNDS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Mixture grid: `{MIXTURE_GRID_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Scenario cache: `{SCENARIO_CACHE_PATH.relative_to(PROJECT_ROOT)}`.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n")


def first_row(frame: pd.DataFrame, mask: pd.Series) -> pd.Series:
    subset = frame[mask]
    if subset.empty:
        raise ValueError("Expected row is missing.")
    return subset.iloc[0]


def rel_plot(path: Path) -> str:
    rel_path = Path(os.path.relpath(path, EXPLORATION_DIR))
    return f"![{path.stem}]({rel_path.as_posix()})"


def format_frequency(value: float) -> str:
    if not np.isfinite(value):
        return "unbounded"
    if value <= 0:
        return "0"
    if value >= 1:
        return "not bounded"
    if value < 1e-3:
        return f"{value:.1e}"
    return f"{100 * value:.3g}%"


def main() -> None:
    args = parse_args()
    parallel = not args.no_parallel
    current_metadata = metadata(args.n, args.mixture_component_n, parallel)

    rows = run_missing_scenarios(
        n=args.n,
        parallel=parallel,
        force=args.force,
        plots_only=args.plots_only,
        current_metadata=current_metadata,
    )
    component_curves = run_mixture_components(
        mixture_component_n=args.mixture_component_n,
        parallel=parallel,
        force=args.force,
        plots_only=args.plots_only,
        current_metadata=current_metadata,
    )
    hmd = load_hmd_conditional_survival()
    mixture_bounds, mixture_grid = compute_mixture_bounds(component_curves, hmd)
    export_csvs(rows, mixture_bounds, mixture_grid)

    plot_paths = [
        plot_tail_collapse(rows),
        plot_truncation_heatmap(rows),
        plot_correlated_eta_beta(rows),
        plot_mixture_bounds(mixture_bounds, mixture_grid, component_curves, hmd),
    ]
    write_report(rows, mixture_bounds, mixture_grid, component_curves, hmd, plot_paths, current_metadata)

    print(f"Saved report: {REPORT_PATH}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
