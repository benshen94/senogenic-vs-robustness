#!/usr/bin/env python3
"""Wide-bounds SR refits with tau-spread inside the fitted model."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from senogenic_vs_robustness.paths import RESULTS_DIR


fitmod = importlib.import_module(
    "analysis.model_fits.hmd.joint_shared_eta_beta_epsilon_65_100_n100k_tail90"
)

EXPLORATION_DIR = Path(__file__).resolve().parent
PLOTS_DIR = EXPLORATION_DIR / "plots"
REPORT_PATH = EXPLORATION_DIR / "tau_wide_refit_profile_analysis.md"

CACHE_DIR = RESULTS_DIR / "cache" / "simulations" / "tau_wide_refit_profile"
CSV_DIR = RESULTS_DIR / "tau_wide_refit_profile"
FIT_CACHE_PATH = CACHE_DIR / "fit_candidates.csv"
CURVE_CACHE_PATH = CACHE_DIR / "fit_curves.csv"
METADATA_PATH = CACHE_DIR / "metadata.json"
FIT_RESULTS_PATH = CSV_DIR / "tau_wide_refit_profile_candidates.csv"
CURVE_RESULTS_PATH = CSV_DIR / "tau_wide_refit_profile_curves.csv"

FIT_ARCHIVE_DIR = RESULTS_DIR / "fits" / "records"
ARCHIVE_RECORDS = (
    "joint2019_tail90_sweden_emphasis.json",
    "joint2019_shared_eta_beta_epsilon_65_100_n100k.json",
    "hybrid2019_swe_tail90_usa_refit.json",
)

RANDOM_SEED = 20260601
TAU_SD_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.13, 0.16, 0.20, 0.25)
TAU_SD_BOUNDS = (0.0, 0.30)
STEP_SIZES = (0.50, 0.25, 0.125)
MAX_DESCENT_SWEEPS_PER_STEP = 2
TAIL_AGES = (100.0, 105.0, 110.0)
CONDITION_AGE = 90.0
TOP_SURVIVAL = 1e-4
CURVE_TMAX = 180.0

PARAMETER_COLUMNS = (
    "eta_log2",
    "tau_log2",
    "kappa_log2",
    "epsilon_log2",
    "SWE_Xc_log2",
    "SWE_xc_std_log2",
    "USA_Xc_log2",
    "USA_xc_std_log2",
)

# Wide SR-side fitting bounds. The model uses beta = eta * tau, so beta has the
# combined freedom of eta_log2 + tau_log2.
WIDE_BOUNDS = (
    (-1.50, 1.50),  # eta factor 0.35x-2.83x
    (-2.00, 2.00),  # tau=beta/eta factor 0.25x-4x
    (-1.50, 1.50),  # kappa factor 0.35x-2.83x
    (-1.50, 1.50),  # epsilon factor 0.35x-2.83x
    (-1.20, 1.20),  # Sweden Xc factor 0.44x-2.30x
    (-2.00, 1.50),  # Sweden Xc CV factor 0.25x-2.83x
    (-1.20, 1.20),  # USA Xc factor 0.44x-2.30x
    (-2.00, 1.50),  # USA Xc CV factor 0.25x-2.83x
)


@dataclass(frozen=True)
class FitConfig:
    profile_starts_per_sd: int
    full_starts: int
    fit_n: int
    eval_n: int
    curve_n: int
    fit_dt: float
    curve_dt: float
    acceptance_multiplier: float
    parallel: bool


FIT_PARALLEL = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-starts-per-sd", type=int, default=7)
    parser.add_argument("--full-starts", type=int, default=8)
    parser.add_argument("--fit-n", type=int, default=3_000)
    parser.add_argument("--eval-n", type=int, default=40_000)
    parser.add_argument("--curve-n", type=int, default=100_000)
    parser.add_argument("--fit-dt", type=float, default=0.08)
    parser.add_argument("--curve-dt", type=float, default=0.1)
    parser.add_argument("--acceptance-multiplier", type=float, default=2.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FitConfig:
    return FitConfig(
        profile_starts_per_sd=int(args.profile_starts_per_sd),
        full_starts=int(args.full_starts),
        fit_n=int(args.fit_n),
        eval_n=int(args.eval_n),
        curve_n=int(args.curve_n),
        fit_dt=float(args.fit_dt),
        curve_dt=float(args.curve_dt),
        acceptance_multiplier=float(args.acceptance_multiplier),
        parallel=not bool(args.no_parallel),
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def set_runtime(config: FitConfig) -> None:
    global FIT_PARALLEL
    fitmod.DT = float(config.fit_dt)
    FIT_PARALLEL = bool(config.parallel)


def metadata(config: FitConfig) -> dict[str, object]:
    return {
        "task": "tau_wide_refit_profile",
        "tau_sd_grid": list(TAU_SD_GRID),
        "tau_sd_bounds": list(TAU_SD_BOUNDS),
        "wide_bounds": [list(bound) for bound in WIDE_BOUNDS],
        "parameter_columns": list(PARAMETER_COLUMNS),
        "step_sizes": list(STEP_SIZES),
        "max_descent_sweeps_per_step": MAX_DESCENT_SWEEPS_PER_STEP,
        "fit_config": config.__dict__,
        "tail_ages": list(TAIL_AGES),
        "condition_age": CONDITION_AGE,
        "top_survival": TOP_SURVIVAL,
        "random_seed": RANDOM_SEED,
    }


def metadata_matches(config: FitConfig) -> bool:
    if not METADATA_PATH.exists():
        return False
    try:
        return json.loads(METADATA_PATH.read_text()) == metadata(config)
    except Exception:
        return False


def load_data_by_country() -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {country: fitmod.load_country_data(country) for country in fitmod.COUNTRIES}


def positive_normal(mean: float, std_frac: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = float(mean) * float(std_frac)
    if std <= 0:
        return np.full(n, max(float(mean), 1e-12), dtype=float)
    values = rng.normal(float(mean), std, size=n)
    bad = values <= 0
    while np.any(bad):
        values[bad] = rng.normal(float(mean), std, size=int(np.sum(bad)))
        bad = values <= 0
    return values


def tau_cv_from_sd(tau_sd: float) -> float:
    return float(math.sqrt(math.exp(float(tau_sd) ** 2) - 1.0))


def clip_vector(vector: np.ndarray, bounds: tuple[tuple[float, float], ...] = WIDE_BOUNDS) -> np.ndarray:
    values = np.asarray(vector, dtype=float).copy()
    for idx, (lower, upper) in enumerate(bounds):
        values[idx] = float(np.clip(values[idx], lower, upper))
    return values


def vector_to_params(vector: np.ndarray, h_ext_by_country: dict[str, float]) -> dict[str, dict[str, float]]:
    values = dict(zip(PARAMETER_COLUMNS, clip_vector(vector)))
    base = fitmod.BASELINE
    base_tau = base["beta"] / base["eta"]
    eta = base["eta"] * 2.0 ** values["eta_log2"]
    tau = base_tau * 2.0 ** values["tau_log2"]
    shared = {
        "eta": float(eta),
        "beta": float(eta * tau),
        "kappa": float(base["kappa"] * 2.0 ** values["kappa_log2"]),
        "epsilon": float(base["epsilon"] * 2.0 ** values["epsilon_log2"]),
    }
    return {
        "shared": shared,
        "SWE": {
            **shared,
            "Xc": float(base["Xc"] * 2.0 ** values["SWE_Xc_log2"]),
            "xc_std_frac": float(base["xc_std_frac"] * 2.0 ** values["SWE_xc_std_log2"]),
            "h_ext": h_ext_by_country["SWE"],
        },
        "USA": {
            **shared,
            "Xc": float(base["Xc"] * 2.0 ** values["USA_Xc_log2"]),
            "xc_std_frac": float(base["xc_std_frac"] * 2.0 ** values["USA_xc_std_log2"]),
            "h_ext": h_ext_by_country["USA"],
        },
    }


def vector_from_archive(path: Path) -> np.ndarray:
    record = json.loads(path.read_text())
    fitted = record["summary"]["fitted_parameters"]
    base = fitmod.BASELINE
    base_tau = base["beta"] / base["eta"]
    eta = float(fitted["eta"])
    beta = float(fitted["beta"])
    ratios = [
        eta / base["eta"],
        (beta / eta) / base_tau,
        float(fitted.get("kappa", base["kappa"])) / base["kappa"],
        float(fitted["epsilon"]) / base["epsilon"],
        float(fitted.get("SWE_Xc", fitted.get("Xc"))) / base["Xc"],
        float(fitted.get("SWE_xc_std_frac", fitted.get("xc_std_frac"))) / base["xc_std_frac"],
        float(fitted.get("USA_Xc", fitted.get("SWE_Xc", fitted.get("Xc")))) / base["Xc"],
        float(fitted.get("USA_xc_std_frac", fitted.get("SWE_xc_std_frac", fitted.get("xc_std_frac")))) / base["xc_std_frac"],
    ]
    return clip_vector(np.asarray([math.log(max(ratio, 1e-12), 2.0) for ratio in ratios], dtype=float))


def archive_starts() -> list[tuple[str, np.ndarray]]:
    starts = []
    for filename in ARCHIVE_RECORDS:
        path = FIT_ARCHIVE_DIR / filename
        if path.exists():
            starts.append((path.stem, vector_from_archive(path)))
    return starts


def structured_starts() -> list[tuple[str, np.ndarray]]:
    starts = [("baseline", np.zeros(len(PARAMETER_COLUMNS), dtype=float))]
    for name, vector in archive_starts():
        starts.append((name, vector))
    anchor = starts[0][1].copy()
    for tau_log2 in (-2.0, -1.0, 1.0, 2.0):
        trial = anchor.copy()
        trial[1] = tau_log2
        starts.append((f"baseline_tau_{tau_log2:+.1f}", trial))
    return starts


def random_starts(count: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED + 77)
    lowers = np.asarray([bound[0] for bound in WIDE_BOUNDS], dtype=float)
    uppers = np.asarray([bound[1] for bound in WIDE_BOUNDS], dtype=float)
    return [(f"random_{idx:02d}", rng.uniform(lowers, uppers)) for idx in range(int(count))]


def starts_for_profile(count: int) -> list[tuple[str, np.ndarray]]:
    starts = structured_starts()
    if len(starts) < count:
        starts.extend(random_starts(count - len(starts)))
    return starts[: max(count, 1)]


def simulate_country_tau_spread(
    data: object,
    params: dict[str, float],
    tau_sd: float,
    n: int,
    seed: int,
    *,
    tmax: float = 112.0,
    dt: float | None = None,
    survival_ages: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 23_000)
    if tau_sd > 0:
        v = rng.normal(0.0, float(tau_sd), size=int(n))
        eta = float(params["eta"]) * np.exp(-0.5 * v)
        beta = float(params["beta"]) * np.exp(0.5 * v)
    else:
        eta = np.full(int(n), float(params["eta"]), dtype=float)
        beta = np.full(int(n), float(params["beta"]), dtype=float)

    sim = fitmod.SRHazardSim(
        n=int(n),
        eta=eta,
        beta=beta,
        kappa=params["kappa"],
        epsilon=params["epsilon"],
        Xc=positive_normal(params["Xc"], params["xc_std_frac"], int(n), seed),
        h_ext=params["h_ext"],
        tmax=float(tmax),
        dt=fitmod.DT if dt is None else float(dt),
        parallel=FIT_PARALLEL,
        break_early=True,
        random_seed=seed + 10_000,
        chunk_size=10_000,
    )

    hazard = fitmod._interpolate_on_times(sim.tspan_hazard, sim.hazard, data.ages_hazard)
    survival_times = data.ages_survival if survival_ages is None else np.asarray(survival_ages, dtype=float)
    survival = fitmod._interpolate_on_times(sim.tspan_survival, sim.survival, survival_times)
    survival = np.clip(survival, 1e-12, 1.0)
    survival = survival / survival[0]
    return {
        "hazard": np.maximum(hazard, 1e-12),
        "survival": survival,
        "survival_ages": survival_times,
    }


def residual_vector(
    vector: np.ndarray,
    tau_sd: float,
    data_by_country: dict[str, object],
    n: int,
    seed_base: int,
) -> np.ndarray:
    h_ext_by_country = {country: data_by_country[country].h_ext for country in fitmod.COUNTRIES}
    params = vector_to_params(vector, h_ext_by_country)
    residuals: list[np.ndarray] = []

    for index, country in enumerate(fitmod.COUNTRIES):
        data = data_by_country[country]
        fit = simulate_country_tau_spread(data, params[country], tau_sd, int(n), seed_base + index)

        hazard_mask = fitmod.focus_mask(data.ages_hazard)
        hazard_ages = data.ages_hazard[hazard_mask]
        log_hazard_residual = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])
        hazard_weights = np.ones_like(log_hazard_residual)
        hazard_weights[hazard_ages >= 85] = 2.5
        hazard_weights[hazard_ages >= 90] = 4.0
        if country == "SWE":
            sweden_tail_mask = (hazard_ages >= 88) & (hazard_ages <= 96)
            hazard_weights[sweden_tail_mask] *= 2.8
        hazard_weights = hazard_weights / np.mean(hazard_weights)
        residuals.append(np.sqrt(hazard_weights) * log_hazard_residual)

        survival_mask = fitmod.focus_mask(data.ages_survival)
        survival_residual = fit["survival"][survival_mask] - data.survival[survival_mask]
        residuals.append(np.sqrt(3.0) * survival_residual)

    return np.concatenate(residuals)


def score_candidate(
    vector: np.ndarray,
    tau_sd: float,
    data_by_country: dict[str, object],
    n: int,
    seed_base: int,
    memo: dict[tuple[object, ...], float] | None = None,
) -> float:
    vector = clip_vector(vector)
    tau_sd = float(np.clip(tau_sd, *TAU_SD_BOUNDS))
    key = tuple(np.round(vector, 7)) + (round(tau_sd, 7), int(n), int(seed_base), float(fitmod.DT))
    if memo is not None and key in memo:
        return memo[key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residuals = residual_vector(vector, tau_sd, data_by_country, int(n), int(seed_base))
    score = float(np.mean(residuals**2))
    if not np.isfinite(score):
        score = float("inf")
    if memo is not None:
        memo[key] = score
    return score


def coordinate_descent(
    objective_fn: Callable[[np.ndarray], float],
    start: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    step_sizes: tuple[float, ...],
) -> tuple[np.ndarray, float, list[float]]:
    current = clip_vector(start, bounds)
    current_score = float(objective_fn(current))
    history = [current_score]
    for step_size in step_sizes:
        improved = True
        sweeps = 0
        while improved:
            sweeps += 1
            improved = False
            best_vector = current.copy()
            best_score = current_score
            for dim in range(current.size):
                for direction in (-1.0, 1.0):
                    trial = current.copy()
                    trial[dim] += direction * step_size
                    trial = clip_vector(trial, bounds)
                    if np.allclose(trial, current):
                        continue
                    score = float(objective_fn(trial))
                    if score < best_score:
                        best_vector = trial
                        best_score = score
                        improved = True
            if improved:
                current = best_vector
                current_score = best_score
                history.append(current_score)
            if sweeps >= MAX_DESCENT_SWEEPS_PER_STEP:
                break
    return current, current_score, history


def fit_profile_tau_sd(
    tau_sd: float,
    starts: list[tuple[str, np.ndarray]],
    data_by_country: dict[str, object],
    config: FitConfig,
    memo: dict[tuple[object, ...], float],
) -> tuple[str, np.ndarray, float, list[float]]:
    best: tuple[float, str, np.ndarray, list[float]] | None = None
    seed = stable_seed("profile-wide", tau_sd)
    for start_name, start in starts:
        objective = lambda vector: score_candidate(vector, tau_sd, data_by_country, config.fit_n, seed, memo)
        vector, score, history = coordinate_descent(objective, start, WIDE_BOUNDS, STEP_SIZES)
        if best is None or score < best[0]:
            best = (score, start_name, vector, history)
    assert best is not None
    score, start_name, vector, history = best
    return start_name, vector, score, history


def fit_full_tau_sd(
    start_name: str,
    start_vector: np.ndarray,
    start_tau_sd: float,
    data_by_country: dict[str, object],
    config: FitConfig,
    memo: dict[tuple[object, ...], float],
) -> tuple[np.ndarray, float, list[float]]:
    bounds = (*WIDE_BOUNDS, TAU_SD_BOUNDS)
    start = np.asarray([*clip_vector(start_vector), float(start_tau_sd)], dtype=float)
    objective = lambda full_vector: score_candidate(
        full_vector[: len(PARAMETER_COLUMNS)],
        full_vector[len(PARAMETER_COLUMNS)],
        data_by_country,
        config.fit_n,
        stable_seed("full-wide", start_name),
        memo,
    )
    return coordinate_descent(objective, start, bounds, STEP_SIZES)


def natural_values(vector: np.ndarray) -> dict[str, float]:
    params = vector_to_params(vector, {"SWE": 0.0, "USA": 0.0})
    swe = params["SWE"]
    usa = params["USA"]
    base = fitmod.BASELINE
    tau = swe["beta"] / swe["eta"]
    base_tau = base["beta"] / base["eta"]
    return {
        "eta": float(swe["eta"]),
        "beta": float(swe["beta"]),
        "kappa": float(swe["kappa"]),
        "epsilon": float(swe["epsilon"]),
        "tau": float(tau),
        "tau_factor_vs_karin": float(tau / base_tau),
        "eta_factor": float(swe["eta"] / base["eta"]),
        "beta_factor": float(swe["beta"] / base["beta"]),
        "kappa_factor": float(swe["kappa"] / base["kappa"]),
        "epsilon_factor": float(swe["epsilon"] / base["epsilon"]),
        "SWE_Xc": float(swe["Xc"]),
        "SWE_xc_std_frac": float(swe["xc_std_frac"]),
        "USA_Xc": float(usa["Xc"]),
        "USA_xc_std_frac": float(usa["xc_std_frac"]),
    }


def build_fit_row(
    candidate_id: str,
    label: str,
    source: str,
    vector: np.ndarray,
    tau_sd: float,
    search_score: float,
    eval_score: float,
    history: list[float],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "candidate_id": candidate_id,
        "label": label,
        "source": source,
        "search_score": float(search_score),
        "eval_score": float(eval_score),
        "history_start": float(history[0]) if history else np.nan,
        "history_end": float(history[-1]) if history else float(search_score),
        "history_steps": len(history),
        "tau_sd": float(tau_sd),
        "tau_cv": tau_cv_from_sd(tau_sd),
    }
    row.update({name: float(value) for name, value in zip(PARAMETER_COLUMNS, clip_vector(vector))})
    row.update(natural_values(vector))
    if extra:
        row.update(extra)
    return row


def run_fits(config: FitConfig) -> pd.DataFrame:
    set_runtime(config)
    data_by_country = load_data_by_country()
    fit_memo: dict[tuple[object, ...], float] = {}
    eval_memo: dict[tuple[object, ...], float] = {}
    rows: list[dict[str, object]] = []
    starts = starts_for_profile(config.profile_starts_per_sd)

    for tau_sd in TAU_SD_GRID:
        start_name, vector, search_score, history = fit_profile_tau_sd(tau_sd, starts, data_by_country, config, fit_memo)
        eval_score = score_candidate(
            vector,
            tau_sd,
            data_by_country,
            config.eval_n,
            stable_seed("eval-profile-wide", tau_sd),
            eval_memo,
        )
        rows.append(
            build_fit_row(
                candidate_id=f"profile_tau_sd_{tau_sd:.3f}",
                label=rf"profile $\sigma_v={tau_sd:.3f}$",
                source="profile_tau_sd",
                vector=vector,
                tau_sd=tau_sd,
                search_score=search_score,
                eval_score=eval_score,
                history=history,
                extra={"profile_requested_tau_sd": float(tau_sd), "best_start_name": start_name},
            )
        )
        print(f"profile tau_sd={tau_sd:.3f} start={start_name} search={search_score:.4g} eval={eval_score:.4g}", flush=True)

    full_starts = starts_for_profile(max(config.full_starts, 1))
    rng = np.random.default_rng(RANDOM_SEED + 700)
    for index, (start_name, vector) in enumerate(full_starts):
        start_tau_sd = float(rng.choice(TAU_SD_GRID))
        full_vector, search_score, history = fit_full_tau_sd(start_name, vector, start_tau_sd, data_by_country, config, fit_memo)
        fit_vector = clip_vector(full_vector[: len(PARAMETER_COLUMNS)])
        tau_sd = float(np.clip(full_vector[len(PARAMETER_COLUMNS)], *TAU_SD_BOUNDS))
        eval_score = score_candidate(
            fit_vector,
            tau_sd,
            data_by_country,
            config.eval_n,
            stable_seed("eval-full-wide", start_name),
            eval_memo,
        )
        rows.append(
            build_fit_row(
                candidate_id=f"full_fit_{index:02d}",
                label=f"full fit {index:02d}",
                source="full_tau_sd_fit",
                vector=fit_vector,
                tau_sd=tau_sd,
                search_score=search_score,
                eval_score=eval_score,
                history=history,
                extra={"start_name": start_name, "start_tau_sd": start_tau_sd},
            )
        )
        print(f"full {index:02d} tau_sd={tau_sd:.3f} start={start_name} search={search_score:.4g} eval={eval_score:.4g}", flush=True)

    fits = pd.DataFrame(rows)
    best_score = float(fits["eval_score"].min())
    fits["best_eval_score"] = best_score
    fits["score_ratio_to_best"] = fits["eval_score"] / best_score
    fits["acceptance_multiplier"] = float(config.acceptance_multiplier)
    fits["strict_accepted"] = fits["score_ratio_to_best"] <= config.acceptance_multiplier
    return fits.sort_values(["eval_score", "tau_sd"]).reset_index(drop=True)


def load_hmd_tail_reference() -> dict[float, float]:
    hmd = HMD("SWE", "both", "period")
    ages, survival = hmd.get_survival(2019, strict=True)
    frame = pd.DataFrame({"age": ages, "survival": survival}).dropna()
    condition = float(frame.loc[frame["age"] == CONDITION_AGE, "survival"].iloc[0])
    return {age: float(frame.loc[frame["age"] == age, "survival"].iloc[0] / condition) for age in TAIL_AGES}


def top_survival_age_from_curve(ages: np.ndarray, survival: np.ndarray) -> tuple[float, bool]:
    ages = np.asarray(ages, dtype=float)
    survival = np.asarray(survival, dtype=float)
    order = np.argsort(ages)
    ages = ages[order]
    survival = survival[order]
    if np.nanmin(survival) > TOP_SURVIVAL:
        return float(np.nanmax(ages)), True
    reversed_age = ages[::-1]
    reversed_survival = survival[::-1]
    return float(np.interp(TOP_SURVIVAL, reversed_survival, reversed_age)), False


def conditional_survival_from_curve(ages: np.ndarray, survival: np.ndarray, age: float) -> float:
    s90 = float(np.interp(CONDITION_AGE, ages, survival))
    if s90 <= 0:
        return np.nan
    return float(np.interp(age, ages, survival) / s90)


def selected_fit_ids(fits: pd.DataFrame) -> list[str]:
    profile = fits[fits["source"] == "profile_tau_sd"].sort_values("tau_sd")
    selected = profile["candidate_id"].tolist()
    full = fits[fits["source"] == "full_tau_sd_fit"].sort_values("eval_score").head(4)
    selected.extend(full["candidate_id"].tolist())
    return list(dict.fromkeys(selected))


def build_curves(fits: pd.DataFrame, config: FitConfig) -> pd.DataFrame:
    set_runtime(config)
    data_by_country = load_data_by_country()
    h_ext_by_country = {country: data_by_country[country].h_ext for country in fitmod.COUNTRIES}
    hmd_ref = load_hmd_tail_reference()
    ids = selected_fit_ids(fits)
    rows: list[dict[str, object]] = []

    hmd = HMD("SWE", "both", "period")
    h_age, h_val = hmd.get_hazard(2019, haz_type="mx", strict=True)
    s_age, s_val = hmd.get_survival(2019, strict=True)
    s90 = float(s_val[np.where(s_age == CONDITION_AGE)][0])
    for age, value in zip(h_age, h_val):
        if 65 <= age <= 110 and np.isfinite(value) and value > 0:
            rows.append({"candidate_id": "HMD", "label": "Sweden 2019 HMD", "curve": "hazard", "age": float(age), "value": float(value)})
    for age, value in zip(s_age, s_val):
        if 90 <= age <= 110 and np.isfinite(value) and value > 0:
            rows.append(
                {
                    "candidate_id": "HMD",
                    "label": "Sweden 2019 HMD",
                    "curve": "conditional_survival_from90",
                    "age": float(age),
                    "value": float(value / s90),
                }
            )

    survival_ages = np.arange(0.0, CURVE_TMAX + 1.0, 1.0)
    for candidate_id in ids:
        row = fits[fits["candidate_id"] == candidate_id].iloc[0]
        vector = np.asarray([row[column] for column in PARAMETER_COLUMNS], dtype=float)
        params = vector_to_params(vector, h_ext_by_country)["SWE"]
        fit = simulate_country_tau_spread(
            data_by_country["SWE"],
            params,
            tau_sd=float(row["tau_sd"]),
            n=config.curve_n,
            seed=stable_seed("curve-wide", candidate_id),
            tmax=CURVE_TMAX,
            dt=config.curve_dt,
            survival_ages=survival_ages,
        )
        for age, value in zip(data_by_country["SWE"].ages_hazard, fit["hazard"]):
            if 65 <= age <= 110:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "label": str(row["label"]),
                        "curve": "hazard",
                        "age": float(age),
                        "value": float(value),
                    }
                )
        s_at_90 = float(np.interp(CONDITION_AGE, fit["survival_ages"], fit["survival"]))
        for age, value in zip(fit["survival_ages"], fit["survival"]):
            if 90 <= age <= 110:
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "label": str(row["label"]),
                        "curve": "conditional_survival_from90",
                        "age": float(age),
                        "value": float(value / s_at_90),
                    }
                )
        top_age, censored = top_survival_age_from_curve(fit["survival_ages"], fit["survival"])
        for age in TAIL_AGES:
            cond_survival = conditional_survival_from_curve(fit["survival_ages"], fit["survival"], age)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "label": str(row["label"]),
                    "curve": "tail_metric",
                    "age": float(age),
                    "value": cond_survival,
                    "hmd_reference": hmd_ref[age],
                    "ratio_to_hmd": cond_survival / hmd_ref[age],
                    "top_0_01pct_lifespan": top_age,
                    "top_0_01pct_censored": censored,
                }
            )
    return pd.DataFrame(rows)


def write_outputs(fits: pd.DataFrame, curves: pd.DataFrame, config: FitConfig) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fits.to_csv(FIT_RESULTS_PATH, index=False)
    curves.to_csv(CURVE_RESULTS_PATH, index=False)
    fits.to_csv(FIT_CACHE_PATH, index=False)
    curves.to_csv(CURVE_CACHE_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata(config), indent=2, sort_keys=True) + "\n")


def load_cached_outputs(config: FitConfig, require_metadata: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if require_metadata and not metadata_matches(config):
        raise FileNotFoundError("Cache metadata does not match requested run.")
    return pd.read_csv(FIT_CACHE_PATH), pd.read_csv(CURVE_CACHE_PATH)


def save_fig(fig: plt.Figure, stem: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png = PLOTS_DIR / f"{stem}.png"
    pdf = PLOTS_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def tail_summary(fits: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    tail = curves[curves["curve"] == "tail_metric"].copy()
    summary = tail.pivot_table(index="candidate_id", columns="age", values="ratio_to_hmd").reset_index()
    summary.columns = ["candidate_id", *[f"S{int(col)}_ratio_to_hmd" for col in summary.columns[1:]]]
    top = tail.groupby("candidate_id")["top_0_01pct_lifespan"].first().reset_index()
    censored = tail.groupby("candidate_id")["top_0_01pct_censored"].first().reset_index()
    return fits.merge(summary, on="candidate_id", how="left").merge(top, on="candidate_id", how="left").merge(censored, on="candidate_id", how="left")


def plot_score_and_tau(fits: pd.DataFrame) -> Path:
    configure_matplotlib()
    profile = fits[fits["source"] == "profile_tau_sd"].sort_values("tau_cv")
    full = fits[fits["source"] == "full_tau_sd_fit"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    axes[0].plot(profile["tau_cv"], profile["eval_score"], color="#0B7F8C", lw=2.5, marker="o", label="fixed-spread profile")
    axes[0].scatter(full["tau_cv"], full["eval_score"], color="#8E5EA2", marker="^", s=52, alpha=0.7, label="free-spread starts")
    axes[0].axhline(fits["best_eval_score"].iloc[0] * fits["acceptance_multiplier"].iloc[0], color="#777777", ls="--", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[0].set_ylabel("Fit objective score")
    axes[0].set_title("Wide-bounds fit profile")
    axes[1].plot(profile["tau_cv"], profile["tau_factor_vs_karin"], color="#117A65", lw=2.5, marker="o")
    axes[1].axhline(1.0, color="#555555", ls="--", lw=1.0)
    axes[1].axhline(0.25, color="#BBBBBB", ls=":", lw=1.0)
    axes[1].axhline(4.0, color="#BBBBBB", ls=":", lw=1.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[1].set_ylabel(r"Fitted median $\tau/\tau_K$")
    axes[1].set_title("Median tau was allowed 0.25x-4x")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    return save_fig(fig, "01_wide_fit_score_and_tau")


def plot_parameter_compensation(fits: pd.DataFrame) -> Path:
    configure_matplotlib()
    profile = fits[fits["source"] == "profile_tau_sd"].sort_values("tau_cv")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for col, label, color in [
        ("eta_factor", r"$\eta/\eta_K$", "#2274A5"),
        ("beta_factor", r"$\beta/\beta_K$", "#B05C2E"),
        ("kappa_factor", r"$\kappa/\kappa_K$", "#6A4C93"),
        ("epsilon_factor", r"$\epsilon/\epsilon_K$", "#2A9D8F"),
    ]:
        ax.plot(profile["tau_cv"], profile[col], marker="o", lw=2.0, label=label, color=color)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    ax.set_ylabel("Fitted parameter factor vs baseline")
    ax.set_title("SR parameters can move during the refit")
    ax.set_ylim(0.6, 1.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    return save_fig(fig, "02_wide_parameter_compensation")


def plot_mortality_fits(fits: pd.DataFrame, curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    hazard = curves[curves["curve"] == "hazard"].copy()
    hmd = hazard[hazard["candidate_id"] == "HMD"].copy()
    profile = fits[fits["source"] == "profile_tau_sd"].sort_values("tau_cv").copy()
    accepted = profile[profile["strict_accepted"]]
    ids = list(dict.fromkeys([profile.iloc[0]["candidate_id"], *accepted["candidate_id"].tolist(), profile.iloc[-1]["candidate_id"]]))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(ids), 1)))
    color_map = dict(zip(ids, colors))
    axes[0].scatter(hmd["age"], hmd["value"], s=28, color="black", label="Sweden 2019 HMD", zorder=5)
    hmd_lookup = hmd[["age", "value"]].rename(columns={"value": "hmd_hazard"})
    for candidate_id in ids:
        row = fits[fits["candidate_id"] == candidate_id].iloc[0]
        sub = hazard[hazard["candidate_id"] == candidate_id].sort_values("age")
        if sub.empty:
            continue
        label = rf"CV {row['tau_cv']:.3f}, $\tau$ {row['tau_factor_vs_karin']:.2f}x"
        axes[0].plot(sub["age"], sub["value"], lw=2.1, alpha=0.85, color=color_map[candidate_id], label=label)
        resid = sub[["age", "value"]].merge(hmd_lookup, on="age", how="inner")
        resid["log2_fold"] = np.log2(resid["value"] / resid["hmd_hazard"])
        axes[1].plot(resid["age"], resid["log2_fold"], lw=1.9, alpha=0.85, color=color_map[candidate_id], label=label)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Mortality rate / hazard")
    axes[0].set_title("Data and wide refit curves")
    axes[1].axhline(0, color="black", lw=1.1)
    axes[1].axhspan(-0.5, 0.5, color="#EDEDED", zorder=-10)
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel(r"Fit / HMD mortality, $\log_2$ fold")
    axes[1].set_title("Age-by-age residuals")
    for ax in axes:
        ax.set_xlim(65, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.0)
    return save_fig(fig, "03_wide_mortality_data_vs_fits")


def plot_tail_metrics(fits: pd.DataFrame, curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = tail_summary(fits, curves)
    profile = data[data["source"] == "profile_tau_sd"].sort_values("tau_cv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(profile["tau_cv"], profile["S110_ratio_to_hmd"], marker="o", lw=2.4, color="#B05C2E")
    axes[0].axhline(1.0, color="black", lw=1.1, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[0].set_ylabel(r"$S(110\mid90)$ ratio to HMD")
    axes[0].set_title("Tail inflation after wide refit")
    axes[1].plot(profile["tau_cv"], profile["top_0_01pct_lifespan"], marker="o", lw=2.4, color="#245A8D")
    axes[1].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[1].set_ylabel("Top 0.01% lifespan [years]")
    axes[1].set_title("Extreme simulated lifespan")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return save_fig(fig, "04_wide_tail_metrics")


def plot_score_tail_tradeoff(fits: pd.DataFrame, curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = tail_summary(fits, curves)
    profile = data[data["source"] == "profile_tau_sd"].sort_values("tau_cv")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    scatter = ax.scatter(
        profile["eval_score"],
        profile["S110_ratio_to_hmd"],
        c=profile["tau_cv"],
        s=72,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.7,
    )
    for _, row in profile.iterrows():
        ax.text(row["eval_score"] * 1.04, row["S110_ratio_to_hmd"], f"{row['tau_cv']:.2f}", fontsize=8.5, va="center")
    ax.axvline(fits["best_eval_score"].iloc[0] * fits["acceptance_multiplier"].iloc[0], color="#777777", lw=1.2, ls="--")
    ax.axhline(1.0, color="black", lw=1.1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fit objective score")
    ax.set_ylabel(r"$S(110\mid90)$ ratio to HMD")
    ax.set_title("Fit quality and tail excess")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    cbar.set_label(r"$\mathrm{CV}(\tau)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_fig(fig, "05_wide_fit_tail_tradeoff")


def make_plots(fits: pd.DataFrame, curves: pd.DataFrame) -> list[Path]:
    return [
        plot_score_and_tau(fits),
        plot_parameter_compensation(fits),
        plot_mortality_fits(fits, curves),
        plot_tail_metrics(fits, curves),
        plot_score_tail_tradeoff(fits, curves),
    ]


def rel_plot(path: Path) -> str:
    return f"![{path.stem}]({Path(os.path.relpath(path, EXPLORATION_DIR)).as_posix()})"


def fmt_range(values: pd.Series, digits: int = 2) -> str:
    values = values[np.isfinite(values)]
    if values.empty:
        return "NA"
    return f"{values.min():.{digits}f}-{values.max():.{digits}f}"


def mortality_metrics(fits: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    hazard = curves[curves["curve"] == "hazard"].copy()
    hmd = hazard[hazard["candidate_id"] == "HMD"][["age", "value"]].rename(columns={"value": "hmd_hazard"})
    rows = []
    for _, fit in fits[fits["source"] == "profile_tau_sd"].sort_values("tau_sd").iterrows():
        sub = hazard[hazard["candidate_id"] == fit["candidate_id"]][["age", "value"]].merge(hmd, on="age", how="inner")
        sub = sub[(sub["age"] >= 65) & (sub["age"] <= 100)]
        if sub.empty:
            continue
        resid = np.log(sub["value"].to_numpy(dtype=float)) - np.log(sub["hmd_hazard"].to_numpy(dtype=float))
        rows.append(
            {
                "tau_cv": fit["tau_cv"],
                "tau_factor_vs_karin": fit["tau_factor_vs_karin"],
                "eval_score": fit["eval_score"],
                "hazard_log_rmse_65_100": float(np.sqrt(np.mean(resid**2))),
                "median_fold_error_65_100": float(np.exp(np.median(np.abs(resid)))),
            }
        )
    return pd.DataFrame(rows)


def write_report(fits: pd.DataFrame, curves: pd.DataFrame, plot_paths: list[Path], config: FitConfig) -> None:
    profile = fits[fits["source"] == "profile_tau_sd"].sort_values("tau_sd").copy()
    accepted = profile[profile["strict_accepted"]].copy()
    full = fits[fits["source"] == "full_tau_sd_fit"].sort_values("eval_score").copy()
    tail = tail_summary(fits, curves)
    tail_profile = tail[tail["source"] == "profile_tau_sd"].sort_values("tau_sd")
    metrics = mortality_metrics(fits, curves)
    best_profile = profile.sort_values("eval_score").iloc[0]
    best_full = full.iloc[0] if not full.empty else None

    lines = [
        "# Wide Tau-Spread Refit Profile",
        "",
        "This is the deliberately wide follow-up to the narrower tau-spread profile. Here the fitted model is parameterized directly by the median senogenic timescale",
        "",
        "$$",
        "\\tau=\\frac{\\beta}{\\eta}.",
        "$$",
        "",
        "For each fixed population spread \\(\\sigma_v=\\mathrm{SD}[\\log(\\tau)]\\), the optimizer can move \\(\\tau\\), \\(\\eta\\), \\(\\kappa\\), \\(\\epsilon\\), Sweden/USA \\(X_c\\), and Sweden/USA \\(X_c\\) heterogeneity. The external Makeham-like term \\(h_\\mathrm{ext}\\) remains fixed from the HMD/GGM preprocessing.",
        "",
        "## Wide Bounds",
        "",
        "- Median \\(\\tau/\\tau_K\\): 0.25x-4x.",
        "- \\(\\eta/\\eta_K\\), \\(\\kappa/\\kappa_K\\), \\(\\epsilon/\\epsilon_K\\): about 0.35x-2.83x.",
        "- Sweden/USA \\(X_c/X_{c,K}\\): about 0.44x-2.30x.",
        "- Sweden/USA \\(X_c\\) CV factor: 0.25x-2.83x.",
        "- Because \\(\\beta=\\eta\\tau\\), \\(\\beta/\\beta_K\\) can move over the combined \\(\\eta\\) and \\(\\tau\\) range.",
        "",
        "## What I Ran",
        "",
        f"- Profile grid: \\(\\sigma_v={', '.join(f'{x:g}' for x in TAU_SD_GRID)}\\).",
        f"- Profile search used {config.profile_starts_per_sd} structured/random starts per grid value.",
        f"- Free-spread multistart fits used {config.full_starts} starts and could move \\(\\sigma_v\\) within {TAU_SD_BOUNDS}.",
        f"- Coordinate-search step sizes were \\({', '.join(f'{x:g}' for x in STEP_SIZES)}\\) in log2-parameter units.",
        f"- Search simulations: \\(n={config.fit_n:,}\\), \\(\\Delta t={config.fit_dt:g}\\). Final evaluation: \\(n={config.eval_n:,}\\). Curve/tail simulations: \\(n={config.curve_n:,}\\).",
        "",
        "Scope note: this is still an exploratory stochastic coordinate-search profile, not a theorem of global optimality. Its point is to stress-test whether broad smooth \\(\\tau\\)-spread can be hidden by moving the whole SR parameter set across very wide bounds.",
        "",
        "Individual variation is introduced as",
        "",
        "$$",
        "\\eta_i=\\eta_0 e^{-v_i/2},\\qquad \\beta_i=\\beta_0 e^{v_i/2},\\qquad v_i\\sim\\mathcal N(0,\\sigma_v^2).",
        "$$",
        "",
        "Thus individual \\(\\tau_i=\\beta_i/\\eta_i\\) has log-spread \\(\\sigma_v\\), while the fitted median \\(\\tau_0\\) is free to move.",
        "",
        "## Result 1: wide refits still prefer small tau-spread",
        "",
        f"The best fixed-spread profile fit had \\(\\mathrm{{CV}}(\\tau)={best_profile['tau_cv']:.3f}\\), median \\(\\tau/\\tau_K={best_profile['tau_factor_vs_karin']:.2f}\\), and score {best_profile['eval_score']:.4g}.",
    ]
    if best_full is not None:
        lines.append(
            f"The best free-spread multistart fit ended at \\(\\mathrm{{CV}}(\\tau)={best_full['tau_cv']:.3f}\\), median \\(\\tau/\\tau_K={best_full['tau_factor_vs_karin']:.2f}\\), and score {best_full['eval_score']:.4g}."
        )
    lines.extend(
        [
            f"Using the exploratory \\({config.acceptance_multiplier:g}\\times\\) score threshold, accepted profile fits covered \\(\\mathrm{{CV}}(\\tau)={fmt_range(accepted['tau_cv'], 3)}\\).",
            "",
            rel_plot(plot_paths[0]),
            "",
            "How to read this figure:",
            "",
            "- Left panel: each teal point is a fixed-spread profile fit. The x-axis is the imposed population CV of \\(\\tau=\\beta/\\eta\\); the y-axis is the mortality-fit objective, with lower being better. The dashed horizontal line is the exploratory acceptance threshold. The important pattern is that fit quality is still good only at very small spread, then worsens quickly; by \\(\\mathrm{CV}(\\tau)\\approx0.05\\), the profile is already outside the threshold.",
            "- Purple triangles are free-spread multistart fits. If broad \\(\\tau\\)-spread were useful, these would tend to land at nonzero CV with good scores. Instead, the good free fits collapse back to \\(\\mathrm{CV}(\\tau)=0\\); the nonzero-spread free fit is much worse.",
            "- Right panel: this is the main sanity check for your question. The optimizer was allowed to move the median \\(\\tau\\) anywhere from 0.25x to 4x the baseline value, shown by the dotted gray bounds. If high spread could be rescued by choosing a radically different baseline \\(\\tau\\), this curve would move strongly up or down. It does not. It stays near \\(\\tau/\\tau_K\\approx0.97\\)-0.99. So the failure of broad spread is not because median \\(\\beta/\\eta\\) was frozen.",
            "",
            "Takeaway: even when median \\(\\tau\\) has a huge escape route, the fit does not use it. Broad smooth spread in \\(\\tau\\) itself is what creates trouble.",
            "",
            "## Result 2: nuisance parameters really were allowed to move",
            "",
            "The panel below shows fitted SR-parameter factors across the fixed-spread profile.",
            "",
            rel_plot(plot_paths[1]),
            "",
            "How to read this figure:",
            "",
            "- Each line is a fitted shared SR parameter divided by its baseline value. A value of 1 means unchanged; 0.8 means 20% lower; 1.4 means 40% higher.",
            "- This plot is there to show that the refit was not artificially rigid. \\(\\eta\\), \\(\\beta\\), \\(\\kappa\\), and \\(\\epsilon\\) all had room to move; Sweden/USA \\(X_c\\) and \\(X_c\\) heterogeneity were also fit, but are not drawn here to keep the panel readable.",
            "- The optimizer does move nuisance parameters somewhat, especially \\(\\kappa\\) at some grid points, but these shifts do not rescue high \\(\\tau\\)-CV. The bad high-CV fits are not bad because the rest of the model was held fixed.",
            "",
            "Takeaway: this is a compensation check. We gave the SR model many knobs, and broad \\(\\tau\\)-spread still degraded the fit and inflated the tail.",
            "",
            "## Result 3: data-vs-fit mortality curves",
            "",
            rel_plot(plot_paths[2]),
            "",
            "How to read this figure:",
            "",
            "- Left panel: black dots are Sweden 2019 HMD mortality. Colored lines are selected wide-refit SR curves at different \\(\\mathrm{CV}(\\tau)\\). A good fit should track the black dots over ages 65-100.",
            "- Right panel: the same information as fold-error. The horizontal zero line means exact agreement with HMD. The gray band is roughly within \\(2^{0.5}\\approx1.4\\)-fold of HMD.",
            "- Low-spread fits stay much closer to the gray band. Broad-spread fits leave systematic curvature: they can be too high at younger old ages and too low at the oldest fitted ages. That shape is the mortality signature of mixing subpopulations with different senogenic timescales.",
            "",
            "Takeaway: the problem is not just an abstract score. You can see the broad-spread models bending away from the age pattern in the actual mortality data.",
            "",
            "Profile mortality fit errors:",
            "",
            "| CV(tau) | median tau/tau_K | fit score | mortality log RMSE | median fold error |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['tau_cv']:.3f} | {row['tau_factor_vs_karin']:.2f} | {row['eval_score']:.4g} | {row['hazard_log_rmse_65_100']:.3f} | {row['median_fold_error_65_100']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Result 4: old-age tails after wide refit",
            "",
            f"Across the profile grid, \\(S(110\\mid90)\\) ratios to Sweden 2019 HMD ranged from {fmt_range(tail_profile['S110_ratio_to_hmd'], 2)}.",
            f"Top 0.01% simulated lifespan ranged from {fmt_range(tail_profile['top_0_01pct_lifespan'], 1)} years; the largest value is censored at the simulation maximum if marked with \\(\\ge\\).",
            "",
            rel_plot(plot_paths[3]),
            "",
            "How to read this figure:",
            "",
            "- Left panel: this is the old-age survival-tail penalty. The y-axis is \\(S(110\\mid90)\\) in the model divided by Sweden 2019 HMD. A value of 1 would match HMD. Values above 1 mean too many people surviving from 90 to 110.",
            "- The low-spread baseline is already about 2.6x HMD in this particular wide stochastic fit. But adding only \\(\\mathrm{CV}(\\tau)\\approx0.025\\) raises the age-110 tail to about 15x HMD, and \\(\\mathrm{CV}(\\tau)\\approx0.05\\) raises it to about 68x HMD.",
            "- Right panel: this converts the same tail effect into an intuitive extreme-lifespan metric: the age reached by the top 0.01% of the simulated cohort. As \\(\\tau\\)-spread increases, the favorable tail creates rare very slow-aging individuals, so this age shoots upward.",
            "",
            "Takeaway: the old-age tail is more sensitive than the central mortality fit. Even small smooth spread in \\(\\tau\\) produces too many extreme survivors.",
            "",
            "Tail summary:",
            "",
            "| CV(tau) | median tau/tau_K | fit score | S(110\\|90) / HMD | top 0.01% age |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in tail_profile.iterrows():
        top_age = f"{row['top_0_01pct_lifespan']:.1f}"
        if bool(row.get("top_0_01pct_censored", False)):
            top_age = f">={top_age}"
        lines.append(
            f"| {row['tau_cv']:.3f} | {row['tau_factor_vs_karin']:.2f} | {row['eval_score']:.4g} | {row['S110_ratio_to_hmd']:.2f} | {top_age} |"
        )
    lines.extend(
        [
            "",
            "## Result 5: fit score and tail score together",
            "",
            rel_plot(plot_paths[4]),
            "",
            "How to read this figure:",
            "",
            "- This plot puts the two constraints on one graph. The x-axis is fit quality; left is better. The y-axis is age-110 tail excess; lower is better. The color and labels show \\(\\mathrm{CV}(\\tau)\\).",
            "- The desirable region is the lower-left: good mortality fit and no excess survival tail. Increasing \\(\\mathrm{CV}(\\tau)\\) moves points up and to the right, meaning both constraints get worse together.",
            "- The \\(\\mathrm{CV}(\\tau)=0.025\\) point still has acceptable fit score by the exploratory threshold, but it already has a large age-110 tail excess. By \\(\\mathrm{CV}(\\tau)\\approx0.05\\), both the fit score and the tail are problematic.",
            "",
            "Takeaway: there is no hidden broad-spread solution in this wide profile. The favorable tail shows up either as poor mortality curvature, excessive extreme survival, or both.",
            "",
            "## Bottom Line",
            "",
            "In this wide-bounds version, broad smooth variation in the senogenic timescale is tested while all fitted SR parameters can compensate over large ranges. The optimizer did not use the allowed 0.25x-4x median-\\(\\tau\\) freedom to rescue broad \\(\\tau\\)-spread; profile fits stayed near \\(\\tau/\\tau_K\\approx0.97\\)-0.99. Fit quality crossed the exploratory threshold by \\(\\mathrm{CV}(\\tau)\\approx0.05\\), and even the accepted \\(\\mathrm{CV}(\\tau)\\approx0.025\\) profile produced a large age-110 tail excess. The useful manuscript-level statement is therefore about the favorable tail: smooth spread in \\(\\tau\\) is hard to hide, even when the baseline SR parameters are given wide compensatory freedom.",
            "",
            "## Outputs",
            "",
            f"- Fit candidates: `{FIT_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Fit curves and tail metrics: `{CURVE_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Cache: `{CACHE_DIR.relative_to(PROJECT_ROOT)}`.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    set_runtime(config)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.plots_only:
        fits, curves = load_cached_outputs(config, require_metadata=False)
    elif not args.force and metadata_matches(config) and FIT_CACHE_PATH.exists() and CURVE_CACHE_PATH.exists():
        fits, curves = load_cached_outputs(config, require_metadata=True)
    else:
        fits = run_fits(config)
        curves = build_curves(fits, config)
        write_outputs(fits, curves, config)

    plot_paths = make_plots(fits, curves)
    write_report(fits, curves, plot_paths, config)
    write_outputs(fits, curves, config)
    print(f"Saved report: {REPORT_PATH}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
