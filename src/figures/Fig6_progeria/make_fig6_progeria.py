#!/usr/bin/env python3
"""Make the new Fig. 6 progeria panels from the Sweden 2019 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
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
from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Fig6_progeria"
SOURCE_DIR = SAVED_RESULTS_DIR / "csv"
CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "Fig6_progeria"

PANEL_A_PNG_PATH = OUTPUT_DIR / "fig6a_progeria_hgps_survival.png"
PANEL_A_PDF_PATH = OUTPUT_DIR / "fig6a_progeria_hgps_survival.pdf"
PANEL_B_PNG_PATH = OUTPUT_DIR / "fig6b_single_parameter_fits.png"
PANEL_B_PDF_PATH = OUTPUT_DIR / "fig6b_single_parameter_fits.pdf"
PANEL_C_PNG_PATH = OUTPUT_DIR / "fig6c_two_parameter_fits.png"
PANEL_C_PDF_PATH = OUTPUT_DIR / "fig6c_two_parameter_fits.pdf"
COMPOSITE_PNG_PATH = OUTPUT_DIR / "fig6_progeria_composite.png"
COMPOSITE_PDF_PATH = OUTPUT_DIR / "fig6_progeria_composite.pdf"

HGPS_ENVELOPE_PATH = SOURCE_DIR / "fig6_progeria_hgps_bootstrap_envelope.csv"
FIT_RESULTS_PATH = SOURCE_DIR / "fig6_progeria_fit_results.csv"
SURVIVAL_CURVES_PATH = SOURCE_DIR / "fig6_progeria_survival_curves.csv"
MODEL_ENVELOPES_PATH = SOURCE_DIR / "fig6_progeria_model_ci_envelopes.csv"
PERIOD_SURVIVAL_PATH = SOURCE_DIR / "fig6_progeria_period_survival.csv"

FIT_CACHE_PATH = CACHE_DIR / "fit_results.json"
CURVE_CACHE_PATH = CACHE_DIR / "survival_curves.csv"
ENVELOPE_CACHE_PATH = CACHE_DIR / "model_ci_envelopes.csv"
METADATA_PATH = CACHE_DIR / "metadata.json"

PROGERIA_DATA_PATH = SAVED_RESULTS_DIR / "progeria_data.pkl"
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

SINGLE_FITS = (("eta",), ("beta",), ("Xc",), ("epsilon",))
PAIR_FITS = (
    ("Xc", "epsilon"),
    ("eta", "Xc"),
    ("eta", "epsilon"),
    ("beta", "Xc"),
    ("beta", "epsilon"),
    ("eta", "beta"),
)
SENOGENIC_PARAMS = {"eta", "beta"}

PARAMETER_SHORT_LABELS = {
    "eta": r"$\eta$",
    "beta": r"$\beta$",
    "Xc": r"$X_c$",
    "epsilon": r"$\epsilon$",
}
PARAMETER_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
}
PAIR_COLORS = {
    "Xc+epsilon": "#D62728",
    "eta+Xc": "#0B7F8C",
    "eta+epsilon": "#44A6AE",
    "beta+Xc": "#173A6A",
    "beta+epsilon": "#4A6FA5",
    "eta+beta": "#532B88",
}

FIT_BOUNDS_LOG2 = {
    "eta": (-1.0, 3.0),
    "beta": (-4.0, 1.0),
    "Xc": (-3.0, 0.5),
    "epsilon": (-4.0, 3.0),
}
OLD_RESULT_NAMES = {
    ("eta",): "eta",
    ("beta",): "beta",
    ("Xc",): "Xc",
    ("epsilon",): "epsilon",
    ("Xc", "epsilon"): "Xc + epsilon",
    ("eta", "Xc"): "eta + Xc",
    ("eta", "epsilon"): "eta + epsilon",
    ("beta", "Xc"): "beta + Xc",
    ("beta", "epsilon"): "beta + epsilon",
    ("eta", "beta"): "beta + eta",
}
CI_ROW_BY_PARAM = {
    "eta": "eta",
    "beta": "beta",
    "epsilon": "epsilon",
    "Xc": "SWE_Xc",
}

TMAX = 35.0
DT = 0.025
SAVE_STEP = 0.25
RANDOM_SEED = 20260520
TARGET_MIN_SURVIVAL = 0.015
MODEL_BAND_ALPHA = 0.12
HGPS_BAND_ALPHA = 0.18
HGPS_SAMPLE_SIZE = 202
PAIR_SCREEN_GRID_SIZE = 9
LOCAL_SEARCH_STEPS = (0.5, 0.25, 0.125, 0.0625)


@dataclass(frozen=True)
class HGPSData:
    times: np.ndarray
    survival: np.ndarray
    envelope: pd.DataFrame


@dataclass(frozen=True)
class PeriodSurvival:
    country: str
    label: str
    ages: np.ndarray
    survival: np.ndarray
    median_lifespan: float
    color: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-n", type=int, default=4_000)
    parser.add_argument("--curve-n", type=int, default=35_000)
    parser.add_argument("--bootstrap-n", type=int, default=2_000)
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument("--force-curves", action="store_true")
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12.4,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 8.6,
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


def load_sweden_baseline() -> dict[str, float]:
    record = json.loads(BASELINE_FIT_PATH.read_text())
    fitted = record["summary"]["fitted_parameters"]
    return {
        "eta": float(fitted["eta"]),
        "beta": float(fitted["beta"]),
        "kappa": 0.5,
        "epsilon": float(fitted["epsilon"]),
        "Xc": float(fitted["SWE_Xc"]),
        "xc_std_frac": float(fitted["SWE_xc_std_frac"]),
    }


def load_sweden_ci_bounds() -> dict[str, tuple[float, float]]:
    ci = pd.read_csv(BASELINE_CI_PATH).set_index("parameter")
    bounds = {}
    for param_name, row_name in CI_ROW_BY_PARAM.items():
        row = ci.loc[row_name]
        bounds[param_name] = (float(row["ci95_lower"]), float(row["ci95_upper"]))
    return bounds


def load_hgps_data(bootstrap_n: int) -> HGPSData:
    with PROGERIA_DATA_PATH.open("rb") as handle:
        data = pickle.load(handle)

    times = np.asarray(data["survival_curve"]["t_pred"], dtype=float)
    survival = np.clip(np.asarray(data["survival_curve"]["s_pred"], dtype=float), 0.0, 1.0)
    envelope = build_hgps_bootstrap_envelope(times, survival, bootstrap_n)
    return HGPSData(times=times, survival=survival, envelope=envelope)


def build_hgps_bootstrap_envelope(
    target_times: np.ndarray,
    smooth_survival: np.ndarray,
    bootstrap_n: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    curves = []
    for _ in range(bootstrap_n):
        lifespans = sample_lifespans_from_survival(
            times=target_times,
            survival=smooth_survival,
            sample_size=HGPS_SAMPLE_SIZE,
            rng=rng,
        )
        curves.append(empirical_survival_on_grid(lifespans, target_times))

    values = np.vstack(curves)
    envelope = pd.DataFrame(
        {
            "age": target_times,
            "survival": smooth_survival,
            "ci_lower": np.quantile(values, 0.025, axis=0),
            "ci_upper": np.quantile(values, 0.975, axis=0),
        }
    )
    HGPS_ENVELOPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    envelope.to_csv(HGPS_ENVELOPE_PATH, index=False)
    return envelope


def sample_lifespans_from_survival(
    times: np.ndarray,
    survival: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    clean_times = np.asarray(times, dtype=float)
    clean_survival = np.clip(np.asarray(survival, dtype=float), 0.0, 1.0)

    if clean_survival[-1] > 0:
        clean_times = np.append(clean_times, TMAX)
        clean_survival = np.append(clean_survival, 0.0)

    survival_draws = rng.uniform(0.0, 1.0, int(sample_size))
    return np.interp(survival_draws, clean_survival[::-1], clean_times[::-1])


def empirical_survival_on_grid(lifespans: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.array([np.mean(lifespans > age) for age in grid], dtype=float)


def build_metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "baseline_fit_path": str(BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)),
        "baseline_ci_path": str(BASELINE_CI_PATH.relative_to(PROJECT_ROOT)),
        "progeria_data_path": str(PROGERIA_DATA_PATH.relative_to(PROJECT_ROOT)),
        "fit_n": int(args.fit_n),
        "curve_n": int(args.curve_n),
        "bootstrap_n": int(args.bootstrap_n),
        "tmax": TMAX,
        "dt": DT,
        "save_step": SAVE_STEP,
        "single_fits": [list(item) for item in SINGLE_FITS],
        "pair_fits": [list(item) for item in PAIR_FITS],
        "fit_bounds_log2": FIT_BOUNDS_LOG2,
        "random_seed": RANDOM_SEED,
        "hgps_sample_size": HGPS_SAMPLE_SIZE,
        "parallel": False,
        "ci_method": "fitted_fold_factor_applied_to_sweden_2019_fit_ci_endpoint_combinations",
    }


def load_period_survival_curves() -> list[PeriodSurvival]:
    period_curves = [
        build_period_survival(country="SWE", label="Sweden 2019 period data", color="#111111"),
    ]
    save_period_survival_curves(period_curves)
    return period_curves


def build_period_survival(country: str, label: str, color: str) -> PeriodSurvival:
    hmd = HMD(country, "both", "period")
    ages, survival = hmd.get_survival(2019, strict=True)
    ages = np.asarray(ages, dtype=float)
    survival = np.asarray(survival, dtype=float)
    valid = np.isfinite(ages) & np.isfinite(survival) & (survival >= 0)
    ages = ages[valid]
    survival = np.clip(survival[valid], 0.0, 1.0)
    median = interpolate_age_at_survival(ages, survival, 0.5)
    return PeriodSurvival(
        country=country,
        label=label,
        ages=ages,
        survival=survival,
        median_lifespan=median,
        color=color,
    )


def interpolate_age_at_survival(ages: np.ndarray, survival: np.ndarray, value: float) -> float:
    return float(np.interp(value, survival[::-1], ages[::-1]))


def save_period_survival_curves(period_curves: list[PeriodSurvival]) -> None:
    rows = []
    for curve in period_curves:
        for age, survival in zip(curve.ages, curve.survival):
            rows.append(
                {
                    "country": curve.country,
                    "label": curve.label,
                    "age": float(age),
                    "survival": float(survival),
                    "median_lifespan": float(curve.median_lifespan),
                    "age_over_median": float(age / curve.median_lifespan),
                }
            )

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PERIOD_SURVIVAL_PATH, index=False)


def metadata_matches(current: dict[str, object], ignore_keys: set[str] | None = None) -> bool:
    if not METADATA_PATH.exists():
        return False
    cached = json.loads(METADATA_PATH.read_text())
    normalised_current = json.loads(json.dumps(current))
    if ignore_keys:
        cached = {key: value for key, value in cached.items() if key not in ignore_keys}
        normalised_current = {key: value for key, value in normalised_current.items() if key not in ignore_keys}
    return cached == normalised_current


def load_old_warm_starts(baseline: dict[str, float]) -> dict[tuple[str, ...], dict[str, float]]:
    path = SAVED_RESULTS_DIR / "progeria_fitting_results.pkl"
    if not path.exists():
        return {}

    with path.open("rb") as handle:
        old = pickle.load(handle)

    starts = {}
    for fit_params, old_name in OLD_RESULT_NAMES.items():
        result = find_old_result(old["results"], old_name)
        if result is None:
            continue
        starts[fit_params] = {
            param: safe_log2(result["best_params"][param] / baseline[param])
            for param in fit_params
        }
    return starts


def find_old_result(results: list[dict[str, object]], name: str) -> dict[str, object] | None:
    compact_name = name.replace(" ", "")
    for result in results:
        result_name = str(result["param_name"])
        if result_name == name or result_name.replace(" ", "") == compact_name:
            return result
    return None


def safe_log2(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(np.log2(value))


def fit_all_models(
    hgps: HGPSData,
    baseline: dict[str, float],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    cached_fits = []
    if FIT_CACHE_PATH.exists() and metadata_matches(build_metadata(args), ignore_keys={"curve_n", "bootstrap_n"}):
        fits = json.loads(FIT_CACHE_PATH.read_text())["fits"]
        expected_count = len(SINGLE_FITS) + len(PAIR_FITS)
        if len(fits) == expected_count and not args.force_fit:
            return fits
        cached_fits = fits

    warm_starts = load_old_warm_starts(baseline)
    fits = [] if args.force_fit and len(cached_fits) == len(SINGLE_FITS) + len(PAIR_FITS) else cached_fits
    completed_names = {fit["fit_name"] for fit in fits}
    for fit_params in [*SINGLE_FITS, *PAIR_FITS]:
        if fit_name(fit_params) in completed_names:
            continue
        fit = fit_parameter_set(
            fit_params=fit_params,
            hgps=hgps,
            baseline=baseline,
            warm_start=warm_starts.get(fit_params, {}),
            n=args.fit_n,
        )
        fits.append(fit)
        save_fit_cache(fits, args)
        print(f"fit {fit['fit_name']}: cost={fit['cost']:.5f}")

    return fits


def fit_parameter_set(
    fit_params: tuple[str, ...],
    hgps: HGPSData,
    baseline: dict[str, float],
    warm_start: dict[str, float],
    n: int,
) -> dict[str, object]:
    objective_seed = stable_seed(RANDOM_SEED, "fit-objective", fit_name(fit_params))
    factors = fit_log2_factors(
        fit_params=fit_params,
        hgps=hgps,
        baseline=baseline,
        warm_start=warm_start,
        n=n,
        seed=objective_seed,
    )

    params = apply_log2_factors(baseline, factors)
    cost = survival_cost(params, hgps, n, objective_seed)
    return {
        "fit_name": fit_name(fit_params),
        "fit_params": list(fit_params),
        "log2_factors": {key: float(value) for key, value in factors.items()},
        "factors": {key: float(2.0 ** value) for key, value in factors.items()},
        "params": {key: float(value) for key, value in params.items()},
        "cost": float(cost),
        "contains_senogenic": any(param in SENOGENIC_PARAMS for param in fit_params),
    }


def fit_log2_factors(
    fit_params: tuple[str, ...],
    hgps: HGPSData,
    baseline: dict[str, float],
    warm_start: dict[str, float],
    n: int,
    seed: int,
) -> dict[str, float]:
    scored = score_factor_grid(
        fit_params=fit_params,
        hgps=hgps,
        baseline=baseline,
        warm_start=warm_start,
        n=n,
        seed=seed,
    )
    best_cost, best_factors = scored[0]

    starts = [factors for _, factors in scored[:4]]
    if all(param in warm_start for param in fit_params):
        starts.append({param: clip_log2_factor(param, warm_start[param]) for param in fit_params})

    for start in starts:
        local_factors, local_cost = local_pattern_search(
            start=start,
            fit_params=fit_params,
            hgps=hgps,
            baseline=baseline,
            n=n,
            seed=seed,
        )
        if local_cost < best_cost:
            best_cost = local_cost
            best_factors = local_factors

    return best_factors


def score_factor_grid(
    fit_params: tuple[str, ...],
    hgps: HGPSData,
    baseline: dict[str, float],
    warm_start: dict[str, float],
    n: int,
    seed: int,
) -> list[tuple[float, dict[str, float]]]:
    value_lists = [candidate_log2_values(param, warm_start) for param in fit_params]
    scored = []
    for values in cartesian_product_float(value_lists):
        factors = {param: float(value) for param, value in zip(fit_params, values)}
        params = apply_log2_factors(baseline, factors)
        cost = survival_cost(params, hgps, n, seed)
        scored.append((cost, factors))

    scored.sort(key=lambda item: item[0])
    return scored


def candidate_log2_values(param: str, warm_start: dict[str, float]) -> np.ndarray:
    lower, upper = FIT_BOUNDS_LOG2[param]
    values = np.linspace(lower, upper, PAIR_SCREEN_GRID_SIZE)
    values = np.append(values, 0.0)
    if param in warm_start:
        values = np.append(values, clip_log2_factor(param, warm_start[param]))
    return np.unique(np.round(values, 6))


def cartesian_product_float(items: list[np.ndarray]) -> list[list[float]]:
    if not items:
        return [[]]
    rest = cartesian_product_float(items[1:])
    return [[float(value), *tail] for value in items[0] for tail in rest]


def local_pattern_search(
    start: dict[str, float],
    fit_params: tuple[str, ...],
    hgps: HGPSData,
    baseline: dict[str, float],
    n: int,
    seed: int,
) -> tuple[dict[str, float], float]:
    current = {param: clip_log2_factor(param, start[param]) for param in fit_params}
    current_cost = survival_cost(apply_log2_factors(baseline, current), hgps, n, seed)

    for step in LOCAL_SEARCH_STEPS:
        improved = True
        while improved:
            improved = False
            for param in fit_params:
                for direction in (-1.0, 1.0):
                    trial = dict(current)
                    trial[param] = clip_log2_factor(param, trial[param] + direction * step)
                    if trial[param] == current[param]:
                        continue

                    params = apply_log2_factors(baseline, trial)
                    cost = survival_cost(params, hgps, n, seed)
                    if cost >= current_cost:
                        continue

                    current = trial
                    current_cost = cost
                    improved = True

    return current, current_cost


def clip_log2_factor(param: str, value: float) -> float:
    lower, upper = FIT_BOUNDS_LOG2[param]
    return float(np.clip(value, lower, upper))


def apply_log2_factors(baseline: dict[str, float], factors: dict[str, float]) -> dict[str, float]:
    params = dict(baseline)
    for param, log2_factor in factors.items():
        params[param] = baseline[param] * (2.0 ** float(log2_factor))
    return params


def survival_cost(params: dict[str, float], hgps: HGPSData, n: int, seed: int) -> float:
    ages, survival = simulate_survival(params, n=n, seed=seed)
    model = np.interp(hgps.times, ages, survival)
    target = np.clip(hgps.survival, TARGET_MIN_SURVIVAL, 1.0)
    model = np.clip(model, TARGET_MIN_SURVIVAL, 1.0)
    weights = 0.4 + 0.6 * (1.0 - target)
    return float(np.sqrt(np.average((model - target) ** 2, weights=weights)))


def simulate_survival(params: dict[str, float], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    params_dict = build_simulation_params(params, n=n, seed=seed)
    sim = utils.create_sr_simulation(
        params_dict=params_dict,
        n=int(n),
        h_ext=0.0,
        tmax=TMAX,
        dt=DT,
        save_times=np.arange(0.0, TMAX + 1e-9, SAVE_STEP),
        parallel=False,
        break_early=True,
        random_seed=int(seed),
    )
    ages = np.arange(0.0, TMAX + 1e-9, SAVE_STEP)
    death_times = np.asarray(sim.death_times, dtype=float)
    survival = np.array([np.mean(death_times > age) for age in ages], dtype=float)
    return ages, survival


def build_simulation_params(params: dict[str, float], n: int, seed: int) -> dict[str, object]:
    params_dict: dict[str, object] = {
        "eta": float(params["eta"]),
        "beta": float(params["beta"]),
        "kappa": float(params["kappa"]),
        "epsilon": float(params["epsilon"]),
        "Xc": float(params["Xc"]),
    }
    xc_std_frac = float(params.get("xc_std_frac", 0.0))
    if xc_std_frac <= 0:
        return params_dict

    params_dict["Xc"] = sample_positive_gaussian(
        mean=float(params["Xc"]),
        rel_std=xc_std_frac,
        n=n,
        seed=stable_seed(seed, "Xc-vector"),
    )
    return params_dict


def sample_positive_gaussian(mean: float, rel_std: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = mean * rel_std
    values = rng.normal(loc=mean, scale=std, size=int(n))

    while True:
        bad = values <= 0
        if not np.any(bad):
            return values

        values[bad] = rng.normal(loc=mean, scale=std, size=int(np.sum(bad)))


def fit_name(fit_params: Iterable[str]) -> str:
    return "+".join(fit_params)


def save_fit_cache(fits: list[dict[str, object]], args: argparse.Namespace) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": build_metadata(args), "fits": fits}
    FIT_CACHE_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    METADATA_PATH.write_text(json.dumps(build_metadata(args), indent=2) + "\n")


def load_or_build_curves(
    hgps: HGPSData,
    baseline: dict[str, float],
    ci_bounds: dict[str, tuple[float, float]],
    fits: list[dict[str, object]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        CURVE_CACHE_PATH.exists()
        and ENVELOPE_CACHE_PATH.exists()
        and metadata_matches(build_metadata(args))
        and not args.force_curves
    ):
        curves = pd.read_csv(CURVE_CACHE_PATH)
        envelopes = pd.read_csv(ENVELOPE_CACHE_PATH)
        copy_curve_outputs(curves, envelopes, fits)
        return curves, envelopes

    curve_rows = []
    envelope_rows = []
    for fit in fits:
        fit_params = tuple(fit["fit_params"])
        central_params = {key: float(value) for key, value in fit["params"].items()}
        ages, survival = simulate_survival(
            central_params,
            n=args.curve_n,
            seed=stable_seed(RANDOM_SEED, "curve", fit["fit_name"], "central"),
        )
        curve_rows.extend(curve_records(fit, ages, survival))

        endpoint_curves = [survival]
        for endpoint_params in build_ci_endpoint_params(baseline, ci_bounds, fit):
            _, endpoint_survival = simulate_survival(
                endpoint_params,
                n=args.curve_n,
                seed=stable_seed(RANDOM_SEED, "curve", fit["fit_name"], endpoint_key(endpoint_params, fit_params)),
            )
            endpoint_curves.append(endpoint_survival)

        endpoint_values = np.vstack(endpoint_curves)
        envelope_rows.extend(
            {
                "fit_name": fit["fit_name"],
                "age": float(age),
                "ci_lower": float(low),
                "ci_upper": float(high),
            }
            for age, low, high in zip(ages, np.min(endpoint_values, axis=0), np.max(endpoint_values, axis=0))
        )
        print(f"curves {fit['fit_name']}")

    curves = pd.DataFrame(curve_rows)
    envelopes = pd.DataFrame(envelope_rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(CURVE_CACHE_PATH, index=False)
    envelopes.to_csv(ENVELOPE_CACHE_PATH, index=False)
    copy_curve_outputs(curves, envelopes, fits)
    return curves, envelopes


def curve_records(fit: dict[str, object], ages: np.ndarray, survival: np.ndarray) -> list[dict[str, object]]:
    return [
        {
            "fit_name": fit["fit_name"],
            "fit_params": "+".join(fit["fit_params"]),
            "contains_senogenic": bool(fit["contains_senogenic"]),
            "age": float(age),
            "survival": float(value),
        }
        for age, value in zip(ages, survival)
    ]


def build_ci_endpoint_params(
    baseline: dict[str, float],
    ci_bounds: dict[str, tuple[float, float]],
    fit: dict[str, object],
) -> list[dict[str, float]]:
    fit_params = tuple(fit["fit_params"])
    factors = {key: float(value) for key, value in fit["factors"].items()}
    scenarios = []
    endpoint_choices = [["lower", "upper"] for _ in fit_params]
    for choice in cartesian_product(endpoint_choices):
        params = dict(baseline)
        for param, endpoint in zip(fit_params, choice):
            lower, upper = ci_bounds[param]
            params[param] = factors[param] * (lower if endpoint == "lower" else upper)
        scenarios.append(params)
    return scenarios


def cartesian_product(items: list[list[str]]) -> list[list[str]]:
    if not items:
        return [[]]
    rest = cartesian_product(items[1:])
    return [[value, *tail] for value in items[0] for tail in rest]


def endpoint_key(params: dict[str, float], fit_params: tuple[str, ...]) -> str:
    return "|".join(f"{param}:{params[param]:.6g}" for param in fit_params)


def copy_curve_outputs(curves: pd.DataFrame, envelopes: pd.DataFrame, fits: list[dict[str, object]]) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    curves.to_csv(SURVIVAL_CURVES_PATH, index=False)
    envelopes.to_csv(MODEL_ENVELOPES_PATH, index=False)
    pd.DataFrame(fits).to_csv(FIT_RESULTS_PATH, index=False)


def style_axis(
    ax: plt.Axes,
    *,
    xlim: tuple[float, float] = (0, 31),
    xlabel: str = "Age",
    xticks: list[float] | None = None,
) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", pad=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Survival")
    ax.set_xticks([0, 10, 20, 30] if xticks is None else xticks)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])


def draw_hgps(
    ax: plt.Axes,
    hgps: HGPSData,
    label: str = "HGPS patients",
    x_scale: float = 1.0,
) -> None:
    envelope_ages = np.asarray(hgps.envelope["age"], dtype=float) / x_scale
    ax.fill_between(
        envelope_ages,
        hgps.envelope["ci_lower"],
        hgps.envelope["ci_upper"],
        color="#111111",
        alpha=HGPS_BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )
    ax.plot(hgps.times / x_scale, hgps.survival, color="#111111", lw=2.7, label=label, zorder=5)


def draw_period_survival(ax: plt.Axes, period_curves: list[PeriodSurvival]) -> None:
    for curve in period_curves:
        ax.plot(
            curve.ages / curve.median_lifespan,
            curve.survival,
            color=curve.color,
            linestyle="--",
            lw=2.7,
            label=curve.label,
            zorder=4,
        )


def draw_model_fit(
    ax: plt.Axes,
    curves: pd.DataFrame,
    envelopes: pd.DataFrame,
    fit_name_value: str,
    color: str,
    label: str,
    lw: float = 2.3,
) -> None:
    curve = curves[curves["fit_name"] == fit_name_value]
    envelope = envelopes[envelopes["fit_name"] == fit_name_value]
    ax.fill_between(
        envelope["age"],
        envelope["ci_lower"],
        envelope["ci_upper"],
        color=color,
        alpha=MODEL_BAND_ALPHA,
        linewidth=0,
        zorder=2,
    )
    ax.plot(curve["age"], curve["survival"], color=color, lw=lw, label=label, zorder=4)


def make_panel_a(hgps: HGPSData, period_curves: list[PeriodSurvival]) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.7))
    hgps_median = median_lifespan_from_curve(hgps.times, hgps.survival)
    draw_hgps(ax, hgps, x_scale=hgps_median)
    draw_period_survival(ax, period_curves)
    style_axis(
        ax,
        xlim=(0, 1.95),
        xlabel="Age / median lifespan",
        xticks=[0, 0.5, 1.0, 1.5],
    )
    ax.set_title("HGPS survival is shallow relative to controls", pad=7)
    ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    fig.tight_layout(pad=0.7)
    fig.savefig(PANEL_A_PNG_PATH, dpi=300)
    fig.savefig(PANEL_A_PDF_PATH)
    plt.close(fig)


def median_lifespan_from_curve(times: np.ndarray, survival: np.ndarray) -> float:
    return interpolate_age_at_survival(times, np.clip(survival, 0.0, 1.0), 0.5)


def make_panel_b(
    hgps: HGPSData,
    curves: pd.DataFrame,
    envelopes: pd.DataFrame,
    fits: list[dict[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(5.9, 4.1))
    draw_hgps(ax, hgps)
    fit_lookup = {fit["fit_name"]: fit for fit in fits}
    for param in ("eta", "beta", "Xc", "epsilon"):
        draw_model_fit(
            ax=ax,
            curves=curves,
            envelopes=envelopes,
            fit_name_value=param,
            color=PARAMETER_COLORS[param],
            label=single_fit_label(param, fit_lookup[param]),
        )
    style_axis(ax)
    ax.set_title("Single-parameter fits miss HGPS shape", pad=7)
    build_single_parameter_legend(ax, fit_lookup)
    fig.tight_layout(pad=0.75)
    fig.savefig(PANEL_B_PNG_PATH, dpi=300)
    fig.savefig(PANEL_B_PDF_PATH)
    plt.close(fig)


def build_single_parameter_legend(ax: plt.Axes, fit_lookup: dict[str, dict[str, object]]) -> None:
    handles = [
        Line2D([0], [0], color="#111111", lw=2.7, label="HGPS patients"),
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=PARAMETER_COLORS["eta"], lw=2.5, label=single_fit_label("eta", fit_lookup["eta"])),
        Line2D([0], [0], color=PARAMETER_COLORS["beta"], lw=2.5, label=single_fit_label("beta", fit_lookup["beta"])),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=PARAMETER_COLORS["Xc"], lw=2.5, label=single_fit_label("Xc", fit_lookup["Xc"])),
        Line2D([0], [0], color=PARAMETER_COLORS["epsilon"], lw=2.5, label=single_fit_label("epsilon", fit_lookup["epsilon"])),
    ]
    legend = ax.legend(
        handles=handles,
        frameon=False,
        handlelength=2.1,
        labelspacing=0.16,
        borderpad=0.0,
        fontsize=7.4,
        loc="upper right",
    )
    for idx, text in enumerate(legend.get_texts()):
        if idx in (1, 4):
            text.set_fontweight("bold")


def single_fit_label(param: str, fit: dict[str, object]) -> str:
    factor = float(fit["factors"][param])
    return f"{PARAMETER_SHORT_LABELS[param]} x{factor:.2g}"


def make_panel_c(
    hgps: HGPSData,
    curves: pd.DataFrame,
    envelopes: pd.DataFrame,
    fits: list[dict[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    draw_hgps(ax, hgps)
    fit_lookup = {fit["fit_name"]: fit for fit in fits}
    for fit_params in PAIR_FITS:
        name = fit_name(fit_params)
        draw_model_fit(
            ax=ax,
            curves=curves,
            envelopes=envelopes,
            fit_name_value=name,
            color=PAIR_COLORS[name],
            label=pair_label(fit_params, fit_lookup[name]),
            lw=2.2 if fit_params != ("Xc", "epsilon") else 2.7,
        )
    style_axis(ax)
    ax.set_title("Two-parameter fits require senogenic change", pad=7)
    build_pair_legend(ax, fit_lookup)
    fig.tight_layout(pad=0.75)
    fig.savefig(PANEL_C_PNG_PATH, dpi=300)
    fig.savefig(PANEL_C_PDF_PATH)
    plt.close(fig)


def build_pair_legend(ax: plt.Axes, fit_lookup: dict[str, dict[str, object]]) -> None:
    handles = [
        Line2D([0], [0], color="#111111", lw=2.7, label="HGPS patients"),
        Line2D([], [], color="none", label="Without senogenic parameter"),
        Line2D(
            [0],
            [0],
            color=PAIR_COLORS["Xc+epsilon"],
            lw=2.7,
            label=pair_label(("Xc", "epsilon"), fit_lookup["Xc+epsilon"]),
        ),
        Line2D([], [], color="none", label="With senogenic parameter"),
    ]
    for fit_params in PAIR_FITS[1:]:
        name = fit_name(fit_params)
        handles.append(Line2D([0], [0], color=PAIR_COLORS[name], lw=2.3, label=pair_label(fit_params, fit_lookup[name])))

    legend = ax.legend(
        handles=handles,
        frameon=False,
        handlelength=2.0,
        labelspacing=0.12,
        borderpad=0.0,
        ncol=1,
        fontsize=6.9,
        loc="upper right",
    )
    for idx, text in enumerate(legend.get_texts()):
        if idx in (1, 3):
            text.set_fontweight("bold")


def pair_label(fit_params: tuple[str, ...], fit: dict[str, object]) -> str:
    factors = fit["factors"]
    parts = [f"{PARAMETER_SHORT_LABELS[param]} x{float(factors[param]):.2g}" for param in fit_params]
    return " + ".join(parts)


def make_composite(
    hgps: HGPSData,
    period_curves: list[PeriodSurvival],
    curves: pd.DataFrame,
    envelopes: pd.DataFrame,
    fits: list[dict[str, object]],
) -> None:
    fig = plt.figure(figsize=(14.2, 4.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[0.92, 1.16, 1.22], wspace=0.36)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])
    fit_lookup = {fit["fit_name"]: fit for fit in fits}

    hgps_median = median_lifespan_from_curve(hgps.times, hgps.survival)
    draw_hgps(ax_a, hgps, x_scale=hgps_median)
    draw_period_survival(ax_a, period_curves)
    style_axis(
        ax_a,
        xlim=(0, 1.95),
        xlabel="Age / median lifespan",
        xticks=[0, 0.5, 1.0, 1.5],
    )
    ax_a.set_title("HGPS survival is shallow relative to controls", pad=8)
    ax_a.legend(frameon=False, loc="upper right", handlelength=2.2, fontsize=7.2)

    draw_hgps(ax_b, hgps)
    for param in ("eta", "beta", "Xc", "epsilon"):
        draw_model_fit(
            ax_b,
            curves,
            envelopes,
            param,
            PARAMETER_COLORS[param],
            single_fit_label(param, fit_lookup[param]),
            lw=2.2,
        )
    style_axis(ax_b)
    ax_b.set_title("Single-parameter fits miss HGPS shape", pad=8)
    build_single_parameter_legend(ax_b, fit_lookup)

    draw_hgps(ax_c, hgps)
    for fit_params in PAIR_FITS:
        name = fit_name(fit_params)
        draw_model_fit(ax_c, curves, envelopes, name, PAIR_COLORS[name], pair_label(fit_params, fit_lookup[name]), lw=2.1)
    style_axis(ax_c)
    ax_c.set_title("Two-parameter fits require senogenic change", pad=8)
    build_pair_legend(ax_c, fit_lookup)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")
    fig.savefig(COMPOSITE_PNG_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(COMPOSITE_PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.05,
        1.16,
        label,
        transform=ax.transAxes,
        fontsize=18,
        fontfamily="Arial",
        fontweight="normal",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def update_output_index() -> None:
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
            existing_rows = list(csv.DictReader(handle))

    artifacts = [
        ("figure", PANEL_A_PNG_PATH, "PNG preview of Fig6a HGPS survival"),
        ("figure", PANEL_A_PDF_PATH, "Vector PDF of Fig6a HGPS survival"),
        ("figure", PANEL_B_PNG_PATH, "PNG preview of Fig6b single-parameter HGPS fits"),
        ("figure", PANEL_B_PDF_PATH, "Vector PDF of Fig6b single-parameter HGPS fits"),
        ("figure", PANEL_C_PNG_PATH, "PNG preview of Fig6c two-parameter HGPS fits"),
        ("figure", PANEL_C_PDF_PATH, "Vector PDF of Fig6c two-parameter HGPS fits"),
        ("figure", COMPOSITE_PNG_PATH, "PNG preview of the Fig6 progeria composite"),
        ("figure", COMPOSITE_PDF_PATH, "Vector PDF of the Fig6 progeria composite"),
        ("csv", HGPS_ENVELOPE_PATH, "HGPS survival curve and bootstrap envelope"),
        ("csv", FIT_RESULTS_PATH, "Fitted parameter values and costs"),
        ("csv", SURVIVAL_CURVES_PATH, "Central model survival curves"),
        ("csv", MODEL_ENVELOPES_PATH, "Model fit-CI survival envelopes"),
        ("csv", PERIOD_SURVIVAL_PATH, "Sweden 2019 period survival curve for Fig6a"),
    ]
    rows = []
    for artifact_type, path, description in artifacts:
        rows.append(
            {
                "date": "2026-05-20",
                "task": "fig6_progeria",
                "artifact_type": artifact_type,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "source_script": "src/figures/Fig6_progeria/make_fig6_progeria.py",
                "input_paths": (
                    "saved_results/progeria_data.pkl; "
                    "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json; "
                    "saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv"
                ),
                "description": description,
                "notes": "Sweden 2019 baseline with fixed Xc heterogeneity; HGPS bootstrap samples n=202 from smooth survival; model bands use Sweden fit-CI endpoint envelopes",
            }
        )

    paths = {row["path"] for row in rows}
    kept_rows = [row for row in existing_rows if row.get("path") not in paths]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_sweden_baseline()
    ci_bounds = load_sweden_ci_bounds()
    hgps = load_hgps_data(args.bootstrap_n)
    period_curves = load_period_survival_curves()
    fits = fit_all_models(hgps, baseline, args)
    curves, envelopes = load_or_build_curves(hgps, baseline, ci_bounds, fits, args)

    make_panel_a(hgps, period_curves)
    make_panel_b(hgps, curves, envelopes, fits)
    make_panel_c(hgps, curves, envelopes, fits)
    make_composite(hgps, period_curves, curves, envelopes, fits)
    update_output_index()

    print(f"saved {COMPOSITE_PNG_PATH}")


if __name__ == "__main__":
    main()
