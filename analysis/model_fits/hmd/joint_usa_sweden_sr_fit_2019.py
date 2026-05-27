"""
Joint SR fit for USA and Sweden 2019 period HMD curves.

The fit uses one shared eta and beta for both countries. Each country gets its
own epsilon, Xc, and Xc heterogeneity. The objective fits log hazard with extra
weight on ages 60-90, and also keeps survival close enough to reject hazard-only
solutions that look wrong in survival space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.SR_models.hazard_only import SRHazardSim
from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.utils.sr_fitter import (
    _interpolate_on_times,
    _normalize_survival_to_start,
)


OUTPUT_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = OUTPUT_DIR / "joint_usa_sweden_sr_fit_2019_summary.json"
PLOT_PATH = OUTPUT_DIR / "joint_usa_sweden_sr_fit_2019.png"

COUNTRIES = ("SWE", "USA")
COUNTRY_LABELS = {"SWE": "Sweden", "USA": "USA"}

AGE_START = 30
AGE_END = 100
HAZARD_IMPORTANT_START = 60
HAZARD_IMPORTANT_END = 90
FIT_DT = 0.025
FIT_N = 50_000
FINAL_N = 90_000
SURVIVAL_SCORE_WEIGHT = 30.0

BASELINE = {
    "eta": 0.54,
    "beta": 54.75,
    "kappa": 0.5,
    "epsilon": 51.83,
    "Xc": 21.0,
    "xc_std_frac": 0.20,
}

PARAMETER_NAMES = (
    "eta_log2",
    "beta_log2",
    "swe_epsilon_log2",
    "swe_xc_log2",
    "swe_xc_std_log2",
    "swe_h_ext_log2",
    "usa_epsilon_log2",
    "usa_xc_log2",
    "usa_xc_std_log2",
    "usa_h_ext_log2",
)

BOUNDS = (
    (-0.45, 0.45),  # eta
    (-0.45, 0.45),  # beta
    (-0.60, 0.60),  # Sweden epsilon
    (-0.60, 0.60),  # Sweden Xc
    (-0.75, 0.75),  # Sweden Xc heterogeneity
    (-2.00, 2.00),  # Sweden m_ex / h_ext
    (-0.60, 0.60),  # USA epsilon
    (-0.60, 0.60),  # USA Xc
    (-0.75, 0.75),  # USA Xc heterogeneity
    (-2.00, 2.00),  # USA m_ex / h_ext
)

REFERENCE_VECTORS = {
    "baseline": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "hazard_focused": (0.0, 0.0, 0.0, 0.0, -0.15, 0.0, 0.0, 0.0, 0.075, 0.0),
    "hazard_focused_low_swe_m": (0.0, 0.0, 0.0, 0.0, -0.15, -0.3, 0.0, 0.0, 0.075, 0.0),
}


@dataclass(frozen=True)
class CountryData:
    country: str
    ages_hazard: np.ndarray
    hazard: np.ndarray
    hazard_weights: np.ndarray
    ages_survival: np.ndarray
    survival: np.ndarray
    survival_weights: np.ndarray
    h_ext: float


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    clean = np.asarray(weights, dtype=float)
    clean = np.where(np.isfinite(clean), clean, 0.0)
    clean = np.maximum(clean, 0.0)

    total = float(clean.sum())
    if total <= 0:
        return np.full(clean.size, 1.0 / clean.size)

    return clean / total


def load_country_data(country: str) -> CountryData:
    hmd = HMD(country, "both", "period")

    ages_hazard, hazard = hmd.get_hazard(2019, haz_type="mx", strict=True)
    ages_survival, survival = hmd.get_survival(2019, strict=True)

    hazard_mask = (
        (ages_hazard >= AGE_START)
        & (ages_hazard <= AGE_END)
        & np.isfinite(hazard)
        & (hazard > 0)
    )
    survival_mask = (
        (ages_survival >= AGE_START)
        & (ages_survival <= AGE_END)
        & np.isfinite(survival)
        & (survival > 0)
    )

    fit_hazard_ages = np.asarray(ages_hazard[hazard_mask], dtype=float)
    fit_hazard = np.asarray(hazard[hazard_mask], dtype=float)
    fit_survival_ages = np.asarray(ages_survival[survival_mask], dtype=float)
    fit_survival = np.asarray(survival[survival_mask], dtype=float)
    fit_survival_ages, fit_survival = _normalize_survival_to_start(
        fit_survival_ages,
        fit_survival,
        start_time=AGE_START,
    )

    hazard_weights = np.ones_like(fit_hazard_ages, dtype=float)
    important = (
        (fit_hazard_ages >= HAZARD_IMPORTANT_START)
        & (fit_hazard_ages <= HAZARD_IMPORTANT_END)
    )
    hazard_weights[important] = 6.0

    survival_weights = np.ones_like(fit_survival_ages, dtype=float)
    survival_weights[fit_survival_ages > 90] = 0.35

    ggm = hmd.fit_ggm(2019, age_start=20, age_end=100)
    h_ext = float(ggm["m"])
    if not np.isfinite(h_ext) or h_ext < 0:
        h_ext = 0.0

    return CountryData(
        country=country,
        ages_hazard=fit_hazard_ages,
        hazard=fit_hazard,
        hazard_weights=normalize_weights(hazard_weights),
        ages_survival=fit_survival_ages,
        survival=fit_survival,
        survival_weights=normalize_weights(survival_weights),
        h_ext=h_ext,
    )


def build_xc_vector(mean_xc: float, std_frac: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = float(mean_xc) * float(std_frac)
    values = rng.normal(loc=mean_xc, scale=std, size=n)

    bad = values <= 0
    while np.any(bad):
        values[bad] = rng.normal(loc=mean_xc, scale=std, size=int(bad.sum()))
        bad = values <= 0

    return values


def vector_to_parameters(vector: Iterable[float]) -> dict[str, object]:
    values = dict(zip(PARAMETER_NAMES, np.asarray(vector, dtype=float)))

    eta = BASELINE["eta"] * 2.0 ** values["eta_log2"]
    beta = BASELINE["beta"] * 2.0 ** values["beta_log2"]

    return {
        "shared": {
            "eta": float(eta),
            "beta": float(beta),
            "kappa": BASELINE["kappa"],
        },
        "SWE": {
            "epsilon": float(BASELINE["epsilon"] * 2.0 ** values["swe_epsilon_log2"]),
            "Xc": float(BASELINE["Xc"] * 2.0 ** values["swe_xc_log2"]),
            "xc_std_frac": float(BASELINE["xc_std_frac"] * 2.0 ** values["swe_xc_std_log2"]),
            "h_ext_factor": float(2.0 ** values["swe_h_ext_log2"]),
        },
        "USA": {
            "epsilon": float(BASELINE["epsilon"] * 2.0 ** values["usa_epsilon_log2"]),
            "Xc": float(BASELINE["Xc"] * 2.0 ** values["usa_xc_log2"]),
            "xc_std_frac": float(BASELINE["xc_std_frac"] * 2.0 ** values["usa_xc_std_log2"]),
            "h_ext_factor": float(2.0 ** values["usa_h_ext_log2"]),
        },
    }


def simulate_country(
    data: CountryData,
    params: dict[str, object],
    n: int,
    seed: int,
    dt: float,
    parallel: bool,
) -> dict[str, np.ndarray]:
    country_params = params[data.country]
    shared = params["shared"]

    sr_params = {
        "eta": shared["eta"],
        "beta": shared["beta"],
        "kappa": shared["kappa"],
        "epsilon": country_params["epsilon"],
        "Xc": build_xc_vector(
            mean_xc=country_params["Xc"],
            std_frac=country_params["xc_std_frac"],
            n=n,
            seed=seed,
        ),
    }
    sim = SRHazardSim(
        n=n,
        eta=sr_params["eta"],
        beta=sr_params["beta"],
        kappa=sr_params["kappa"],
        epsilon=sr_params["epsilon"],
        Xc=sr_params["Xc"],
        h_ext=data.h_ext * country_params["h_ext_factor"],
        tmax=112,
        dt=dt,
        parallel=parallel,
        break_early=True,
        random_seed=seed + 10_000,
        chunk_size=10_000,
    )

    fitted_hazard = _interpolate_on_times(sim.tspan_hazard, sim.hazard, data.ages_hazard)
    fitted_survival = _interpolate_on_times(sim.tspan_survival, sim.survival, data.ages_survival)
    fitted_survival = np.clip(fitted_survival, 1e-12, 1.0)
    fitted_survival = fitted_survival / fitted_survival[0]

    return {
        "hazard": np.maximum(fitted_hazard, 1e-12),
        "survival": fitted_survival,
    }


def country_score(data: CountryData, fit: dict[str, np.ndarray]) -> float:
    hazard_residual = np.log(fit["hazard"]) - np.log(data.hazard)
    survival_residual = fit["survival"] - data.survival

    hazard_score = float(np.sum(data.hazard_weights * hazard_residual**2))
    survival_score = float(np.sum(data.survival_weights * survival_residual**2))
    return hazard_score + SURVIVAL_SCORE_WEIGHT * survival_score


def joint_score(
    vector: Iterable[float],
    data_by_country: dict[str, CountryData],
    n: int,
    dt: float,
    parallel: bool,
) -> float:
    params = vector_to_parameters(vector)
    total = 0.0

    for index, country in enumerate(COUNTRIES):
        fit = simulate_country(
            data=data_by_country[country],
            params=params,
            n=n,
            seed=20260512 + index,
            dt=dt,
            parallel=parallel,
        )
        total += country_score(data_by_country[country], fit)

    return total


def benchmark_parallel_mode(data_by_country: dict[str, CountryData]) -> bool:
    import time

    vector = np.zeros(len(PARAMETER_NAMES), dtype=float)
    timings = {}
    for parallel in (False, True):
        start = time.perf_counter()
        _ = joint_score(
            vector=vector,
            data_by_country=data_by_country,
            n=FIT_N,
            dt=FIT_DT,
            parallel=parallel,
        )
        timings[parallel] = time.perf_counter() - start
        print(f"Benchmark parallel={parallel}: {timings[parallel]:.2f} seconds")

    return timings[True] < timings[False]


def clip_vector(vector: np.ndarray) -> np.ndarray:
    lower = np.array([bound[0] for bound in BOUNDS], dtype=float)
    upper = np.array([bound[1] for bound in BOUNDS], dtype=float)
    return np.clip(vector, lower, upper)


def coordinate_search(
    start_vector: np.ndarray,
    data_by_country: dict[str, CountryData],
    parallel: bool,
) -> np.ndarray:
    current = clip_vector(start_vector)
    current_score = joint_score(
        current,
        data_by_country=data_by_country,
        n=FIT_N,
        dt=FIT_DT,
        parallel=parallel,
    )
    print(f"Initial score: {current_score:.4f}")

    for step_size in (0.30, 0.15, 0.075):
        improved = True
        while improved:
            improved = False
            best_vector = current.copy()
            best_score = current_score

            for dim, name in enumerate(PARAMETER_NAMES):
                for direction in (-1.0, 1.0):
                    trial = current.copy()
                    trial[dim] += direction * step_size
                    trial = clip_vector(trial)
                    if np.allclose(trial, current):
                        continue

                    score = joint_score(
                        trial,
                        data_by_country=data_by_country,
                        n=FIT_N,
                        dt=FIT_DT,
                        parallel=parallel,
                    )
                    print(f"step={step_size:.3f} {name} {direction:+.0f}: {score:.4f}")

                    if score >= best_score:
                        continue

                    best_vector = trial
                    best_score = score
                    improved = True

            if improved:
                current = best_vector
                current_score = best_score
                print(f"Accepted score: {current_score:.4f}")

    return current


def fit_parameters(data_by_country: dict[str, CountryData], parallel: bool) -> np.ndarray:
    print("Stage 1: coordinate search at n=50000")
    baseline = np.zeros(len(PARAMETER_NAMES), dtype=float)
    coordinate_vector = coordinate_search(
        start_vector=baseline,
        data_by_country=data_by_country,
        parallel=parallel,
    )
    return choose_best_final_vector(
        coordinate_vector=coordinate_vector,
        data_by_country=data_by_country,
        parallel=parallel,
    )


def vector_selection_score(
    vector: np.ndarray,
    data_by_country: dict[str, CountryData],
    parallel: bool,
) -> float:
    params = vector_to_parameters(vector)
    total = 0.0

    for index, country in enumerate(COUNTRIES):
        data = data_by_country[country]
        fit = simulate_country(
            data=data,
            params=params,
            n=FINAL_N,
            seed=20260612 + index,
            dt=FIT_DT,
            parallel=parallel,
        )
        log_error = np.log(fit["hazard"]) - np.log(data.hazard)
        important = (
            (data.ages_hazard >= HAZARD_IMPORTANT_START)
            & (data.ages_hazard <= HAZARD_IMPORTANT_END)
        )
        hazard_rmse = float(np.sqrt(np.mean(log_error[important] ** 2)))
        survival_rmse = float(np.sqrt(np.mean((fit["survival"] - data.survival) ** 2)))
        total += hazard_rmse + 2.0 * survival_rmse

    return total


def choose_best_final_vector(
    coordinate_vector: np.ndarray,
    data_by_country: dict[str, CountryData],
    parallel: bool,
) -> np.ndarray:
    candidates = {"coordinate": coordinate_vector}
    for name, vector in REFERENCE_VECTORS.items():
        candidates[name] = np.asarray(vector, dtype=float)

    scored = []
    for name, vector in candidates.items():
        score = vector_selection_score(vector, data_by_country, parallel)
        scored.append((score, name, vector))
        print(f"Final-resolution candidate {name}: {score:.4f}")

    scored.sort(key=lambda item: item[0])
    print(f"Selected final candidate: {scored[0][1]}")
    return scored[0][2]


def evaluate_final_fit(
    best_vector: np.ndarray,
    data_by_country: dict[str, CountryData],
    parallel: bool,
) -> dict[str, object]:
    params = vector_to_parameters(best_vector)
    country_results = {}

    for index, country in enumerate(COUNTRIES):
        fit = simulate_country(
            data=data_by_country[country],
            params=params,
            n=FINAL_N,
            seed=20260612 + index,
            dt=FIT_DT,
            parallel=parallel,
        )
        country_results[country] = {
            "score": country_score(data_by_country[country], fit),
            "fit": fit,
        }

    return {
        "vector": best_vector.tolist(),
        "params": params,
        "country_results": country_results,
    }


def plot_final_fit(
    final: dict[str, object],
    data_by_country: dict[str, CountryData],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for row, country in enumerate(("SWE", "USA")):
        data = data_by_country[country]
        fit = final["country_results"][country]["fit"]
        label = COUNTRY_LABELS[country]

        survival_ax = axes[row, 0]
        hazard_ax = axes[row, 1]

        survival_ax.plot(data.ages_survival, data.survival, "o", ms=3, label=f"{label} HMD")
        survival_ax.plot(data.ages_survival, fit["survival"], lw=2.5, label="SR fit")
        survival_ax.set_ylabel(f"{label}\nSurvival from age {AGE_START}")
        survival_ax.grid(alpha=0.25)
        survival_ax.legend(frameon=False)

        hazard_ax.plot(data.ages_hazard, data.hazard, "o", ms=3, label=f"{label} HMD")
        hazard_ax.plot(data.ages_hazard, fit["hazard"], lw=2.5, label="SR fit")
        hazard_ax.axvspan(60, 90, color="0.90", zorder=-1)
        hazard_ax.set_yscale("log")
        hazard_ax.set_ylabel("Hazard [1/year]")
        hazard_ax.grid(alpha=0.25)
        hazard_ax.legend(frameon=False)

    axes[0, 0].set_title("Survival")
    axes[0, 1].set_title("Hazard")
    axes[1, 0].set_xlabel("Age [years]")
    axes[1, 1].set_xlabel("Age [years]")

    fig.suptitle("Joint SR fit: Sweden and USA 2019 period, both genders")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(final: dict[str, object], data_by_country: dict[str, CountryData]) -> dict[str, object]:
    summary = {
        "objective": (
            "Jointly fit Sweden and USA HMD 2019 period both-gender hazard curves "
            "with shared eta and beta, separate epsilon, Xc, and Xc heterogeneity."
        ),
        "baseline": BASELINE,
        "age_window": [AGE_START, AGE_END],
        "extra_hazard_weight_window": [HAZARD_IMPORTANT_START, HAZARD_IMPORTANT_END],
        "parameter_names": PARAMETER_NAMES,
        "best_vector_log2_folds": final["vector"],
        "fitted_parameters": final["params"],
        "countries": {},
        "plot_path": str(PLOT_PATH),
    }

    for country in COUNTRIES:
        data = data_by_country[country]
        fit = final["country_results"][country]["fit"]
        log_error_60_90 = np.log(fit["hazard"]) - np.log(data.hazard)
        important = (
            (data.ages_hazard >= HAZARD_IMPORTANT_START)
            & (data.ages_hazard <= HAZARD_IMPORTANT_END)
        )

        summary["countries"][country] = {
            "h_ext_initial_from_ggm": data.h_ext,
            "h_ext_fitted": data.h_ext * summary["fitted_parameters"][country]["h_ext_factor"],
            "joint_score": final["country_results"][country]["score"],
            "hazard_log_rmse_60_90": float(np.sqrt(np.mean(log_error_60_90[important] ** 2))),
            "hazard_median_fold_error_60_90": float(np.exp(np.median(np.abs(log_error_60_90[important])))),
            "survival_rmse": float(np.sqrt(np.mean((fit["survival"] - data.survival) ** 2))),
        }

    return summary


def main() -> None:
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}

    use_parallel = benchmark_parallel_mode(data_by_country)
    print(f"Using parallel={use_parallel} for the fit")

    best_vector = fit_parameters(data_by_country, parallel=use_parallel)
    final = evaluate_final_fit(best_vector, data_by_country, parallel=use_parallel)
    plot_final_fit(final, data_by_country)

    summary = build_summary(final, data_by_country)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["fitted_parameters"], indent=2))
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
