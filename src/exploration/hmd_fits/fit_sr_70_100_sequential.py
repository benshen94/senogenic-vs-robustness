"""
70-100-focused sequential SR fit for USA and Sweden 2019 period HMD curves.

Workflow:
1. Fit USA first, including eta and beta.
2. Reuse the USA eta and beta for Sweden.
3. Fit Sweden using only epsilon, Xc, Xc heterogeneity, and h_ext.

The objective is intentionally concentrated on ages 70-100, because this script
is for visual hazard agreement in the old-age region rather than a broad
all-adult compromise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.SR_models.hazard_only import SRHazardSim
from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.utils.sr_fitter import _interpolate_on_times, _normalize_survival_to_start


OUTPUT_DIR = PROJECT_ROOT / "saved_results"
SUMMARY_PATH = OUTPUT_DIR / "sequential_70_100_usa_sweden_sr_fit_2019_summary.json"
PLOT_PATH = OUTPUT_DIR / "sequential_70_100_usa_sweden_sr_fit_2019.png"

AGE_START = 30
AGE_END = 100
FOCUS_START = 70
FOCUS_END = 100

OPT_N = 35_000
FINAL_N = 140_000
DT = 0.025
RUN_POWELL_POLISH = False

USA_COORDINATE_BEST = np.array([0.08, 0.08, 0.09, 0.0, -0.36, -0.135])

BASELINE = {
    "eta": 0.54,
    "beta": 54.75,
    "kappa": 0.5,
    "epsilon": 51.83,
    "Xc": 21.0,
    "xc_std_frac": 0.20,
}


@dataclass(frozen=True)
class CountryData:
    country: str
    ages_hazard: np.ndarray
    hazard: np.ndarray
    ages_survival: np.ndarray
    survival: np.ndarray
    h_ext: float


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

    fit_survival_ages, fit_survival = _normalize_survival_to_start(
        np.asarray(ages_survival[survival_mask], dtype=float),
        np.asarray(survival[survival_mask], dtype=float),
        start_time=AGE_START,
    )

    ggm = hmd.fit_ggm(2019, age_start=20, age_end=100)
    h_ext = float(ggm["m"])
    if not np.isfinite(h_ext) or h_ext < 0:
        h_ext = 0.0

    return CountryData(
        country=country,
        ages_hazard=np.asarray(ages_hazard[hazard_mask], dtype=float),
        hazard=np.asarray(hazard[hazard_mask], dtype=float),
        ages_survival=fit_survival_ages,
        survival=fit_survival,
        h_ext=h_ext,
    )


def clipped_normal(mean: float, std_frac: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = mean * std_frac
    values = rng.normal(mean, std, size=n)

    bad = values <= 0
    while np.any(bad):
        values[bad] = rng.normal(mean, std, size=int(bad.sum()))
        bad = values <= 0

    return values


def simulate(
    data: CountryData,
    params: dict[str, float],
    n: int,
    seed: int,
    parallel: bool = True,
) -> dict[str, np.ndarray]:
    sim = SRHazardSim(
        n=n,
        eta=params["eta"],
        beta=params["beta"],
        kappa=BASELINE["kappa"],
        epsilon=params["epsilon"],
        Xc=clipped_normal(params["Xc"], params["xc_std_frac"], n, seed),
        h_ext=params["h_ext"],
        tmax=112,
        dt=DT,
        parallel=parallel,
        break_early=True,
        random_seed=seed + 10_000,
        chunk_size=10_000,
    )

    hazard = _interpolate_on_times(sim.tspan_hazard, sim.hazard, data.ages_hazard)
    survival = _interpolate_on_times(sim.tspan_survival, sim.survival, data.ages_survival)
    survival = np.clip(survival, 1e-12, 1.0)
    survival = survival / survival[0]

    return {"hazard": np.maximum(hazard, 1e-12), "survival": survival}


def focus_mask(ages: np.ndarray) -> np.ndarray:
    return (ages >= FOCUS_START) & (ages <= FOCUS_END)


def old_age_score(data: CountryData, fit: dict[str, np.ndarray]) -> float:
    hazard_mask = focus_mask(data.ages_hazard)
    survival_mask = focus_mask(data.ages_survival)

    hazard_residual = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])
    survival_residual = fit["survival"][survival_mask] - data.survival[survival_mask]

    age_weights = np.ones(np.sum(hazard_mask), dtype=float)
    old_tail = data.ages_hazard[hazard_mask] >= 85
    age_weights[old_tail] = 1.8
    age_weights = age_weights / age_weights.sum()

    hazard_score = float(np.sum(age_weights * hazard_residual**2))
    survival_score = float(np.mean(survival_residual**2))
    return hazard_score + 12.0 * survival_score


def params_from_usa_vector(vector: Iterable[float], usa_h_ext: float) -> dict[str, float]:
    eta_log2, beta_log2, eps_log2, xc_log2, std_log2, h_ext_log2 = np.asarray(vector, dtype=float)
    return {
        "eta": BASELINE["eta"] * 2.0**eta_log2,
        "beta": BASELINE["beta"] * 2.0**beta_log2,
        "epsilon": BASELINE["epsilon"] * 2.0**eps_log2,
        "Xc": BASELINE["Xc"] * 2.0**xc_log2,
        "xc_std_frac": BASELINE["xc_std_frac"] * 2.0**std_log2,
        "h_ext": usa_h_ext * 2.0**h_ext_log2,
    }


def params_from_country_vector(
    vector: Iterable[float],
    shared: dict[str, float],
    country_h_ext: float,
) -> dict[str, float]:
    eps_log2, xc_log2, std_log2, h_ext_log2 = np.asarray(vector, dtype=float)
    return {
        "eta": shared["eta"],
        "beta": shared["beta"],
        "epsilon": BASELINE["epsilon"] * 2.0**eps_log2,
        "Xc": BASELINE["Xc"] * 2.0**xc_log2,
        "xc_std_frac": BASELINE["xc_std_frac"] * 2.0**std_log2,
        "h_ext": country_h_ext * 2.0**h_ext_log2,
    }


def bounded(vector: np.ndarray, bounds: tuple[tuple[float, float], ...]) -> np.ndarray:
    low = np.asarray([bound[0] for bound in bounds], dtype=float)
    high = np.asarray([bound[1] for bound in bounds], dtype=float)
    return np.clip(vector, low, high)


def coordinate_polish(objective, start, bounds, step_sizes) -> np.ndarray:
    current = bounded(np.asarray(start, dtype=float), bounds)
    current_score = float(objective(current))
    print(f"start score: {current_score:.4f}")

    for step_size in step_sizes:
        improved = True
        while improved:
            improved = False
            best_vector = current.copy()
            best_score = current_score

            for dim in range(current.size):
                for direction in (-1.0, 1.0):
                    trial = current.copy()
                    trial[dim] += direction * step_size
                    trial = bounded(trial, bounds)
                    if np.allclose(trial, current):
                        continue

                    score = float(objective(trial))
                    print(f"  step={step_size:.3f} dim={dim} dir={direction:+.0f} score={score:.4f}")
                    if score >= best_score:
                        continue

                    best_score = score
                    best_vector = trial
                    improved = True

            if improved:
                current = best_vector
                current_score = best_score
                print(f"accepted score: {current_score:.4f}, vector={current.tolist()}")

    return current


def fit_usa(usa: CountryData) -> tuple[np.ndarray, dict[str, float]]:
    bounds = (
        (-0.35, 0.35),  # eta
        (-0.35, 0.35),  # beta
        (-0.45, 0.45),  # epsilon
        (-0.45, 0.45),  # Xc
        (-0.60, 0.60),  # Xc heterogeneity
        (-1.00, 1.00),  # h_ext
    )

    def objective(vector):
        params = params_from_usa_vector(vector, usa.h_ext)
        fit = simulate(usa, params, n=OPT_N, seed=20260711)
        return old_age_score(usa, fit)

    starts = [
        USA_COORDINATE_BEST,
        np.zeros(6),
        np.array([0.0, 0.0, 0.0, 0.0, 0.075, 0.0]),
    ]

    polished = []
    for start in starts:
        print("USA start", start.tolist())
        vector = coordinate_polish(objective, start, bounds, step_sizes=(0.18, 0.09, 0.045))
        polished.append((objective(vector), vector))

    polished.sort(key=lambda item: item[0])
    best_vector = polished[0][1]

    if RUN_POWELL_POLISH:
        from scipy.optimize import minimize

        def scipy_objective(vector):
            return objective(bounded(vector, bounds))

        result = minimize(
            scipy_objective,
            best_vector,
            method="Powell",
            options={"maxiter": 35, "xtol": 0.03, "ftol": 0.02, "disp": True},
        )
        best_vector = bounded(result.x, bounds)

    return best_vector, params_from_usa_vector(best_vector, usa.h_ext)


def fit_country_with_shared_eta_beta(
    data: CountryData,
    shared: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    bounds = (
        (-0.50, 0.50),  # epsilon
        (-0.50, 0.50),  # Xc
        (-0.75, 0.75),  # Xc heterogeneity
        (-1.25, 1.25),  # h_ext
    )

    def objective(vector):
        params = params_from_country_vector(vector, shared, data.h_ext)
        fit = simulate(data, params, n=OPT_N, seed=20260712)
        return old_age_score(data, fit)

    starts = [
        np.zeros(4),
        np.array([0.0, 0.0, -0.15, 0.0]),
        np.array([0.0, 0.15, 0.30, -0.30]),
        np.array([0.15, 0.0, 0.0, 0.0]),
        np.array([-0.15, 0.0, 0.0, 0.0]),
    ]

    polished = []
    for start in starts:
        print(data.country, "start", start.tolist())
        vector = coordinate_polish(objective, start, bounds, step_sizes=(0.20, 0.10, 0.05))
        polished.append((objective(vector), vector))

    polished.sort(key=lambda item: item[0])
    best_vector = polished[0][1]

    if RUN_POWELL_POLISH:
        from scipy.optimize import minimize

        def scipy_objective(vector):
            return objective(bounded(vector, bounds))

        result = minimize(
            scipy_objective,
            best_vector,
            method="Powell",
            options={"maxiter": 30, "xtol": 0.03, "ftol": 0.02, "disp": True},
        )
        best_vector = bounded(result.x, bounds)

    return best_vector, params_from_country_vector(best_vector, shared, data.h_ext)


def fit_metrics(data: CountryData, fit: dict[str, np.ndarray]) -> dict[str, float]:
    hazard_mask = focus_mask(data.ages_hazard)
    survival_mask = focus_mask(data.ages_survival)
    log_error = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])

    return {
        "hazard_log_rmse_70_100": float(np.sqrt(np.mean(log_error**2))),
        "hazard_median_fold_error_70_100": float(np.exp(np.median(np.abs(log_error)))),
        "survival_rmse_70_100": float(
            np.sqrt(np.mean((fit["survival"][survival_mask] - data.survival[survival_mask]) ** 2))
        ),
    }


def plot_results(data_by_country, fits) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    order = [("SWE", "Sweden"), ("USA", "USA")]

    for row, (country, label) in enumerate(order):
        data = data_by_country[country]
        fit = fits[country]["fit"]

        survival_ax = axes[row, 0]
        hazard_ax = axes[row, 1]

        survival_ax.plot(data.ages_survival, data.survival, "o", ms=3, label=f"{label} HMD")
        survival_ax.plot(data.ages_survival, fit["survival"], lw=2.5, label="SR fit")
        survival_ax.axvspan(FOCUS_START, FOCUS_END, color="0.92", zorder=-1)
        survival_ax.set_ylabel(f"{label}\nSurvival from age {AGE_START}")
        survival_ax.grid(alpha=0.25)
        survival_ax.legend(frameon=False)

        hazard_ax.plot(data.ages_hazard, data.hazard, "o", ms=3, label=f"{label} HMD")
        hazard_ax.plot(data.ages_hazard, fit["hazard"], lw=2.5, label="SR fit")
        hazard_ax.axvspan(FOCUS_START, FOCUS_END, color="0.92", zorder=-1)
        hazard_ax.set_yscale("log")
        hazard_ax.set_ylabel("Hazard [1/year]")
        hazard_ax.grid(alpha=0.25)
        hazard_ax.legend(frameon=False)

    axes[0, 0].set_title("Survival")
    axes[0, 1].set_title("Hazard")
    axes[1, 0].set_xlabel("Age [years]")
    axes[1, 1].set_xlabel("Age [years]")
    fig.suptitle("Sequential SR fit focused on ages 70-100: Sweden and USA 2019 period")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_by_country = {country: load_country_data(country) for country in ("USA", "SWE")}

    usa_vector, usa_params = fit_usa(data_by_country["USA"])
    shared = {"eta": usa_params["eta"], "beta": usa_params["beta"]}
    swe_vector, swe_params = fit_country_with_shared_eta_beta(data_by_country["SWE"], shared)

    params_by_country = {"USA": usa_params, "SWE": swe_params}
    fits = {}
    for index, country in enumerate(("USA", "SWE")):
        fit = simulate(
            data_by_country[country],
            params_by_country[country],
            n=FINAL_N,
            seed=20260720 + index,
        )
        fits[country] = {
            "fit": fit,
            "metrics": fit_metrics(data_by_country[country], fit),
        }

    plot_results(data_by_country, fits)

    summary = {
        "workflow": "USA first; Sweden fitted after fixing USA eta and beta.",
        "focus_age_window": [FOCUS_START, FOCUS_END],
        "baseline": BASELINE,
        "vectors_log2": {
            "USA": usa_vector.tolist(),
            "SWE": swe_vector.tolist(),
        },
        "fitted_parameters": params_by_country,
        "metrics": {
            country: fits[country]["metrics"]
            for country in ("SWE", "USA")
        },
        "plot_path": str(PLOT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["fitted_parameters"], indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
