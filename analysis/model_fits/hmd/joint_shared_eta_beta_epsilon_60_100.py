"""
Joint SR fit for USA and Sweden 2019 period HMD curves, focused on ages 60-100.

Constraint:
- eta, beta, epsilon are shared.
- Sweden and USA differ only in Xc and Xc heterogeneity.
- kappa is fixed at 0.5.
- h_ext is fixed per country from an HMD/GGM Makeham fit and is not an SR fit
  degree of freedom here.

The script also estimates local curvature-based confidence intervals for the
fitted parameters in log2-parameter space and converts them to natural units.
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
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.SR_models.hazard_only import SRHazardSim
from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.utils.sr_fitter import _interpolate_on_times, _normalize_survival_to_start


OUTPUT_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = OUTPUT_DIR / "joint_shared_eta_beta_epsilon_60_100_fit_2019_summary.json"
PLOT_PATH = OUTPUT_DIR / "joint_shared_eta_beta_epsilon_60_100_fit_2019.png"
CI_PATH = OUTPUT_DIR / "joint_shared_eta_beta_epsilon_60_100_fit_2019_ci.csv"
METHODS_PATH = OUTPUT_DIR / "joint_shared_eta_beta_epsilon_60_100_methods.md"

COUNTRIES = ("SWE", "USA")
COUNTRY_LABELS = {"SWE": "Sweden", "USA": "USA"}

AGE_START = 30
AGE_END = 100
FOCUS_START = 60
FOCUS_END = 100
DT = 0.025
OPT_N = 50_000
FINAL_N = 160_000
CI_N = 160_000

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
    "epsilon_log2",
    "swe_xc_log2",
    "swe_xc_std_log2",
    "usa_xc_log2",
    "usa_xc_std_log2",
)

BOUNDS = (
    (-0.35, 0.35),  # eta
    (-0.35, 0.35),  # beta
    (-0.35, 0.35),  # epsilon
    (-0.45, 0.55),  # Sweden Xc
    (-0.80, 0.60),  # Sweden Xc heterogeneity
    (-0.55, 0.45),  # USA Xc
    (-0.80, 0.70),  # USA Xc heterogeneity
)


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

    survival_ages, survival_values = _normalize_survival_to_start(
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
        ages_survival=survival_ages,
        survival=survival_values,
        h_ext=h_ext,
    )


def positive_normal(mean: float, std_frac: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = mean * std_frac
    values = rng.normal(mean, std, size=n)

    bad = values <= 0
    while np.any(bad):
        values[bad] = rng.normal(mean, std, size=int(bad.sum()))
        bad = values <= 0

    return values


def vector_to_params(vector: Iterable[float], h_ext_by_country: dict[str, float]) -> dict[str, dict[str, float]]:
    values = dict(zip(PARAMETER_NAMES, np.asarray(vector, dtype=float)))

    shared = {
        "eta": BASELINE["eta"] * 2.0 ** values["eta_log2"],
        "beta": BASELINE["beta"] * 2.0 ** values["beta_log2"],
        "epsilon": BASELINE["epsilon"] * 2.0 ** values["epsilon_log2"],
        "kappa": BASELINE["kappa"],
    }

    return {
        "shared": shared,
        "SWE": {
            **shared,
            "Xc": BASELINE["Xc"] * 2.0 ** values["swe_xc_log2"],
            "xc_std_frac": BASELINE["xc_std_frac"] * 2.0 ** values["swe_xc_std_log2"],
            "h_ext": h_ext_by_country["SWE"],
        },
        "USA": {
            **shared,
            "Xc": BASELINE["Xc"] * 2.0 ** values["usa_xc_log2"],
            "xc_std_frac": BASELINE["xc_std_frac"] * 2.0 ** values["usa_xc_std_log2"],
            "h_ext": h_ext_by_country["USA"],
        },
    }


def simulate_country(
    data: CountryData,
    params: dict[str, float],
    n: int,
    seed: int,
) -> dict[str, np.ndarray]:
    sim = SRHazardSim(
        n=n,
        eta=params["eta"],
        beta=params["beta"],
        kappa=params["kappa"],
        epsilon=params["epsilon"],
        Xc=positive_normal(params["Xc"], params["xc_std_frac"], n, seed),
        h_ext=params["h_ext"],
        tmax=112,
        dt=DT,
        parallel=True,
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


def residual_vector(
    vector: Iterable[float],
    data_by_country: dict[str, CountryData],
    n: int,
    seed_base: int,
    include_survival: bool = True,
) -> np.ndarray:
    h_ext_by_country = {country: data_by_country[country].h_ext for country in COUNTRIES}
    params = vector_to_params(vector, h_ext_by_country)
    residuals: list[np.ndarray] = []

    for index, country in enumerate(COUNTRIES):
        data = data_by_country[country]
        fit = simulate_country(data, params[country], n=n, seed=seed_base + index)

        hazard_mask = focus_mask(data.ages_hazard)
        hazard_ages = data.ages_hazard[hazard_mask]
        log_hazard_residual = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])

        hazard_weights = np.ones_like(log_hazard_residual)
        hazard_weights[hazard_ages >= 85] = 1.4
        hazard_weights = hazard_weights / np.mean(hazard_weights)
        residuals.append(np.sqrt(hazard_weights) * log_hazard_residual)

        if include_survival:
            survival_mask = focus_mask(data.ages_survival)
            survival_residual = fit["survival"][survival_mask] - data.survival[survival_mask]
            residuals.append(np.sqrt(10.0) * survival_residual)

    return np.concatenate(residuals)


def score_vector(vector: Iterable[float], data_by_country: dict[str, CountryData], n: int, seed_base: int) -> float:
    residuals = residual_vector(vector, data_by_country, n=n, seed_base=seed_base)
    return float(np.mean(residuals**2))


def clip_vector(vector: Iterable[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    lower = np.asarray([bound[0] for bound in BOUNDS], dtype=float)
    upper = np.asarray([bound[1] for bound in BOUNDS], dtype=float)
    return np.clip(vector, lower, upper)


def coordinate_search(
    data_by_country: dict[str, CountryData],
    start_vector: Iterable[float],
    step_sizes: tuple[float, ...],
) -> tuple[np.ndarray, float]:
    current = clip_vector(start_vector)
    current_score = score_vector(current, data_by_country, n=OPT_N, seed_base=20260513)
    print(f"start score={current_score:.5f}, vector={current.tolist()}")

    for step_size in step_sizes:
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

                    score = score_vector(trial, data_by_country, n=OPT_N, seed_base=20260513)
                    print(f"  step={step_size:.3f} {name} {direction:+.0f}: {score:.5f}")

                    if score >= best_score:
                        continue

                    best_vector = trial
                    best_score = score
                    improved = True

            if improved:
                current = best_vector
                current_score = best_score
                print(f"accepted score={current_score:.5f}, vector={current.tolist()}")

    return current, current_score


def fit_parameters(data_by_country: dict[str, CountryData]) -> tuple[np.ndarray, float]:
    starts = [
        np.zeros(len(PARAMETER_NAMES)),
        np.array([0.08, 0.08, 0.09, 0.05, -0.35, 0.00, -0.36]),
        np.array([0.08, 0.08, 0.09, 0.10, -0.45, -0.05, -0.30]),
        np.array([0.00, 0.00, 0.00, 0.10, -0.25, -0.05, -0.10]),
    ]

    candidates = []
    for start in starts:
        print("Fit start", start.tolist())
        vector, score = coordinate_search(data_by_country, start, step_sizes=(0.18, 0.09, 0.045))
        candidates.append((score, vector))

    candidates.sort(key=lambda item: item[0])
    best_vector = candidates[0][1]

    final_score = score_vector(best_vector, data_by_country, n=FINAL_N, seed_base=20260613)
    print(f"final-resolution score={final_score:.5f}, vector={best_vector.tolist()}")
    return best_vector, final_score


def evaluate_fit(
    vector: Iterable[float],
    data_by_country: dict[str, CountryData],
    n: int,
    seed_base: int,
) -> dict[str, object]:
    h_ext_by_country = {country: data_by_country[country].h_ext for country in COUNTRIES}
    params = vector_to_params(vector, h_ext_by_country)
    results: dict[str, object] = {"params": params, "countries": {}}

    for index, country in enumerate(COUNTRIES):
        data = data_by_country[country]
        fit = simulate_country(data, params[country], n=n, seed=seed_base + index)
        results["countries"][country] = {"fit": fit, "metrics": metrics_for_country(data, fit)}

    return results


def metrics_for_country(data: CountryData, fit: dict[str, np.ndarray]) -> dict[str, float]:
    hazard_mask = focus_mask(data.ages_hazard)
    survival_mask = focus_mask(data.ages_survival)
    log_error = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])

    return {
        "hazard_log_rmse_60_100": float(np.sqrt(np.mean(log_error**2))),
        "hazard_median_fold_error_60_100": float(np.exp(np.median(np.abs(log_error)))),
        "survival_rmse_60_100": float(
            np.sqrt(np.mean((fit["survival"][survival_mask] - data.survival[survival_mask]) ** 2))
        ),
    }


def jacobian_ci(
    vector: np.ndarray,
    data_by_country: dict[str, CountryData],
    delta: float = 0.035,
) -> list[dict[str, float | str]]:
    base_residuals = residual_vector(
        vector,
        data_by_country,
        n=CI_N,
        seed_base=20260713,
        include_survival=True,
    )
    jacobian = np.zeros((base_residuals.size, vector.size), dtype=float)

    for dim, name in enumerate(PARAMETER_NAMES):
        plus = vector.copy()
        minus = vector.copy()
        plus[dim] += delta
        minus[dim] -= delta
        plus = clip_vector(plus)
        minus = clip_vector(minus)

        plus_residuals = residual_vector(plus, data_by_country, n=CI_N, seed_base=20260713)
        minus_residuals = residual_vector(minus, data_by_country, n=CI_N, seed_base=20260713)
        width = plus[dim] - minus[dim]
        if width <= 0:
            jacobian[:, dim] = np.nan
            continue

        jacobian[:, dim] = (plus_residuals - minus_residuals) / width
        print(f"CI Jacobian column done: {name}")

    valid_columns = np.all(np.isfinite(jacobian), axis=0)
    usable_jacobian = jacobian[:, valid_columns]
    dof = max(base_residuals.size - int(np.sum(valid_columns)), 1)
    sigma2 = float(np.sum(base_residuals**2) / dof)
    covariance = sigma2 * np.linalg.pinv(usable_jacobian.T @ usable_jacobian)
    se_log2_valid = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    se_log2 = np.full(vector.size, np.nan, dtype=float)
    se_log2[valid_columns] = se_log2_valid

    return build_ci_rows(vector, se_log2)


def natural_parameter_values(vector: Iterable[float]) -> dict[str, float]:
    h_ext_dummy = {"SWE": 0.0, "USA": 0.0}
    params = vector_to_params(vector, h_ext_dummy)
    return {
        "eta": params["shared"]["eta"],
        "beta": params["shared"]["beta"],
        "epsilon": params["shared"]["epsilon"],
        "SWE_Xc": params["SWE"]["Xc"],
        "SWE_xc_std_frac": params["SWE"]["xc_std_frac"],
        "USA_Xc": params["USA"]["Xc"],
        "USA_xc_std_frac": params["USA"]["xc_std_frac"],
    }


def build_ci_rows(vector: np.ndarray, se_log2: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    natural_names = (
        "eta",
        "beta",
        "epsilon",
        "SWE_Xc",
        "SWE_xc_std_frac",
        "USA_Xc",
        "USA_xc_std_frac",
    )
    baseline_values = (
        BASELINE["eta"],
        BASELINE["beta"],
        BASELINE["epsilon"],
        BASELINE["Xc"],
        BASELINE["xc_std_frac"],
        BASELINE["Xc"],
        BASELINE["xc_std_frac"],
    )

    for idx, name in enumerate(natural_names):
        estimate = baseline_values[idx] * 2.0 ** vector[idx]
        lower = baseline_values[idx] * 2.0 ** (vector[idx] - 1.96 * se_log2[idx])
        upper = baseline_values[idx] * 2.0 ** (vector[idx] + 1.96 * se_log2[idx])
        rows.append(
            {
                "parameter": name,
                "estimate": float(estimate),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "log2_estimate": float(vector[idx]),
                "log2_se": float(se_log2[idx]),
            }
        )

    return rows


def write_ci_csv(rows: list[dict[str, float | str]]) -> None:
    header = ["parameter", "estimate", "ci95_lower", "ci95_upper", "log2_estimate", "log2_se"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in header))
    CI_PATH.write_text("\n".join(lines) + "\n")


def plot_results(results: dict[str, object], data_by_country: dict[str, CountryData]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    for row, country in enumerate(("SWE", "USA")):
        data = data_by_country[country]
        fit = results["countries"][country]["fit"]
        label = COUNTRY_LABELS[country]

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
    fig.suptitle("Joint SR fit focused on ages 60-100: shared eta, beta, epsilon")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_methods_markdown(ci_rows: list[dict[str, float | str]]) -> None:
    METHODS_PATH.write_text(
        f"""# Joint SR Fit Methods: USA and Sweden 2019 Period Curves

## Data

Period life table data for 2019 were loaded from the local Human Mortality
Database files using `HMD(country, "both", "period")`. The fitted hazard target
was the both-sex central death rate, `mx`, for ages {FOCUS_START}-{FOCUS_END}.
Survival curves were read from `lx` and normalized to survival from age {AGE_START}.

## Model Constraint

The stochastic repair model used fixed \\(\\kappa = {BASELINE["kappa"]}\\). The
parameters \\(\\eta\\), \\(\\beta\\), and \\(\\epsilon\\) were constrained to be
identical for USA and Sweden. Country-specific variation was allowed only for
the threshold mean \\(X_c\\) and the Gaussian fractional heterogeneity of
\\(X_c\\). The country-specific external Makeham term, \\(h_{{ext}}\\), was fixed
from a Gamma-Gompertz-Makeham fit to each country's HMD hazard curve and was not
optimized as an SR parameter in this fit.

## Objective Function

For each candidate parameter vector, SR hazard and survival curves were
simulated with \\(n={OPT_N:,}\\) particles and \\(dt={DT}\\) years during
optimization. The objective minimized weighted residuals over ages
{FOCUS_START}-{FOCUS_END}. Hazard residuals were computed on the log scale:

\\[
r_h(a) = \\log h_{{SR}}(a) - \\log h_{{HMD}}(a).
\\]

A mild extra weight was applied to ages \\(a \\ge 85\\) within the old-age window.
Survival residuals over the same age window were included as a secondary
regularizer so that the fitted hazards also reproduced cumulative survival:

\\[
r_S(a) = S_{{SR}}(a) - S_{{HMD}}(a).
\\]

The final reported curves were re-simulated with \\(n={FINAL_N:,}\\) particles.

## Confidence Intervals

Approximate 95% confidence intervals were estimated from the local curvature of
the fitted residual surface in log2-parameter space. Around the final parameter
vector, each fitted parameter was perturbed by \\(\\Delta = 0.035\\) in log2
units, using common random seeds for the plus and minus simulations. A numerical
Jacobian \\(J\\) of the residual vector was computed by central differences.
The covariance matrix was approximated as:

\\[
\\widehat{{\\mathrm{{Cov}}}}(\\theta) =
\\hat\\sigma^2 (J^T J)^+,
\\]

where \\((J^T J)^+\\) is the Moore-Penrose pseudoinverse and
\\(\\hat\\sigma^2\\) is the residual variance divided by residual degrees of
freedom. Intervals were computed in log2 space and transformed back to natural
parameter units. These intervals are curvature-based uncertainty intervals for
the fitting criterion; they do not include all possible HMD sampling or model
misspecification uncertainty.

## CI Output

The numeric CI table is saved at `{CI_PATH}`.
"""
    )


def build_summary(
    vector: np.ndarray,
    results: dict[str, object],
    ci_rows: list[dict[str, float | str]],
    data_by_country: dict[str, CountryData],
) -> dict[str, object]:
    return {
        "constraint": "shared eta, beta, epsilon; country-specific Xc and Xc heterogeneity only",
        "focus_age_window": [FOCUS_START, FOCUS_END],
        "baseline": BASELINE,
        "parameter_names": PARAMETER_NAMES,
        "best_vector_log2": vector.tolist(),
        "fitted_parameters": natural_parameter_values(vector),
        "fixed_h_ext": {country: data_by_country[country].h_ext for country in COUNTRIES},
        "metrics": {country: results["countries"][country]["metrics"] for country in COUNTRIES},
        "ci": ci_rows,
        "plot_path": str(PLOT_PATH),
        "ci_path": str(CI_PATH),
        "methods_path": str(METHODS_PATH),
    }


def main() -> None:
    data_by_country = {country: load_country_data(country) for country in COUNTRIES}
    best_vector, _ = fit_parameters(data_by_country)
    results = evaluate_fit(best_vector, data_by_country, n=FINAL_N, seed_base=20260613)
    ci_rows = jacobian_ci(best_vector, data_by_country)

    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_results(results, data_by_country)
    write_ci_csv(ci_rows)
    write_methods_markdown(ci_rows)

    summary = build_summary(best_vector, results, ci_rows, data_by_country)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["fitted_parameters"], indent=2))
    print(json.dumps(summary["fixed_h_ext"], indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Saved summary: {SUMMARY_PATH}")
    print(f"Saved CI table: {CI_PATH}")
    print(f"Saved methods markdown: {METHODS_PATH}")
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
