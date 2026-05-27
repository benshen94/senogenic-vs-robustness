"""
Fit Sweden historical HMD curves with fixed eta and beta.

Targets:
- Sweden cohort 1900
- Sweden cohort 1920
- Sweden period 1900

Fixed from the current Sweden-tail/USA-refit model:
- eta
- beta
- kappa

Fitted per target:
- epsilon
- Xc
- Xc fractional heterogeneity
- h_ext, the external mortality term
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


OUTPUT_DIR = PROJECT_ROOT / "saved_results" / "sweden_historical_fixed_eta_beta"

COUNTRY = "SWE"
GENDER = "both"
AGE_START = 30
AGE_END = 100
FOCUS_START = 65
FOCUS_END = 100
DT = 0.025
OPT_N = 100_000
FINAL_N = 250_000
CI_N = 200_000

FIXED = {
    "eta": 0.5868368257640714,
    "beta": 57.87173772073557,
    "kappa": 0.5,
}

BASELINE = {
    "epsilon": 49.718659304628446,
    "Xc": 21.74056340066893,
    "xc_std_frac": 0.14142135623730953,
    "h_ext": 0.001,
}

PARAMETER_NAMES = (
    "epsilon_log2",
    "xc_log2",
    "xc_std_log2",
    "h_ext_log2",
)

BOUNDS = (
    (-0.70, 0.70),  # epsilon
    (-0.60, 0.70),  # Xc
    (-1.20, 0.90),  # Xc heterogeneity
    (-8.00, 3.00),  # h_ext
)

TARGETS = (
    ("swe_cohort_1900", "cohort", 1900, "Sweden 1900 cohort"),
    ("swe_cohort_1920", "cohort", 1920, "Sweden 1920 cohort"),
    ("swe_period_1900", "period", 1900, "Sweden 1900 period"),
)


@dataclass(frozen=True)
class TargetData:
    name: str
    data_type: str
    year: int
    label: str
    ages_hazard: np.ndarray
    hazard: np.ndarray
    ages_survival: np.ndarray
    survival: np.ndarray


def load_target(name: str, data_type: str, year: int, label: str) -> TargetData:
    hmd = HMD(COUNTRY, GENDER, data_type)
    ages_hazard, hazard = hmd.get_hazard(year, haz_type="mx", strict=True)
    ages_survival, survival = hmd.get_survival(year, strict=True)

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

    return TargetData(
        name=name,
        data_type=data_type,
        year=year,
        label=label,
        ages_hazard=np.asarray(ages_hazard[hazard_mask], dtype=float),
        hazard=np.asarray(hazard[hazard_mask], dtype=float),
        ages_survival=survival_ages,
        survival=survival_values,
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


def vector_to_params(vector: Iterable[float]) -> dict[str, float]:
    values = dict(zip(PARAMETER_NAMES, np.asarray(vector, dtype=float)))
    return {
        **FIXED,
        "epsilon": BASELINE["epsilon"] * 2.0 ** values["epsilon_log2"],
        "Xc": BASELINE["Xc"] * 2.0 ** values["xc_log2"],
        "xc_std_frac": BASELINE["xc_std_frac"] * 2.0 ** values["xc_std_log2"],
        "h_ext": BASELINE["h_ext"] * 2.0 ** values["h_ext_log2"],
    }


def simulate_target(data: TargetData, params: dict[str, float], n: int, seed: int) -> dict[str, np.ndarray]:
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
    data: TargetData,
    n: int,
    seed: int,
    include_survival: bool = True,
) -> np.ndarray:
    params = vector_to_params(vector)
    fit = simulate_target(data, params, n=n, seed=seed)

    hazard_mask = focus_mask(data.ages_hazard)
    hazard_ages = data.ages_hazard[hazard_mask]
    log_hazard_residual = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])

    hazard_weights = np.ones_like(log_hazard_residual)
    hazard_weights[hazard_ages >= 80] = 1.5
    hazard_weights[hazard_ages >= 90] = 2.5
    hazard_weights = hazard_weights / np.mean(hazard_weights)
    residuals = [np.sqrt(hazard_weights) * log_hazard_residual]

    if include_survival:
        survival_mask = focus_mask(data.ages_survival)
        survival_residual = fit["survival"][survival_mask] - data.survival[survival_mask]
        residuals.append(np.sqrt(10.0) * survival_residual)

    return np.concatenate(residuals)


def score_vector(vector: Iterable[float], data: TargetData, n: int, seed: int) -> float:
    residuals = residual_vector(vector, data, n=n, seed=seed)
    return float(np.mean(residuals**2))


def clip_vector(vector: Iterable[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    lower = np.asarray([bound[0] for bound in BOUNDS], dtype=float)
    upper = np.asarray([bound[1] for bound in BOUNDS], dtype=float)
    return np.clip(vector, lower, upper)


def coordinate_search(
    data: TargetData,
    start_vector: Iterable[float],
    step_sizes: tuple[float, ...],
    seed: int,
) -> tuple[np.ndarray, float]:
    current = clip_vector(start_vector)
    current_score = score_vector(current, data, n=OPT_N, seed=seed)
    print(f"{data.name} start score={current_score:.5f}, vector={current.tolist()}")

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

                    score = score_vector(trial, data, n=OPT_N, seed=seed)
                    print(f"  {data.name} step={step_size:.3f} {name} {direction:+.0f}: {score:.5f}")

                    if score >= best_score:
                        continue

                    best_vector = trial
                    best_score = score
                    improved = True

            if improved:
                current = best_vector
                current_score = best_score
                print(f"{data.name} accepted score={current_score:.5f}, vector={current.tolist()}")

    return current, current_score


def fit_target(data: TargetData, target_index: int) -> tuple[np.ndarray, float]:
    starts = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-0.15, -0.05, 0.20, 1.0]),
        np.array([0.15, 0.10, 0.35, 2.0]),
        np.array([-0.30, -0.10, 0.50, 0.0]),
    ]

    seed = 20260513 + target_index * 1_000
    candidates = []
    for start in starts:
        vector, score = coordinate_search(data, start, step_sizes=(0.16, 0.08, 0.04, 0.02), seed=seed)
        candidates.append((score, vector))

    candidates.sort(key=lambda item: item[0])
    best_vector = candidates[0][1]
    final_score = score_vector(best_vector, data, n=FINAL_N, seed=seed + 100)
    print(f"{data.name} final-resolution score={final_score:.5f}, vector={best_vector.tolist()}")
    return best_vector, final_score


def evaluate_fit(data: TargetData, vector: Iterable[float], n: int, seed: int) -> dict[str, object]:
    params = vector_to_params(vector)
    fit = simulate_target(data, params, n=n, seed=seed)
    return {"params": params, "fit": fit, "metrics": metrics_for_target(data, fit)}


def metrics_for_target(data: TargetData, fit: dict[str, np.ndarray]) -> dict[str, float]:
    hazard_mask = focus_mask(data.ages_hazard)
    survival_mask = focus_mask(data.ages_survival)
    log_error = np.log(fit["hazard"][hazard_mask]) - np.log(data.hazard[hazard_mask])

    return {
        f"hazard_log_rmse_{FOCUS_START}_{FOCUS_END}": float(np.sqrt(np.mean(log_error**2))),
        f"hazard_median_fold_error_{FOCUS_START}_{FOCUS_END}": float(np.exp(np.median(np.abs(log_error)))),
        f"survival_rmse_{FOCUS_START}_{FOCUS_END}": float(
            np.sqrt(np.mean((fit["survival"][survival_mask] - data.survival[survival_mask]) ** 2))
        ),
    }


def jacobian_ci(data: TargetData, vector: np.ndarray, seed: int, delta: float = 0.035) -> list[dict[str, float | str]]:
    base_residuals = residual_vector(vector, data, n=CI_N, seed=seed, include_survival=True)
    jacobian = np.zeros((base_residuals.size, vector.size), dtype=float)

    for dim, name in enumerate(PARAMETER_NAMES):
        plus = vector.copy()
        minus = vector.copy()
        plus[dim] += delta
        minus[dim] -= delta
        plus = clip_vector(plus)
        minus = clip_vector(minus)

        plus_residuals = residual_vector(plus, data, n=CI_N, seed=seed)
        minus_residuals = residual_vector(minus, data, n=CI_N, seed=seed)
        width = plus[dim] - minus[dim]
        if width <= 0:
            jacobian[:, dim] = np.nan
            continue

        jacobian[:, dim] = (plus_residuals - minus_residuals) / width
        print(f"{data.name} CI Jacobian column done: {name}")

    valid_columns = np.all(np.isfinite(jacobian), axis=0)
    usable_jacobian = jacobian[:, valid_columns]
    dof = max(base_residuals.size - int(np.sum(valid_columns)), 1)
    sigma2 = float(np.sum(base_residuals**2) / dof)
    covariance = sigma2 * np.linalg.pinv(usable_jacobian.T @ usable_jacobian)
    se_log2_valid = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    se_log2 = np.full(vector.size, np.nan, dtype=float)
    se_log2[valid_columns] = se_log2_valid
    return build_ci_rows(vector, se_log2)


def build_ci_rows(vector: np.ndarray, se_log2: np.ndarray) -> list[dict[str, float | str]]:
    rows = []
    natural_names = ("epsilon", "Xc", "xc_std_frac", "h_ext")
    baseline_values = (
        BASELINE["epsilon"],
        BASELINE["Xc"],
        BASELINE["xc_std_frac"],
        BASELINE["h_ext"],
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


def write_ci_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    header = ["parameter", "estimate", "ci95_lower", "ci95_upper", "log2_estimate", "log2_se"]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[name]) for name in header))
    path.write_text("\n".join(lines) + "\n")


def plot_target(data: TargetData, result: dict[str, object], path: Path) -> None:
    fit = result["fit"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    axes[0].plot(data.ages_survival, data.survival, "o", ms=3, label="HMD")
    axes[0].plot(data.ages_survival, fit["survival"], lw=2.5, label="SR fit")
    axes[0].axvspan(FOCUS_START, FOCUS_END, color="0.92", zorder=-1)
    axes[0].set_title("Survival")
    axes[0].set_ylabel(f"Survival from age {AGE_START}")
    axes[0].set_xlabel("Age [years]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(data.ages_hazard, data.hazard, "o", ms=3, label="HMD")
    axes[1].plot(data.ages_hazard, fit["hazard"], lw=2.5, label="SR fit")
    axes[1].axvspan(FOCUS_START, FOCUS_END, color="0.92", zorder=-1)
    axes[1].set_yscale("log")
    axes[1].set_title("Hazard")
    axes[1].set_ylabel("Hazard [1/year]")
    axes[1].set_xlabel("Age [years]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle(f"{data.label}: fixed eta and beta")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_methods(path: Path) -> None:
    path.write_text(
        f"""# Sweden Historical SR Fits With Fixed Eta and Beta

## Data

The fits used local Human Mortality Database life tables for Sweden, both sexes.
The fitted targets were Sweden cohort 1900, Sweden cohort 1920, and Sweden
period 1900. Hazard targets used central death rates, \\(m_x\\), over ages
{FOCUS_START}-{FOCUS_END}. Survival curves used \\(l_x\\), normalized to
survival from age {AGE_START}.

## Model Constraint

The stochastic repair model fixed \\(\\eta = {FIXED["eta"]}\\),
\\(\\beta = {FIXED["beta"]}\\), and \\(\\kappa = {FIXED["kappa"]}\\) from the
2019 Sweden-tail/USA-refit model. For each Sweden historical target, the fitted
parameters were \\(\\epsilon\\), \\(X_c\\), fractional Gaussian heterogeneity in
\\(X_c\\), and the external mortality term \\(h_{{ext}}\\).

## Objective

During optimization, each candidate parameter vector was simulated with
\\(n={OPT_N:,}\\) particles and \\(dt={DT}\\) years. The objective minimized
weighted residuals over ages {FOCUS_START}-{FOCUS_END}. Hazard residuals were
computed on the log scale:

\\[
r_h(a) = \\log h_{{SR}}(a) - \\log h_{{HMD}}(a).
\\]

Additional hazard weight was applied for ages \\(a \\ge 80\\), with stronger
weight for ages \\(a \\ge 90\\). Survival residuals over the same age window
were included as a secondary regularizer:

\\[
r_S(a) = S_{{SR}}(a) - S_{{HMD}}(a).
\\]

Final reported curves were re-simulated with \\(n={FINAL_N:,}\\) particles.

## Confidence Intervals

Approximate 95% confidence intervals were estimated from local curvature of the
residual surface in log2-parameter space. Each fitted parameter was perturbed by
\\(\\Delta = 0.035\\) in log2 units with common random seeds, and a numerical
Jacobian was computed by central differences. The covariance approximation was:

\\[
\\widehat{{\\mathrm{{Cov}}}}(\\theta) =
\\hat\\sigma^2 (J^T J)^+.
\\]

These are fitting-criterion uncertainty intervals and do not include all
possible HMD sampling or model misspecification uncertainty.
"""
    )


def build_summary(
    results: dict[str, dict[str, object]],
    ci_by_target: dict[str, list[dict[str, float | str]]],
) -> dict[str, object]:
    return {
        "constraint": "fixed eta, beta, kappa; fit epsilon, Xc, Xc heterogeneity, and h_ext per Sweden historical target",
        "fixed": FIXED,
        "baseline": BASELINE,
        "focus_age_window": [FOCUS_START, FOCUS_END],
        "opt_n": OPT_N,
        "final_n": FINAL_N,
        "ci_n": CI_N,
        "targets": {
            name: {
                "label": result["data"].label,
                "country": COUNTRY,
                "gender": GENDER,
                "data_type": result["data"].data_type,
                "year": result["data"].year,
                "best_vector_log2": result["vector"].tolist(),
                "fitted_parameters": result["result"]["params"],
                "metrics": result["result"]["metrics"],
                "ci": ci_by_target[name],
                "plot_path": str(result["plot_path"]),
                "ci_path": str(result["ci_path"]),
            }
            for name, result in results.items()
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    methods_path = OUTPUT_DIR / "methods.md"
    write_methods(methods_path)

    results: dict[str, dict[str, object]] = {}
    ci_by_target: dict[str, list[dict[str, float | str]]] = {}

    for target_index, target in enumerate(TARGETS):
        data = load_target(*target)
        vector, _ = fit_target(data, target_index)
        result = evaluate_fit(data, vector, n=FINAL_N, seed=20260613 + target_index * 1_000)
        ci_rows = jacobian_ci(data, vector, seed=20260713 + target_index * 1_000)

        plot_path = OUTPUT_DIR / f"{data.name}.png"
        ci_path = OUTPUT_DIR / f"{data.name}_ci.csv"
        plot_target(data, result, plot_path)
        write_ci_csv(ci_path, ci_rows)

        results[data.name] = {
            "data": data,
            "vector": vector,
            "result": result,
            "plot_path": plot_path,
            "ci_path": ci_path,
        }
        ci_by_target[data.name] = ci_rows

        print(data.name)
        print(json.dumps(result["params"], indent=2))
        print(json.dumps(result["metrics"], indent=2))

    summary = build_summary(results, ci_by_target)
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")
    print(f"Saved methods: {methods_path}")


if __name__ == "__main__":
    main()
