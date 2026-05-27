"""
SR fitter built around the full SR simulator.

The fitter searches around the Karin baseline parameters and supports two entry
points:

- `fit_sr_to_hmd`
- `fit_sr_to_arrays`

The default setup matches the user's requested workflow:

- fit ages 30-100
- `dt = 0.025`
- fixed `h_ext` seeded from an MGG fit
- fixed `kappa`
- fixed Gaussian `Xc` heterogeneity at 18%
- hazard objective on log hazard
- survival objective on linear survival
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from ..mortality_data_analysis.HMD_lifetables import HMD
from .sr_fits import build_sr_fit_record, save_sr_fit_record
from .sr_utils import create_sr_simulation, load_baseline_human_params_dict


FIT_PARAM_ORDER = ("eta", "beta", "epsilon", "Xc", "xc_std_frac")
CORE_FIT_PARAMS = ("eta", "beta", "epsilon", "Xc")
DEFAULT_HAZARD_OBJECTIVES = ("log_rmse", "log_mae", "log_huber")
DEFAULT_SURVIVAL_OBJECTIVES = ("linear_rmse", "linear_mae", "ks")
EVAL_SIMULATION_SEED = 20260401
XC_VECTOR_SEED = 20260402
EPS = 1e-12


@dataclass(frozen=True)
class SRBenchmarkCase:
    country: str
    gender: str
    data_type: str
    year: int
    label: str


@dataclass(frozen=True)
class SRFitConfig:
    age_start: int = 30
    age_end: int = 100
    dt: float = 0.025
    tmax: float = 110.0
    fit_params: tuple[str, ...] = CORE_FIT_PARAMS
    fit_kappa: bool = False
    xc_std_frac: float = 0.18
    fit_xc_std: bool = False
    h_ext_mode: str = "fixed_from_mgg"
    stage1_screen_size: int = 6
    stage1_hazard_n: int = 12_000
    stage1_survival_n: int = 3_000
    stage3_hazard_n: int = 50_000
    stage3_survival_n: int = 6_000
    stage2_top_k: int = 2
    stage3_top_k: int = 1
    stage2_step_sizes: tuple[float, ...] = (0.25, 0.125)
    stage4_step_sizes: tuple[float, ...] = (0.0625,)
    hazard_objective: str = "log_rmse"
    survival_objective: str = "linear_rmse"
    hazard_objective_candidates: tuple[str, ...] = DEFAULT_HAZARD_OBJECTIVES
    survival_objective_candidates: tuple[str, ...] = DEFAULT_SURVIVAL_OBJECTIVES
    save_dir: str | None = None
    initial_vectors: tuple[tuple[float, ...], ...] | None = None
    parallel_simulation: bool | None = None


@dataclass
class SRTarget:
    target: str
    times: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    label: str


@dataclass
class SRFitResult:
    case_label: str
    target: str
    objective_name: str
    score: float
    baseline_score: float
    cross_target_score: float | None
    baseline_cross_target_score: float | None
    h_ext: float
    fit_params: Dict[str, float]
    scalar_params: Dict[str, float]
    heterogeneity: Dict[str, Any]
    config: SRFitConfig
    candidate_vector: List[float]
    bound_hits: List[str]
    accepted_by_metrics: bool
    target_curve: Dict[str, List[float]]
    fitted_curve: Dict[str, List[float]]
    baseline_curve: Dict[str, List[float]]
    cross_target_curve: Dict[str, List[float]] | None
    cross_fitted_curve: Dict[str, List[float]] | None = None
    cross_baseline_curve: Dict[str, List[float]] | None = None
    fitted_summary_path: str | None = None
    fitted_plot_path: str | None = None
    saved_fit_name: str | None = None


DEFAULT_BENCHMARK_CASES = (
    SRBenchmarkCase("USA", "both", "period", 2019, "USA period 2019"),
    SRBenchmarkCase("DAN", "both", "cohort", 1920, "DAN cohort 1920"),
    SRBenchmarkCase("ENG", "both", "period", 2010, "ENG period 2010"),
)


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_save_dir(config: SRFitConfig) -> Path:
    if config.save_dir is not None:
        return Path(config.save_dir).expanduser().resolve()

    return _root_dir() / "saved_results" / "sr_fitter"


def _baseline_scalar_params() -> Dict[str, float]:
    raw = load_baseline_human_params_dict()
    baseline: Dict[str, float] = {}
    for name, value in raw.items():
        if np.isscalar(value):
            baseline[name] = float(value)
            continue

        baseline[name] = float(np.asarray(value, dtype=float).ravel()[0])

    return baseline


def _prepare_fit_params(config: SRFitConfig) -> tuple[str, ...]:
    names = list(config.fit_params)

    if config.fit_kappa and "kappa" not in names:
        names.append("kappa")

    if config.fit_xc_std and "xc_std_frac" not in names:
        names.append("xc_std_frac")

    return tuple(names)


def _resolve_objective_name(target: str, config: SRFitConfig) -> str:
    if target == "hazard":
        return config.hazard_objective

    return config.survival_objective


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    weights_array = np.where(np.isfinite(weights_array), weights_array, 0.0)
    weights_array = np.maximum(weights_array, 0.0)

    if weights_array.size == 0:
        return weights_array

    total = float(np.sum(weights_array))
    if total <= 0:
        return np.full(weights_array.size, 1.0 / weights_array.size, dtype=float)

    return weights_array / total


def _clip_positive(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return np.maximum(array, EPS)


def _resample_negative_gaussian(
    rng: np.random.Generator,
    mean: float,
    std: float,
    size: int,
) -> np.ndarray:
    values = rng.normal(loc=mean, scale=std, size=size)

    if std <= 0:
        return np.full(size, max(mean, EPS), dtype=float)

    negative_mask = values <= 0
    while np.any(negative_mask):
        values[negative_mask] = rng.normal(
            loc=mean,
            scale=std,
            size=int(np.sum(negative_mask)),
        )
        negative_mask = values <= 0

    return values


def _build_xc_vector(mean_xc: float, std_frac: float, n: int) -> np.ndarray:
    std = float(std_frac) * float(mean_xc)
    rng = np.random.default_rng(XC_VECTOR_SEED)
    return _resample_negative_gaussian(rng=rng, mean=float(mean_xc), std=std, size=int(n))


def _vector_to_params(
    vector: Sequence[float],
    fit_params: Sequence[str],
    baseline: Dict[str, float],
    config: SRFitConfig,
    n: int,
    h_ext: float,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    scalar_params = dict(baseline)
    scalar_params["h_ext"] = float(h_ext)
    scalar_params["xc_std_frac"] = float(config.xc_std_frac)

    for name, log2_fold in zip(fit_params, vector):
        if name == "xc_std_frac":
            scalar_params[name] = float(config.xc_std_frac) * (2.0 ** float(log2_fold))
            continue

        baseline_value = float(baseline[name])
        scalar_params[name] = baseline_value * (2.0 ** float(log2_fold))

    params_dict: Dict[str, Any] = {
        "eta": scalar_params["eta"],
        "beta": scalar_params["beta"],
        "kappa": scalar_params["kappa"],
        "epsilon": scalar_params["epsilon"],
        "Xc": _build_xc_vector(
            mean_xc=scalar_params["Xc"],
            std_frac=scalar_params["xc_std_frac"],
            n=n,
        ),
    }

    return params_dict, scalar_params


def _extract_survival_curve(sim: Any) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(sim.survival.index.values, dtype=float)
    values = np.asarray(sim.survival.iloc[:, 0].values, dtype=float)
    return times, np.clip(values, EPS, 1.0)


def _extract_hazard_curve(sim: Any) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(sim.tspan_hazard, dtype=float).reshape(-1)
    values = np.asarray(sim.hazard, dtype=float).reshape(-1)
    return times, np.maximum(values, EPS)


def _normalize_survival_to_start(
    times: np.ndarray,
    values: np.ndarray,
    start_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(times) & np.isfinite(values) & (times >= start_time)
    filtered_times = np.asarray(times[mask], dtype=float)
    filtered_values = np.asarray(values[mask], dtype=float)

    if filtered_times.size == 0:
        raise ValueError("No survival values remain after the start-time filter.")

    baseline = float(filtered_values[0])
    if baseline <= 0:
        raise ValueError("Survival normalization point must be positive.")

    return filtered_times, np.clip(filtered_values / baseline, EPS, 1.0)


def _interpolate_on_times(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    if source_times.size == 0:
        raise ValueError("Cannot interpolate from an empty source curve.")

    return np.interp(
        target_times,
        source_times,
        source_values,
        left=float(source_values[0]),
        right=float(source_values[-1]),
    )


def _huber_loss(residuals: np.ndarray, delta: float = 0.25) -> np.ndarray:
    absolute = np.abs(residuals)
    quadratic = np.minimum(absolute, delta)
    linear = absolute - quadratic
    return 0.5 * quadratic**2 + delta * linear


def _score_hazard(
    target_values: np.ndarray,
    sim_values: np.ndarray,
    weights: np.ndarray,
    objective_name: str,
) -> float:
    residuals = np.log(_clip_positive(sim_values)) - np.log(_clip_positive(target_values))

    if objective_name == "log_rmse":
        return float(np.sqrt(np.sum(weights * residuals**2)))

    if objective_name == "log_mae":
        return float(np.sum(weights * np.abs(residuals)))

    if objective_name == "log_huber":
        return float(np.sum(weights * _huber_loss(residuals)))

    raise ValueError(f"Unsupported hazard objective: {objective_name}")


def _score_survival(
    target_values: np.ndarray,
    sim_values: np.ndarray,
    weights: np.ndarray,
    objective_name: str,
) -> float:
    residuals = np.asarray(sim_values, dtype=float) - np.asarray(target_values, dtype=float)

    if objective_name == "linear_rmse":
        return float(np.sqrt(np.sum(weights * residuals**2)))

    if objective_name == "linear_mae":
        return float(np.sum(weights * np.abs(residuals)))

    if objective_name == "ks":
        return float(np.max(np.abs(residuals)))

    raise ValueError(f"Unsupported survival objective: {objective_name}")


def _score_curve(
    target: SRTarget,
    sim_times: np.ndarray,
    sim_values: np.ndarray,
    objective_name: str,
) -> float:
    model_values = _interpolate_on_times(sim_times, sim_values, target.times)

    if target.target == "hazard":
        return _score_hazard(
            target_values=target.values,
            sim_values=model_values,
            weights=target.weights,
            objective_name=objective_name,
        )

    model_values = np.clip(model_values, EPS, 1.0)
    model_values = model_values / model_values[0]
    return _score_survival(
        target_values=target.values,
        sim_values=model_values,
        weights=target.weights,
        objective_name=objective_name,
    )


def _build_hmd_targets(
    country: str,
    gender: str,
    data_type: str,
    year: int,
    config: SRFitConfig,
) -> Dict[str, Any]:
    hmd = HMD(country, gender, data_type)

    ages_hazard, hazards = hmd.get_hazard(year, haz_type="mx")
    ages_survival, survival = hmd.get_survival(year)

    hazard_mask = (
        (ages_hazard >= config.age_start)
        & (ages_hazard <= config.age_end)
        & np.isfinite(hazards)
        & (hazards > 0)
    )
    survival_mask = (
        (ages_survival >= config.age_start)
        & (ages_survival <= config.age_end)
        & np.isfinite(survival)
        & (survival > 0)
    )

    hazard_times = np.asarray(ages_hazard[hazard_mask], dtype=float)
    hazard_values = np.asarray(hazards[hazard_mask], dtype=float)
    survival_times = np.asarray(ages_survival[survival_mask], dtype=float)
    survival_values = np.asarray(survival[survival_mask], dtype=float)

    if hazard_times.size == 0 or survival_times.size == 0:
        raise ValueError("No HMD data left after age filtering.")

    survival_times, survival_values = _normalize_survival_to_start(
        times=survival_times,
        values=survival_values,
        start_time=float(config.age_start),
    )

    hazard_weights = _normalize_weights(survival[survival_mask])
    survival_weights = _normalize_weights(survival_values)
    mgg_fit = hmd.fit_ggm(year, age_start=20, age_end=100)
    h_ext = float(mgg_fit["m"])

    if not np.isfinite(h_ext):
        raise ValueError("MGG fit did not return a finite Makeham term.")

    return {
        "hmd": hmd,
        "h_ext": h_ext,
        "hazard_target": SRTarget(
            target="hazard",
            times=hazard_times,
            values=hazard_values,
            weights=hazard_weights,
            label=f"{country} {gender} {data_type} {year} hazard",
        ),
        "survival_target": SRTarget(
            target="survival",
            times=survival_times,
            values=survival_values,
            weights=survival_weights,
            label=f"{country} {gender} {data_type} {year} survival",
        ),
    }


def _build_array_target(
    times: Sequence[float],
    values: Sequence[float],
    target: str,
    config: SRFitConfig,
    weights: Sequence[float] | None = None,
) -> SRTarget:
    times_array = np.asarray(times, dtype=float).reshape(-1)
    values_array = np.asarray(values, dtype=float).reshape(-1)

    if times_array.size != values_array.size:
        raise ValueError("times and values must have the same length.")

    mask = np.isfinite(times_array) & np.isfinite(values_array)
    mask &= (times_array >= config.age_start) & (times_array <= config.age_end)

    if target == "hazard":
        mask &= values_array > 0
    else:
        mask &= values_array > 0

    filtered_times = np.asarray(times_array[mask], dtype=float)
    filtered_values = np.asarray(values_array[mask], dtype=float)

    if filtered_times.size == 0:
        raise ValueError("No target values remain after filtering the age window.")

    if target == "survival":
        filtered_times, filtered_values = _normalize_survival_to_start(
            times=filtered_times,
            values=filtered_values,
            start_time=float(filtered_times[0]),
        )

    if weights is None:
        filtered_weights = np.full(filtered_times.size, 1.0 / filtered_times.size, dtype=float)
    else:
        raw_weights = np.asarray(weights, dtype=float).reshape(-1)
        if raw_weights.size != times_array.size:
            raise ValueError("weights must match the length of times and values.")
        filtered_weights = _normalize_weights(raw_weights[mask])

    return SRTarget(
        target=target,
        times=filtered_times,
        values=filtered_values,
        weights=filtered_weights,
        label=f"array {target}",
    )


def _simulation_n_for_stage(target: str, stage: str, config: SRFitConfig) -> int:
    if target == "hazard":
        if stage == "stage1":
            return int(config.stage1_hazard_n)
        return int(config.stage3_hazard_n)

    if stage == "stage1":
        return int(config.stage1_survival_n)
    return int(config.stage3_survival_n)


def _make_bounds(fit_params: Sequence[str]) -> list[tuple[float, float]]:
    return [(-1.0, 1.0) for _ in fit_params]


def _clip_vector(vector: np.ndarray, bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    clipped = np.asarray(vector, dtype=float).copy()
    for idx, (lower, upper) in enumerate(bounds):
        clipped[idx] = np.clip(clipped[idx], lower, upper)
    return clipped


def _candidate_key(
    vector: Sequence[float],
    fit_params: Sequence[str],
    target: str,
    objective_name: str,
    n: int,
) -> tuple[Any, ...]:
    rounded = tuple(np.round(np.asarray(vector, dtype=float), 8))
    return tuple(fit_params) + (target, objective_name, int(n)) + rounded


def _resolve_parallel_simulation(config: SRFitConfig) -> bool:
    if config.parallel_simulation is not None:
        return bool(config.parallel_simulation)

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if main_file is None:
        return False

    return True


def _evaluate_candidate(
    *,
    vector: Sequence[float],
    fit_params: Sequence[str],
    baseline: Dict[str, float],
    h_ext: float,
    target: SRTarget,
    objective_name: str,
    config: SRFitConfig,
    n: int,
    cache: Dict[tuple[Any, ...], tuple[float, Any, Dict[str, Any], Dict[str, float]]],
) -> tuple[float, Any, Dict[str, Any], Dict[str, float]]:
    key = _candidate_key(
        vector=vector,
        fit_params=fit_params,
        target=target.target,
        objective_name=objective_name,
        n=n,
    )
    if key in cache:
        return cache[key]

    params_dict, scalar_params = _vector_to_params(
        vector=vector,
        fit_params=fit_params,
        baseline=baseline,
        config=config,
        n=n,
        h_ext=h_ext,
    )
    sim = create_sr_simulation(
        species="human",
        n=n,
        params_dict=params_dict,
        h_ext=h_ext,
        tmax=config.tmax,
        dt=config.dt,
        save_times=np.arange(0, config.tmax + 1e-9, 1.0),
        parallel=_resolve_parallel_simulation(config),
        break_early=True,
        random_seed=EVAL_SIMULATION_SEED,
    )

    if target.target == "hazard":
        sim_times, sim_values = _extract_hazard_curve(sim)
    else:
        sim_times, sim_values = _extract_survival_curve(sim)

    score = _score_curve(
        target=target,
        sim_times=sim_times,
        sim_values=sim_values,
        objective_name=objective_name,
    )
    cache[key] = (score, sim, params_dict, scalar_params)
    return cache[key]


def fit_coordinate_descent(
    objective_fn: Any,
    start_vector: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    step_sizes: Sequence[float],
) -> tuple[np.ndarray, float, list[float]]:
    current = _clip_vector(np.asarray(start_vector, dtype=float), bounds)
    current_score = float(objective_fn(current))
    history = [current_score]

    for step_size in step_sizes:
        improved = True
        while improved:
            improved = False
            best_vector = current
            best_score = current_score

            for dim in range(current.size):
                for delta in (-step_size, step_size):
                    trial = current.copy()
                    trial[dim] += delta
                    trial = _clip_vector(trial, bounds)

                    if np.allclose(trial, current):
                        continue

                    trial_score = float(objective_fn(trial))
                    history.append(trial_score)

                    if trial_score >= best_score:
                        continue

                    best_vector = trial
                    best_score = trial_score
                    improved = True

            if not improved:
                continue

            current = best_vector
            current_score = best_score

    return current, current_score, history


def _screen_candidates(
    *,
    target: SRTarget,
    objective_name: str,
    fit_params: Sequence[str],
    baseline: Dict[str, float],
    h_ext: float,
    config: SRFitConfig,
    cache: Dict[tuple[Any, ...], tuple[float, Any, Dict[str, Any], Dict[str, float]]],
) -> list[tuple[np.ndarray, float]]:
    sampler = qmc.LatinHypercube(d=len(fit_params), seed=EVAL_SIMULATION_SEED)
    samples = qmc.scale(
        sampler.random(config.stage1_screen_size),
        -1.0,
        1.0,
    )
    candidates = [np.zeros(len(fit_params), dtype=float)]
    candidates.extend(np.asarray(sample, dtype=float) for sample in samples)

    stage1_n = _simulation_n_for_stage(target.target, "stage1", config)
    scored: list[tuple[np.ndarray, float]] = []
    for vector in candidates:
        score, _, _, _ = _evaluate_candidate(
            vector=vector,
            fit_params=fit_params,
            baseline=baseline,
            h_ext=h_ext,
            target=target,
            objective_name=objective_name,
            config=config,
            n=stage1_n,
            cache=cache,
        )
        scored.append((np.asarray(vector, dtype=float), float(score)))

    scored.sort(key=lambda item: item[1])
    return scored


def _polish_stage(
    *,
    starts: Sequence[np.ndarray],
    target: SRTarget,
    objective_name: str,
    fit_params: Sequence[str],
    baseline: Dict[str, float],
    h_ext: float,
    config: SRFitConfig,
    step_sizes: Sequence[float],
    n: int,
    cache: Dict[tuple[Any, ...], tuple[float, Any, Dict[str, Any], Dict[str, float]]],
) -> list[tuple[np.ndarray, float]]:
    bounds = _make_bounds(fit_params)
    polished: list[tuple[np.ndarray, float]] = []

    def objective_fn(vector: Sequence[float]) -> float:
        score, _, _, _ = _evaluate_candidate(
            vector=vector,
            fit_params=fit_params,
            baseline=baseline,
            h_ext=h_ext,
            target=target,
            objective_name=objective_name,
            config=config,
            n=n,
            cache=cache,
        )
        return score

    for start in starts:
        best_vector, best_score, _ = fit_coordinate_descent(
            objective_fn=objective_fn,
            start_vector=start,
            bounds=bounds,
            step_sizes=step_sizes,
        )
        polished.append((best_vector, best_score))

    polished.sort(key=lambda item: item[1])
    return polished


def _evaluate_full_resolution(
    *,
    candidates: Sequence[np.ndarray],
    target: SRTarget,
    objective_name: str,
    fit_params: Sequence[str],
    baseline: Dict[str, float],
    h_ext: float,
    config: SRFitConfig,
    cache: Dict[tuple[Any, ...], tuple[float, Any, Dict[str, Any], Dict[str, float]]],
) -> list[tuple[np.ndarray, float]]:
    full_n = _simulation_n_for_stage(target.target, "stage3", config)
    scored: list[tuple[np.ndarray, float]] = []

    for vector in candidates:
        score, _, _, _ = _evaluate_candidate(
            vector=vector,
            fit_params=fit_params,
            baseline=baseline,
            h_ext=h_ext,
            target=target,
            objective_name=objective_name,
            config=config,
            n=full_n,
            cache=cache,
        )
        scored.append((np.asarray(vector, dtype=float), float(score)))

    scored.sort(key=lambda item: item[1])
    return scored


def _bound_hits(vector: Sequence[float], fit_params: Sequence[str], tol: float = 1e-3) -> list[str]:
    hits: list[str] = []
    array = np.asarray(vector, dtype=float)
    for idx, name in enumerate(fit_params):
        if abs(array[idx] - (-1.0)) <= tol or abs(array[idx] - 1.0) <= tol:
            hits.append(name)
    return hits


def _compute_acceptance(
    *,
    score: float,
    baseline_score: float,
    cross_score: float | None,
    baseline_cross_score: float | None,
    bound_hits: Sequence[str],
) -> bool:
    if baseline_score <= 0:
        return False

    fit_improvement = (baseline_score - score) / baseline_score
    if fit_improvement < 0.10:
        return False

    if cross_score is not None and baseline_cross_score is not None and baseline_cross_score > 0:
        cross_worsening = (cross_score - baseline_cross_score) / baseline_cross_score
        if cross_worsening > 0.05:
            return False

    if len(bound_hits) > 0:
        return False

    return True


def _build_curve_payload(times: np.ndarray, values: np.ndarray) -> Dict[str, List[float]]:
    return {
        "times": np.asarray(times, dtype=float).tolist(),
        "values": np.asarray(values, dtype=float).tolist(),
    }


def _save_result_summary(result: SRFitResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["config"] = asdict(result.config)
    path.write_text(json.dumps(payload, indent=2))


def _format_case_slug(
    country: str,
    gender: str,
    data_type: str,
    year: int,
    target: str,
    config: SRFitConfig,
) -> str:
    return (
        f"{country.lower()}_{data_type}_{year}_{gender}_{target}_"
        f"age{config.age_start}_{config.age_end}_xc18"
    )


def _build_saved_fit_name(
    country: str,
    gender: str,
    data_type: str,
    year: int,
    target: str,
    config: SRFitConfig,
) -> str:
    return (
        f"{country.lower()}_{data_type}_{year}_{gender}_{target}_"
        f"age{config.age_start}_{config.age_end}_xc18_naf3"
    )


def plot_sr_fit_result(result: SRFitResult, save_path: str | Path) -> Path:
    save_path = Path(save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    hazard_ax = axes[0]
    survival_ax = axes[1]

    if result.target == "hazard":
        hazard_target = result.target_curve
        hazard_fit = result.fitted_curve
        hazard_baseline = result.baseline_curve
        survival_target = result.cross_target_curve
        survival_fit = result.cross_fitted_curve
        survival_baseline = result.cross_baseline_curve
    else:
        hazard_target = result.cross_target_curve
        hazard_fit = result.cross_fitted_curve
        hazard_baseline = result.cross_baseline_curve
        survival_target = result.target_curve
        survival_fit = result.fitted_curve
        survival_baseline = result.baseline_curve

    if hazard_target is not None:
        hazard_ax.plot(hazard_target["times"], hazard_target["values"], "o", markersize=3, label="Target")
    if hazard_baseline is not None:
        hazard_ax.plot(hazard_baseline["times"], hazard_baseline["values"], "--", linewidth=1.5, label="Baseline")
    if hazard_fit is not None:
        hazard_ax.plot(hazard_fit["times"], hazard_fit["values"], "-", linewidth=2, label="SR fit")

    hazard_ax.set_xlabel("Age [years]")
    hazard_ax.set_ylabel("Hazard [1/year]")
    hazard_ax.set_yscale("log")
    hazard_ax.set_xlim(result.config.age_start, result.config.age_end)
    hazard_ax.grid(alpha=0.3)
    hazard_ax.legend(frameon=False)
    hazard_ax.set_title("Hazard")

    if survival_target is not None:
        survival_ax.plot(survival_target["times"], survival_target["values"], "o", markersize=3, label="Target")
    if survival_baseline is not None:
        survival_ax.plot(survival_baseline["times"], survival_baseline["values"], "--", linewidth=1.5, label="Baseline")
    if survival_fit is not None:
        survival_ax.plot(survival_fit["times"], survival_fit["values"], "-", linewidth=2, label="SR fit")

    survival_ax.set_xlabel("Age [years]")
    survival_ax.set_ylabel(f"Survival from age {result.config.age_start}")
    survival_ax.set_xlim(result.config.age_start, result.config.age_end)
    survival_ax.grid(alpha=0.3)
    survival_ax.legend(frameon=False)
    survival_ax.set_title("Survival")

    fig.suptitle(result.case_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _result_from_candidate(
    *,
    case_label: str,
    target_name: str,
    objective_name: str,
    best_vector: np.ndarray,
    baseline_vector: np.ndarray,
    fit_params: Sequence[str],
    baseline: Dict[str, float],
    h_ext: float,
    primary_target: SRTarget,
    cross_target: SRTarget | None,
    config: SRFitConfig,
    cache: Dict[tuple[Any, ...], tuple[float, Any, Dict[str, Any], Dict[str, float]]],
) -> SRFitResult:
    full_n = _simulation_n_for_stage(target_name, "stage3", config)

    score, sim, _, scalar_params = _evaluate_candidate(
        vector=best_vector,
        fit_params=fit_params,
        baseline=baseline,
        h_ext=h_ext,
        target=primary_target,
        objective_name=objective_name,
        config=config,
        n=full_n,
        cache=cache,
    )
    baseline_score, baseline_sim, _, baseline_scalar_params = _evaluate_candidate(
        vector=baseline_vector,
        fit_params=fit_params,
        baseline=baseline,
        h_ext=h_ext,
        target=primary_target,
        objective_name=objective_name,
        config=config,
        n=full_n,
        cache=cache,
    )

    if target_name == "hazard":
        fitted_times, fitted_values = _extract_hazard_curve(sim)
        baseline_times, baseline_values = _extract_hazard_curve(baseline_sim)
    else:
        fitted_times, fitted_values = _extract_survival_curve(sim)
        baseline_times, baseline_values = _extract_survival_curve(baseline_sim)

    fitted_values = _interpolate_on_times(fitted_times, fitted_values, primary_target.times)
    baseline_values = _interpolate_on_times(baseline_times, baseline_values, primary_target.times)
    if target_name == "survival":
        fitted_values = np.clip(fitted_values, EPS, 1.0)
        baseline_values = np.clip(baseline_values, EPS, 1.0)
        fitted_values = fitted_values / fitted_values[0]
        baseline_values = baseline_values / baseline_values[0]

    cross_score: float | None = None
    baseline_cross_score: float | None = None
    cross_target_curve: Dict[str, List[float]] | None = None
    cross_fitted_curve: Dict[str, List[float]] | None = None
    cross_baseline_curve: Dict[str, List[float]] | None = None

    if cross_target is not None:
        if cross_target.target == "hazard":
            cross_times, cross_values = _extract_hazard_curve(sim)
            baseline_cross_times, baseline_cross_values = _extract_hazard_curve(baseline_sim)
        else:
            cross_times, cross_values = _extract_survival_curve(sim)
            baseline_cross_times, baseline_cross_values = _extract_survival_curve(baseline_sim)

        cross_score = _score_curve(
            target=cross_target,
            sim_times=cross_times,
            sim_values=cross_values,
            objective_name=(
                config.hazard_objective
                if cross_target.target == "hazard"
                else config.survival_objective
            ),
        )
        baseline_cross_score = _score_curve(
            target=cross_target,
            sim_times=baseline_cross_times,
            sim_values=baseline_cross_values,
            objective_name=(
                config.hazard_objective
                if cross_target.target == "hazard"
                else config.survival_objective
            ),
        )
        cross_target_curve = _build_curve_payload(
            times=cross_target.times,
            values=cross_target.values,
        )
        cross_fitted_values = _interpolate_on_times(cross_times, cross_values, cross_target.times)
        cross_baseline_values = _interpolate_on_times(
            baseline_cross_times,
            baseline_cross_values,
            cross_target.times,
        )
        if cross_target.target == "survival":
            cross_fitted_values = np.clip(cross_fitted_values, EPS, 1.0)
            cross_baseline_values = np.clip(cross_baseline_values, EPS, 1.0)
            cross_fitted_values = cross_fitted_values / cross_fitted_values[0]
            cross_baseline_values = cross_baseline_values / cross_baseline_values[0]

        cross_fitted_curve = _build_curve_payload(
            times=cross_target.times,
            values=cross_fitted_values,
        )
        cross_baseline_curve = _build_curve_payload(
            times=cross_target.times,
            values=cross_baseline_values,
        )

    bound_hits = _bound_hits(best_vector, fit_params)
    accepted = _compute_acceptance(
        score=score,
        baseline_score=baseline_score,
        cross_score=cross_score,
        baseline_cross_score=baseline_cross_score,
        bound_hits=bound_hits,
    )

    return SRFitResult(
        case_label=case_label,
        target=target_name,
        objective_name=objective_name,
        score=float(score),
        baseline_score=float(baseline_score),
        cross_target_score=None if cross_score is None else float(cross_score),
        baseline_cross_target_score=None if baseline_cross_score is None else float(baseline_cross_score),
        h_ext=float(h_ext),
        fit_params={name: float(value) for name, value in scalar_params.items() if name in FIT_PARAM_ORDER or name == "kappa"},
        scalar_params={key: float(value) for key, value in scalar_params.items()},
        heterogeneity={
            "param": "Xc",
            "dist_type": "gaussian",
            "std_frac": float(scalar_params["xc_std_frac"]),
        },
        config=config,
        candidate_vector=np.asarray(best_vector, dtype=float).tolist(),
        bound_hits=bound_hits,
        accepted_by_metrics=accepted,
        target_curve=_build_curve_payload(primary_target.times, primary_target.values),
        fitted_curve=_build_curve_payload(primary_target.times, fitted_values),
        baseline_curve=_build_curve_payload(primary_target.times, baseline_values),
        cross_target_curve=cross_target_curve,
        cross_fitted_curve=cross_fitted_curve,
        cross_baseline_curve=cross_baseline_curve,
    )


def _run_fit(
    *,
    case_label: str,
    primary_target: SRTarget,
    cross_target: SRTarget | None,
    h_ext: float,
    config: SRFitConfig,
) -> SRFitResult:
    baseline = _baseline_scalar_params()
    fit_params = _prepare_fit_params(config)
    cache: Dict[tuple[Any, ...], tuple[float, Any, Dict[str, Any], Dict[str, float]]] = {}

    if config.initial_vectors:
        stage1_best = [
            _clip_vector(np.asarray(vector, dtype=float), _make_bounds(fit_params))
            for vector in config.initial_vectors
        ]
    else:
        screened = _screen_candidates(
            target=primary_target,
            objective_name=_resolve_objective_name(primary_target.target, config),
            fit_params=fit_params,
            baseline=baseline,
            h_ext=h_ext,
            config=config,
            cache=cache,
        )
        stage1_best = [vector for vector, _ in screened[: config.stage2_top_k]]

    if len(stage1_best) == 0:
        stage1_best = [np.zeros(len(fit_params), dtype=float)]

    stage1_n = _simulation_n_for_stage(primary_target.target, "stage1", config)
    stage2 = _polish_stage(
        starts=stage1_best,
        target=primary_target,
        objective_name=_resolve_objective_name(primary_target.target, config),
        fit_params=fit_params,
        baseline=baseline,
        h_ext=h_ext,
        config=config,
        step_sizes=config.stage2_step_sizes,
        n=stage1_n,
        cache=cache,
    )
    rerank_candidates = [vector for vector, _ in stage2[: config.stage3_top_k]]
    stage3 = _evaluate_full_resolution(
        candidates=rerank_candidates,
        target=primary_target,
        objective_name=_resolve_objective_name(primary_target.target, config),
        fit_params=fit_params,
        baseline=baseline,
        h_ext=h_ext,
        config=config,
        cache=cache,
    )
    best_stage3_vector = stage3[0][0]
    full_n = _simulation_n_for_stage(primary_target.target, "stage3", config)
    stage4 = _polish_stage(
        starts=[best_stage3_vector],
        target=primary_target,
        objective_name=_resolve_objective_name(primary_target.target, config),
        fit_params=fit_params,
        baseline=baseline,
        h_ext=h_ext,
        config=config,
        step_sizes=config.stage4_step_sizes,
        n=full_n,
        cache=cache,
    )

    return _result_from_candidate(
        case_label=case_label,
        target_name=primary_target.target,
        objective_name=_resolve_objective_name(primary_target.target, config),
        best_vector=stage4[0][0],
        baseline_vector=np.zeros(len(fit_params), dtype=float),
        fit_params=fit_params,
        baseline=baseline,
        h_ext=h_ext,
        primary_target=primary_target,
        cross_target=cross_target,
        config=config,
        cache=cache,
    )


def fit_sr_to_hmd(
    country: str,
    gender: str,
    data_type: str,
    year: int,
    target: str,
    config: SRFitConfig | None = None,
) -> SRFitResult:
    config = SRFitConfig() if config is None else config
    prepared = _build_hmd_targets(
        country=country,
        gender=gender,
        data_type=data_type,
        year=year,
        config=config,
    )

    if target == "hazard":
        primary_target = prepared["hazard_target"]
        cross_target = prepared["survival_target"]
    elif target == "survival":
        primary_target = prepared["survival_target"]
        cross_target = prepared["hazard_target"]
    else:
        raise ValueError("target must be 'hazard' or 'survival'.")

    result = _run_fit(
        case_label=f"{country} {gender} {data_type} {year}",
        primary_target=primary_target,
        cross_target=cross_target,
        h_ext=float(prepared["h_ext"]),
        config=config,
    )

    save_dir = _default_save_dir(config)
    slug = _format_case_slug(country, gender, data_type, year, target, config)
    summary_path = save_dir / f"{slug}.json"
    plot_path = save_dir / f"{slug}.png"
    _save_result_summary(result, summary_path)
    plot_sr_fit_result(result, plot_path)
    result.fitted_summary_path = str(summary_path)
    result.fitted_plot_path = str(plot_path)
    return result


def fit_sr_to_arrays(
    times: Sequence[float],
    values: Sequence[float],
    target: str,
    config: SRFitConfig | None = None,
    weights: Sequence[float] | None = None,
) -> SRFitResult:
    config = SRFitConfig() if config is None else config
    prepared_target = _build_array_target(
        times=times,
        values=values,
        target=target,
        config=config,
        weights=weights,
    )
    result = _run_fit(
        case_label=f"array {target}",
        primary_target=prepared_target,
        cross_target=None,
        h_ext=0.0,
        config=config,
    )

    save_dir = _default_save_dir(config)
    slug = f"array_{target}_age{config.age_start}_{config.age_end}"
    summary_path = save_dir / f"{slug}.json"
    plot_path = save_dir / f"{slug}.png"
    _save_result_summary(result, summary_path)
    plot_sr_fit_result(result, plot_path)
    result.fitted_summary_path = str(summary_path)
    result.fitted_plot_path = str(plot_path)
    return result


def build_fit_name_from_result(result: SRFitResult) -> str:
    parts = result.case_label.split()

    if len(parts) < 4:
        return f"sr_{result.target}_fit"

    country = parts[0]
    gender = parts[1]
    data_type = parts[2]
    year = int(parts[3])
    return _build_saved_fit_name(
        country=country,
        gender=gender,
        data_type=data_type,
        year=year,
        target=result.target,
        config=result.config,
    )


def save_accepted_fit(
    result: SRFitResult,
    fit_name: str | None = None,
    visual_ok: bool = False,
) -> bool:
    if not result.accepted_by_metrics:
        return False

    if not visual_ok:
        return False

    fit_name = build_fit_name_from_result(result) if fit_name is None else fit_name
    country, gender, data_type, year = result.case_label.split()[:4]
    fit_record = build_sr_fit_record(
        label=f"{country} {year} {result.target.title()} Fit",
        params=result.scalar_params,
        h_ext=result.h_ext,
        hetero_std=result.heterogeneity["std_frac"],
        source="Direct SR fitter benchmark",
        country=country,
        gender=gender,
        data_type=data_type,
        year=int(year),
        fit_target=result.target,
        age_start=result.config.age_start,
        age_end=result.config.age_end,
        hazard_mode="naf_bw3",
        notes=(
            "Fitted with fixed kappa and fixed h_ext from the MGG Makeham term. "
            "Uses Gaussian Xc heterogeneity with fixed std_frac=0.18."
        ),
    )
    save_sr_fit_record(fit_name=fit_name, fit_record=fit_record)
    result.saved_fit_name = fit_name
    return True


def run_objective_bakeoff(config: SRFitConfig | None = None) -> Dict[str, Any]:
    config = SRFitConfig() if config is None else config
    base_case = DEFAULT_BENCHMARK_CASES[0]

    bakeoff: Dict[str, Any] = {
        "case": asdict(base_case),
        "hazard": {},
        "survival": {},
        "recommended_hazard_objective": config.hazard_objective,
        "recommended_survival_objective": config.survival_objective,
    }

    baseline_hazard = fit_sr_to_hmd(
        country=base_case.country,
        gender=base_case.gender,
        data_type=base_case.data_type,
        year=base_case.year,
        target="hazard",
        config=config,
    )
    baseline_survival = fit_sr_to_hmd(
        country=base_case.country,
        gender=base_case.gender,
        data_type=base_case.data_type,
        year=base_case.year,
        target="survival",
        config=config,
    )

    bakeoff["hazard"][config.hazard_objective] = {
        "score": baseline_hazard.score,
        "cross_score": baseline_hazard.cross_target_score,
    }
    bakeoff["survival"][config.survival_objective] = {
        "score": baseline_survival.score,
        "cross_score": baseline_survival.cross_target_score,
    }

    for objective_name in config.hazard_objective_candidates:
        if objective_name == config.hazard_objective:
            continue

        candidate_config = replace(config, hazard_objective=objective_name)
        result = fit_sr_to_hmd(
            country=base_case.country,
            gender=base_case.gender,
            data_type=base_case.data_type,
            year=base_case.year,
            target="hazard",
            config=candidate_config,
        )
        bakeoff["hazard"][objective_name] = {
            "score": result.score,
            "cross_score": result.cross_target_score,
        }

        fit_gain = (baseline_hazard.score - result.score) / baseline_hazard.score
        cross_gain = 0.0
        if baseline_hazard.cross_target_score and result.cross_target_score:
            cross_gain = (
                baseline_hazard.cross_target_score - result.cross_target_score
            ) / baseline_hazard.cross_target_score

        if fit_gain >= 0.05 and cross_gain >= 0.05:
            bakeoff["recommended_hazard_objective"] = objective_name

    for objective_name in config.survival_objective_candidates:
        if objective_name == config.survival_objective:
            continue

        candidate_config = replace(config, survival_objective=objective_name)
        result = fit_sr_to_hmd(
            country=base_case.country,
            gender=base_case.gender,
            data_type=base_case.data_type,
            year=base_case.year,
            target="survival",
            config=candidate_config,
        )
        bakeoff["survival"][objective_name] = {
            "score": result.score,
            "cross_score": result.cross_target_score,
        }

        fit_gain = (baseline_survival.score - result.score) / baseline_survival.score
        cross_gain = 0.0
        if baseline_survival.cross_target_score and result.cross_target_score:
            cross_gain = (
                baseline_survival.cross_target_score - result.cross_target_score
            ) / baseline_survival.cross_target_score

        if fit_gain >= 0.05 and cross_gain >= 0.05:
            bakeoff["recommended_survival_objective"] = objective_name

    save_dir = _default_save_dir(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    bakeoff_path = save_dir / "objective_bakeoff.json"
    bakeoff_path.write_text(json.dumps(bakeoff, indent=2))
    return bakeoff


def run_default_benchmarks(
    config: SRFitConfig | None = None,
    cases: Iterable[SRBenchmarkCase] = DEFAULT_BENCHMARK_CASES,
) -> List[SRFitResult]:
    config = SRFitConfig() if config is None else config
    bakeoff = run_objective_bakeoff(config)
    benchmark_config = replace(
        config,
        hazard_objective=bakeoff["recommended_hazard_objective"],
        survival_objective=bakeoff["recommended_survival_objective"],
    )

    results: List[SRFitResult] = []
    for case in cases:
        for target in ("hazard", "survival"):
            result = fit_sr_to_hmd(
                country=case.country,
                gender=case.gender,
                data_type=case.data_type,
                year=case.year,
                target=target,
                config=benchmark_config,
            )
            results.append(result)

    summary = []
    for result in results:
        summary.append(
            {
                "case_label": result.case_label,
                "target": result.target,
                "score": result.score,
                "baseline_score": result.baseline_score,
                "cross_target_score": result.cross_target_score,
                "accepted_by_metrics": result.accepted_by_metrics,
                "plot_path": result.fitted_plot_path,
            }
        )

    save_dir = _default_save_dir(config)
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    return results


class SRFitter:
    def __init__(self, config: SRFitConfig | None = None):
        self.config = SRFitConfig() if config is None else config

    def fit_to_hmd(
        self,
        country: str,
        gender: str,
        data_type: str,
        year: int,
        target: str,
    ) -> SRFitResult:
        return fit_sr_to_hmd(
            country=country,
            gender=gender,
            data_type=data_type,
            year=year,
            target=target,
            config=self.config,
        )

    def fit_to_arrays(
        self,
        times: Sequence[float],
        values: Sequence[float],
        target: str,
        weights: Sequence[float] | None = None,
    ) -> SRFitResult:
        return fit_sr_to_arrays(
            times=times,
            values=values,
            target=target,
            config=self.config,
            weights=weights,
        )

    def run_objective_bakeoff(self) -> Dict[str, Any]:
        return run_objective_bakeoff(self.config)

    def run_default_benchmarks(self) -> List[SRFitResult]:
        return run_default_benchmarks(self.config)


def fit_with_restarts(
    *,
    target_times: Sequence[float],
    target_array: Sequence[float],
    fit_params: Sequence[str],
    initial_params: Dict[str, float] | None = None,
    fit_to: str = "survival",
    n: int = 3_000,
    n_restarts: int = 3,
    sample_weights: Sequence[float] | None = None,
    **_: Any,
) -> tuple[Dict[str, float], list[float]]:
    config = SRFitConfig(
        fit_params=tuple(fit_params),
        stage1_screen_size=max(int(n_restarts), 0),
        stage1_hazard_n=int(n) if fit_to == "hazard" else 30_000,
        stage1_survival_n=int(n) if fit_to == "survival" else 3_000,
        stage3_hazard_n=int(n) if fit_to == "hazard" else 50_000,
        stage3_survival_n=int(n) if fit_to == "survival" else 6_000,
    )
    result = fit_sr_to_arrays(
        times=target_times,
        values=target_array,
        target=fit_to,
        config=config,
        weights=sample_weights,
    )
    return result.scalar_params, [result.score]


def fit_hybrid_two_stage(**kwargs: Any) -> tuple[Dict[str, float], list[float]]:
    return fit_with_restarts(**kwargs)
