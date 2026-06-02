#!/usr/bin/env python3
"""Stress-grid refits that overwrite the canonical wide tau-spread profile.

This script is the all-CV version of the targeted 10% stress refit. For each
forced spread in tau=beta/eta, it lets all wide SR-side fitted parameters move:
eta, median tau, kappa, epsilon, country-specific Xc, and country-specific Xc
heterogeneity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.model_fits.tau_wide_refit_profile import run_tau_wide_refit_profile as wide


EXPLORATION_DIR = Path(__file__).resolve().parent
PLOTS_DIR = wide.PLOTS_DIR
REPORT_PATH = wide.REPORT_PATH

CACHE_DIR = wide.CACHE_DIR
METADATA_PATH = wide.METADATA_PATH
FIT_RESULTS_PATH = wide.FIT_RESULTS_PATH
ALL_CANDIDATES_PATH = wide.CSV_DIR / "tau_wide_refit_profile_all_candidates.csv"
BEST_RESULTS_PATH = wide.CSV_DIR / "tau_wide_refit_profile_best.csv"
CURVE_RESULTS_PATH = wide.CURVE_RESULTS_PATH
HIGH_N_CURVE_RESULTS_PATH = wide.CSV_DIR / "tau_wide_refit_profile_curves_n1000000.csv"

TAU_SD_GRID = wide.TAU_SD_GRID
LOCAL_STEPS = (0.25, 0.125, 0.0625, 0.03125)
LOCAL_SWEEPS = 3
ACCEPTANCE_MULTIPLIER = 2.5


@dataclass(frozen=True)
class StressConfig:
    screen_random: int
    screen_keep: int
    de_maxiter: int
    de_popsize: int
    fit_n: int
    screen_n: int
    eval_n: int
    eval_repeats: int
    curve_n: int
    fit_dt: float
    curve_dt: float
    no_parallel: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-random", type=int, default=14)
    parser.add_argument("--screen-keep", type=int, default=7)
    parser.add_argument("--de-maxiter", type=int, default=3)
    parser.add_argument("--de-popsize", type=int, default=3)
    parser.add_argument("--screen-n", type=int, default=1_200)
    parser.add_argument("--fit-n", type=int, default=4_000)
    parser.add_argument("--eval-n", type=int, default=50_000)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--curve-n", type=int, default=1_000_000)
    parser.add_argument("--fit-dt", type=float, default=0.08)
    parser.add_argument("--curve-dt", type=float, default=0.1)
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-parallel", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> StressConfig:
    return StressConfig(
        screen_random=int(args.screen_random),
        screen_keep=int(args.screen_keep),
        de_maxiter=int(args.de_maxiter),
        de_popsize=int(args.de_popsize),
        screen_n=int(args.screen_n),
        fit_n=int(args.fit_n),
        eval_n=int(args.eval_n),
        eval_repeats=int(args.eval_repeats),
        curve_n=int(args.curve_n),
        fit_dt=float(args.fit_dt),
        curve_dt=float(args.curve_dt),
        no_parallel=bool(args.no_parallel),
    )


def wide_config(config: StressConfig) -> wide.FitConfig:
    return wide.FitConfig(
        profile_starts_per_sd=0,
        full_starts=0,
        fit_n=config.fit_n,
        eval_n=config.eval_n,
        curve_n=config.curve_n,
        fit_dt=config.fit_dt,
        curve_dt=config.curve_dt,
        acceptance_multiplier=ACCEPTANCE_MULTIPLIER,
        parallel=not config.no_parallel,
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def tau_cv_from_sd(tau_sd: float) -> float:
    return float(math.sqrt(math.exp(float(tau_sd) ** 2) - 1.0))


def metadata(config: StressConfig) -> dict[str, object]:
    return {
        "task": "tau_wide_refit_profile_stress_grid",
        "description": "All-CV stress-grid refits; canonical wide outputs overwritten with best candidate per forced CV.",
        "tau_sd_grid": list(TAU_SD_GRID),
        "wide_bounds": [list(bound) for bound in wide.WIDE_BOUNDS],
        "parameter_columns": list(wide.PARAMETER_COLUMNS),
        "local_steps": list(LOCAL_STEPS),
        "local_sweeps": LOCAL_SWEEPS,
        "acceptance_multiplier": ACCEPTANCE_MULTIPLIER,
        "config": config.__dict__,
    }


def metadata_matches(config: StressConfig) -> bool:
    if not METADATA_PATH.exists():
        return False
    try:
        return json.loads(METADATA_PATH.read_text()) == metadata(config)
    except Exception:
        return False


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 9.5,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def vector_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray([row[col] for col in wide.PARAMETER_COLUMNS], dtype=float)


def existing_wide_starts() -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = []
    paths = [
        wide.CSV_DIR / "tau_cv010_stress_refit_candidates.csv",
        wide.RESULTS_DIR / "tables" / "extended_data_figure1_tau_spread_constraint_source_data.csv",
    ]
    seen_paths: set[Path] = set()
    for path in paths:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        fits = pd.read_csv(path)
        if not all(col in fits.columns for col in wide.PARAMETER_COLUMNS):
            continue
        fits = fits.dropna(subset=list(wide.PARAMETER_COLUMNS))
        for _, row in fits.iterrows():
            candidate_id = row["candidate_id"] if "candidate_id" in row.index else path.stem
            starts.append((f"{path.stem}_{candidate_id}", vector_from_row(row)))
    return starts


def random_starts(count: int, tau_sd: float) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(stable_seed("stress-random", tau_sd, count))
    lowers = np.asarray([bound[0] for bound in wide.WIDE_BOUNDS], dtype=float)
    uppers = np.asarray([bound[1] for bound in wide.WIDE_BOUNDS], dtype=float)
    return [(f"random_{idx:02d}", rng.uniform(lowers, uppers)) for idx in range(int(count))]


def start_bank(tau_sd: float, config: StressConfig, carry: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = []
    starts.extend(wide.structured_starts())
    starts.extend(existing_wide_starts())
    starts.extend(carry)
    starts.extend(random_starts(config.screen_random, tau_sd))

    deduped: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, vector in starts:
        vector = wide.clip_vector(vector)
        key = tuple(np.round(vector, 5))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, vector))
    return deduped


def score_fn(data_by_country: dict[str, object], tau_sd: float, n: int, seed_label: str, memo: dict[tuple[object, ...], float]):
    seed = stable_seed("stress-grid", tau_sd, seed_label)

    def objective(vector: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return wide.score_candidate(vector, tau_sd, data_by_country, n, seed, memo)

    return objective


def local_coordinate_refine(objective, start: np.ndarray) -> tuple[np.ndarray, float, list[float]]:
    current = wide.clip_vector(start)
    current_score = float(objective(current))
    history = [current_score]
    for step_size in LOCAL_STEPS:
        improved = True
        sweeps = 0
        while improved and sweeps < LOCAL_SWEEPS:
            sweeps += 1
            improved = False
            best_vector = current.copy()
            best_score = current_score
            for dim in range(current.size):
                for direction in (-1.0, 1.0):
                    trial = current.copy()
                    trial[dim] += direction * step_size
                    trial = wide.clip_vector(trial)
                    if np.allclose(trial, current):
                        continue
                    score = float(objective(trial))
                    if score < best_score:
                        best_vector = trial
                        best_score = score
                        improved = True
            if improved:
                current = best_vector
                current_score = best_score
                history.append(current_score)
    return current, current_score, history


def screen_starts(
    tau_sd: float,
    starts: list[tuple[str, np.ndarray]],
    data_by_country: dict[str, object],
    config: StressConfig,
) -> pd.DataFrame:
    memo: dict[tuple[object, ...], float] = {}
    objective = score_fn(data_by_country, tau_sd, config.screen_n, "screen", memo)
    rows = []
    for name, vector in starts:
        rows.append(
            {
                "name": name,
                "screen_score": float(objective(vector)),
                **{col: float(val) for col, val in zip(wide.PARAMETER_COLUMNS, vector)},
            }
        )
    return pd.DataFrame(rows).sort_values("screen_score").reset_index(drop=True)


def differential_evolution_start(
    tau_sd: float,
    starts: list[tuple[str, np.ndarray]],
    data_by_country: dict[str, object],
    config: StressConfig,
) -> tuple[np.ndarray, float]:
    memo: dict[tuple[object, ...], float] = {}
    objective = score_fn(data_by_country, tau_sd, config.screen_n, "de", memo)
    required = max(5, config.de_popsize * len(wide.PARAMETER_COLUMNS))
    init_vectors = [vector for _, vector in starts[:required]]
    if len(init_vectors) < required:
        init_vectors.extend([vector for _, vector in random_starts(required - len(init_vectors), tau_sd + 991.0)])
    init = np.vstack([wide.clip_vector(vector) for vector in init_vectors])
    result = differential_evolution(
        objective,
        bounds=list(wide.WIDE_BOUNDS),
        maxiter=config.de_maxiter,
        popsize=config.de_popsize,
        init=init,
        polish=False,
        updating="immediate",
        workers=1,
        seed=stable_seed("stress-grid-de", tau_sd),
        tol=0.0,
        atol=0.0,
        disp=False,
    )
    return wide.clip_vector(result.x), float(result.fun)


def priority_starts(tau_sd: float, starts: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    """Force scientifically relevant previous candidates into local refinement."""
    selected: list[tuple[str, np.ndarray]] = []
    profile_tag = f"profile_tau_sd_{tau_sd:.3f}"
    for name, vector in starts:
        if "figS_tau_spread_constraint_source_data" in name and profile_tag in name:
            selected.append((name, vector))
        elif tau_sd <= 0.05 and "figS_tau_spread_constraint_source_data_full_fit_02" in name:
            selected.append((name, vector))
        elif abs(tau_sd - 0.10) <= 0.025 and "tau_cv010_stress_refit_candidates" in name:
            selected.append((name, vector))

    deduped: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, vector in selected:
        key = tuple(np.round(wide.clip_vector(vector), 5))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, wide.clip_vector(vector)))
    return deduped


def dedupe_start_list(starts: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
    deduped: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for name, vector in starts:
        vector = wide.clip_vector(vector)
        key = tuple(np.round(vector, 5))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((name, vector))
    return deduped


def natural_row(vector: np.ndarray) -> dict[str, float]:
    values = wide.natural_values(vector)
    values.update({col: float(val) for col, val in zip(wide.PARAMETER_COLUMNS, wide.clip_vector(vector))})
    return values


def refine_one_tau_sd(
    tau_sd: float,
    data_by_country: dict[str, object],
    config: StressConfig,
    carry: list[tuple[str, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray]:
    starts = start_bank(tau_sd, config, carry)
    screened = screen_starts(tau_sd, starts, data_by_country, config)
    top_starts = [
        (row["name"], np.asarray([row[col] for col in wide.PARAMETER_COLUMNS], dtype=float))
        for _, row in screened.head(config.screen_keep).iterrows()
    ]
    forced_starts = priority_starts(tau_sd, starts)
    de_vector, de_score = differential_evolution_start(tau_sd, forced_starts + top_starts + starts, data_by_country, config)
    refine_starts = dedupe_start_list([("de_best", de_vector), *forced_starts, *top_starts])

    memo: dict[tuple[object, ...], float] = {}
    objective = score_fn(data_by_country, tau_sd, config.fit_n, "local", memo)
    refined = []
    for name, vector in refine_starts:
        fit_vector, search_score, history = local_coordinate_refine(objective, vector)
        eval_scores = [
            wide.score_candidate(
                fit_vector,
                tau_sd,
                data_by_country,
                config.eval_n,
                stable_seed("stress-grid-eval", tau_sd, name, repeat),
                memo=None,
            )
            for repeat in range(config.eval_repeats)
        ]
        row = {
            "candidate_id": f"stress_tau_sd_{tau_sd:.3f}_{name}",
            "start_name": name,
            "tau_sd": float(tau_sd),
            "tau_cv": tau_cv_from_sd(tau_sd),
            "screen_de_score": float(de_score) if name == "de_best" else np.nan,
            "search_score": float(search_score),
            "eval_score_mean": float(np.mean(eval_scores)),
            "eval_score_sd": float(np.std(eval_scores, ddof=1)) if len(eval_scores) > 1 else 0.0,
            "eval_score_min": float(np.min(eval_scores)),
            "eval_score_max": float(np.max(eval_scores)),
            "history_start": float(history[0]) if history else np.nan,
            "history_end": float(history[-1]) if history else float(search_score),
            "history_steps": len(history),
        }
        row.update(natural_row(fit_vector))
        refined.append(row)
        print(
            f"tau_sd={tau_sd:.3f} {name}: search={search_score:.4g} eval={row['eval_score_mean']:.4g}",
            flush=True,
        )

    frame = pd.DataFrame(refined).sort_values("eval_score_mean").reset_index(drop=True)
    best_vector = np.asarray([frame.iloc[0][col] for col in wide.PARAMETER_COLUMNS], dtype=float)
    return frame, best_vector


def canonical_profile_from_best(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    best_score = float(best["eval_score_mean"].min())
    for _, row in best.sort_values("tau_sd").iterrows():
        out = {
            "candidate_id": f"profile_tau_sd_{row['tau_sd']:.3f}",
            "label": rf"profile $\sigma_v={row['tau_sd']:.3f}$",
            "source": "profile_tau_sd",
            "search_score": float(row["search_score"]),
            "eval_score": float(row["eval_score_mean"]),
            "eval_score_mean": float(row["eval_score_mean"]),
            "eval_score_sd": float(row["eval_score_sd"]),
            "eval_score_min": float(row["eval_score_min"]),
            "eval_score_max": float(row["eval_score_max"]),
            "history_start": float(row["history_start"]),
            "history_end": float(row["history_end"]),
            "history_steps": int(row["history_steps"]),
            "tau_sd": float(row["tau_sd"]),
            "tau_cv": float(row["tau_cv"]),
            "best_eval_score": best_score,
            "score_ratio_to_best": float(row["eval_score_mean"]) / best_score,
            "acceptance_multiplier": ACCEPTANCE_MULTIPLIER,
            "strict_accepted": float(row["eval_score_mean"]) / best_score <= ACCEPTANCE_MULTIPLIER,
            "profile_requested_tau_sd": float(row["tau_sd"]),
            "best_start_name": str(row["start_name"]),
            "stress_origin_candidate_id": str(row["candidate_id"]),
            "screen_de_score": float(row["screen_de_score"]) if np.isfinite(row["screen_de_score"]) else np.nan,
        }
        for col in wide.PARAMETER_COLUMNS:
            out[col] = float(row[col])
        for col in [
            "eta",
            "beta",
            "kappa",
            "epsilon",
            "tau",
            "tau_factor_vs_karin",
            "eta_factor",
            "beta_factor",
            "kappa_factor",
            "epsilon_factor",
            "SWE_Xc",
            "SWE_xc_std_frac",
            "USA_Xc",
            "USA_xc_std_frac",
        ]:
            out[col] = float(row[col])
        rows.append(out)
    return pd.DataFrame(rows).sort_values("tau_sd").reset_index(drop=True)


def run_stress_grid(config: StressConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide.set_runtime(wide_config(config))
    data_by_country = wide.load_data_by_country()
    all_candidates = []
    carry: list[tuple[str, np.ndarray]] = []

    for tau_sd in TAU_SD_GRID:
        frame, best_vector = refine_one_tau_sd(tau_sd, data_by_country, config, carry)
        all_candidates.append(frame)
        carry.append((f"carry_tau_sd_{tau_sd:.3f}", best_vector))

    candidates = pd.concat(all_candidates, ignore_index=True)
    best_score = float(candidates["eval_score_mean"].min())
    candidates["best_eval_score"] = best_score
    candidates["score_ratio_to_best"] = candidates["eval_score_mean"] / best_score
    candidates["acceptance_multiplier"] = ACCEPTANCE_MULTIPLIER
    candidates["strict_accepted"] = candidates["score_ratio_to_best"] <= ACCEPTANCE_MULTIPLIER
    candidates = candidates.sort_values(["tau_sd", "eval_score_mean"]).reset_index(drop=True)

    best = candidates.groupby("tau_sd", as_index=False).first()
    profile = canonical_profile_from_best(best)
    return candidates, profile


def curve_fits_from_profile(profile: pd.DataFrame, config: StressConfig) -> pd.DataFrame:
    return wide.build_curves(profile, wide_config(config))


def write_outputs(candidates: pd.DataFrame, profile: pd.DataFrame, curves: pd.DataFrame, config: StressConfig) -> None:
    wide.CSV_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(ALL_CANDIDATES_PATH, index=False)
    profile.to_csv(FIT_RESULTS_PATH, index=False)
    profile.to_csv(BEST_RESULTS_PATH, index=False)
    curves.to_csv(CURVE_RESULTS_PATH, index=False)

    # Keep the cache filenames used by the original wide-profile script current.
    profile.to_csv(wide.FIT_CACHE_PATH, index=False)
    curves.to_csv(wide.CURVE_CACHE_PATH, index=False)

    if int(config.curve_n) == 1_000_000:
        curves.to_csv(HIGH_N_CURVE_RESULTS_PATH, index=False)

    METADATA_PATH.write_text(json.dumps(metadata(config), indent=2, sort_keys=True) + "\n")


def load_outputs(config: StressConfig, require_metadata: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if require_metadata and not metadata_matches(config):
        raise FileNotFoundError("Cached stress-grid metadata does not match requested config.")
    profile = pd.read_csv(FIT_RESULTS_PATH)
    curves = pd.read_csv(CURVE_RESULTS_PATH)
    candidates = pd.read_csv(ALL_CANDIDATES_PATH) if ALL_CANDIDATES_PATH.exists() else profile.copy()
    return candidates, profile, curves


def save_fig(fig: plt.Figure, stem: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    png = PLOTS_DIR / f"{stem}.png"
    pdf = PLOTS_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def tail_summary(profile: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    return wide.tail_summary(profile, curves)


def fmt_range(values: pd.Series, digits: int = 2) -> str:
    values = values[np.isfinite(values)]
    if values.empty:
        return "none"
    return f"{values.min():.{digits}f}-{values.max():.{digits}f}"


def plot_score_tau(profile: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = profile.sort_values("tau_cv")
    eval_sd = data["eval_score_sd"] if "eval_score_sd" in data.columns else pd.Series(np.zeros(len(data)), index=data.index)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))

    axes[0].plot(data["tau_cv"], data["eval_score"], color="#0B7F8C", lw=2.5, marker="o", label="stress-grid best")
    axes[0].fill_between(
        data["tau_cv"],
        data["eval_score"] - eval_sd,
        data["eval_score"] + eval_sd,
        color="#0B7F8C",
        alpha=0.18,
        linewidth=0,
        label="eval repeat SD",
    )
    axes[0].axhline(data["best_eval_score"].iloc[0] * ACCEPTANCE_MULTIPLIER, color="#777777", ls="--", lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[0].set_ylabel("Fit objective score")
    axes[0].set_title("Best stress refit at each forced CV")
    axes[0].legend(frameon=False)

    axes[1].plot(data["tau_cv"], data["tau_factor_vs_karin"], color="#117A65", lw=2.5, marker="o")
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
    return save_fig(fig, "01_wide_fit_score_and_tau")


def plot_parameters(profile: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = profile.sort_values("tau_cv")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for col, label, color in [
        ("eta_factor", r"$\eta/\eta_K$", "#2274A5"),
        ("beta_factor", r"$\beta/\beta_K$", "#B05C2E"),
        ("kappa_factor", r"$\kappa/\kappa_K$", "#6A4C93"),
        ("epsilon_factor", r"$\epsilon/\epsilon_K$", "#2A9D8F"),
    ]:
        ax.plot(data["tau_cv"], data[col], marker="o", lw=2.0, label=label, color=color)
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    ax.set_ylabel("Fitted parameter factor vs baseline")
    ax.set_title("Shared SR parameters were refit")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    return save_fig(fig, "02_wide_parameter_compensation")


def plot_tail(profile: pd.DataFrame, curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = tail_summary(profile, curves).sort_values("tau_cv")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(data["tau_cv"], data["S110_ratio_to_hmd"], marker="o", lw=2.4, color="#B05C2E")
    axes[0].axhline(1.0, color="black", lw=1.1, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[0].set_ylabel(r"$S(110\mid90)$ ratio to HMD")
    axes[0].set_title("Old-age tail after best refit")

    axes[1].plot(data["tau_cv"], data["top_0_01pct_lifespan"], marker="o", lw=2.4, color="#245A8D")
    axes[1].set_xlabel(r"Timescale CV, $\mathrm{CV}(\tau)$")
    axes[1].set_ylabel("Top 0.01% lifespan [years]")
    axes[1].set_title("Extreme simulated lifespan")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return save_fig(fig, "04_wide_tail_metrics")


def nearest_candidate_id(profile: pd.DataFrame, cv: float) -> str:
    idx = (profile["tau_cv"] - cv).abs().idxmin()
    return str(profile.loc[idx, "candidate_id"])


def selected_mortality_ids(profile: pd.DataFrame) -> list[str]:
    data = profile.sort_values("tau_cv").reset_index(drop=True)
    targets = [0.0, 0.025, 0.05, 0.10, float(data["tau_cv"].max())]
    return list(dict.fromkeys(nearest_candidate_id(data, target) for target in targets))


def plot_mortality_fits(profile: pd.DataFrame, curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    hazard = curves[curves["curve"] == "hazard"].copy()
    hmd = hazard[hazard["candidate_id"] == "HMD"].copy()
    ids = selected_mortality_ids(profile)
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, max(len(ids), 1)))
    color_map = dict(zip(ids, colors))

    axes[0].scatter(hmd["age"], hmd["value"], s=28, color="black", label="Sweden 2019 HMD", zorder=5)
    hmd_lookup = hmd[["age", "value"]].rename(columns={"value": "hmd_hazard"})
    for candidate_id in ids:
        row = profile[profile["candidate_id"] == candidate_id].iloc[0]
        sub = hazard[hazard["candidate_id"] == candidate_id].sort_values("age")
        if sub.empty:
            continue
        label = rf"CV {row['tau_cv']:.3f}, $\tau$ {row['tau_factor_vs_karin']:.2f}x"
        axes[0].plot(sub["age"], sub["value"], lw=2.1, alpha=0.88, color=color_map[candidate_id], label=label)
        resid = sub[["age", "value"]].merge(hmd_lookup, on="age", how="inner")
        resid["log2_fold"] = np.log2(resid["value"] / resid["hmd_hazard"])
        axes[1].plot(resid["age"], resid["log2_fold"], lw=1.9, alpha=0.88, color=color_map[candidate_id], label=label)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Mortality rate / hazard")
    axes[0].set_title("Data and best stress-refit curves")
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


def plot_tradeoff(profile: pd.DataFrame, curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    data = tail_summary(profile, curves).sort_values("tau_cv")
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    scatter = ax.scatter(
        data["score_ratio_to_best"],
        data["S110_ratio_to_hmd"],
        c=data["tau_cv"],
        s=76,
        cmap="viridis",
        edgecolor="white",
        linewidth=0.7,
    )
    for _, row in data.iterrows():
        ax.text(row["score_ratio_to_best"] * 1.04, row["S110_ratio_to_hmd"], f"{row['tau_cv']:.2f}", fontsize=8.5, va="center")
    ax.axvline(ACCEPTANCE_MULTIPLIER, color="#777777", lw=1.2, ls="--")
    ax.axhline(1.0, color="black", lw=1.1, ls="--")
    ax.axhline(5.0, color="#999999", lw=1.1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fit score / best stress-grid fit")
    ax.set_ylabel(r"$S(110\mid90)$ ratio to HMD")
    ax.set_title("Fit and tail constraints together")
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.03)
    cbar.set_label(r"$\mathrm{CV}(\tau)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return save_fig(fig, "05_wide_fit_tail_tradeoff")


def make_plots(profile: pd.DataFrame, curves: pd.DataFrame) -> list[Path]:
    return [
        plot_score_tau(profile),
        plot_parameters(profile),
        plot_mortality_fits(profile, curves),
        plot_tail(profile, curves),
        plot_tradeoff(profile, curves),
    ]


def rel_plot(path: Path) -> str:
    return f"![{path.stem}](plots/{path.name})"


def write_report(candidates: pd.DataFrame, profile: pd.DataFrame, curves: pd.DataFrame, plot_paths: list[Path], config: StressConfig) -> None:
    tails = tail_summary(profile, curves).sort_values("tau_cv")
    accepted = profile[profile["score_ratio_to_best"] <= ACCEPTANCE_MULTIPLIER]
    metrics = wide.mortality_metrics(profile, curves)
    best_profile = profile.sort_values("eval_score").iloc[0]
    accepted_range = fmt_range(accepted["tau_cv"], 3)

    lines = [
        "# Wide Tau-Spread Stress-Grid Refit Profile",
        "",
        "This exploration overwrites the previous wide-profile outputs with a stronger search. It is the all-CV version of the forced 10% stress test: for each fixed population spread in the senogenic timescale,",
        "",
        "$$",
        "\\tau=\\frac{\\beta}{\\eta},",
        "$$",
        "",
        "the optimizer was allowed to move the median \\(\\tau\\), \\(\\eta\\), \\(\\kappa\\), \\(\\epsilon\\), Sweden/USA \\(X_c\\), and Sweden/USA \\(X_c\\) heterogeneity. The external Makeham-like term \\(h_\\mathrm{ext}\\) remains fixed from the HMD/GGM preprocessing.",
        "",
        "## What I Ran",
        "",
        f"- Forced \\(\\sigma_v=\\mathrm{{SD}}[\\log(\\tau)]\\) grid: {', '.join(f'{x:g}' for x in TAU_SD_GRID)}.",
        f"- At every grid point, screened structured/archive/source-data/targeted-stress/carryover/random starts, including {config.screen_random} new random starts.",
        f"- Ran differential evolution with maxiter={config.de_maxiter}, popsize={config.de_popsize} at \\(n={config.screen_n:,}\\).",
        f"- Locally refined differential-evolution and top screened starts using log2 steps \\({', '.join(f'{x:g}' for x in LOCAL_STEPS)}\\) at \\(n={config.fit_n:,}\\).",
        f"- Evaluated each refined candidate with {config.eval_repeats} independent scores at \\(n={config.eval_n:,}\\).",
        f"- Built Sweden mortality curves and tail metrics with \\(n={config.curve_n:,}\\), \\(\\Delta t={config.curve_dt:g}\\), and maximum simulated age {wide.CURVE_TMAX:g} years.",
        "",
        "This is still a stochastic numerical search, not a mathematical proof of global optimality. The point is a hard stress test: if broad smooth \\(\\tau\\)-spread can be hidden by moving the SR baseline and nuisance parameters over wide bounds, this search gives it several ways to do that.",
        "",
        "Individual variation is introduced as",
        "",
        "$$",
        "\\eta_i=\\eta_0 e^{-v_i/2},\\qquad \\beta_i=\\beta_0 e^{v_i/2},\\qquad v_i\\sim\\mathcal N(0,\\sigma_v^2).",
        "$$",
        "",
        "Thus individual \\(\\tau_i=\\beta_i/\\eta_i\\) has log-spread \\(\\sigma_v\\), while the fitted median \\(\\tau_0\\) is free to move.",
        "",
        "## Result 1: fit penalty rises with forced tau-spread",
        "",
        f"The best fixed-spread stress-grid fit had \\(\\mathrm{{CV}}(\\tau)={best_profile['tau_cv']:.3f}\\), median \\(\\tau/\\tau_K={best_profile['tau_factor_vs_karin']:.2f}\\), and score {best_profile['eval_score']:.4g}.",
        f"As a rough visual guide, a \\({ACCEPTANCE_MULTIPLIER:g}\\times\\) relative-score line covers \\(\\mathrm{{CV}}(\\tau)={accepted_range}\\). This line is not the scientific criterion; the question is whether large forced CV can still fit the mortality data after all compensatory parameters are refit.",
        "",
        rel_plot(plot_paths[0]),
        "",
        "How to read this figure:",
        "",
        "- Left panel: each teal point is the best candidate found for that forced \\(\\mathrm{CV}(\\tau)\\), after screening, differential evolution, local refinement, and repeated stochastic evaluation. Lower score is better. The translucent band is the between-repeat score SD.",
        "- The dashed horizontal line is only a relative-score guide. The important pattern is the profile shape: 2.5% CV can be partly absorbed, 5% CV is already meaningfully worse, and 7.5-10%+ CV remains worse despite the rescue scan.",
        "- Right panel: the median \\(\\tau=\\beta/\\eta\\) was allowed to move from 0.25x to 4x baseline, shown by the dotted gray bounds. If high spread could be rescued by choosing a radically different median timescale, this panel would show that escape route.",
        "",
        "Takeaway: the result is not that nonzero CV must beat the zero-spread fit. The point is that larger smooth \\(\\tau\\)-spread still leaves a growing fit penalty after a serious compensatory search.",
        "",
        "## Result 2: all fitted SR parameters were allowed to compensate",
        "",
        rel_plot(plot_paths[1]),
        "",
        "How to read this figure:",
        "",
        "- Each line is a shared fitted SR parameter divided by its baseline value. A value of 1 means unchanged; 0.5 means half; 2 means doubled.",
        "- This is not a post-hoc perturbation around a fixed baseline. \\(\\eta\\), \\(\\beta\\), \\(\\kappa\\), and \\(\\epsilon\\) move during the fit. Sweden/USA \\(X_c\\) and \\(X_c\\) heterogeneity also move and are saved in the CSV, but they are omitted from this plot to keep it readable.",
        "- Compensation is real, but it is not magic: the model can improve central mortality for some forced spreads, yet the same smooth favorable tail still creates too many extreme survivors.",
        "",
        "Takeaway: high-CV failures here are not because \\(\\beta\\), \\(\\eta\\), \\(\\epsilon\\), \\(X_c\\), or \\(X_c\\)-heterogeneity were frozen.",
        "",
        "## Result 3: data-vs-fit mortality curves",
        "",
        rel_plot(plot_paths[2]),
        "",
        "How to read this figure:",
        "",
        "- Left panel: black dots are Sweden 2019 HMD mortality. Colored lines are selected best stress-grid SR curves at 0%, 2.5%, 5%, 10%, and the widest forced CV. A good fit tracks the black dots over ages 65-100.",
        "- Right panel: the same comparison as fold-error. The horizontal zero line means exact agreement with HMD. The gray band is roughly within \\(2^{0.5}\\approx1.4\\)-fold of HMD.",
        "- Broad-spread fits can be partially compensated, but they still tend to bend the old-age mortality curve in a way the data do not like.",
        "",
        "Takeaway: the score penalty is visible in the age pattern, not just in an abstract objective function.",
        "",
        "Profile mortality fit errors:",
        "",
        "| CV(tau) | median tau/tau_K | fit score | mortality log RMSE | median fold error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            f"| {row['tau_cv']:.3f} | {row['tau_factor_vs_karin']:.2f} | {row['eval_score']:.4g} | {row['hazard_log_rmse_65_100']:.3f} | {row['median_fold_error_65_100']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Result 4: old-age tails after best refit at each CV",
            "",
            f"Across the forced-CV grid, \\(S(110\\mid90)\\) ratios to Sweden 2019 HMD ranged from {fmt_range(tails['S110_ratio_to_hmd'], 2)}.",
            f"Top 0.01% simulated lifespan ranged from {fmt_range(tails['top_0_01pct_lifespan'], 1)} years; values marked with \\(\\ge\\) are censored at the simulation maximum.",
            "",
            rel_plot(plot_paths[3]),
            "",
            "How to read this figure:",
            "",
            "- Left panel: this is the old-age survival-tail penalty. The y-axis is \\(S(110\\mid90)\\) in the model divided by Sweden 2019 HMD. A value of 1 would match HMD. Values above 1 mean too many people surviving from 90 to 110.",
            "- Right panel: this converts the same tail behavior into the age reached by the top 0.01% of the simulated cohort.",
            "- This is the reviewer-relevant part. Even when the central mortality fit is re-optimized, smooth spread in \\(\\tau\\) puts probability mass in the favorable tail, and those rare slow-aging individuals dominate extreme survival.",
            "",
            "Takeaway: the constraint is most naturally a constraint on the favorable tail of \\(\\tau=\\beta/\\eta\\), not a distribution-free ban on every possible central variance.",
            "",
            "Tail summary:",
            "",
            "| CV(tau) | median tau/tau_K | fit score | score ratio | SWE Xc CV | USA Xc CV | S(110\\|90) / HMD | top 0.01% age |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in tails.iterrows():
        top_age = f"{row['top_0_01pct_lifespan']:.1f}"
        if bool(row.get("top_0_01pct_censored", False)):
            top_age = f">={top_age}"
        lines.append(
            f"| {row['tau_cv']:.3f} | {row['tau_factor_vs_karin']:.2f} | {row['eval_score']:.4g} | {row['score_ratio_to_best']:.2f} | {row['SWE_xc_std_frac']:.3f} | {row['USA_xc_std_frac']:.3f} | {row['S110_ratio_to_hmd']:.2f} | {top_age} |"
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
            "- Lower-left is the good region: low fit-score ratio and HMD-like age-110 survival.",
            "- Moving right means the central old-age mortality fit gets worse. Moving up means the extreme survival tail gets worse.",
            "- The labels show \\(\\mathrm{CV}(\\tau)\\). A forced-CV point is only scientifically persuasive if it remains reasonable on both the mortality shape and the old-age survival tail.",
            "",
            "Takeaway: broad smooth \\(\\tau\\)-spread can sometimes be made less bad in the central fit, but it still pays a tail cost. That is exactly the argument for a favorable-tail constraint.",
            "",
            "## Bottom Line",
            "",
            "This all-CV stress scan gives the model broad compensatory freedom: median \\(\\tau\\), \\(\\eta\\), \\(\\beta\\), \\(\\kappa\\), \\(\\epsilon\\), Sweden/USA \\(X_c\\), and Sweden/USA \\(X_c\\)-heterogeneity all move. The best available numerical rescue is still limited. Smooth variation in \\(\\tau=\\beta/\\eta\\) quickly creates a favorable tail of slow-aging individuals, and that tail inflates extreme survival even when the central mortality fit is re-optimized.",
            "",
            "The manuscript-safe conclusion is therefore: human late-life data constrain the favorable tail of the senogenic timescale. Larger apparent central variation is possible only if the favorable tail is sharply truncated, depleted, or compensated in a way that preserves \\(\\tau\\).",
            "",
            "## Outputs",
            "",
            f"- Canonical best candidate per forced CV: `{FIT_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- All refined stress-search candidates: `{ALL_CANDIDATES_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Convenience copy of best candidates: `{BEST_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
            f"- Canonical curves and tail metrics: `{CURVE_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.",
        ]
    )
    if int(config.curve_n) == 1_000_000:
        lines.append(f"- High-n supplement curves: `{HIGH_N_CURVE_RESULTS_PATH.relative_to(PROJECT_ROOT)}`.")
    lines.append(f"- Cache: `{CACHE_DIR.relative_to(PROJECT_ROOT)}`.")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    wide.set_runtime(wide_config(config))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plots_only = bool(args.plots_only)
    if plots_only:
        candidates, profile, curves = load_outputs(config, require_metadata=False)
    elif not args.force and metadata_matches(config) and FIT_RESULTS_PATH.exists() and ALL_CANDIDATES_PATH.exists() and CURVE_RESULTS_PATH.exists():
        candidates, profile, curves = load_outputs(config, require_metadata=True)
    else:
        candidates, profile = run_stress_grid(config)
        curves = curve_fits_from_profile(profile, config)
        write_outputs(candidates, profile, curves, config)

    plot_paths = make_plots(profile, curves)
    write_report(candidates, profile, curves, plot_paths, config)
    if not plots_only:
        write_outputs(candidates, profile, curves, config)
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved canonical candidates: {FIT_RESULTS_PATH}")
    print(f"Saved all candidates: {ALL_CANDIDATES_PATH}")
    print(f"Saved curves: {CURVE_RESULTS_PATH}")
    for path in plot_paths:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
