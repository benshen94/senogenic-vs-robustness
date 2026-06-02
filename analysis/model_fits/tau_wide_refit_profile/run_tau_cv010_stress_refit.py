#!/usr/bin/env python3
"""Targeted stronger refit with forced 10% CV in tau."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
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


REPORT_PATH = Path(__file__).resolve().parent / "tau_cv010_stress_refit_analysis.md"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
FIT_RESULTS_PATH = wide.CSV_DIR / "tau_cv010_stress_refit_candidates.csv"
CURVE_RESULTS_PATH = wide.CSV_DIR / "tau_cv010_stress_refit_curves.csv"

TARGET_CV = 0.10
TARGET_TAU_SD = math.sqrt(math.log1p(TARGET_CV**2))
LOCAL_STEPS = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
LOCAL_SWEEPS = 4
EVAL_REPEATS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-random", type=int, default=24)
    parser.add_argument("--screen-keep", type=int, default=10)
    parser.add_argument("--de-maxiter", type=int, default=5)
    parser.add_argument("--de-popsize", type=int, default=4)
    parser.add_argument("--screen-n", type=int, default=1_500)
    parser.add_argument("--fit-n", type=int, default=5_000)
    parser.add_argument("--eval-n", type=int, default=60_000)
    parser.add_argument("--curve-n", type=int, default=100_000)
    parser.add_argument("--fit-dt", type=float, default=0.08)
    parser.add_argument("--curve-dt", type=float, default=0.1)
    parser.add_argument("--no-parallel", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> wide.FitConfig:
    return wide.FitConfig(
        profile_starts_per_sd=0,
        full_starts=0,
        fit_n=args.fit_n,
        eval_n=args.eval_n,
        curve_n=args.curve_n,
        fit_dt=args.fit_dt,
        curve_dt=args.curve_dt,
        acceptance_multiplier=2.5,
        parallel=not args.no_parallel,
    )


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def vector_from_row(row: pd.Series) -> np.ndarray:
    return np.asarray([row[col] for col in wide.PARAMETER_COLUMNS], dtype=float)


def existing_candidate_starts() -> list[tuple[str, np.ndarray]]:
    starts: list[tuple[str, np.ndarray]] = []
    if wide.FIT_RESULTS_PATH.exists():
        fits = pd.read_csv(wide.FIT_RESULTS_PATH)
        for _, row in fits.iterrows():
            starts.append((f"wide_{row['candidate_id']}", vector_from_row(row)))
    return starts


def random_starts(count: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(wide.RANDOM_SEED + 10_010)
    lowers = np.asarray([bound[0] for bound in wide.WIDE_BOUNDS], dtype=float)
    uppers = np.asarray([bound[1] for bound in wide.WIDE_BOUNDS], dtype=float)
    return [(f"stress_random_{idx:02d}", rng.uniform(lowers, uppers)) for idx in range(int(count))]


def make_start_bank(random_count: int) -> list[tuple[str, np.ndarray]]:
    starts = []
    starts.extend(wide.structured_starts())
    starts.extend(existing_candidate_starts())
    starts.extend(random_starts(random_count))
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


def local_coordinate_refine(
    objective,
    start: np.ndarray,
    *,
    step_sizes: tuple[float, ...] = LOCAL_STEPS,
    max_sweeps: int = LOCAL_SWEEPS,
) -> tuple[np.ndarray, float, list[float]]:
    current = wide.clip_vector(start)
    current_score = float(objective(current))
    history = [current_score]
    for step_size in step_sizes:
        improved = True
        sweeps = 0
        while improved and sweeps < max_sweeps:
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


def score_fn(data_by_country: dict[str, object], n: int, seed_label: str, memo: dict[tuple[object, ...], float]):
    seed = wide.stable_seed("cv010-stress", seed_label)

    def objective(vector: np.ndarray) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return wide.score_candidate(vector, TARGET_TAU_SD, data_by_country, n, seed, memo)

    return objective


def screen_starts(
    starts: list[tuple[str, np.ndarray]],
    data_by_country: dict[str, object],
    args: argparse.Namespace,
) -> pd.DataFrame:
    memo: dict[tuple[object, ...], float] = {}
    objective = score_fn(data_by_country, args.screen_n, "screen", memo)
    rows = []
    for name, vector in starts:
        rows.append({"stage": "screen", "name": name, "score": float(objective(vector)), **{col: val for col, val in zip(wide.PARAMETER_COLUMNS, vector)}})
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def differential_evolution_start(
    start_bank: list[tuple[str, np.ndarray]],
    data_by_country: dict[str, object],
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    memo: dict[tuple[object, ...], float] = {}
    objective = score_fn(data_by_country, args.screen_n, "de", memo)
    required = max(5, args.de_popsize * len(wide.PARAMETER_COLUMNS))
    init_vectors = [vector for _, vector in start_bank[:required]]
    if len(init_vectors) < required:
        init_vectors.extend([vector for _, vector in random_starts(required - len(init_vectors))])
    init = np.vstack([wide.clip_vector(vector) for vector in init_vectors])
    result = differential_evolution(
        objective,
        bounds=list(wide.WIDE_BOUNDS),
        maxiter=args.de_maxiter,
        popsize=args.de_popsize,
        init=init,
        polish=False,
        updating="immediate",
        workers=1,
        seed=wide.stable_seed("cv010-de-seed"),
        tol=0.0,
        atol=0.0,
        disp=False,
    )
    return wide.clip_vector(result.x), float(result.fun)


def evaluate_repeats(
    vectors: list[tuple[str, np.ndarray, float, list[float]]],
    data_by_country: dict[str, object],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    for name, vector, search_score, history in vectors:
        scores = []
        for repeat in range(EVAL_REPEATS):
            score = wide.score_candidate(
                vector,
                TARGET_TAU_SD,
                data_by_country,
                args.eval_n,
                wide.stable_seed("cv010-eval-repeat", name, repeat),
                memo=None,
            )
            scores.append(score)
        values = wide.natural_values(vector)
        row = {
            "candidate_id": name,
            "source": "cv010_stress_refit",
            "tau_sd": TARGET_TAU_SD,
            "tau_cv": TARGET_CV,
            "search_score": float(search_score),
            "eval_score_mean": float(np.mean(scores)),
            "eval_score_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "eval_score_min": float(np.min(scores)),
            "eval_score_max": float(np.max(scores)),
            "history_start": float(history[0]) if history else np.nan,
            "history_end": float(history[-1]) if history else search_score,
            "history_steps": len(history),
        }
        row.update({col: float(val) for col, val in zip(wide.PARAMETER_COLUMNS, wide.clip_vector(vector))})
        row.update(values)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("eval_score_mean").reset_index(drop=True)
    best_score = float(pd.read_csv(wide.FIT_RESULTS_PATH)["eval_score"].min()) if wide.FIT_RESULTS_PATH.exists() else np.nan
    frame["wide_best_eval_score"] = best_score
    frame["score_ratio_to_wide_best"] = frame["eval_score_mean"] / best_score
    return frame


def build_best_curve(best: pd.Series, config: wide.FitConfig) -> pd.DataFrame:
    row = {
        "candidate_id": "cv010_stress_best",
        "label": "10% CV stress best",
        "source": "profile_tau_sd",
        "eval_score": float(best["eval_score_mean"]),
        "search_score": float(best["search_score"]),
        "tau_sd": TARGET_TAU_SD,
        "tau_cv": TARGET_CV,
        "best_eval_score": float(best["wide_best_eval_score"]),
        "score_ratio_to_best": float(best["score_ratio_to_wide_best"]),
        "acceptance_multiplier": 2.5,
        "strict_accepted": bool(best["score_ratio_to_wide_best"] <= 2.5),
    }
    for col in wide.PARAMETER_COLUMNS:
        row[col] = float(best[col])
    row.update({key: best[key] for key in ["eta", "beta", "kappa", "epsilon", "tau", "tau_factor_vs_karin", "eta_factor", "beta_factor", "kappa_factor", "epsilon_factor", "SWE_Xc", "SWE_xc_std_frac", "USA_Xc", "USA_xc_std_frac"]})
    return wide.build_curves(pd.DataFrame([row]), config)


def plot_best_mortality(curves: pd.DataFrame) -> Path:
    configure_matplotlib()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    hazard = curves[curves["curve"] == "hazard"].copy()
    hmd = hazard[hazard["candidate_id"] == "HMD"].copy()
    fit = hazard[hazard["candidate_id"] == "cv010_stress_best"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), gridspec_kw={"width_ratios": [1.15, 1.0]})
    axes[0].scatter(hmd["age"], hmd["value"], s=28, color="black", label="Sweden 2019 HMD", zorder=5)
    axes[0].plot(fit["age"], fit["value"], lw=2.4, color="#0B7F8C", label="10% CV stress best")
    hmd_lookup = hmd[["age", "value"]].rename(columns={"value": "hmd_hazard"})
    resid = fit[["age", "value"]].merge(hmd_lookup, on="age", how="inner")
    resid["log2_fold"] = np.log2(resid["value"] / resid["hmd_hazard"])
    axes[1].plot(resid["age"], resid["log2_fold"], lw=2.2, color="#0B7F8C")
    axes[1].axhline(0, color="black", lw=1.1)
    axes[1].axhspan(-0.5, 0.5, color="#EDEDED", zorder=-10)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Mortality rate / hazard")
    axes[0].set_title("Forced 10% CV: best stress refit")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel(r"Fit / HMD mortality, $\log_2$ fold")
    axes[1].set_title("Age-by-age residuals")
    for ax in axes:
        ax.set_xlim(65, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)
    path = PLOTS_DIR / "06_tau_cv010_stress_mortality.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / "06_tau_cv010_stress_mortality.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(candidates: pd.DataFrame, curves: pd.DataFrame, plot_path: Path, args: argparse.Namespace) -> None:
    best = candidates.iloc[0]
    tail = curves[(curves["curve"] == "tail_metric") & (curves["candidate_id"] == "cv010_stress_best")]
    s110 = float(tail.loc[tail["age"] == 110.0, "ratio_to_hmd"].iloc[0])
    top_age = float(tail["top_0_01pct_lifespan"].iloc[0])
    censored = bool(tail["top_0_01pct_censored"].iloc[0])
    top_text = f">={top_age:.1f}" if censored else f"{top_age:.1f}"
    previous = pd.read_csv(wide.FIT_RESULTS_PATH)
    prev_010 = previous[previous["candidate_id"] == "profile_tau_sd_0.100"].iloc[0]
    lines = [
        "# Forced 10% Tau-CV Stress Refit",
        "",
        "Question: if \\(\\mathrm{CV}(\\tau)\\) is forced to 10%, can a stronger targeted search find a good compensating SR fit?",
        "",
        "This run fixes",
        "",
        "$$",
        f"\\mathrm{{CV}}(\\tau)={TARGET_CV:.2f},\\qquad \\sigma_v=\\sqrt{{\\log(1+{TARGET_CV:.2f}^2)}}={TARGET_TAU_SD:.5f}.",
        "$$",
        "",
        "All wide SR parameters were still free: median \\(\\tau\\), \\(\\eta\\), \\(\\kappa\\), \\(\\epsilon\\), Sweden/USA \\(X_c\\), and Sweden/USA \\(X_c\\) heterogeneity.",
        "",
        "## Search Strategy",
        "",
        f"- Screened structured/archive/wide-profile/random starts, including {args.screen_random} new random wide-bounds starts.",
        f"- Ran differential evolution with maxiter={args.de_maxiter}, popsize={args.de_popsize} at \\(n={args.screen_n:,}\\).",
        f"- Locally refined the differential-evolution result plus the best screened starts with steps {LOCAL_STEPS}, using \\(n={args.fit_n:,}\\).",
        f"- Evaluated top refined candidates with {EVAL_REPEATS} independent high-\\(n\\) scores at \\(n={args.eval_n:,}\\).",
        f"- Built the final mortality/tail curve with \\(n={args.curve_n:,}\\).",
        "",
        "## Result",
        "",
        f"Best stress candidate mean evaluation score: {best['eval_score_mean']:.4g} +/- {best['eval_score_sd']:.4g}.",
        f"Best score ratio to the best wide-profile fit: {best['score_ratio_to_wide_best']:.2f}x.",
        f"Previous wide-grid 10% profile score: {prev_010['eval_score']:.4g}, ratio {prev_010['score_ratio_to_best']:.2f}x.",
        "",
        "Interpretation: the original wide-grid 10% point was **not** the best 10% solution. A harder search found a much better forced-10% fit. But the improved fit is still just above the exploratory \\(2.5\\times\\) fit threshold, and its tail remains far too large.",
        "",
        f"Best stress candidate median \\(\\tau/\\tau_K={best['tau_factor_vs_karin']:.2f}\\). So the optimizer did move the median timescale down by about 20%, but it did not find a radically shifted median-\\(\\tau\\) solution that makes 10% spread look acceptable.",
        "",
        f"Tail for best stress candidate: \\(S(110\\mid90)\\) is {s110:.1f}x HMD; top 0.01% simulated lifespan is {top_text} years.",
        "",
        f"![{plot_path.stem}](plots/{plot_path.name})",
        "",
        "How to read this figure:",
        "",
        "- Left panel compares the best forced-10% fit to Sweden 2019 HMD mortality. A good rescue would closely track the black points across ages 65-100.",
        "- Right panel shows the same fit as log2 fold-error. The gray band is roughly within 1.4-fold of HMD.",
        "- The fit still has systematic curvature rather than random small errors, which is the same qualitative failure seen in the wide profile.",
        "",
        "Takeaway: this stronger targeted search improved the 10% forced-CV fit somewhat if it found a lower score than the original grid point, but it did not make 10% CV look like an acceptable solution. The score remains far above the best low-spread fit and the old-age tail remains strongly inflated.",
        "",
        "## Top Candidates",
        "",
        "| candidate | eval score mean | eval sd | ratio to best wide fit | tau/tau_K | eta factor | beta factor | kappa factor | epsilon factor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in candidates.head(8).iterrows():
        lines.append(
            f"| {row['candidate_id']} | {row['eval_score_mean']:.4g} | {row['eval_score_sd']:.3g} | {row['score_ratio_to_wide_best']:.2f} | {row['tau_factor_vs_karin']:.2f} | {row['eta_factor']:.2f} | {row['beta_factor']:.2f} | {row['kappa_factor']:.2f} | {row['epsilon_factor']:.2f} |"
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    wide.set_runtime(config)
    data_by_country = wide.load_data_by_country()

    starts = make_start_bank(args.screen_random)
    screened = screen_starts(starts, data_by_country, args)
    top_starts = [(row["name"], np.asarray([row[col] for col in wide.PARAMETER_COLUMNS], dtype=float)) for _, row in screened.head(args.screen_keep).iterrows()]

    de_vector, de_score = differential_evolution_start(top_starts + starts, data_by_country, args)
    refine_starts = [("de_best", de_vector), *top_starts]
    memo: dict[tuple[object, ...], float] = {}
    objective = score_fn(data_by_country, args.fit_n, "local-refine", memo)
    refined = []
    for name, vector in refine_starts:
        fit_vector, search_score, history = local_coordinate_refine(objective, vector)
        refined.append((name, fit_vector, search_score, history))
        print(f"refined {name}: start_screen={de_score if name == 'de_best' else np.nan:.4g} search={search_score:.4g}", flush=True)
    refined.sort(key=lambda item: item[2])
    candidates = evaluate_repeats(refined[: min(6, len(refined))], data_by_country, args)
    config_for_curve = wide.FitConfig(
        profile_starts_per_sd=0,
        full_starts=0,
        fit_n=args.fit_n,
        eval_n=args.eval_n,
        curve_n=args.curve_n,
        fit_dt=args.fit_dt,
        curve_dt=args.curve_dt,
        acceptance_multiplier=2.5,
        parallel=not args.no_parallel,
    )
    curves = build_best_curve(candidates.iloc[0], config_for_curve)

    wide.CSV_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(FIT_RESULTS_PATH, index=False)
    curves.to_csv(CURVE_RESULTS_PATH, index=False)
    plot_path = plot_best_mortality(curves)
    write_report(candidates, curves, plot_path, args)
    print(f"Saved {REPORT_PATH}")
    print(f"Saved {FIT_RESULTS_PATH}")
    print(f"Saved {CURVE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
