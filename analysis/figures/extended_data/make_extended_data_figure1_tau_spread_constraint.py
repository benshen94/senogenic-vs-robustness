#!/usr/bin/env python3
"""Make Extended Data Fig. 1 for forced tau-senogenic heterogeneity."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR


OUTPUT_DIR = FIGURES_DIR / "ExtendedDataFigure1"
PNG_PATH = OUTPUT_DIR / "extended_data_figure1_tau_spread_constraint.png"
PDF_PATH = OUTPUT_DIR / "extended_data_figure1_tau_spread_constraint.pdf"

TAU_RESULTS_DIR = RESULTS_DIR / "tau_wide_refit_profile"
CANDIDATES_PATH = TAU_RESULTS_DIR / "tau_wide_refit_profile_candidates.csv"
PANEL_B_CURVES_PATH = TAU_RESULTS_DIR / "figS_tau_spread_constraint_panel_b_curves_n1000000_age55.csv"
BASELINE_CURVE_PATH = TAU_RESULTS_DIR / "figS_tau_spread_constraint_baseline_sweden_fit_curve.csv"
BASELINE_CI_PATH = TAU_RESULTS_DIR / "figS_tau_spread_constraint_baseline_sweden_fit_ci_envelope.csv"
SOURCE_DATA_PATH = RESULTS_DIR / "tables" / "extended_data_figure1_tau_spread_constraint_source_data.csv"

SELECTED_MORTALITY_CVS = [0.025, 0.05, 0.10, 0.15]
MORTALITY_MIN_AGE = 55.0
MORTALITY_MAX_AGE = 100.0

PROFILE_COLOR = "#0B8793"
XC_COLOR = "#D77A16"
CI_BAND_ALPHA = 0.16


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 13,
            "axes.labelsize": 19,
            "axes.titlesize": 21,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 10.5,
            "axes.linewidth": 1.15,
            "xtick.major.width": 1.15,
            "ytick.major.width": 1.15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="normal",
        va="top",
        ha="left",
    )


def format_percent(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))}%"
    return f"{value:g}%"


def percent_formatter(value: float, _pos: int) -> str:
    return format_percent(value)


def profile_fits(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates[candidates["source"] == "profile_tau_sd"].copy()
    data["heterogeneity_pct"] = 100.0 * data["tau_cv"]
    return data.sort_values("heterogeneity_pct")


def heterogeneity_fits_for_plot(candidates: pd.DataFrame) -> pd.DataFrame:
    profile = profile_fits(candidates)
    zero = candidates[(candidates["tau_cv"].abs() < 1e-12) & candidates["eval_score"].notna()].copy()
    if zero.empty:
        return profile
    best_zero = zero.sort_values("eval_score").iloc[[0]].copy()
    best_zero["heterogeneity_pct"] = 0.0
    positive_profile = profile[profile["tau_cv"] > 1e-12]
    return (
        pd.concat([best_zero, positive_profile], ignore_index=True, sort=False)
        .sort_values("heterogeneity_pct")
        .reset_index(drop=True)
    )


def nearest_profile_id(candidates: pd.DataFrame, cv: float) -> str:
    profiles = candidates[candidates["source"] == "profile_tau_sd"].copy()
    idx = (profiles["tau_cv"] - cv).abs().idxmin()
    return str(profiles.loc[idx, "candidate_id"])


def candidate_id_for_heterogeneity(candidates: pd.DataFrame, cv: float) -> str:
    if abs(cv) < 1e-12:
        fits = heterogeneity_fits_for_plot(candidates)
        return str(fits.iloc[0]["candidate_id"])
    return nearest_profile_id(candidates, cv)


def selected_cv_label(cv: float) -> str:
    return f"{format_percent(100.0 * cv)} heterogeneity"


def build_source_data(
    candidates: pd.DataFrame,
    curves: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    baseline_ci: pd.DataFrame,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    profile = heterogeneity_fits_for_plot(candidates).copy()
    profile["panel"] = "A_fit_objective"
    parts.append(profile)

    hazard = curves[curves["curve"] == "hazard"].copy()
    hazard["panel"] = "B_mortality_curves"
    parts.append(hazard)

    baseline_curve = baseline_curve.copy()
    baseline_curve["panel"] = "B_mortality_curves"
    parts.append(baseline_curve)

    baseline_ci = baseline_ci.copy()
    baseline_ci["panel"] = "B_baseline_fit_ci_envelope"
    parts.append(baseline_ci)

    return pd.concat(parts, ignore_index=True, sort=False)


def style_percent_axis(ax: plt.Axes, xmax: float = 26.5) -> None:
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(percent_formatter))
    ax.set_xlim(-1.0, xmax)
    ax.set_xticks([0, 5, 10, 15, 20, 25])


def plot_fit_profile(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    profile = heterogeneity_fits_for_plot(candidates)
    baseline_score = float(candidates["best_eval_score"].dropna().iloc[0])

    ax.plot(
        profile["heterogeneity_pct"],
        profile["eval_score"],
        color=PROFILE_COLOR,
        linewidth=2.7,
        marker="o",
        markersize=7.4,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.axhline(baseline_score, color="#5F5F5F", linestyle="--", linewidth=1.7)
    ax.text(
        11.0,
        baseline_score * 1.16,
        f"baseline fit score = {baseline_score:.3f}",
        color="#4C4C4C",
        fontsize=13.0,
        ha="left",
        va="bottom",
    )
    ax.set_yscale("log")
    ax.set_ylim(0.006, 1.60)
    style_percent_axis(ax)
    ax.set_title("Imposed senogenic heterogeneity worsens fit")
    ax.set_xlabel(r"Imposed $\tau_{\rm sen}$ heterogeneity (%)")
    ax.set_ylabel("Fit objective score")
    add_panel_label(ax, "a")


def plot_mortality_curves(
    ax: plt.Axes,
    candidates: pd.DataFrame,
    curves: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    baseline_ci: pd.DataFrame,
) -> None:
    hazard = curves[curves["curve"] == "hazard"].copy()
    hmd = hazard[hazard["candidate_id"] == "HMD"].sort_values("age")
    hmd = hmd[hmd["age"].between(MORTALITY_MIN_AGE, MORTALITY_MAX_AGE)]
    ax.plot(
        hmd["age"],
        hmd["value"],
        color="black",
        linewidth=4.0,
        label="Sweden 2019 HMD",
        zorder=5,
    )

    baseline_curve = baseline_curve.sort_values("age")
    baseline_ci = baseline_ci.sort_values("age")
    ax.fill_between(
        baseline_ci["age"],
        baseline_ci["ci_lower"],
        baseline_ci["ci_upper"],
        color=XC_COLOR,
        alpha=CI_BAND_ALPHA,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        baseline_curve["age"],
        baseline_curve["value"],
        color=XC_COLOR,
        linewidth=4.0,
        alpha=0.98,
        label="Baseline fit",
        zorder=4,
    )

    gradient = mpl.colormaps["viridis"](np.linspace(0.28, 0.82, len(SELECTED_MORTALITY_CVS)))
    for cv, color in zip(SELECTED_MORTALITY_CVS, gradient):
        candidate_id = candidate_id_for_heterogeneity(candidates, cv)
        subset = hazard[hazard["candidate_id"] == candidate_id].sort_values("age")
        subset = subset[subset["age"].between(MORTALITY_MIN_AGE, MORTALITY_MAX_AGE)]
        ax.plot(
            subset["age"],
            subset["value"],
            color=color,
            linewidth=1.8,
            alpha=0.98,
            label=selected_cv_label(cv),
        )

    ax.set_yscale("log")
    ax.set_xlim(MORTALITY_MIN_AGE, MORTALITY_MAX_AGE)
    ax.set_ylim(0.0025, 0.60)
    ax.set_xticks(np.arange(55, 101, 5))
    ax.set_title("Senogenic heterogeneity misses old-age mortality")
    ax.set_xlabel("Age [years]")
    ax.set_ylabel(r"Mortality rate [year$^{-1}$]")
    ax.legend(loc="upper left", frameon=False, handlelength=2.2)
    add_panel_label(ax, "b")


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATES_PATH)
    curves = pd.read_csv(PANEL_B_CURVES_PATH)
    baseline_curve = pd.read_csv(BASELINE_CURVE_PATH)
    baseline_ci = pd.read_csv(BASELINE_CI_PATH)

    source_data = build_source_data(candidates, curves, baseline_curve, baseline_ci)
    source_data.to_csv(SOURCE_DATA_PATH, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)
    plot_fit_profile(axes[0], candidates)
    plot_mortality_curves(axes[1], candidates, curves, baseline_curve, baseline_ci)

    fig.savefig(PDF_PATH, bbox_inches="tight")
    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {PNG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {PDF_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {SOURCE_DATA_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
