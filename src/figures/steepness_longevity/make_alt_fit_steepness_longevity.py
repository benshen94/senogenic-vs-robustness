#!/usr/bin/env python3
"""Generate a steepness-longevity plane for the alternative SR fit."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from src.shared.thresholds import thresholds_functions as th
from src.shared.thresholds.paths import FIGURES_DIR


RESULT_STEM = "alt_fit_naveh"
RESULTS_PATH = Path(th.saved_results_path) / f"param_variation_results_{RESULT_STEM}.pkl"
PNG_PATH = FIGURES_DIR / "steepness_longevity_alt_fit_naveh.png"
PDF_PATH = FIGURES_DIR / "steepness_longevity_alt_fit_naveh.pdf"

N_SIM = int(1e5)
TMAX = 120
DT = 1 / (365 * 2)
SAVE_TIMES = TMAX
BASE_H_EXT = 0.0
FROM_T = 0

ALT_PARAMS = {
    "eta": 1.3,
    "beta": 173.9,
    "kappa": 0.5,
    "epsilon": 0.833,
    "Xc": 1.23,
}

XC_STD_REL = 0.27
FACTOR_VALUES = np.arange(0.5, 1.51, 0.1)
PARAMS_TO_VARY = ["eta", "beta", "kappa", "epsilon", "Xc"]
RNG_SEED = 7
INCLUDE_H_EXT = True


def sample_positive_gaussian(mean: float, rel_std: float, n: int, seed: int) -> np.ndarray:
    """Draw positive samples from a Gaussian by resampling negatives."""
    rng = np.random.default_rng(seed)
    std = mean * rel_std
    values = rng.normal(loc=mean, scale=std, size=n)

    while True:
        negative_mask = values <= 0
        if not np.any(negative_mask):
            return values

        values[negative_mask] = rng.normal(
            loc=mean,
            scale=std,
            size=int(np.sum(negative_mask)),
        )


def build_alt_baseline_dict(n: int) -> dict[str, np.ndarray]:
    """Build the alternative-fit baseline with Xc heterogeneity."""
    baseline = {}

    for param_name, value in ALT_PARAMS.items():
        baseline[param_name] = np.full(n, value, dtype=float)

    baseline["Xc"] = sample_positive_gaussian(
        mean=ALT_PARAMS["Xc"],
        rel_std=XC_STD_REL,
        n=n,
        seed=RNG_SEED,
    )
    return baseline


def ensure_results_exist() -> None:
    """Run the parameter study once and reuse the saved pickle afterward."""
    if RESULTS_PATH.exists():
        print(f"Using existing results: {RESULTS_PATH}")
        return

    baseline_dict = build_alt_baseline_dict(N_SIM)

    th.run_parameter_study(
        study_type="variation",
        baseline_dict=baseline_dict,
        factors=FACTOR_VALUES,
        name=RESULT_STEM,
        n=N_SIM,
        params=PARAMS_TO_VARY,
        include_h_ext=INCLUDE_H_EXT,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        h_ext=BASE_H_EXT,
        break_early=True,
    )


def apply_plot_style() -> None:
    """Use a clean, readable style that matches the existing plane."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )


def save_figure() -> None:
    """Render and save the alternative-fit steepness-longevity plane."""
    apply_plot_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    legend = th.plot_steepness_longevity(
        pkl_file=RESULTS_PATH.name,
        param_type="variation",
        from_t=FROM_T,
        longevity_metric="t_median_absolute",
        steepness_metric="steepness_iqr_absolute",
        ignore_kappa=False,
        ax=ax,
        title="Alternative-fit steepness-longevity plane",
        value_type="normalized",
        alpha=0.95,
        marker_size_range=(40, 140),
        linewidth=2.4,
        line_alpha=0.85,
        h_ext=INCLUDE_H_EXT,
        legend_fontsize=12,
        legend_loc="upper left",
        legend_title="Parameter change",
    )

    ax.set_xlabel("Normalized median lifespan")
    ax.set_ylabel("Normalized IQR steepness")
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.5, 1.5)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=6, width=1.2)
    ax.margins(x=0.06, y=0.08)

    if legend is not None:
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_edgecolor("none")

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {PNG_PATH}")


def main() -> None:
    ensure_results_exist()
    save_figure()


if __name__ == "__main__":
    main()
