#!/usr/bin/env python3
"""Make the supplementary Gompertz and Fedichev-Gruber model comparison.

This script replaces the legacy notebook figure from
`src/notebooks/fedichev_gompertz_models.ipynb`.
"""

from __future__ import annotations

import csv
import hashlib
import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_models.gamma_gompertz import GammaGompertz as gg
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Supp_Figgs"
PNG_PATH = OUTPUT_DIR / "supp_model_comparison.png"
PDF_PATH = OUTPUT_DIR / "supp_model_comparison.pdf"

MAX_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "supp_model_comparison_max_lifespan.csv"
SHAPE_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "supp_model_comparison_shape_response.csv"

GOMPERTZ_SHAPE_PATH = SAVED_RESULTS_DIR / "gamma_factor_sweep.pkl"
FEDICHEV_SHAPE_PATH = SAVED_RESULTS_DIR / "fedichev_model_steepness_longevity_data.pkl"

LEGACY_NOTEBOOK = "src/notebooks/fedichev_gompertz_models.ipynb"
OUTPUT_INDEX_PATH = SAVED_RESULTS_DIR / "index" / "outputs.csv"

RANDOM_SEED = 20260520
GOMPERTZ_N = 300_000
FEDICHEV_N = 300_000
FEDICHEV_DT = 0.05
FEDICHEV_TMAX = 1000.0
TAIL_SURVIVAL_FRACTION = 1e-4
GOMPERTZ_STD_VALUES = np.arange(0.0, 0.2001, 0.025)
FEDICHEV_STD_VALUES = np.arange(0.0, 0.2001, 0.025)
FEDICHEV_FACTORS = np.arange(0.6, 1.4001, 0.1)

GOMPERTZ_MAX_PARAMS = ("a_linear", "a_power", "b", "coupled_ab")
GOMPERTZ_SHAPE_PARAMS = (
    "intercept_a_linear",
    "intercept_a_exp",
    "slope_b",
    "coupled_intercept_slope",
    "makeham_m",
)
FEDICHEV_PARAMS = (
    "beta_prime",
    "epsilon_0_init",
    "gamma",
    "beta",
    "g",
    "D0",
)

GOMPERTZ_LABELS = {
    "a_linear": r"Intercept $a$ (linear)",
    "a_power": r"Intercept $a$ (exponent)",
    "b": r"Slope $b$",
    "coupled_ab": r"Coupled $a+b$",
    "intercept_a_linear": r"Intercept $a$ (linear)",
    "intercept_a_exp": r"Intercept $a$ (exponent)",
    "slope_b": r"Slope $b$",
    "coupled_intercept_slope": r"Coupled $a+b$",
    "makeham_m": r"Makeham $m$",
}

FEDICHEV_LABELS = {
    "beta_prime": r"$\beta'$ resilience decay",
    "epsilon_0_init": r"$\epsilon_0$ initial resilience",
    "gamma": r"$\gamma$ damage accumulation",
    "beta": r"$\beta$ damage coupling",
    "g": r"$g$ nonlinearity",
    "D0": r"$D_0$ noise",
}

GOMPERTZ_COLORS = {
    "a_linear": "#0B7F8C",
    "a_power": "#1F78B4",
    "b": "#173A6A",
    "coupled_ab": "#D77A16",
    "intercept_a_linear": "#0B7F8C",
    "intercept_a_exp": "#1F78B4",
    "slope_b": "#173A6A",
    "coupled_intercept_slope": "#D77A16",
    "makeham_m": "#C51F2F",
}

FEDICHEV_COLORS = {
    "beta_prime": "#173A6A",
    "epsilon_0_init": "#0097A7",
    "gamma": "#2F80ED",
    "beta": "#007F5F",
    "g": "#6C63B5",
    "D0": "#E5A100",
}

SENOGENIC_PARAMS = ("beta_prime", "epsilon_0_init", "gamma", "beta", "g")
ROBUSTNESS_PARAMS = ("D0",)

MAX_X_LIMIT = (-0.35, 20.0)
GOMPERTZ_MAX_Y_LIMIT = (98.0, 150.0)
FEDICHEV_MAX_Y_LIMIT = (118.0, 150.0)
SHAPE_X_LIMIT = (0.45, 1.58)
SHAPE_Y_LIMIT = (0.4, 1.62)


def configure_matplotlib() -> None:
    """Use the manuscript figure style used by the newer figure scripts."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 11.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.8,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-sim",
        action="store_true",
        help="Regenerate the expensive Gompertz/Fedichev extreme-lifespan simulations instead of using cached CSVs.",
    )
    return parser.parse_args()


def fit_legacy_gompertz_model() -> gg:
    """Fit the Gompertz baseline used by the legacy notebook."""
    model = gg()
    model.fit_params(
        country="dan",
        year=1880,
        gender="male",
        data_type="cohort",
        haz_type="mx",
        filter_from=20,
        filter_to=105,
        print_out=False,
    )
    model.c = 100
    model.m = 0
    return model


def positive_normal(rng: np.random.Generator, mean: float, rel_std: float, n: int) -> np.ndarray:
    """Draw positive values from a normal distribution."""
    values = rng.normal(loc=mean, scale=rel_std * mean, size=n)
    while np.any(values <= 0):
        bad = values <= 0
        values[bad] = rng.normal(loc=mean, scale=rel_std * mean, size=int(np.sum(bad)))
    return values


def sample_gompertz_m0_deaths(
    rng: np.random.Generator,
    a_values: np.ndarray | float,
    b_values: np.ndarray | float,
    n: int,
) -> np.ndarray:
    """Sample death times from the m=0 Gompertz limit used in the old panel."""
    thresholds = -np.log(rng.random(n))
    return np.log1p((b_values * thresholds) / a_values) / b_values


def tail_survivor_rank(n: int) -> int:
    """Return the empirical rank corresponding to S(t)=0.0001."""
    return max(1, int(round(n * TAIL_SURVIVAL_FRACTION)))


def tail_lifespan(values: np.ndarray) -> float:
    """Return the simulated lifespan at the S(t)=0.0001 tail."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    rank = tail_survivor_rank(finite.size)
    if finite.size < rank:
        return float("nan")
    return float(np.partition(finite, -rank)[-rank])


def gompertz_parameter_values(
    rng: np.random.Generator,
    model: gg,
    param_name: str,
    std: float,
    n: int,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return per-person a and b values for one Gompertz heterogeneity setting."""
    if param_name == "a_linear":
        return positive_normal(rng, model.a, std, n), model.b

    if param_name == "a_power":
        exponents = rng.normal(loc=1.0, scale=std, size=n)
        return model.a**exponents, model.b

    if param_name == "b":
        return model.a, positive_normal(rng, model.b, std, n)

    if param_name == "coupled_ab":
        b_values = positive_normal(rng, model.b, std, n)
        relative_b = b_values / model.b
        return model.a**relative_b, b_values

    raise ValueError(f"Unknown Gompertz parameter: {param_name}")


def make_gompertz_max_lifespan_data() -> pd.DataFrame:
    """Regenerate the missing legacy Gompertz maximum-lifespan panel data."""
    model = fit_legacy_gompertz_model()
    rows = []

    for param_name in GOMPERTZ_MAX_PARAMS:
        for std in GOMPERTZ_STD_VALUES:
            seed = stable_seed(param_name, round(float(std), 4))
            rng = np.random.default_rng(seed)
            a_values, b_values = gompertz_parameter_values(
                rng=rng,
                model=model,
                param_name=param_name,
                std=float(std),
                n=GOMPERTZ_N,
            )
            deaths = sample_gompertz_m0_deaths(rng, a_values, b_values, GOMPERTZ_N)
            rows.append(
                {
                    "model": "Gompertz",
                    "parameter": param_name,
                    "parameter_label": GOMPERTZ_LABELS[param_name],
                    "variation_percent": 100 * float(std),
                    "max_lifespan": tail_lifespan(deaths),
                }
            )

    return pd.DataFrame(rows)


def load_or_make_max_lifespan_data(force_sim: bool) -> pd.DataFrame:
    if MAX_DATA_PATH.exists() and not force_sim:
        return pd.read_csv(MAX_DATA_PATH)

    max_data = pd.concat(
        [make_gompertz_max_lifespan_data(), make_fedichev_max_lifespan_data()],
        ignore_index=True,
    )
    return add_smoothed_max_lifespan(max_data)


def load_or_make_shape_data(force_sim: bool) -> pd.DataFrame:
    if SHAPE_DATA_PATH.exists() and not force_sim:
        return pd.read_csv(SHAPE_DATA_PATH)

    return pd.concat(
        [make_gompertz_shape_data(), make_fedichev_shape_data()],
        ignore_index=True,
    )


def parameter_or_scalar(values: np.ndarray | float, indices: np.ndarray) -> np.ndarray | float:
    """Return per-alive values while preserving scalars for speed."""
    if isinstance(values, np.ndarray):
        return values[indices]
    return values


def subset_or_scalar(values: np.ndarray | float, mask: np.ndarray) -> np.ndarray | float:
    """Return per-stable values while preserving scalars for speed."""
    if isinstance(values, np.ndarray):
        return values[mask]
    return values


def fedichev_parameter_values(
    rng: np.random.Generator,
    param_name: str,
    std: float,
    n: int,
) -> dict[str, np.ndarray | float]:
    """Return baseline Fedichev-Gruber parameters with one heterogeneous parameter."""
    params: dict[str, np.ndarray | float] = {
        "epsilon_0": 4.0,
        "D0": 1.1,
        "beta": 0.015,
        "g": 0.8,
        "gamma": 1.0,
        "beta_prime": 0.013333,
    }
    params[param_name] = positive_normal(rng, float(params[param_name]), std, n)
    return params


def simulate_fedichev_tail_lifespan(
    rng: np.random.Generator,
    param_name: str,
    std: float,
    n: int,
    dt: float,
    tmax: float,
) -> float:
    """Simulate the Fedichev-Gruber model until the S(t)=0.0001 tail is known."""
    params = fedichev_parameter_values(rng, param_name, std, n)
    target_rank = tail_survivor_rank(n)
    alive_indices = np.arange(n)
    z_alive = np.zeros(n)
    sqrt_dt = np.sqrt(dt)
    noise_strength = np.sqrt(2 * params["D0"]) if isinstance(params["D0"], np.ndarray) else np.sqrt(2 * float(params["D0"]))

    for step in range(int(tmax / dt)):
        t = step * dt
        if alive_indices.size == 0:
            return float("nan")

        epsilon_0 = parameter_or_scalar(params["epsilon_0"], alive_indices)
        d0_noise = parameter_or_scalar(noise_strength, alive_indices)
        beta = parameter_or_scalar(params["beta"], alive_indices)
        g = parameter_or_scalar(params["g"], alive_indices)
        gamma = parameter_or_scalar(params["gamma"], alive_indices)
        beta_prime = parameter_or_scalar(params["beta_prime"], alive_indices)

        z_driver = gamma * t
        epsilon_eff = epsilon_0 - beta_prime * z_driver
        discriminant = epsilon_eff**2 - 4 * g * (beta * z_driver)
        stable_mask = np.full(alive_indices.size, bool(discriminant > 0)) if np.ndim(discriminant) == 0 else discriminant > 0
        dead_mask = ~stable_mask

        if np.any(stable_mask):
            stable_positions = np.where(stable_mask)[0]
            eps_stable = subset_or_scalar(epsilon_eff, stable_mask)
            disc_stable = subset_or_scalar(discriminant, stable_mask)
            g_stable = subset_or_scalar(g, stable_mask)
            beta_stable = subset_or_scalar(beta, stable_mask)
            z_driver_stable = subset_or_scalar(z_driver, stable_mask)
            noise_stable = subset_or_scalar(d0_noise, stable_mask)

            z_unstable = (eps_stable + np.sqrt(disc_stable)) / (2 * g_stable)
            z_current = z_alive[stable_positions]
            drift = beta_stable * z_driver_stable - eps_stable * z_current + g_stable * z_current**2
            diffusion = noise_stable * rng.normal(0, sqrt_dt, size=stable_positions.size)
            z_new = z_current + drift * dt + diffusion
            z_alive[stable_positions] = z_new
            dead_mask[stable_positions[z_new > z_unstable]] = True

        alive_indices = alive_indices[~dead_mask]
        z_alive = z_alive[~dead_mask]
        if alive_indices.size < target_rank:
            return float(t)

    return float(tmax)


def stable_seed(*parts: object) -> int:
    """Create a reproducible uint32 seed from readable metadata."""
    text = "|".join(str(part) for part in (RANDOM_SEED, *parts))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_fedichev_max_lifespan_data() -> pd.DataFrame:
    """Regenerate the Fedichev-Gruber heterogeneity panel data."""
    rows = []
    for param_name in FEDICHEV_PARAMS:
        source_param = "epsilon_0" if param_name == "epsilon_0_init" else param_name
        for std in FEDICHEV_STD_VALUES:
            print(f"Running Fedichev-Gruber extreme simulation for {source_param}, CV={std:.3f}")
            seed = stable_seed("fedichev", source_param, round(float(std), 4))
            rng = np.random.default_rng(seed)
            rows.append(
                {
                    "model": "Fedichev-Gruber",
                    "parameter": param_name,
                    "parameter_label": FEDICHEV_LABELS[param_name],
                    "variation_percent": 100 * float(std),
                    "max_lifespan": simulate_fedichev_tail_lifespan(
                        rng=rng,
                        param_name=source_param,
                        std=float(std),
                        n=FEDICHEV_N,
                        dt=FEDICHEV_DT,
                        tmax=FEDICHEV_TMAX,
                    ),
                }
            )

    return pd.DataFrame(rows)


def add_smoothed_max_lifespan(data: pd.DataFrame) -> pd.DataFrame:
    """Add the smoothed curves used for display."""
    smoothed = []
    for _, group in data.groupby(["model", "parameter"], sort=False):
        group = group.sort_values("variation_percent").copy()
        group["max_lifespan_smoothed"] = gaussian_filter1d(group["max_lifespan"], sigma=0.85)
        smoothed.append(group)
    return pd.concat(smoothed, ignore_index=True)


def make_gompertz_shape_data() -> pd.DataFrame:
    """Load the cached Gompertz steepness-longevity data."""
    import pickle

    with GOMPERTZ_SHAPE_PATH.open("rb") as handle:
        payload = pickle.load(handle)

    baseline = payload["results"]["baseline"][0]
    summary = payload["summary"].copy()
    summary = summary[summary["parameter"].isin(GOMPERTZ_SHAPE_PARAMS)].copy()
    summary["x_norm"] = summary["t_median_abs"] / baseline["t_median_absolute"]
    summary["y_norm"] = summary["steepness_abs"] / baseline["steepness_iqr_absolute"]
    summary["model"] = "Gompertz"
    summary["parameter_label"] = summary["parameter"].map(GOMPERTZ_LABELS)
    return summary[
        [
            "model",
            "parameter",
            "parameter_label",
            "factor",
            "x_norm",
            "y_norm",
        ]
    ]


def make_fedichev_shape_data() -> pd.DataFrame:
    """Load the cached Fedichev-Gruber steepness-longevity data."""
    import pickle

    with FEDICHEV_SHAPE_PATH.open("rb") as handle:
        data = pickle.load(handle)

    rows = []
    for param_name in FEDICHEV_PARAMS:
        values = data["results"][param_name]
        for factor, x_value, y_value in zip(FEDICHEV_FACTORS, values["x"], values["y"]):
            rows.append(
                {
                    "model": "Fedichev-Gruber",
                    "parameter": param_name,
                    "parameter_label": FEDICHEV_LABELS[param_name],
                    "factor": float(factor),
                    "x_norm": float(x_value),
                    "y_norm": float(y_value),
                }
            )

    return pd.DataFrame(rows)


def save_plot_data(max_data: pd.DataFrame, shape_data: pd.DataFrame) -> None:
    """Save the central data used in the figure."""
    MAX_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    max_data.to_csv(MAX_DATA_PATH, index=False)
    shape_data.to_csv(SHAPE_DATA_PATH, index=False)


def marker_sizes(values: pd.Series, size_range: tuple[float, float]) -> np.ndarray:
    """Map curve values to readable marker areas."""
    numeric = values.to_numpy(dtype=float)
    low, high = np.nanmin(numeric), np.nanmax(numeric)
    if np.isclose(low, high):
        return np.full_like(numeric, np.mean(size_range), dtype=float)
    scaled = (numeric - low) / (high - low)
    return size_range[0] + (size_range[1] - size_range[0]) * scaled


def marker_size_for_factor(factor: float) -> float:
    """Map a multiplicative factor to the shape-panel marker area."""
    min_size, max_size = 16.0, 58.0
    scaled = (factor - 0.6) / (1.4 - 0.6)
    scaled = float(np.clip(scaled, 0.0, 1.0))
    return min_size + (max_size - min_size) * scaled


def style_axes(ax: plt.Axes) -> None:
    """Apply common axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(length=5.5, width=1.2, color="#222222", pad=3)
    ax.grid(False)
    ax.set_facecolor("white")


def plot_max_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_name: str,
    param_order: tuple[str, ...],
    colors: dict[str, str],
    y_limit: tuple[float, float],
) -> None:
    """Draw one maximum-lifespan panel."""
    subset = data[data["model"] == model_name]
    for param_name in param_order:
        rows = subset[subset["parameter"] == param_name].sort_values("variation_percent")
        if rows.empty:
            continue
        alpha = 1.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_ab" else 0.78
        linewidth = 3.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_ab" else 2.2
        ax.plot(
            rows["variation_percent"],
            rows["max_lifespan_smoothed"],
            color=colors[param_name],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
        )

    ax.set_xlim(*MAX_X_LIMIT)
    ax.set_ylim(*y_limit)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Parameter heterogeneity (CV, %)")
    ax.set_ylabel("Top 0.01% survivors [years]")
    style_axes(ax)


def plot_shape_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    model_name: str,
    param_order: tuple[str, ...],
    colors: dict[str, str],
) -> None:
    """Draw one steepness-longevity plane."""
    subset = data[data["model"] == model_name]
    ax.axhline(1.0, color="#B8B8B8", linewidth=1.2, linestyle=(0, (2.2, 2.2)), zorder=0)
    ax.axvline(1.0, color="#B8B8B8", linewidth=1.2, linestyle=(0, (2.2, 2.2)), zorder=0)

    for param_name in param_order:
        rows = subset[subset["parameter"] == param_name].sort_values("factor")
        if rows.empty:
            continue
        alpha = 1.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_intercept_slope" else 0.76
        linewidth = 3.0 if param_name in ROBUSTNESS_PARAMS or param_name == "coupled_intercept_slope" else 2.15
        ax.plot(
            rows["x_norm"],
            rows["y_norm"],
            color=colors[param_name],
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            rows["x_norm"],
            rows["y_norm"],
            s=marker_sizes(rows["factor"], (16, 58)) if param_name != "makeham_m" else 28,
            color=colors[param_name],
            edgecolor="white",
            linewidth=0.45,
            alpha=alpha,
            zorder=3,
        )

    ax.set_xlim(*SHAPE_X_LIMIT)
    ax.set_ylim(*SHAPE_Y_LIMIT)
    ax.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([0.4, 0.7, 1.0, 1.3, 1.6])
    ax.set_xlabel("Median lifespan relative to baseline")
    ax.set_ylabel("Steepness relative to baseline")
    style_axes(ax)


def build_factor_legend(ax: plt.Axes) -> None:
    """Add a Fig1-style legend explaining factor marker size."""
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=np.sqrt(marker_size_for_factor(0.6)),
            label="0.6x",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=np.sqrt(marker_size_for_factor(1.4)),
            label="1.4x",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        title="Factor change",
        loc="lower right",
        frameon=True,
        fontsize=9.5,
        title_fontsize=10.5,
        borderpad=0.5,
        labelspacing=0.42,
        handletextpad=0.7,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E3E3E3")
    legend.get_frame().set_alpha(0.92)


def panel_label(ax: plt.Axes, label: str) -> None:
    """Place a consistent panel label."""
    ax.text(
        -0.18,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_legend_heading(ax: plt.Axes, y: float, text: str) -> float:
    """Draw a manual legend heading and return the next y position."""
    ax.text(0.0, y, text, fontsize=12.0, fontweight="bold", ha="left", va="top")
    return y - 0.085


def draw_legend_item(ax: plt.Axes, y: float, color: str, label: str, linewidth: float = 3.0) -> float:
    """Draw one manual legend line item."""
    ax.add_line(Line2D([0.0, 0.15], [y - 0.014, y - 0.014], color=color, linewidth=linewidth))
    ax.text(0.19, y, label, fontsize=10.2, ha="left", va="top")
    return y - 0.073


def draw_gompertz_legend(ax: plt.Axes) -> None:
    """Draw the row legend for the Gompertz-Makeham panels."""
    ax.axis("off")
    y = 0.98
    y = draw_legend_heading(ax, y, "Gompertz-Makeham")
    for param_name in ("a_linear", "a_power", "b", "makeham_m", "coupled_ab"):
        y = draw_legend_item(
            ax,
            y,
            GOMPERTZ_COLORS[param_name],
            GOMPERTZ_LABELS[param_name],
            linewidth=3.2 if param_name == "coupled_ab" else 2.6,
        )


def draw_fedichev_legend(ax: plt.Axes) -> None:
    """Draw the row legend for the Fedichev-Gruber panels."""
    ax.axis("off")
    y = 0.98
    y = draw_legend_heading(ax, y, "Fedichev-Gruber")
    y = draw_legend_heading(ax, y, "Senogenic parameters")
    for param_name in SENOGENIC_PARAMS:
        y = draw_legend_item(ax, y, FEDICHEV_COLORS[param_name], FEDICHEV_LABELS[param_name], linewidth=2.6)
    y -= 0.015
    y = draw_legend_heading(ax, y, "Robustness parameter")
    draw_legend_item(ax, y, FEDICHEV_COLORS["D0"], FEDICHEV_LABELS["D0"], linewidth=3.2)


def make_figure(max_data: pd.DataFrame, shape_data: pd.DataFrame) -> None:
    """Build and save the polished supplementary figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 8.9))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.43],
        height_ratios=[1.0, 1.0],
        wspace=0.28,
        hspace=0.34,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_gompertz_legend = fig.add_subplot(grid[0, 2])
    ax_fedichev_legend = fig.add_subplot(grid[1, 2])

    plot_max_panel(
        ax=ax_a,
        data=max_data,
        model_name="Gompertz",
        param_order=GOMPERTZ_MAX_PARAMS,
        colors=GOMPERTZ_COLORS,
        y_limit=GOMPERTZ_MAX_Y_LIMIT,
    )
    plot_shape_panel(
        ax=ax_b,
        data=shape_data,
        model_name="Gompertz",
        param_order=GOMPERTZ_SHAPE_PARAMS,
        colors=GOMPERTZ_COLORS,
    )
    plot_max_panel(
        ax=ax_c,
        data=max_data,
        model_name="Fedichev-Gruber",
        param_order=FEDICHEV_PARAMS,
        colors=FEDICHEV_COLORS,
        y_limit=FEDICHEV_MAX_Y_LIMIT,
    )
    plot_shape_panel(
        ax=ax_d,
        data=shape_data,
        model_name="Fedichev-Gruber",
        param_order=FEDICHEV_PARAMS,
        colors=FEDICHEV_COLORS,
    )

    ax_a.set_title("Gompertz-Makeham: extreme lifespan", loc="left", pad=10)
    ax_b.set_title("Gompertz-Makeham: shape response", loc="left", pad=10)
    ax_c.set_title("Fedichev-Gruber: extreme lifespan", loc="left", pad=10)
    ax_d.set_title("Fedichev-Gruber: shape response", loc="left", pad=10)

    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), "ABCD"):
        panel_label(ax, label)

    build_factor_legend(ax_b)
    build_factor_legend(ax_d)
    draw_gompertz_legend(ax_gompertz_legend)
    draw_fedichev_legend(ax_fedichev_legend)

    fig.patch.set_facecolor("white")
    fig.savefig(PNG_PATH, dpi=350, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def update_output_index() -> None:
    """Record the figure outputs in the project output index."""
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
    rows = []
    if OUTPUT_INDEX_PATH.exists():
        with OUTPUT_INDEX_PATH.open(newline="") as handle:
            rows = list(csv.DictReader(handle))

    source_script = "src/figures/Supp_Figgs/make_supp_model_comparison.py"
    input_paths = "; ".join(
        [
            "saved_results/gamma_factor_sweep.pkl",
            "saved_results/fedichev_model_steepness_longevity_data.pkl",
            LEGACY_NOTEBOOK,
        ]
    )
    new_rows = [
        {
            "date": date.today().isoformat(),
            "task": "supp_model_comparison",
            "artifact_type": "figure",
            "path": str(PNG_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": input_paths,
            "description": "PNG preview of supplementary Gompertz and Fedichev-Gruber model comparison.",
            "notes": "Central curves only; no fit-CI envelope is drawn because comparable fit-CI endpoint data are not available for the legacy model calculations.",
        },
        {
            "date": date.today().isoformat(),
            "task": "supp_model_comparison",
            "artifact_type": "figure",
            "path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": input_paths,
            "description": "Vector PDF of supplementary Gompertz and Fedichev-Gruber model comparison.",
            "notes": "Supplementary figure output with external grouped legends.",
        },
        {
            "date": date.today().isoformat(),
            "task": "supp_model_comparison",
            "artifact_type": "csv",
            "path": str(MAX_DATA_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": input_paths,
            "description": "Central maximum-lifespan source data for the supplementary model comparison.",
            "notes": "Gompertz-Makeham and Fedichev-Gruber maximum-lifespan data regenerated deterministically at N=300000.",
        },
        {
            "date": date.today().isoformat(),
            "task": "supp_model_comparison",
            "artifact_type": "csv",
            "path": str(SHAPE_DATA_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": input_paths,
            "description": "Central steepness-longevity source data for the supplementary model comparison.",
            "notes": "Loaded from the cached legacy Gompertz and Fedichev-Gruber shape-response runs.",
        },
    ]

    paths_to_replace = {row["path"] for row in new_rows}
    kept_rows = [row for row in rows if row.get("path") not in paths_to_replace]
    OUTPUT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_INDEX_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        writer.writerows(new_rows)


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    max_data = load_or_make_max_lifespan_data(args.force_sim)
    shape_data = load_or_make_shape_data(args.force_sim)

    save_plot_data(max_data, shape_data)
    make_figure(max_data, shape_data)
    update_output_index()

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")
    print(f"Saved max-lifespan data: {MAX_DATA_PATH}")
    print(f"Saved shape-response data: {SHAPE_DATA_PATH}")


if __name__ == "__main__":
    main()
