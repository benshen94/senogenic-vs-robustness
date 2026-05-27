#!/usr/bin/env python3
"""Rebuild the supplemental parameter-distribution figure from the legacy notebook."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


TASK_NAME = "figs1_parameter_distribution_supplement"
OUTPUT_DIR = FIGURES_NEW_DIR / "FigS1"
PNG_PATH = OUTPUT_DIR / "figs1_parameter_distributions_pretty.png"
PDF_PATH = OUTPUT_DIR / "figs1_parameter_distributions_pretty.pdf"
CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "FigS1"
CACHE_PATH = CACHE_DIR / "parameter_distribution_supplement.npz"
METADATA_PATH = CACHE_DIR / "parameter_distribution_supplement_metadata.json"
INDEX_PATH = SAVED_RESULTS_DIR / "index" / "outputs.csv"

PARAMETER_ORDER = ("eta", "beta", "Xc", "epsilon")
INTERVALS = tuple((start, start + 10) for start in range(40, 160, 10))
N_SIM = 1_000_000
HETERO_CV = 0.20
TMAX = 300.0
DT = 0.025
SAVE_TIMES = 300.0
RANDOM_SEED = 20260520
MIN_INTERVAL_COUNT = 25
MAX_KDE_SAMPLES = 60_000

PARAMETER_LABELS = {
    "eta": {
        "symbol": r"$\eta$",
        "math": r"\eta",
        "name": "Production rate",
        "title": r"Production $\eta$",
        "color": "#0B7F8C",
        "fit": "inverse",
        "fit_range": (80, 160),
        "text_xy": (0.96, 0.91),
        "text_ha": "right",
        "text_va": "top",
        "hazard_text": r"mortality $\sim$ const",
    },
    "beta": {
        "symbol": r"$\beta$",
        "math": r"\beta",
        "name": "Removal rate",
        "title": r"Removal $\beta$",
        "color": "#173A6A",
        "fit": "linear",
        "fit_range": (90, 110),
        "text_xy": (0.06, 0.91),
        "text_ha": "left",
        "text_va": "top",
        "hazard_text": r"mortality $\sim t^2$",
    },
    "Xc": {
        "symbol": r"$X_c$",
        "math": r"X_c",
        "name": "Threshold",
        "title": r"Threshold $X_c$",
        "color": "#D77A16",
        "fit": "exponential",
        "fit_range": (40, 120),
        "text_xy": (0.96, 0.08),
        "text_ha": "right",
        "text_va": "bottom",
        "hazard_text": r"mortality $\sim e^{bt}$",
    },
    "epsilon": {
        "symbol": r"$\epsilon$",
        "math": r"\epsilon",
        "name": "Noise",
        "title": r"Noise $\epsilon$",
        "color": "#E5A100",
        "fit": "linear",
        "fit_range": (40, 120),
        "text_xy": (0.96, 0.91),
        "text_ha": "right",
        "text_va": "top",
        "hazard_text": r"mortality $\sim e^{bt}$",
    },
}


@dataclass(frozen=True)
class SimulationData:
    parameter_values: np.ndarray
    death_times: np.ndarray
    baseline_means: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_SIM, help="Number of simulated people.")
    parser.add_argument("--force-sim", action="store_true", help="Ignore the simulation cache.")
    parser.add_argument("--png-only", action="store_true", help="Skip the PDF export.")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def build_metadata(n: int) -> dict[str, object]:
    base_dict = legacy_baseline()
    return {
        "source_notebook": "src/notebooks/param_distributions_investigation.ipynb",
        "n": n,
        "heterogeneity_cv": HETERO_CV,
        "interval_years": 10,
        "params": list(PARAMETER_ORDER),
        "baseline": {key: float(np.ravel(value)[0]) for key, value in base_dict.items()},
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "random_seed": RANDOM_SEED,
    }


def legacy_baseline() -> dict[str, np.ndarray]:
    base_dict = utils.load_baseline_human_params_dict()
    base_dict["Xc"] = 1.08 * base_dict["Xc"]
    base_dict["eta"] = 1.26 * base_dict["eta"]
    base_dict["beta"] = 1.17 * base_dict["beta"]
    return base_dict


def stable_seed(*parts: object) -> int:
    import hashlib

    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def load_or_run_simulations(n: int, force_sim: bool) -> dict[str, SimulationData]:
    metadata = build_metadata(n)
    if not force_sim and CACHE_PATH.exists() and METADATA_PATH.exists():
        cached_metadata = json.loads(METADATA_PATH.read_text())
        if cached_metadata == metadata:
            return load_cache()

    results = run_simulations(n)
    save_cache(results, metadata)
    return results


def run_simulations(n: int) -> dict[str, SimulationData]:
    base_dict = legacy_baseline()
    baseline_means = {key: float(np.ravel(value)[0]) for key, value in base_dict.items()}
    results = {}

    for parameter in PARAMETER_ORDER:
        print(f"Running {parameter} heterogeneity simulation with n={n}")
        np.random.seed(stable_seed(RANDOM_SEED, parameter, "distribution"))
        param_dict = utils.create_param_distribution_dict(
            params=parameter,
            std=HETERO_CV,
            n=n,
            dist_type="gaussian",
            params_dict=base_dict,
            family="None",
        )
        sim = utils.create_sr_simulation(
            params_dict=param_dict,
            n=n,
            parallel=True,
            tmax=TMAX,
            dt=DT,
            save_times=SAVE_TIMES,
            break_early=True,
            random_seed=stable_seed(RANDOM_SEED, parameter, "simulation"),
        )
        results[parameter] = SimulationData(
            parameter_values=np.asarray(param_dict[parameter], dtype=float),
            death_times=np.asarray(sim.death_times, dtype=float),
            baseline_means=baseline_means,
        )

    return results


def load_cache() -> dict[str, SimulationData]:
    loaded = np.load(CACHE_PATH, allow_pickle=False)
    baseline_means = json.loads(str(loaded["baseline_means"]))
    results = {}
    for parameter in PARAMETER_ORDER:
        results[parameter] = SimulationData(
            parameter_values=np.asarray(loaded[f"{parameter}_values"], dtype=float),
            death_times=np.asarray(loaded[f"{parameter}_death_times"], dtype=float),
            baseline_means=baseline_means,
        )
    return results


def save_cache(results: dict[str, SimulationData], metadata: dict[str, object]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {
        "baseline_means": np.asarray(json.dumps(next(iter(results.values())).baseline_means)),
    }
    for parameter, data in results.items():
        arrays[f"{parameter}_values"] = data.parameter_values
        arrays[f"{parameter}_death_times"] = data.death_times

    np.savez_compressed(CACHE_PATH, **arrays)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def interval_midpoint(interval: tuple[int, int]) -> float:
    start, end = interval
    return (start + end) / 2


def interval_label(interval: tuple[int, int]) -> str:
    start, end = interval
    return f"{start}-{end}"


def interval_color_map() -> tuple[ListedColormap, BoundaryNorm, list[str]]:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(INTERVALS)))
    cmap = ListedColormap(colors)
    boundaries = [INTERVALS[0][0]] + [end for _, end in INTERVALS]
    norm = BoundaryNorm(boundaries, cmap.N)
    labels = [interval_label(interval) for interval in INTERVALS]
    return cmap, norm, labels


def plot_distributions(ax: plt.Axes, data: SimulationData, parameter: str, cmap: ListedColormap) -> tuple[np.ndarray, np.ndarray]:
    label = PARAMETER_LABELS[parameter]
    values = data.parameter_values
    mean_value = float(np.mean(values))
    midpoints = []
    means = []

    x_grid = np.linspace(0.25, 2.05, 320)
    rng = np.random.default_rng(stable_seed(RANDOM_SEED, parameter, "kde-subsample"))

    for index, interval in enumerate(INTERVALS):
        start, end = interval
        mask = (data.death_times >= start) & (data.death_times < end)
        interval_values = values[mask]
        if len(interval_values) < MIN_INTERVAL_COUNT:
            continue

        normalized_values = interval_values / mean_value
        sampled_values = subsample_for_kde(normalized_values, rng)
        density = gaussian_kde(sampled_values)(x_grid)
        ax.plot(x_grid, density, color=cmap(index), lw=1.7, alpha=0.95)

        midpoints.append(interval_midpoint(interval))
        means.append(float(np.mean(interval_values)))

    ax.set_title(f"{label['title']} distribution by lifespan", fontsize=12.5, pad=6)
    ax.set_xlabel(f"Normalized {label['symbol']} ({label['name']})", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_xlim(0.25, 2.05)
    ax.tick_params(labelsize=9.5)
    return np.asarray(midpoints), np.asarray(means)


def subsample_for_kde(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= MAX_KDE_SAMPLES:
        return values

    indices = rng.choice(len(values), size=MAX_KDE_SAMPLES, replace=False)
    return values[indices]


def plot_mean_curve(ax: plt.Axes, parameter: str, midpoints: np.ndarray, means: np.ndarray, baseline_means: dict[str, float]) -> None:
    label = PARAMETER_LABELS[parameter]
    color = str(label["color"])
    ax.plot(midpoints, means, "o-", color=color, lw=2.0, ms=4.5, mec="white", mew=0.7)

    fit_text = plot_fit_line(ax, parameter, midpoints, means)
    annotate_middle_panel(ax, parameter, fit_text, baseline_means)

    ax.set_title(f"Mean {label['symbol']} vs lifespan", fontsize=12.5, pad=6)
    ax.set_xlabel("Lifespan interval midpoint", fontsize=11)
    ax.set_ylabel(f"Mean {label['symbol']}", fontsize=11)
    ax.set_xlim(37, 163)
    ax.tick_params(labelsize=9.5)


def plot_fit_line(ax: plt.Axes, parameter: str, midpoints: np.ndarray, means: np.ndarray) -> str:
    label = PARAMETER_LABELS[parameter]
    fit_range = label["fit_range"]
    mask = (midpoints >= fit_range[0]) & (midpoints <= fit_range[1])
    if int(mask.sum()) < 2:
        return ""

    x_fit = np.linspace(fit_range[0], fit_range[1], 160)
    color = str(label["color"])

    if label["fit"] == "inverse":
        params, _ = curve_fit(lambda t, a, b: a + b / t, midpoints[mask], means[mask])
        y_fit = params[0] + params[1] / x_fit
        ax.plot(x_fit, y_fit, "--", color=color, lw=1.5, alpha=0.65)
        return rf"${label['math']}(t) = {params[1]:.2f}/t {format_signed(params[0], 4)}$"

    if label["fit"] == "exponential":
        params, _ = curve_fit(lambda t, a, b: a * np.exp(b * t), midpoints[mask], means[mask], p0=(means[0], 0.01))
        y_fit = params[0] * np.exp(params[1] * x_fit)
        ax.plot(x_fit, y_fit, "--", color=color, lw=1.5, alpha=0.65)
        return rf"${label['math']}(t) = {params[0]:.2f}e^{{{params[1]:.3f}t}}$"

    coeffs = np.polyfit(midpoints[mask], means[mask], 1)
    y_fit = np.poly1d(coeffs)(x_fit)
    ax.plot(x_fit, y_fit, "--", color=color, lw=1.5, alpha=0.65)
    return rf"${label['math']}(t) = {coeffs[0]:.2f}t {format_signed(coeffs[1], 2)}$"


def format_signed(value: float, decimals: int) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {abs(value):.{decimals}f}"


def annotate_middle_panel(ax: plt.Axes, parameter: str, fit_text: str, baseline_means: dict[str, float]) -> None:
    label = PARAMETER_LABELS[parameter]
    companion = companion_parameter_text(parameter, baseline_means)
    lines = [line for line in (fit_text, companion) if line]
    if not lines:
        return

    ax.text(
        *label["text_xy"],
        "\n".join(lines),
        transform=ax.transAxes,
        ha=str(label["text_ha"]),
        va=str(label["text_va"]),
        fontsize=9.2,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#B8B8B8",
            "linewidth": 0.7,
            "alpha": 0.88,
        },
    )


def companion_parameter_text(parameter: str, baseline_means: dict[str, float]) -> str:
    if parameter == "eta":
        return rf"$\beta$ = {baseline_means['beta']:.2f}"
    if parameter == "beta":
        return rf"$\eta$ = {baseline_means['eta']:.2f}"
    if parameter == "Xc":
        return ""
    if parameter == "epsilon":
        return ""
    return ""


def plot_hazard(ax: plt.Axes, data: SimulationData, parameter: str) -> None:
    label = PARAMETER_LABELS[parameter]
    ages, hazard = smooth_hazard(data.death_times)
    ax.plot(ages, hazard, color="#4D9DE0", lw=2.0)
    ax.set_yscale("log")
    ax.set_xlim(20, 120)
    ax.set_ylim(1e-7, 1.2)
    ax.set_title(f"Mortality rate\n(20% heterogeneity in {label['symbol']})", fontsize=12.0, pad=6)
    ax.set_xlabel("Age", fontsize=11)
    ax.set_ylabel(r"Mortality rate [year$^{-1}$]", fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.text(
        0.52,
        0.62,
        str(label["hazard_text"]),
        transform=ax.transAxes,
        fontsize=12.5,
        ha="left",
        va="center",
        color="#222222",
    )


def smooth_hazard(death_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins = np.arange(20, 122, 1.0)
    centers = (bins[:-1] + bins[1:]) / 2
    alive = np.array([np.count_nonzero(death_times >= start) for start in bins[:-1]], dtype=float)
    deaths = np.array(
        [np.count_nonzero((death_times >= start) & (death_times < end)) for start, end in zip(bins[:-1], bins[1:])],
        dtype=float,
    )
    hazard = np.divide(deaths, alive, out=np.full_like(deaths, np.nan), where=alive > 0)
    hazard = np.clip(hazard, 1e-8, None)
    return centers, gaussian_smooth(hazard, sigma=2.2)


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(4 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def build_figure(results: dict[str, SimulationData], save_pdf: bool) -> None:
    configure_style()
    cmap, norm, _ = interval_color_map()
    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(11.5, 13.2),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.34, "hspace": 0.54},
    )

    for row, parameter in enumerate(PARAMETER_ORDER):
        data = results[parameter]
        midpoints, means = plot_distributions(axes[row, 0], data, parameter, cmap)
        plot_mean_curve(axes[row, 1], parameter, midpoints, means, data.baseline_means)
        plot_hazard(axes[row, 2], data, parameter)

    add_panel_labels(fig, axes)
    add_lifespan_legend(fig, cmap, norm)
    finalize_axes(axes)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=350, bbox_inches="tight")
    if save_pdf:
        fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def add_panel_labels(fig: plt.Figure, axes: np.ndarray) -> None:
    labels = list("abcdefghijkl")
    for label, ax in zip(labels, axes.flat):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.026,
            bbox.y1 + 0.010,
            label,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def add_lifespan_legend(fig: plt.Figure, cmap: ListedColormap, norm: BoundaryNorm) -> None:
    cax = fig.add_axes([0.18, 0.932, 0.40, 0.012])
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    cbar.set_label("Lifespan interval (years)", fontsize=10, labelpad=3)
    cbar.set_ticks([40, 60, 80, 100, 120, 140, 160])
    cbar.ax.tick_params(labelsize=9, length=3, width=0.8)
    cbar.outline.set_linewidth(0.6)


def finalize_axes(axes: np.ndarray) -> None:
    for ax in axes.flat:
        ax.grid(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#333333")


def update_output_index(save_pdf: bool) -> None:
    existing_rows = []
    if INDEX_PATH.exists():
        with INDEX_PATH.open(newline="") as handle:
            existing_rows = list(csv.DictReader(handle))

    source_script = "src/figures/FigS1/make_parameter_distribution_supplement.py"
    input_paths = "src/notebooks/param_distributions_investigation.ipynb"
    rows_to_add = [
        {
            "date": "2026-05-20",
            "task": TASK_NAME,
            "artifact_type": "figure",
            "path": str(PNG_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": input_paths,
            "description": "PNG preview of the refreshed supplemental parameter-distribution figure.",
            "notes": "Legacy 20% one-parameter heterogeneity simulations; distribution panels use 10-year lifespan intervals.",
        }
    ]
    if save_pdf:
        rows_to_add.append(
            {
                "date": "2026-05-20",
                "task": TASK_NAME,
                "artifact_type": "figure",
                "path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
                "source_script": source_script,
                "input_paths": input_paths,
                "description": "Vector PDF of the refreshed supplemental parameter-distribution figure.",
                "notes": "Matplotlib PDF with editable text where supported.",
            }
        )

    known_paths = {row["path"] for row in existing_rows}
    new_rows = [row for row in rows_to_add if row["path"] not in known_paths]
    if not new_rows:
        return

    fieldnames = ["date", "task", "artifact_type", "path", "source_script", "input_paths", "description", "notes"]
    with INDEX_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in new_rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    results = load_or_run_simulations(n=args.n, force_sim=args.force_sim)
    build_figure(results, save_pdf=not args.png_only)
    update_output_index(save_pdf=not args.png_only)
    print(PNG_PATH)
    if not args.png_only:
        print(PDF_PATH)


if __name__ == "__main__":
    main()
