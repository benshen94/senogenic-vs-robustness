#!/usr/bin/env python3
"""Make Supplementary Fig. 1: Gompertz constraints on senogenic heterogeneity."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from senogenic_vs_robustness.paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR


OUTPUT_DIR = FIGURES_DIR / "Supplementary"
PNG_PATH = OUTPUT_DIR / "supp_figure1_gompertz_constraint.png"
PDF_PATH = OUTPUT_DIR / "supp_figure1_gompertz_constraint.pdf"
INDEX_PATH = RESULTS_DIR / "index" / "outputs.csv"

SOURCE_DIR = TABLES_DIR / "supplementary_figure1"
SLOPE_SOURCE = SOURCE_DIR / "sweden2019_decade_slopes.csv"
ALLOWED_SOURCE = SOURCE_DIR / "allowed_parameter_cv_vs_slope_distortion.csv"
MEAN_SOURCE = SOURCE_DIR / "survivor_parameter_means.csv"
HAZARD_SOURCE = SOURCE_DIR / "senogenic_heterogeneity_hazards.csv"

PARAM_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
}
PARAM_LABELS = {
    "eta": r"Production $\eta$",
    "beta": r"Removal $\beta$",
    "Xc": r"Threshold $X_c$",
    "epsilon": r"Noise $\epsilon$",
}

MEAN_PANEL_CONFIG = {
    "eta": {
        "symbol": r"$\eta$",
        "math": r"\eta",
        "fit": "inverse",
        "fit_range": (80.0, 160.0),
        "text_xy": (0.96, 0.91),
        "text_ha": "right",
        "text_va": "top",
        "companion": r"$\beta$ = 64.06",
    },
    "beta": {
        "symbol": r"$\beta$",
        "math": r"\beta",
        "fit": "linear",
        "fit_range": (90.0, 110.0),
        "text_xy": (0.06, 0.91),
        "text_ha": "left",
        "text_va": "top",
        "companion": r"$\eta$ = 0.62",
    },
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10.8,
            "axes.labelsize": 14.0,
            "axes.titlesize": 15.0,
            "xtick.labelsize": 12.2,
            "ytick.labelsize": 12.2,
            "legend.fontsize": 9.0,
            "axes.linewidth": 1.05,
            "xtick.major.width": 1.05,
            "ytick.major.width": 1.05,
            "xtick.major.size": 5.2,
            "ytick.major.size": 5.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=18.0,
        fontweight="normal",
        ha="left",
        va="top",
    )


def header_handle() -> Line2D:
    return Line2D([], [], color="none", lw=0)


def grouped_param_legend(ax: plt.Axes, *, loc: str, fontsize: float = 8.8) -> None:
    handles = [
        header_handle(),
        Line2D([], [], color=PARAM_COLORS["eta"], lw=2.4),
        Line2D([], [], color=PARAM_COLORS["beta"], lw=2.4),
        header_handle(),
        Line2D([], [], color=PARAM_COLORS["Xc"], lw=2.4),
        Line2D([], [], color=PARAM_COLORS["epsilon"], lw=2.4, ls="--"),
    ]
    labels = [
        "Senogenic parameters",
        PARAM_LABELS["eta"],
        PARAM_LABELS["beta"],
        "Robustness parameters",
        PARAM_LABELS["Xc"],
        PARAM_LABELS["epsilon"],
    ]
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.34,
        borderpad=0.45,
    )
    for text in legend.get_texts():
        if text.get_text() in {"Senogenic parameters", "Robustness parameters"}:
            text.set_fontweight("bold")


def plot_slope_panel(ax: plt.Axes, slopes: pd.DataFrame) -> None:
    table = slopes.sort_values("age_mid")
    ax.axhspan(0.80, 1.20, color="#E9EEF1", alpha=0.95, zorder=0)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.35, zorder=1)
    ax.plot(
        table["age_mid"],
        table["slope_ratio_to_mean"],
        color=PARAM_COLORS["eta"],
        linewidth=2.7,
        marker="o",
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.35,
        zorder=3,
    )
    ax.set_xlim(50, 100)
    ax.set_ylim(0.79, 1.25)
    ax.set_xticks(table["age_mid"].to_numpy(dtype=float))
    ax.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2])
    ax.set_xlabel("Age midpoint [years]")
    ax.set_ylabel("Slope / mean slope")
    ax.set_title("Slopes normalized to mean")
    ax.legend(
        [
            Line2D([], [], color="#333333", lw=1.35, ls="--"),
            Line2D([], [], color="#9AA7B2", lw=8, alpha=0.55),
        ],
        ["50-90 mean", r"$\pm$20% band"],
        loc="upper left",
        frameon=False,
        fontsize=9.2,
        handlelength=1.6,
        handletextpad=0.55,
        labelspacing=0.35,
    )
    panel_label(ax, "a")


def plot_allowed_panel(ax: plt.Axes, allowed: pd.DataFrame) -> None:
    for param in ("eta", "beta", "Xc", "epsilon"):
        sub = allowed[allowed["parameter"] == param].sort_values("allowed_slope_distortion_fraction")
        x = sub["allowed_slope_distortion_fraction"].to_numpy(dtype=float)
        y = sub["allowed_parameter_cv"].to_numpy(dtype=float)
        if len(x) and x[0] > 0:
            x = np.r_[0.0, x]
            y = np.r_[0.0, y]
        ax.plot(
            x,
            y,
            color=PARAM_COLORS[param],
            lw=2.0,
            ls="--" if param == "epsilon" else "-",
        )
    ax.axvline(0.20, color="#555555", lw=1.15, ls="--", alpha=0.78, zorder=1)
    ax.set_xlim(0.0, 0.50)
    ax.set_ylim(0.0, 0.75)
    ax.set_xticks(np.arange(0.0, 0.51, 0.10))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Allowed slope distortion at age 90")
    ax.set_ylabel("Allowed parameter heterogeneity")
    ax.set_title("Allowed heterogeneity")
    grouped_param_legend(ax, loc="upper left")
    panel_label(ax, "b")


def format_signed(value: float, decimals: int) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign} {abs(value):.{decimals}f}"


def plot_fit_line(ax: plt.Axes, parameter: str, x: np.ndarray, y: np.ndarray) -> str:
    config = MEAN_PANEL_CONFIG[parameter]
    start, end = config["fit_range"]
    mask = (x >= start) & (x <= end)
    if int(mask.sum()) < 2:
        return ""

    xx = np.linspace(start, end, 160)
    color = PARAM_COLORS[parameter]
    if config["fit"] == "inverse":
        params, _ = curve_fit(lambda t, a, b: a + b / t, x[mask], y[mask])
        ax.plot(xx, params[0] + params[1] / xx, color=color, lw=1.5, ls="--", alpha=0.65)
        return rf"${config['math']}(t) = {params[1]:.2f}/t {format_signed(params[0], 4)}$"

    coeffs = np.polyfit(x[mask], y[mask], 1)
    ax.plot(xx, np.poly1d(coeffs)(xx), color=color, lw=1.5, ls="--", alpha=0.65)
    return rf"${config['math']}(t) = {coeffs[0]:.2f}t {format_signed(coeffs[1], 2)}$"


def annotate_mean_panel(ax: plt.Axes, parameter: str, fit_text: str) -> None:
    config = MEAN_PANEL_CONFIG[parameter]
    lines = [line for line in (fit_text, config["companion"]) if line]
    if not lines:
        return
    ax.text(
        *config["text_xy"],
        "\n".join(lines),
        transform=ax.transAxes,
        ha=str(config["text_ha"]),
        va=str(config["text_va"]),
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


def plot_mean_panel(ax: plt.Axes, means: pd.DataFrame, parameter: str, label: str) -> None:
    config = MEAN_PANEL_CONFIG[parameter]
    sub = means[means["parameter"] == parameter].sort_values("lifespan_midpoint")
    x = sub["lifespan_midpoint"].to_numpy(dtype=float)
    y = sub["mean_parameter"].to_numpy(dtype=float)
    color = PARAM_COLORS[parameter]
    ax.plot(x, y, color=color, lw=2.0, marker="o", ms=4.5, mec="white", mew=0.7)
    fit_text = plot_fit_line(ax, parameter, x, y)
    annotate_mean_panel(ax, parameter, fit_text)
    ax.set_xlabel("Lifespan interval midpoint")
    ax.set_ylabel(f"Mean {config['symbol']}")
    ax.set_title(f"Mean {config['symbol']} vs lifespan")
    ax.set_xlim(37, 163)
    panel_label(ax, label)


def plot_hazard_panel(ax: plt.Axes, hazards: pd.DataFrame, parameter: str, label: str) -> None:
    sub = hazards[hazards["parameter"] == parameter].sort_values("age")
    color = PARAM_COLORS[parameter]
    ax.plot(sub["age"], sub["mortality_rate"], color=color, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlim(20, 120)
    ax.set_ylim(1e-7, 1.2)
    ax.set_xlabel("Age")
    ax.set_ylabel(r"Mortality rate [year$^{-1}$]")
    symbol = MEAN_PANEL_CONFIG[parameter]["symbol"]
    ax.set_title(rf"Mortality rate" + "\n" + rf"(20% heterogeneity in {symbol})")
    ax.text(
        0.40 if parameter == "eta" else 0.50,
        0.62,
        r"mortality $\sim$ const" if parameter == "eta" else r"mortality $\sim t^2$",
        transform=ax.transAxes,
        fontsize=14.0,
        ha="left",
        va="center",
        color="#222222",
    )
    panel_label(ax, label)


def update_output_index() -> None:
    fieldnames = ["date", "task", "artifact_type", "path", "source_script", "input_paths", "description", "notes"]
    existing = []
    if INDEX_PATH.exists():
        with INDEX_PATH.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    source_script = str(Path(__file__).relative_to(PROJECT_ROOT))
    inputs = "; ".join(str(path.relative_to(PROJECT_ROOT)) for path in (SLOPE_SOURCE, ALLOWED_SOURCE, MEAN_SOURCE, HAZARD_SOURCE))
    new_rows = [
        {
            "date": date.today().isoformat(),
            "task": "supp_figure1_gompertz_constraint",
            "artifact_type": "figure",
            "path": str(PNG_PATH.relative_to(PROJECT_ROOT)),
            "source_script": source_script,
            "input_paths": inputs,
            "description": "PNG preview of Supplementary Fig. 1 Gompertz constraint composite.",
            "notes": "Uses cached source tables from the final private analysis package.",
        },
    ]
    replace = {row["path"] for row in new_rows}
    kept = [row for row in existing if row.get("path") not in replace]
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(new_rows)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slopes = pd.read_csv(SLOPE_SOURCE)
    allowed = pd.read_csv(ALLOWED_SOURCE)
    means = pd.read_csv(MEAN_SOURCE)
    hazards = pd.read_csv(HAZARD_SOURCE)

    fig = plt.figure(figsize=(8.7, 12.4), constrained_layout=False)
    fig_w, fig_h = fig.get_size_inches()
    panel = 3.0
    x_gap = 1.05
    y_gap = 1.30
    left = (fig_w - (2 * panel + x_gap)) / 2
    y_bottom = 0.75
    y_mid = y_bottom + panel + y_gap
    y_top = y_mid + panel + y_gap

    def add_square_axis(x_in: float, y_in: float) -> plt.Axes:
        return fig.add_axes([x_in / fig_w, y_in / fig_h, panel / fig_w, panel / fig_h])

    axes = [
        add_square_axis(left, y_top),
        add_square_axis(left + panel + x_gap, y_top),
        add_square_axis(left, y_mid),
        add_square_axis(left + panel + x_gap, y_mid),
        add_square_axis(left, y_bottom),
        add_square_axis(left + panel + x_gap, y_bottom),
    ]

    plot_slope_panel(axes[0], slopes)
    plot_allowed_panel(axes[1], allowed)
    plot_mean_panel(axes[2], means, "eta", "c")
    plot_mean_panel(axes[3], means, "beta", "d")
    plot_hazard_panel(axes[4], hazards, "eta", "e")
    plot_hazard_panel(axes[5], hazards, "beta", "f")

    for ax in axes:
        ax.set_box_aspect(1)
        ax.grid(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#333333")

    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)
    update_output_index()
    print(PNG_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
