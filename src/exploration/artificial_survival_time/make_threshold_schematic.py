#!/usr/bin/env python3
"""Make a standalone schematic for disease and death threshold shifts."""

from __future__ import annotations

import csv
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

from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Supp_Figgs"
PNG_PATH = OUTPUT_DIR / "supp_artificial_survival_threshold_schematic.png"
INDEX_PATH = SAVED_RESULTS_DIR / "index" / "outputs.csv"

DATE = "2026-05-21"
BASE_XD = 0.75
BASE_XC = 1.00
THRESHOLD_FACTOR = 1.20
Y_MAX = 1.28

COLORS = {
    "health": "#DCEFC2",
    "morbidity": "#EE8B91",
    "death": "#B5C1C8",
    "trajectory": "#232A33",
    "xd": "#C5282F",
    "xc": "#20242D",
    "reference": "#6E7781",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.3,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def damage_curve(
    disease_age: float,
    death_age: float,
    xd: float,
    xc: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ages = np.linspace(12.0, death_age, 64)
    values = np.empty_like(ages)

    before = ages <= disease_age
    after = ~before

    before_fraction = (ages[before] - ages[before].min()) / (disease_age - ages[before].min())
    values[before] = 0.05 + (xd - 0.05) * before_fraction**1.8

    after_fraction = (ages[after] - disease_age) / (death_age - disease_age)
    values[after] = xd + (xc - xd) * np.clip(after_fraction, 0, 1) ** 1.35

    noise_scale = np.interp(ages, [12, disease_age, death_age], [0.012, 0.035, 0.015])
    values += rng.normal(0.0, noise_scale, size=ages.size)
    values = np.clip(values, 0.03, Y_MAX - 0.02)

    disease_index = np.argmin(np.abs(ages - disease_age))
    values[disease_index] = xd
    values[-1] = xc
    return ages, values


def draw_bands(ax: plt.Axes, xd: float, xc: float) -> None:
    ax.axhspan(0, xd, color=COLORS["health"], zorder=0)
    ax.axhspan(xd, xc, color=COLORS["morbidity"], zorder=0)
    ax.axhspan(xc, Y_MAX, color=COLORS["death"], zorder=0)
    ax.axhline(xd, color="white", lw=2.0, zorder=1)
    ax.axhline(xc, color="white", lw=2.0, zorder=1)
    ax.axhline(xd, color=COLORS["xd"], lw=2.8, zorder=2)
    ax.axhline(xc, color=COLORS["xc"], lw=2.8, zorder=2)


def draw_reference_thresholds(ax: plt.Axes, show_xd: bool, show_xc: bool) -> None:
    if show_xd:
        ax.axhline(BASE_XD, color=COLORS["xd"], lw=1.9, ls=(0, (4, 3)), zorder=2)
    if show_xc:
        ax.axhline(BASE_XC, color=COLORS["xc"], lw=1.9, ls=(0, (4, 3)), zorder=2)


def draw_threshold_labels(ax: plt.Axes, xd: float, xc: float) -> None:
    ax.text(1.01, xd, r"$X_D$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=15)
    ax.text(1.01, xc, r"$X_c$", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=15)


def draw_trajectory(
    ax: plt.Axes,
    disease_age: float,
    death_age: float,
    xd: float,
    xc: float,
    seed: int,
) -> None:
    ages, values = damage_curve(disease_age=disease_age, death_age=death_age, xd=xd, xc=xc, seed=seed)
    ax.plot(ages, values, color=COLORS["trajectory"], lw=3.0, zorder=4)
    ax.scatter([disease_age], [xd], marker="^", s=86, color="black", zorder=5)
    ax.scatter([disease_age], [xd], marker="^", s=58, color=COLORS["xd"], zorder=6)
    ax.scatter([death_age], [xc], marker="s", s=62, color=COLORS["xc"], zorder=6)


def draw_shift_arrow(ax: plt.Axes, x: float, y0: float, y1: float, label: str) -> None:
    ax.annotate(
        "",
        xy=(x, y1 - 0.015),
        xytext=(x, y0 + 0.015),
        arrowprops={"arrowstyle": "-|>", "lw": 1.8, "color": COLORS["reference"]},
        zorder=6,
    )
    if label:
        ax.text(x + 2.0, (y0 + y1) / 2, label, ha="left", va="center", fontsize=10.5, color=COLORS["reference"])


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 105)
    ax.set_ylim(0, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"Age, $t$")


def add_band_labels(ax: plt.Axes, xd: float, xc: float) -> None:
    x = 5.0
    ax.text(x, xd / 2, "Health", ha="left", va="center", fontsize=13)
    ax.text(x, (xd + xc) / 2, "Morbidity", ha="left", va="center", fontsize=13)
    ax.text(x, (xc + Y_MAX) / 2, "Death", ha="left", va="center", fontsize=13)


def make_figure() -> None:
    scenarios = [
        {
            "title": "Baseline",
            "xd": BASE_XD,
            "xc": BASE_XC,
            "disease_age": 68,
            "death_age": 84,
            "reference_xd": False,
            "reference_xc": False,
            "arrows": [],
        },
        {
            "title": "Increase in death threshold only",
            "xd": BASE_XD,
            "xc": BASE_XC * THRESHOLD_FACTOR,
            "disease_age": 68,
            "death_age": 96,
            "reference_xd": False,
            "reference_xc": True,
            "arrows": [(58, BASE_XC, BASE_XC * THRESHOLD_FACTOR, "")],
        },
        {
            "title": "Increase in death and disease\nthreshold together",
            "xd": BASE_XD * THRESHOLD_FACTOR,
            "xc": BASE_XC * THRESHOLD_FACTOR,
            "disease_age": 80,
            "death_age": 96,
            "reference_xd": True,
            "reference_xc": True,
            "arrows": [
                (50, BASE_XD, BASE_XD * THRESHOLD_FACTOR, ""),
                (62, BASE_XC, BASE_XC * THRESHOLD_FACTOR, ""),
            ],
        },
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9), sharey=True)
    for index, (ax, scenario) in enumerate(zip(axes, scenarios)):
        draw_bands(ax, xd=scenario["xd"], xc=scenario["xc"])
        draw_reference_thresholds(ax, show_xd=scenario["reference_xd"], show_xc=scenario["reference_xc"])
        draw_trajectory(
            ax,
            disease_age=scenario["disease_age"],
            death_age=scenario["death_age"],
            xd=scenario["xd"],
            xc=scenario["xc"],
            seed=20260521 + index,
        )
        for x, y0, y1, label in scenario["arrows"]:
            draw_shift_arrow(ax, x=x, y0=y0, y1=y1, label=label)
        draw_threshold_labels(ax, xd=scenario["xd"], xc=scenario["xc"])
        style_axis(ax)
        ax.set_title(scenario["title"], pad=12)

    axes[0].set_ylabel(r"$X(t)$")
    add_band_labels(axes[0], xd=BASE_XD, xc=BASE_XC)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 1.0), w_pad=2.3)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def update_output_index() -> None:
    if not INDEX_PATH.exists():
        return

    path = str(PNG_PATH.relative_to(PROJECT_ROOT))
    existing_text = INDEX_PATH.read_text()
    if f",{path}," in existing_text or f",{path}\n" in existing_text:
        return

    row = {
        "date": DATE,
        "task": "supp_artificial_survival_threshold_schematic",
        "artifact_type": "figure",
        "path": path,
        "source_script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "input_paths": "",
        "description": "Standalone schematic for disease and death threshold shifts.",
        "notes": "Three panels: baseline, death-threshold-only increase, and proportional disease/death threshold increase.",
    }
    with INDEX_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def main() -> None:
    configure_matplotlib()
    make_figure()
    update_output_index()
    print(PNG_PATH)


if __name__ == "__main__":
    main()
