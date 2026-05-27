#!/usr/bin/env python3
"""Make Supp. Fig. 4 NHANES exposure-group survival curves."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.hetero_analysis import nhanes_analysis as nhanes
from src.shared.thresholds.paths import FIGURES_NEW_DIR, NHANES_DATA_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Supp_Figgs"
PNG_PATH = OUTPUT_DIR / "supp_fig4_nhanes_exposure_groups.png"

NHANES_PATH = str(NHANES_DATA_DIR) + "/"
TIMELINE = np.linspace(0, 120, 721)

BASELINE_COLOR = "#202124"
NEUTRAL_TEXT = "#252A2E"
SPINE_COLOR = "#6E7478"
GRID_COLOR = "#D9DEE2"


@dataclass(frozen=True)
class TopicPanel:
    topic: str
    title: str


TOPIC_PANELS = (
    TopicPanel("diet", "Diet quality"),
    TopicPanel("income", "Income-poverty ratio"),
    TopicPanel("number_of_friends", "Number of friends"),
    TopicPanel("sleep_duration", "Sleep duration"),
    TopicPanel("physical_activity", "Physical activity"),
    TopicPanel("alcohol", "Alcohol consumption"),
    TopicPanel("sleep_frailty", "Sleep frailty index"),
    TopicPanel("church_frequency", "Church attendance"),
    TopicPanel("education_level", "Education level"),
)

GROUP_LABELS = {
    "Good": "Good",
    "Poor": "Poor",
    "Q1 (Lowest)": "Q1 lowest",
    "Q2": "Q2",
    "Q3": "Q3",
    "Q4 (Highest)": "Q4 highest",
    "0 friends": "0 friends",
    "1+ friends": "1+ friends",
    "1-<5 hours": "1-<5 h",
    "5-<7 hours": "5-<7 h",
    "7-<9 hours": "7-<9 h",
    "\u22659 hours": ">=9 h",
    "No Activity": "None",
    "Some Activity": "Some",
    "0-1 drink/day": "0-1/day",
    "2-4 drinks/day": "2-4/day",
    ">4 drinks/day": ">4/day",
    "Q1 (lowest)": "Q1 lowest",
    "Q4 (highest)": "Q4 highest",
    "never": "Never",
    "sometimes": "Sometimes",
    "weekly": "Weekly",
    "no highschool": "No high school",
    "high school": "High school",
    "some college": "Some college",
}

GROUP_COLORS = {
    "Good": "#2D7F5E",
    "Poor": "#C5533D",
    "Q1 (Lowest)": "#B44745",
    "Q2": "#D58735",
    "Q3": "#4F9A94",
    "Q4 (Highest)": "#2D6F95",
    "0 friends": "#C5533D",
    "1+ friends": "#2D7F5E",
    "1-<5 hours": "#B44745",
    "5-<7 hours": "#D58735",
    "7-<9 hours": "#2D7F5E",
    "\u22659 hours": "#7D68A8",
    "No Activity": "#B44745",
    "Some Activity": "#2D7F5E",
    "0-1 drink/day": "#2D7F5E",
    "2-4 drinks/day": "#D58735",
    ">4 drinks/day": "#B44745",
    "Q1 (lowest)": "#2D7F5E",
    "Q4 (highest)": "#B44745",
    "never": "#B44745",
    "sometimes": "#D58735",
    "weekly": "#2D7F5E",
    "no highschool": "#B44745",
    "high school": "#D58735",
    "some college": "#2D7F5E",
}

GROUP_ORDER = {
    "diet": ("Good", "Poor"),
    "income": ("Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"),
    "number_of_friends": ("0 friends", "1+ friends"),
    "sleep_duration": ("1-<5 hours", "5-<7 hours", "7-<9 hours", "\u22659 hours"),
    "physical_activity": ("No Activity", "Some Activity"),
    "alcohol": ("0-1 drink/day", "2-4 drinks/day", ">4 drinks/day"),
    "sleep_frailty": ("Q1 (lowest)", "Q4 (highest)"),
    "church_frequency": ("never", "sometimes", "weekly"),
    "education_level": ("no highschool", "high school", "some college"),
}


def main() -> None:
    """Render the supplemental NHANES exposure-group figure."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    baseline_kmf, baseline_n = fit_baseline()

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(17.2, 12.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    for index, (ax, panel) in enumerate(zip(axes.flat, TOPIC_PANELS)):
        draw_panel(ax, panel, baseline_kmf, baseline_n, index)

    fig.supxlabel("Age (years)", fontsize=21, color=NEUTRAL_TEXT, y=0.04)
    fig.suptitle(
        "NHANES Survival by Exposure Group",
        fontsize=24,
        fontweight="bold",
        color=NEUTRAL_TEXT,
        y=0.985,
    )

    fig.subplots_adjust(left=0.075, right=0.992, bottom=0.105, top=0.9, hspace=0.62, wspace=0.2)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(PNG_PATH)


def configure_matplotlib() -> None:
    """Apply a clean publication style."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 17,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "legend.fontsize": 13.5,
            "legend.title_fontsize": 14,
            "axes.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fit_baseline() -> tuple[KaplanMeierFitter, int]:
    """Fit the all-participant NHANES Kaplan-Meier curve."""
    core = nhanes.load_core(NHANES_PATH)
    core = core.dropna(subset=["entry_age", "exit_age", "event"])

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=core["exit_age"],
        event_observed=core["event"],
        entry=core["entry_age"],
        timeline=TIMELINE,
        label="Baseline",
    )
    return kmf, len(core)


def draw_panel(
    ax: plt.Axes,
    panel: TopicPanel,
    baseline_kmf: KaplanMeierFitter,
    baseline_n: int,
    panel_index: int,
) -> None:
    """Draw one exposure-topic survival panel."""
    draw_baseline(ax, baseline_kmf, baseline_n)

    grouped, group_col = load_grouped_topic(panel.topic)
    for group_name in ordered_groups(panel.topic, grouped, group_col):
        group_data = grouped[grouped[group_col] == group_name]
        if group_data.empty:
            continue

        kmf = fit_group(group_data)
        label = f"{display_label(group_name)} (n={len(group_data):,})"
        color = GROUP_COLORS.get(str(group_name), "#4A6FA5")
        draw_confidence_band(ax, kmf, color=color, alpha=0.14)
        ax.plot(
            TIMELINE,
            kmf.survival_function_.iloc[:, 0],
            color=color,
            lw=2.05,
            alpha=0.96,
            solid_capstyle="round",
            label=label,
        )

    style_axis(ax, panel, panel_index)


def load_grouped_topic(topic: str) -> tuple[pd.DataFrame, str]:
    """Load a topic using the same grouping logic as the original notebook."""
    config = nhanes.TOPIC_CONFIGS[topic]
    topic_df = nhanes.get_topic_df(topic, NHANES_PATH)
    grouped, group_col = nhanes._apply_grouping_strategy(topic_df, config)
    required_cols = [group_col, "entry_age", "exit_age", "event"]
    return grouped.dropna(subset=required_cols), group_col


def ordered_groups(topic: str, grouped: pd.DataFrame, group_col: str) -> list[object]:
    """Return readable, configured group order with any unexpected groups last."""
    present = [group for group in grouped[group_col].dropna().unique()]
    configured = [group for group in GROUP_ORDER.get(topic, ()) if group in present]
    unexpected = sorted([group for group in present if group not in configured], key=str)
    return configured + unexpected


def fit_group(group_data: pd.DataFrame) -> KaplanMeierFitter:
    """Fit one exposure-group Kaplan-Meier curve."""
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=group_data["exit_age"],
        event_observed=group_data["event"],
        entry=group_data["entry_age"],
        timeline=TIMELINE,
    )
    return kmf


def draw_baseline(ax: plt.Axes, kmf: KaplanMeierFitter, n: int) -> None:
    """Draw the all-participant reference curve."""
    draw_confidence_band(ax, kmf, color=BASELINE_COLOR, alpha=0.07)
    ax.plot(
        TIMELINE,
        kmf.survival_function_.iloc[:, 0],
        color=BASELINE_COLOR,
        lw=2.35,
        alpha=0.96,
        solid_capstyle="round",
        label=f"All NHANES (n={n:,})",
    )


def draw_confidence_band(ax: plt.Axes, kmf: KaplanMeierFitter, *, color: str, alpha: float) -> None:
    """Draw the Kaplan-Meier confidence interval for one fitted curve."""
    ci = kmf.confidence_interval_survival_function_
    if ci.empty:
        return

    lower = ci.iloc[:, 0].to_numpy()
    upper = ci.iloc[:, 1].to_numpy()
    ax.fill_between(TIMELINE, lower, upper, color=color, alpha=alpha, lw=0)


def style_axis(ax: plt.Axes, panel: TopicPanel, panel_index: int) -> None:
    """Format one panel after all curves are drawn."""
    panel_letter = chr(ord("a") + panel_index)
    ax.set_title(panel.title, loc="left", pad=9, color=NEUTRAL_TEXT)
    if panel_index % 3 == 0:
        ax.set_ylabel("Survival probability", fontsize=17, color=NEUTRAL_TEXT, labelpad=10)

    ax.text(
        -0.075,
        1.085,
        panel_letter,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=NEUTRAL_TEXT,
        va="top",
    )

    ax.set_xlim(0, 120)
    ax.set_ylim(-0.015, 1.03)
    ax.set_xticks(np.arange(0, 121, 20))
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.55)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.tick_params(axis="both", colors=NEUTRAL_TEXT, width=1.6, length=7.0, pad=5)

    legend = ax.legend(
        loc="lower left",
        frameon=False,
        handlelength=2.0,
        handletextpad=0.55,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    for text in legend.get_texts():
        text.set_color(NEUTRAL_TEXT)


def display_label(group_name: object) -> str:
    """Return compact labels for panel legends."""
    return GROUP_LABELS.get(str(group_name), str(group_name).replace("\u2265", ">="))


if __name__ == "__main__":
    main()
