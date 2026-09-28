"""Drawing helpers for saved NHANES likelihood results; no fitting or simulation."""
from __future__ import annotations

import contextlib

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from matplotlib.lines import Line2D

from scipy.interpolate import PchipInterpolator

from analysis.figures.steepness_longevity import make_fig3_usa_steepness_longevity as fig3_base

from senogenic_vs_robustness.paths import FIGURES_DIR as FIGURES_NEW_DIR

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - figure still renders without label repulsion.
    adjust_text = None

OUTPUT_DIR = FIGURES_NEW_DIR / "Fig3_new"

COMPOSITE_PNG_PATH = OUTPUT_DIR / "Fig3.png"

PNG_DPI = 600

COMPOSITE_LAYOUT_DPI = 350

COMPOSITE_PIXEL_WIDTH = 4486

COMPOSITE_PIXEL_HEIGHT = 6953

COMPOSITE_WIDTH = COMPOSITE_PIXEL_WIDTH / COMPOSITE_LAYOUT_DPI

COMPOSITE_HEIGHT = COMPOSITE_PIXEL_HEIGHT / COMPOSITE_LAYOUT_DPI

PANEL_A_XLIM = (0.90, 1.10)

PANEL_A_YLIM = (0.60, 1.35)

LOWER_XLIM = (60, 103)

LOWER_YLIM = (-10, 10)

LOWER_YTICKS = np.arange(-10, 11, 2)

DISPLAY_LABELS = {
    "Q1 (Lowest)": "Q1 income",
    "Q2": "Q2 income",
    "Q3": "Q3 income",
    "Q4 (Highest)": "Q4 income",
    "Q1 (lowest)": "Q1 sleep frailty",
    "Q4 (highest)": "Q4 sleep frailty",
    "0-1 drink/day": "0-1 drink/day",
    ">4 drinks/day": ">4 drinks/day",
    "1-<5 hours": "1-5 h sleep",
    "5-<7 hours": "5-7 h sleep",
    "7-<9 hours": "7-9 h sleep",
    ">=9 hours": ">=9 h sleep",
    "no highschool": "no high school",
    "some college": "some college",
}

TOPIC_MARKERS = {
    "diet": "s",
    "number_of_friends": "P",
    "income": "o",
    "alcohol": "D",
    "physical_activity": "^",
    "sleep_duration": "v",
    "sleep_frailty": "v",
    "church_frequency": "X",
    "education_level": "<",
}

PANEL_A_EXPOSURE_MARKER_SIZE = 340

PANEL_A_EXPOSURE_LABEL_SIZE = 12.5

PANEL_A_LEGEND_FONTSIZE = 15.5

PANEL_A_EXPOSURE_LEGEND_FONTSIZE = 19.5

PANEL_A_EXPOSURE_LEGEND_MARKER_SIZE = 15.0

PANEL_A_AXIS_LABEL_SIZE = 31

LOWER_AXIS_LABEL_SIZE = 27

LOWER_TITLE_SIZE = 23

LOWER_TICK_SIZE = 15

PANEL_LABEL_SIZE = 46

PANEL_B_LEGEND_FONTSIZE = 13.0

PANEL_B_LEGEND_TITLE_SIZE = 14.3

PANEL_B_LOW_FACTOR_GREY = 0.72

PANEL_B_HIGH_FACTOR_GREY = 0.20

PANEL_B_RIBBON_ALPHA = 0.09

PANEL_B_LABEL_GREY = "0.28"

PANEL_B_EXTREME_LABEL_SIZE = 13.0

EXPOSURE_LEGEND_ITEMS = [
    ("diet", "Diet quality"),
    ("number_of_friends", "Number of friends"),
    ("income", "Income"),
    ("alcohol", "Alcohol consumption"),
    ("physical_activity", "Physical activity"),
    ("sleep_duration", "Sleep duration"),
    ("sleep_frailty", "Sleep frailty"),
    ("church_frequency", "Church attendance"),
    ("education_level", "Education level"),
]

def draw_base_plane(ax: plt.Axes, *, add_parameter_legend: bool = True) -> None:
    """Draw the new USA 2019 Fig. 3 background without factor markers."""
    data = fig3_base.load_normalized_metrics()
    for param in fig3_base.PLOT_PARAMS:
        summary = fig3_base.parameter_summary(data=data, param=param)
        color = fig3_base.PARAM_COLORS[param]
        fig3_base.draw_sensitivity_envelope(ax=ax, summary=summary, color=color)
        ax.plot(
            summary["x_mean"],
            summary["y_mean"],
            color=color,
            linewidth=4.0,
            solid_capstyle="round",
            zorder=3,
        )

    h_ext = fig3_base.h_ext_summary(data)
    color = fig3_base.PARAM_COLORS["h_ext"]
    fig3_base.draw_sensitivity_envelope(ax=ax, summary=h_ext, color=color)
    ax.plot(
        h_ext["x_mean"],
        h_ext["y_mean"],
        color=color,
        linewidth=4.0,
        solid_capstyle="round",
        zorder=2,
    )
    fig3_base.finish_axes(ax)
    if add_parameter_legend:
        build_parameter_legend(ax)

def build_parameter_legend(
    ax: plt.Axes,
    *,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> plt.Legend:
    """Build the grouped SR-parameter legend following the README convention."""
    handles = [
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["eta"], lw=4, label=fig3_base.PARAM_LABELS["eta"]),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["beta"], lw=4, label=fig3_base.PARAM_LABELS["beta"]),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["Xc"], lw=4, label=fig3_base.PARAM_LABELS["Xc"]),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["epsilon"], lw=4, label=fig3_base.PARAM_LABELS["epsilon"]),
        Line2D([], [], color="none", label=" "),
        Line2D([0], [0], color=fig3_base.PARAM_COLORS["h_ext"], lw=4, label=r"Extrinsic mortality ($m_{ex}$)"),
    ]
    legend_kwargs = {
        "handles": handles,
        "loc": "upper left",
        "frameon": True,
        "fontsize": PANEL_A_LEGEND_FONTSIZE,
        "handlelength": 2.7,
        "labelspacing": 0.58,
        "borderpad": 0.45,
    }
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor

    legend = ax.legend(**legend_kwargs)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_alpha(1.0)

    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_fontweight("bold")
            text.set_color("#222222")

    ax.add_artist(legend)
    add_parameter_legend_separator(ax=ax, legend=legend, upper_index=5, lower_index=7)
    return legend

def add_parameter_legend_separator(
    ax: plt.Axes,
    legend: plt.Legend,
    *,
    upper_index: int,
    lower_index: int,
) -> None:
    """Separate extrinsic mortality from parameter curves inside the legend."""
    # Resolve positions during each draw, after the final layout and export DPI.
    # Precomputed axes coordinates drift when subplots_adjust changes the axes.
    class LegendSeparator(Line2D):
        def draw(self, renderer):
            box = legend.get_window_extent(renderer)
            upper = legend.get_texts()[upper_index].get_window_extent(renderer)
            lower = legend.get_texts()[lower_index].get_window_extent(renderer)
            inset = renderer.points_to_pixels(7)
            y = (upper.y0 + lower.y1) / 2
            coords = ax.transAxes.inverted().transform(
                [(box.x0 + inset, y), (box.x0 + 0.80 * box.width, y)]
            )
            self.set_data(coords[:, 0], coords[:, 1])
            super().draw(renderer)

    separator = LegendSeparator(
        [], [], transform=ax.transAxes,
        color="#8A8A8A",
        linewidth=0.85,
        linestyle=(0, (2.2, 2.2)),
        alpha=0.72,
        solid_capstyle="butt",
        clip_on=False,
        zorder=5.5,
    )
    ax.add_line(separator)

def build_exposure_legend(ax: plt.Axes, *, loc: str = "lower right") -> plt.Legend:
    """Build the exposure-group legend for panel A."""
    handles = []
    for topic, label in EXPOSURE_LEGEND_ITEMS:
        marker = TOPIC_MARKERS.get(topic, "o")
        marker_facecolor = "white" if topic == "sleep_frailty" else "black"
        handle = Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markersize=PANEL_A_EXPOSURE_LEGEND_MARKER_SIZE,
            markerfacecolor=marker_facecolor,
            markeredgecolor="black",
            markeredgewidth=1.2,
            color="black",
            label=label,
        )
        handles.append(handle)

    legend = ax.legend(
        handles=handles,
        loc=loc,
        title="Exposures",
        frameon=True,
        fontsize=PANEL_A_EXPOSURE_LEGEND_FONTSIZE,
        handlelength=1.7,
        labelspacing=0.45,
        borderpad=0.55,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#D6D6D6")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(0.94)
    legend.get_title().set_fontsize(PANEL_A_EXPOSURE_LEGEND_FONTSIZE + 1.2)
    legend.get_title().set_fontweight("bold")
    ax.add_artist(legend)
    return legend

def apply_original_panel_a_limits(ax: plt.Axes) -> None:
    """Use the old Fig3A steepness-longevity viewport."""
    ax.set_title("", loc="left")
    ax.set_xlim(*PANEL_A_XLIM)
    ax.set_ylim(*PANEL_A_YLIM)
    ax.set_aspect("auto")
    ax.set_xticks(np.arange(0.90, 1.101, 0.05))
    ax.set_yticks(np.arange(0.60, 1.31, 0.10))
    ax.set_xlabel("Median lifespan exposure / control", fontsize=PANEL_A_AXIS_LABEL_SIZE, labelpad=11)
    ax.set_ylabel("Steepness exposure / control", fontsize=PANEL_A_AXIS_LABEL_SIZE, labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_title(
        "Lifestyle exposures show\nrobustness-like signatures",
        loc="center",
        fontsize=30,
        pad=17,
    )

def draw_exposure_overlay(ax: plt.Axes, projections: pd.DataFrame) -> None:
    """Overlay exposure points, error bars, and labels."""
    texts = []
    off_scale = []
    for _, row in projections.iterrows():
        if row["y"] > ax.get_ylim()[1]:
            off_scale.append(row)
            continue
        color = "#111111" if row["x"] >= 1 else "#404040"
        marker = TOPIC_MARKERS.get(row["topic"], "o")
        ax.errorbar(
            row["x"],
            row["y"],
            xerr=row["x_err"],
            yerr=row["y_err"],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2.0,
            alpha=0.55,
            zorder=6,
        )
        if row["topic"] == "sleep_frailty":
            ax.scatter(
                row["x"],
                row["y"],
                marker=marker,
                s=PANEL_A_EXPOSURE_MARKER_SIZE,
                facecolor="white",
                edgecolor=color,
                linewidth=1.6,
                zorder=7,
            )
        else:
            ax.scatter(
                row["x"],
                row["y"],
                marker=marker,
                s=PANEL_A_EXPOSURE_MARKER_SIZE,
                color=color,
                edgecolor="white",
                linewidth=1.0,
                zorder=7,
            )

        callout_positions = {
            "Q4 sleep frailty": (0.997, 0.817),
            "no high school": (0.944, 0.819),
            ">4 drinks/day": (0.934, 0.846),
        }
        if row["label"] in callout_positions:
            ax.annotate(
                row["label"], xy=(row["x"], row["y"]),
                xytext=callout_positions[row["label"]], ha="right", va="center",
                fontsize=PANEL_A_EXPOSURE_LABEL_SIZE,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5),
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.65),
                zorder=8,
            )
            continue

        text = ax.text(
            row["x"],
            row["y"],
            row["label"],
            fontsize=PANEL_A_EXPOSURE_LABEL_SIZE,
            color="#111111",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
            zorder=8,
        )
        texts.append(text)

    if adjust_text is not None:
        with open(Path("/dev/null"), "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                adjust_text(
                    texts,
                    ax=ax,
                    expand=(1.7, 1.9),
                    force_text=(0.9, 1.0),
                    force_explode=(0.55, 0.8),
                    iter_lim=200,
                    arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45, alpha=0.55),
                )

    # Keep the original panel A viewport while making omitted points explicit.
    for row in off_scale:
        ax.annotate(
            f'{row["label"]}: {row["y"]:.2f} (above scale)',
            xy=(row["x"], ax.get_ylim()[1] - 0.003),
            xytext=(1.096, 1.292),
            ha="right", va="center", fontsize=11.5, color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333333"),
            zorder=9,
        )

def draw_panel_b(
    ax: plt.Axes,
    factor_curves: pd.DataFrame,
    projection_ribbons: pd.DataFrame,
) -> None:
    """Draw the Xc factor sweep with per-line projection-uncertainty ribbons."""
    ax.axhline(0, color="black", linestyle="--", linewidth=1.1, alpha=0.5, zorder=1)

    factors = np.sort(factor_curves["factor"].unique())
    colors = [panel_b_factor_color(factor, factors=factors) for factor in factors]

    for factor, color in zip(factors, colors):
        group = factor_curves[factor_curves["factor"] == factor].sort_values("age")
        ribbon = projection_ribbons[np.isclose(projection_ribbons["factor"], factor)].sort_values("age")
        ax.fill_between(
            ribbon["age"],
            ribbon["gain_projection_low"],
            ribbon["gain_projection_high"],
            color=color,
            alpha=PANEL_B_RIBBON_ALPHA,
            linewidth=0,
            edgecolor="none",
            zorder=2,
        )
        ax.plot(
            group["age"],
            group["gain_years"],
            "-",
            color=color,
            alpha=0.95,
            linewidth=2.2,
            zorder=3,
        )

    build_panel_b_projection_legend(ax, factors=factors, colors=colors)

    ax.text(
        61.2,
        4.85,
        "Largest gains in robustness",
        color=PANEL_B_LABEL_GREY,
        fontsize=PANEL_B_EXTREME_LABEL_SIZE,
        fontweight="bold",
        rotation=-20,
        ha="left",
        va="center",
        zorder=4,
    )
    ax.text(
        67.5,
        -3.85,
        "Largest reductions in robustness",
        color=PANEL_B_LABEL_GREY,
        alpha=0.72,
        fontsize=PANEL_B_EXTREME_LABEL_SIZE,
        fontweight="bold",
        rotation=24,
        ha="left",
        va="center",
        zorder=4,
    )

    ax.set_xlim(*LOWER_XLIM)
    ax.set_ylim(*LOWER_YLIM)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(LOWER_YTICKS)
    ax.set_xlabel("Age [years]", fontsize=LOWER_AXIS_LABEL_SIZE)
    ax.set_ylabel("Median extra years gained", fontsize=LOWER_AXIS_LABEL_SIZE)
    ax.set_title("Extra years from changes\nin robustness (model)", loc="center", pad=19, fontsize=LOWER_TITLE_SIZE)
    ax.tick_params(length=6.5, width=1.25, color="#222222", labelsize=LOWER_TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)

def panel_b_factor_color(factor: float, *, factors: np.ndarray) -> str:
    """Return a monotone grayscale color keyed to the Xc factor."""
    factor_min = float(np.min(factors))
    factor_max = float(np.max(factors))
    if factor_max == factor_min:
        return f"{PANEL_B_HIGH_FACTOR_GREY:.3f}"

    position = (float(factor) - factor_min) / (factor_max - factor_min)
    grey = PANEL_B_LOW_FACTOR_GREY + position * (PANEL_B_HIGH_FACTOR_GREY - PANEL_B_LOW_FACTOR_GREY)
    return f"{grey:.3f}"

def build_panel_b_projection_legend(
    ax: plt.Axes,
    *,
    factors: np.ndarray,
    colors: list,
) -> None:
    """Show the Xc factor color scale."""
    handles = [
        Line2D([0], [0], color=color, lw=2.2, label=f"{factor:.2f}")
        for factor, color in zip(factors, colors)
    ]
    legend = ax.legend(
        handles=handles,
        title="Xc factor",
        loc="upper right",
        frameon=False,
        fontsize=PANEL_B_LEGEND_FONTSIZE,
        title_fontsize=PANEL_B_LEGEND_TITLE_SIZE,
        handlelength=2.2,
        labelspacing=0.34,
        columnspacing=0.9,
        ncol=2,
    )
    ax.add_artist(legend)

def healthy_lifestyle_curves() -> dict[str, np.ndarray]:
    """Return the old Fig3C healthy-lifestyle gain curves."""
    return {
        "0-2 reference": np.array([0, 0, 0, 0, 0, 0, 0], dtype=float),
        "3": np.array([1, 0.8, 0.5, 0.2, 0.3, 0.5, 0.1], dtype=float),
        "4": np.array([3.5, 3.2, 2.6, 1.7, 0.8, 0.4, 0.2], dtype=float),
        "5": np.array([4, 3.6, 3, 2, 0.8, 0.2, 0.1], dtype=float),
        "6": np.array([4.9, 4.5, 3.8, 2.7, 1.2, 0.5, 0.7], dtype=float),
        "7-8": np.array([6.05, 5.6, 4.8, 3.5, 2.2, 1.2, 0.6], dtype=float),
    }

def draw_panel_c(ax: plt.Axes) -> None:
    """Draw the old Fig3C healthy-lifestyle data comparison."""
    x_nodes = np.array([40, 50, 60, 70, 80, 90, 100], dtype=float)
    x_smooth = np.linspace(60, 100, 220)
    curves = healthy_lifestyle_curves()
    curve_order = ["7-8", "6", "5", "4", "3", "0-2 reference"]
    alpha_values = np.linspace(1.0, 0.34, len(curve_order))

    for key, alpha in zip(curve_order, alpha_values):
        y_smooth = PchipInterpolator(x_nodes, curves[key])(x_smooth)
        ax.plot(x_smooth, y_smooth, color="black", alpha=alpha, linewidth=2.2)

    alpha_map = dict(zip(curve_order, alpha_values))
    lifestyle_handles = [
        Line2D([], [], color="black", alpha=alpha_map[key], linewidth=2.2, label=label)
        for key, label in [("7-8", "7–8"), ("6", "6"), ("5", "5"),
                           ("4", "4"), ("3", "3"), ("0-2 reference", "0–2 reference")]
    ]
    lifestyle_legend = ax.legend(
        handles=lifestyle_handles,
        title="Healthy lifestyle factors",
        loc="upper right",
        frameon=False,
        fontsize=12.0,
        title_fontsize=12.5,
        labelspacing=0.28,
        handlelength=1.8,
        borderpad=0.45,
    )
    lifestyle_legend.get_frame().set_facecolor("white")
    lifestyle_legend.get_frame().set_edgecolor("#DDDDDD")
    lifestyle_legend.get_frame().set_alpha(0.96)
    ax.add_artist(lifestyle_legend)

    ax.axhline(0, color="#8F8F8F", linestyle="--", linewidth=1.1, alpha=0.65)
    ax.set_xlim(*LOWER_XLIM)
    ax.set_ylim(*LOWER_YLIM)
    ax.set_xticks(np.arange(60, 101, 10))
    ax.set_yticks(LOWER_YTICKS)
    ax.set_xlabel("Age [years]", fontsize=LOWER_AXIS_LABEL_SIZE)
    ax.set_title("Extra years from\nhealthy lifestyle (Sakaniwa et al)", loc="center", pad=19, fontsize=LOWER_TITLE_SIZE)
    ax.tick_params(length=6.5, width=1.25, color="#222222", labelsize=LOWER_TICK_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)

def save_composite(
    projections: pd.DataFrame,
    factor_curves: pd.DataFrame,
    projection_ribbons: pd.DataFrame,
) -> None:
    """Save the combined two-panel figure."""
    fig3_base.apply_style()
    fig = plt.figure(figsize=(COMPOSITE_WIDTH, COMPOSITE_HEIGHT))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.55, 0.78], hspace=0.28, wspace=0.24)
    top_grid = grid[0, :].subgridspec(1, 3, width_ratios=[0.16, 1.0, 0.30], wspace=0.0)

    ax_a = fig.add_subplot(top_grid[0, 1])
    draw_base_plane(ax_a, add_parameter_legend=False)
    apply_original_panel_a_limits(ax_a)
    draw_exposure_overlay(ax_a, projections)
    build_parameter_legend(ax_a)
    build_exposure_legend(ax_a, loc="lower right")
    ax_a.text(-0.14, 1.06, "a", transform=ax_a.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="normal", va="top")

    ax_b = fig.add_subplot(grid[1, 0])
    draw_panel_b(ax_b, factor_curves, projection_ribbons)
    ax_b.text(-0.16, 1.20, "b", transform=ax_b.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="normal", va="top")

    ax_c = fig.add_subplot(grid[1, 1], sharey=ax_b)
    draw_panel_c(ax_c)
    ax_c.set_ylabel("")
    ax_c.tick_params(labelleft=False)
    ax_c.text(-0.16, 1.20, "c", transform=ax_c.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="normal", va="top")

    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.075)
    fig.savefig(COMPOSITE_PNG_PATH, dpi=PNG_DPI, bbox_inches=None)
    plt.close(fig)
