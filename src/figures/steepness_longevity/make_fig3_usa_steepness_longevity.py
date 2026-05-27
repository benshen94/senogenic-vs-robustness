#!/usr/bin/env python3
"""Make the Fig3 USA steepness-longevity response plane."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


RUN_DIR = SAVED_RESULTS_DIR / "steepness_longevity_usa2019_sensitivity"
METRICS_PATH = RUN_DIR / "metrics_long.csv"
PLOT_DATA_PATH = RUN_DIR / "fig3_usa_steepness_longevity_plot_data.csv"
SUMMARY_PATH = RUN_DIR / "fig3_usa_steepness_longevity_point_intervals.csv"
ENVELOPE_PATH = RUN_DIR / "fig3_usa_steepness_longevity_shaded_envelopes.csv"

OUTPUT_DIR = FIGURES_NEW_DIR / "Fig3_new"
PNG_PATH = OUTPUT_DIR / "fig3_usa_steepness_longevity.png"
PDF_PATH = OUTPUT_DIR / "fig3_usa_steepness_longevity.pdf"

FROM_T = 20
MIN_FACTOR = 0.6
MIN_FACTOR_BY_PARAM = {
    "eta": 0.7,
    "beta": 0.6,
    "Xc": 0.6,
    "epsilon": 0.6,
}
MAX_FACTOR = 1.4
AX_LIMITS = (0.25, 1.65)
H_EXT_MAX_VISIBLE = 0.006
MARKER_SIZE_RANGE = (34, 150)
INTERVAL_LOWER_Q = 0.025
INTERVAL_UPPER_Q = 0.975
TMAX_MEDIAN_MARGIN = 8.0
ENVELOPE_ALPHA = 0.18
ENVELOPE_CAP_FACTOR = 1.0
LEGEND_SEPARATOR_COLOR = "#777777"
LEGEND_SEPARATOR_ALPHA = 0.5

PLOT_PARAMS = ("eta", "beta", "Xc", "epsilon")

PARAM_LABELS = {
    "eta": r"Production $\eta$",
    "beta": r"Removal $\beta$",
    "Xc": r"Threshold $X_c$",
    "epsilon": r"Noise $\epsilon$",
}

PARAM_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
    "h_ext": "#C51F2F",
}


def load_normalized_metrics() -> pd.DataFrame:
    """Load metrics and add normalized x/y values within each scenario."""
    metrics = pd.read_csv(METRICS_PATH)
    metrics = metrics[metrics["from_t"] == FROM_T].copy()

    baseline = metrics[metrics["curve_type"] == "baseline"].set_index("scenario_id")
    metrics["baseline_median"] = metrics["scenario_id"].map(baseline["t_median_absolute"])
    metrics["baseline_steepness"] = metrics["scenario_id"].map(baseline["steepness_iqr_absolute"])

    metrics["x_norm"] = metrics["t_median_absolute"] / metrics["baseline_median"]
    metrics["y_norm"] = metrics["steepness_iqr_absolute"] / metrics["baseline_steepness"]
    return metrics


def valid_interval_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Remove finite-tmax artifacts before summarizing point intervals."""
    valid = (
        np.isfinite(data["x_norm"])
        & np.isfinite(data["y_norm"])
        & (data["t_median_absolute"] <= data["tmax"] - TMAX_MEDIAN_MARGIN)
    )
    return data.loc[valid].copy()


def summarize_points(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize x/y mean positions and empirical 95% sensitivity intervals."""
    summaries = []
    for focal_value, group in rows.groupby("focal_value", sort=True):
        summaries.append(
            {
                "focal_value": focal_value,
                "x_mean": group["x_norm"].mean(),
                "x_low": group["x_norm"].quantile(INTERVAL_LOWER_Q),
                "x_high": group["x_norm"].quantile(INTERVAL_UPPER_Q),
                "y_mean": group["y_norm"].mean(),
                "y_low": group["y_norm"].quantile(INTERVAL_LOWER_Q),
                "y_high": group["y_norm"].quantile(INTERVAL_UPPER_Q),
                "n_scenarios": len(group),
            }
        )
    return pd.DataFrame(summaries)


def parameter_summary(data: pd.DataFrame, param: str) -> pd.DataFrame:
    """Return factor-wise mean points and intervals for one parameter."""
    min_factor = MIN_FACTOR_BY_PARAM[param]
    mask = (
        (data["curve_type"] == "parameter_factor")
        & (data["focal_param"] == param)
        & (data["focal_value"] >= min_factor)
        & (data["focal_value"] <= MAX_FACTOR)
    )
    rows = valid_interval_rows(data.loc[mask])
    summary = summarize_points(rows)
    summary["focal_param"] = param
    summary["curve_type"] = "parameter_factor"
    return summary


def h_ext_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return h_ext-wise mean points and intervals across baseline scenarios."""
    mask = (
        (data["curve_type"] == "h_ext_absolute")
        & (data["focal_param"] == "h_ext")
        & (data["focal_value"] <= H_EXT_MAX_VISIBLE)
    )
    rows = valid_interval_rows(data.loc[mask])
    summary = summarize_points(rows)
    summary["focal_param"] = "h_ext"
    summary["curve_type"] = "h_ext_absolute"
    return summary


def save_plot_data(data: pd.DataFrame) -> None:
    """Save the normalized data used by the plotting script."""
    keep = [
        "run_id",
        "scenario_id",
        "nuisance_param",
        "nuisance_factor",
        "curve_type",
        "focal_param",
        "focal_value",
        "h_ext",
        "from_t",
        "tmax",
        "x_norm",
        "y_norm",
        "t_median_absolute",
        "steepness_iqr_absolute",
    ]
    data[keep].to_csv(PLOT_DATA_PATH, index=False)


def save_summary_data(data: pd.DataFrame) -> None:
    """Save the mean points and sensitivity intervals shown in the figure."""
    summaries = [parameter_summary(data, param) for param in PLOT_PARAMS]
    summaries.append(h_ext_summary(data))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    envelope_tables = [
        envelope_dataframe(
            summary=group,
            focal_param=focal_param,
            curve_type=curve_type,
        )
        for (curve_type, focal_param), group in summary.groupby(["curve_type", "focal_param"])
    ]
    pd.concat(envelope_tables, ignore_index=True).to_csv(ENVELOPE_PATH, index=False)


def apply_style() -> None:
    """Apply a bold but publication-safe figure style."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def marker_size_for_factor(factor: float) -> float:
    """Map multiplicative factor to marker area."""
    min_size, max_size = MARKER_SIZE_RANGE
    scaled = (factor - MIN_FACTOR) / (MAX_FACTOR - MIN_FACTOR)
    scaled = float(np.clip(scaled, 0.0, 1.0))
    return min_size + (max_size - min_size) * scaled


def marker_size_for_h_ext(value: float, values: pd.Series) -> float:
    """Map log-spaced extrinsic mortality values to marker area."""
    min_size, max_size = MARKER_SIZE_RANGE
    log_values = np.log10(values.to_numpy(dtype=float))
    log_value = np.log10(float(value))
    scaled = (log_value - log_values.min()) / (log_values.max() - log_values.min())
    scaled = float(np.clip(scaled, 0.0, 1.0))
    return min_size + (max_size - min_size) * scaled


def local_tangents(points: np.ndarray) -> np.ndarray:
    """Return local tangent directions along an ordered curve."""
    tangents = []
    previous_tangent = None
    for index in range(len(points)):
        left_index = max(index - 1, 0)
        right_index = min(index + 1, len(points) - 1)
        tangent = points[right_index] - points[left_index]
        tangent_length = np.linalg.norm(tangent)

        if tangent_length == 0 and previous_tangent is not None:
            tangent = previous_tangent
        elif tangent_length == 0:
            tangent = np.array([1.0, 0.0])
        else:
            tangent = tangent / tangent_length

        tangents.append(tangent)
        previous_tangent = tangent

    return np.vstack(tangents)


def local_normals(points: np.ndarray) -> np.ndarray:
    """Return consistently oriented local normals along an ordered curve."""
    normals = []
    previous_normal = None
    for tangent in local_tangents(points):
        normal = np.array([-tangent[1], tangent[0]])
        if previous_normal is not None and np.dot(normal, previous_normal) < 0:
            normal = -normal

        normals.append(normal)
        previous_normal = normal

    return np.vstack(normals)


def extend_ribbon_endpoints(
    centers: np.ndarray,
    tangents: np.ndarray,
    normal_width: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend the ribbon beyond the first and last points to cover endpoint markers."""
    start_cap = tangents[0] * normal_width[0] * ENVELOPE_CAP_FACTOR
    end_cap = tangents[-1] * normal_width[-1] * ENVELOPE_CAP_FACTOR

    extended_centers = np.vstack(
        [
            centers[0] - start_cap,
            centers,
            centers[-1] + end_cap,
        ]
    )
    extended_width = np.concatenate(
        [
            [normal_width[0]],
            normal_width,
            [normal_width[-1]],
        ]
    )
    return extended_centers, extended_width


def ribbon_polygon(summary: pd.DataFrame) -> np.ndarray:
    """Convert point-wise x/y intervals into one local-width shaded ribbon."""
    sorted_summary = summary.sort_values("focal_value").reset_index(drop=True)
    centers = sorted_summary[["x_mean", "y_mean"]].to_numpy(dtype=float)
    if len(centers) < 2:
        return np.empty((0, 2))

    tangents = local_tangents(centers)
    normals = local_normals(centers)
    half_x = np.maximum(
        sorted_summary["x_mean"] - sorted_summary["x_low"],
        sorted_summary["x_high"] - sorted_summary["x_mean"],
    ).to_numpy(dtype=float)
    half_y = np.maximum(
        sorted_summary["y_mean"] - sorted_summary["y_low"],
        sorted_summary["y_high"] - sorted_summary["y_mean"],
    ).to_numpy(dtype=float)

    normal_width = np.abs(normals[:, 0]) * half_x + np.abs(normals[:, 1]) * half_y
    centers, normal_width = extend_ribbon_endpoints(centers, tangents, normal_width)
    normals = local_normals(centers)

    upper = centers + normals * normal_width[:, None]
    lower = centers - normals * normal_width[:, None]
    return np.vstack([upper, lower[::-1]])


def envelope_dataframe(summary: pd.DataFrame, focal_param: str, curve_type: str) -> pd.DataFrame:
    """Return polygon coordinates for the shaded sensitivity envelope."""
    polygon = ribbon_polygon(summary)
    rows = []
    for polygon_order, (x_value, y_value) in enumerate(polygon):
        rows.append(
            {
                "focal_param": focal_param,
                "curve_type": curve_type,
                "polygon_order": polygon_order,
                "x": x_value,
                "y": y_value,
            }
        )
    return pd.DataFrame(rows)


def draw_sensitivity_envelope(ax: plt.Axes, summary: pd.DataFrame, color: str) -> None:
    """Draw one shaded envelope from point-wise x/y sensitivity intervals."""
    polygon = ribbon_polygon(summary)
    if len(polygon) < 3:
        return

    patch = PathPatch(
        MplPath(polygon, closed=True),
        facecolor=color,
        edgecolor="none",
        alpha=ENVELOPE_ALPHA,
        zorder=1,
    )
    ax.add_patch(patch)


def draw_curve_markers(ax: plt.Axes, summary: pd.DataFrame, color: str, *, use_h_ext_sizes: bool = False) -> None:
    """Draw the mean points with marker size encoding factor magnitude."""
    markers = summary[~np.isclose(summary["focal_value"], 1.0)].copy()
    if markers.empty:
        return

    h_ext_values = summary["focal_value"] if use_h_ext_sizes else None
    for _, row in markers.iterrows():
        marker_size = (
            marker_size_for_h_ext(row["focal_value"], h_ext_values)
            if use_h_ext_sizes
            else marker_size_for_factor(row["focal_value"])
        )
        ax.scatter(
            row["x_mean"],
            row["y_mean"],
            s=marker_size,
            color=color,
            edgecolor=color,
            linewidth=0.8,
            alpha=0.95,
            zorder=4,
        )


def draw_parameter_curve(ax: plt.Axes, data: pd.DataFrame, param: str) -> None:
    """Draw one mean parameter curve plus factor-wise sensitivity intervals."""
    summary = parameter_summary(data=data, param=param)
    color = PARAM_COLORS[param]

    draw_sensitivity_envelope(ax=ax, summary=summary, color=color)
    ax.plot(
        summary["x_mean"],
        summary["y_mean"],
        color=color,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=3,
    )
    draw_curve_markers(ax=ax, summary=summary, color=color)


def draw_h_ext_curve(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw the mean extrinsic-mortality curve plus sensitivity intervals."""
    summary = h_ext_summary(data)
    if summary.empty:
        return

    color = PARAM_COLORS["h_ext"]
    draw_sensitivity_envelope(ax=ax, summary=summary, color=color)
    ax.plot(
        summary["x_mean"],
        summary["y_mean"],
        color=color,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=2,
    )
    draw_curve_markers(ax=ax, summary=summary, color=color, use_h_ext_sizes=True)


def build_grouped_legend(ax: plt.Axes) -> None:
    """Build an in-panel grouped parameter legend."""
    handles = [
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=PARAM_COLORS["eta"], lw=3, label=PARAM_LABELS["eta"]),
        Line2D([0], [0], color=PARAM_COLORS["beta"], lw=3, label=PARAM_LABELS["beta"]),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=PARAM_COLORS["Xc"], lw=3, label=PARAM_LABELS["Xc"]),
        Line2D([0], [0], color=PARAM_COLORS["epsilon"], lw=3, label=PARAM_LABELS["epsilon"]),
        Line2D([], [], color="none", label=" "),
        Line2D(
            [0],
            [0],
            color=PARAM_COLORS["h_ext"],
            lw=3,
            label="Extrinsic mortality",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        fontsize=10.0,
        handlelength=2.2,
        labelspacing=0.38,
        borderpad=0.45,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_alpha(0.92)

    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_fontweight("bold")
            text.set_color("#222222")

    ax.add_artist(legend)
    add_full_width_legend_separator(ax=ax, legend=legend, row_index=6)


def add_full_width_legend_separator(ax: plt.Axes, legend: plt.Legend, row_index: int) -> None:
    """Draw a subtle full-width separator through one blank legend row."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer=renderer)
    text_box = legend.get_texts()[row_index].get_window_extent(renderer=renderer)

    y_display = (text_box.y0 + text_box.y1) / 2
    start = ax.transAxes.inverted().transform((legend_box.x0, y_display))
    end = ax.transAxes.inverted().transform((legend_box.x1, y_display))

    separator = Line2D(
        [start[0], end[0]],
        [start[1], end[1]],
        transform=ax.transAxes,
        color=LEGEND_SEPARATOR_COLOR,
        linewidth=1.6,
        alpha=LEGEND_SEPARATOR_ALPHA,
        solid_capstyle="butt",
        clip_on=False,
        zorder=5,
    )
    ax.add_line(separator)


def build_factor_legend(ax: plt.Axes) -> None:
    """Add an in-panel legend explaining factor marker size."""
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
        fontsize=12,
        title_fontsize=13,
        borderpad=0.65,
        labelspacing=0.55,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#E3E3E3")
    legend.get_frame().set_alpha(0.92)


def finish_axes(ax: plt.Axes) -> None:
    """Format axes and reference lines."""
    ax.axhline(1.0, color="#B8B8B8", linewidth=1.1, linestyle="--", alpha=0.80, zorder=0)
    ax.axvline(1.0, color="#B8B8B8", linewidth=1.1, linestyle="--", alpha=0.80, zorder=0)

    ax.set_xlim(*AX_LIMITS)
    ax.set_ylim(*AX_LIMITS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0.4, 1.7, 0.2))
    ax.set_yticks(np.arange(0.4, 1.7, 0.2))
    ax.set_xlabel("Median lifespan relative to baseline", labelpad=8)
    ax.set_ylabel("Steepness relative to baseline", labelpad=10)
    ax.set_title(
        "USA 2019 parameter change effect on survival curve shape",
        loc="left",
        fontsize=15,
        pad=11,
    )

    ax.tick_params(length=6.5, width=1.25, color="#222222")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.set_facecolor("white")


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_normalized_metrics()
    save_plot_data(data)
    save_summary_data(data)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor("white")

    for param in PLOT_PARAMS:
        draw_parameter_curve(ax=ax, data=data, param=param)

    draw_h_ext_curve(ax=ax, data=data)
    finish_axes(ax)
    build_grouped_legend(ax)
    build_factor_legend(ax)

    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.92)
    fig.savefig(PNG_PATH, dpi=350, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")
    print(f"Saved plot data: {PLOT_DATA_PATH}")
    print(f"Saved interval data: {SUMMARY_PATH}")
    print(f"Saved envelope data: {ENVELOPE_PATH}")


if __name__ == "__main__":
    main()
