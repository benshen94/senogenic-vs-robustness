#!/usr/bin/env python3
"""Make Fig4 historical steepness-longevity projections."""

from __future__ import annotations

import csv
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import OptimizeWarning


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis import HMD
from ageing_packages.mortality_models.gamma_gompertz import GammaGompertz
from src.figures.steepness_longevity import make_fig1d_new_steepness_longevity as fig1e
from src.shared.thresholds import thresholds_functions as th
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Fig4_new"
RESULTS_DIR = SAVED_RESULTS_DIR / "fig4_new"
OUTPUT_INDEX = SAVED_RESULTS_DIR / "index" / "outputs.csv"

PROJECTION_DATA_PATH = RESULTS_DIR / "sweden_period_steepness_longevity_projection.csv"
DENMARK_PROJECTION_DATA_PATH = RESULTS_DIR / "denmark_period_steepness_longevity_projection.csv"
COMBINED_PROJECTION_DATA_PATH = RESULTS_DIR / "country_period_steepness_longevity_projection.csv"
FIG4C_DATA_PATH = RESULTS_DIR / "fig4c_extrinsic_mortality_projection.csv"
FIG4D_DATA_PATH = RESULTS_DIR / "fig4d_robustness_projection.csv"
FIG4D_EXTRAP_DATA_PATH = RESULTS_DIR / "fig4d_robustness_extrapolation.csv"
MAKEHAM_CI_DATA_PATH = RESULTS_DIR / "sweden_period_makeham_m_fit_ci.csv"
README_PATH = RESULTS_DIR / "README.md"

OLD_RECREATION_PNG = OUTPUT_DIR / "fig4_ab_old_recreated.png"
OLD_RECREATION_PDF = OUTPUT_DIR / "fig4_ab_old_recreated.pdf"
NEW_PROJECTION_PNG = OUTPUT_DIR / "fig4_ab_sweden_period_projection.png"
NEW_PROJECTION_PDF = OUTPUT_DIR / "fig4_ab_sweden_period_projection.pdf"
DENMARK_PROJECTION_PNG = OUTPUT_DIR / "fig4_ab_denmark_period_projection.png"
DENMARK_PROJECTION_PDF = OUTPUT_DIR / "fig4_ab_denmark_period_projection.pdf"
FIG4C_PNG = OUTPUT_DIR / "Fig4C.png"
FIG4C_PDF = OUTPUT_DIR / "Fig4C.pdf"
FIG4D_PNG = OUTPUT_DIR / "Fig4D.png"
FIG4D_PDF = OUTPUT_DIR / "Fig4D.pdf"
FIG4D_EXTRAP_PNG = OUTPUT_DIR / "Fig4D_extrap.png"
FIG4D_EXTRAP_PDF = OUTPUT_DIR / "Fig4D_extrap.pdf"

START_YEAR = 1800
END_YEAR = 2020
REF_YEAR = 2019
OLD_FIG4_REF_YEAR = 2020
FROM_T = 20
GGM_FIT_AGE_START = 20
GGM_FIT_AGE_END = 100
GGM_SURVIVAL_TMAX = 180.0
GGM_SURVIVAL_DT = 0.05
SWEDEN_KEY = "sweden"
NORMAL_CI_Z = 1.959963984540054
ROBUSTNESS_ARTIFACT_YEAR = 1918
EXTRAPOLATION_START_YEAR = 1980
EXTRAPOLATION_END_YEAR = 2100
EXTRINSIC_PANEL_COLOR = "#B84A4F"
ROBUSTNESS_PANEL_COLOR = fig1e.PARAM_COLORS["Xc"]

SURVIVAL_LEVELS = (0.75, 0.5, 0.25)


@dataclass(frozen=True)
class CountryConfig:
    key: str
    name: str
    hmd_code: str
    start_year: int
    data_path: Path
    projection_png: Path
    projection_pdf: Path
    color: str
    marker: str


COUNTRIES = (
    CountryConfig(
        key="sweden",
        name="Sweden",
        hmd_code="swe",
        start_year=1800,
        data_path=PROJECTION_DATA_PATH,
        projection_png=NEW_PROJECTION_PNG,
        projection_pdf=NEW_PROJECTION_PDF,
        color="#111111",
        marker="^",
    ),
    CountryConfig(
        key="denmark",
        name="Denmark",
        hmd_code="dan",
        start_year=1835,
        data_path=DENMARK_PROJECTION_DATA_PATH,
        projection_png=DENMARK_PROJECTION_PNG,
        projection_pdf=DENMARK_PROJECTION_PDF,
        color="#0072B2",
        marker="o",
    ),
)

CONDITION_LABELS = {
    "with_extrinsic": "With extrinsic mortality",
    "extrinsic_removed": "Extrinsic mortality removed",
}
CONDITION_MARKERS = {
    "with_extrinsic": "^",
    "extrinsic_removed": "^",
}
HISTORICAL_CMAP = "Greys"
OLD_AXIS_LIMITS = {
    "with_extrinsic": ((0.5, 1.1), (0.3, 1.4)),
    "extrinsic_removed": ((0.5, 1.1), (0.3, 1.4)),
}
NEW_AXIS_LIMITS = {
    "with_extrinsic": ((0.5, 1.1), (0.3, 1.4)),
    "extrinsic_removed": ((0.5, 1.1), (0.3, 1.4)),
}

REQUIRED_DATA_COLUMNS = {
    "country_key",
    "country_name",
    "x_relative_to_sr",
    "y_relative_to_sr",
    "x_relative_to_ref_year",
    "y_relative_to_ref_year",
    "x_relative_to_old_fig4_ref",
    "y_relative_to_old_fig4_ref",
    "ggm_m",
}

LEGEND_SEPARATOR_COLOR = "#9C9C9C"
CI_LOWER_Q = 0.025
CI_UPPER_Q = 0.975
FIT_TMAX_MEDIAN_MARGIN = 5.0


def load_sr_baseline() -> tuple[float, float]:
    """Return the central SR baseline at the starting age used for Fig4."""
    metrics = pd.read_csv(fig1e.METRICS_PATH)
    baseline = metrics[
        (metrics["curve_type"] == "baseline")
        & (metrics["scenario_id"] == "central")
        & (metrics["from_t"] == FROM_T)
    ]
    if baseline.empty:
        raise ValueError(f"No central SR baseline found for from_t={FROM_T}.")

    row = baseline.iloc[0]
    return float(row["t_median_absolute"]), float(row["steepness_iqr_absolute"])


def hmd_iqr_metrics(hmd: HMD, year: int) -> dict[str, float]:
    """Calculate period HMD median and IQR steepness for one year."""
    quantiles = hmd.calculate_lifespan_quantiles(year, SURVIVAL_LEVELS, FROM_T)
    age_at_75 = float(quantiles[0.75])
    median = float(quantiles[0.5])
    age_at_25 = float(quantiles[0.25])
    return summarize_quantile_ages(age_at_75, median, age_at_25)


def fit_ggm_without_extrinsic(hmd: HMD, year: int) -> tuple[dict[str, float], dict[str, float]]:
    """Fit GGM for one year, set Makeham mortality to zero, and summarize it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params = hmd.fit_ggm(
            year=year,
            age_start=GGM_FIT_AGE_START,
            age_end=GGM_FIT_AGE_END,
        )

    clean_params = {name: float(value) for name, value in params.items()}
    no_ext_params = dict(clean_params)
    no_ext_params["m"] = 0.0

    metrics = ggm_iqr_metrics(no_ext_params)
    return clean_params, metrics


def fit_ggm_makeham_with_ci(hmd: HMD, year: int) -> dict[str, float | int | str]:
    """Fit GGM and return the fitted Makeham term with a 95% interval."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params, stderr = hmd.fit_ggm(
            year=year,
            age_start=GGM_FIT_AGE_START,
            age_end=GGM_FIT_AGE_END,
            return_cov=True,
        )

    return {
        "year": int(year),
        "ggm_m": float(params["m"]),
        "ggm_m_stderr": float(stderr["m"]),
        "ggm_fit_age_start": GGM_FIT_AGE_START,
        "ggm_fit_age_end": GGM_FIT_AGE_END,
        "ci_method": "lognormal delta-method from GGM curve_fit covariance",
    }


def ggm_iqr_metrics(params: dict[str, float]) -> dict[str, float]:
    """Calculate conditional GGM quantile ages deterministically."""
    model = GammaGompertz(params=params)
    ages = np.arange(FROM_T, GGM_SURVIVAL_TMAX + GGM_SURVIVAL_DT, GGM_SURVIVAL_DT)
    _, survival = model.survival_function(ages)
    conditional = survival / survival[0]

    age_at_75 = find_age_at_survival(ages, conditional, 0.75)
    median = find_age_at_survival(ages, conditional, 0.5)
    age_at_25 = find_age_at_survival(ages, conditional, 0.25)
    return summarize_quantile_ages(age_at_75, median, age_at_25)


def find_age_at_survival(ages: np.ndarray, survival: np.ndarray, level: float) -> float:
    """Interpolate the age at which conditional survival reaches ``level``."""
    if len(ages) == 0 or survival[-1] > level:
        return np.nan
    return float(np.interp(level, survival[::-1], ages[::-1]))


def summarize_quantile_ages(age_at_75: float, median: float, age_at_25: float) -> dict[str, float]:
    """Convert survival quantile ages into median, IQR, and steepness."""
    iqr = age_at_25 - age_at_75
    steepness = median / iqr if np.isfinite(iqr) and iqr > 0 else np.nan
    return {
        "age_at_survival_75": age_at_75,
        "median_lifespan": median,
        "age_at_survival_25": age_at_25,
        "iqr": iqr,
        "steepness_iqr_absolute": steepness,
    }


def build_projection_data() -> pd.DataFrame:
    """Load or calculate country historical coordinates for Fig4."""
    country_tables = [build_country_projection_data(country) for country in COUNTRIES]
    combined = pd.concat(country_tables, ignore_index=True)
    combined.to_csv(COMBINED_PROJECTION_DATA_PATH, index=False)
    write_results_readme()
    return combined


def build_country_projection_data(country: CountryConfig) -> pd.DataFrame:
    """Load or calculate one country's historical coordinates for both Fig4 panels."""
    if country.data_path.exists():
        cached = pd.read_csv(country.data_path)
        cached = add_country_columns_if_missing(cached, country)
        if REQUIRED_DATA_COLUMNS.issubset(cached.columns):
            cached.to_csv(country.data_path, index=False)
            print(f"Using cached projection data: {country.data_path}")
            return cached

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building projection data cache: {country.data_path}")
    hmd = HMD(country=country.hmd_code, gender="both", data_type="period")
    sr_median, sr_steepness = load_sr_baseline()
    first_year = max(country.start_year, int(hmd.data["Year"].min()))
    last_year = min(END_YEAR, int(hmd.data["Year"].max()))
    years = range(first_year, last_year + 1)

    ref_with_ext = hmd_iqr_metrics(hmd, REF_YEAR)
    _, ref_no_ext = fit_ggm_without_extrinsic(hmd, REF_YEAR)
    old_ref_with_ext = hmd_iqr_metrics(hmd, OLD_FIG4_REF_YEAR)
    _, old_ref_no_ext = fit_ggm_without_extrinsic(hmd, OLD_FIG4_REF_YEAR)

    rows = []
    for year in years:
        with_ext = hmd_iqr_metrics(hmd, year)
        rows.append(
            projection_row(
                country=country,
                year=year,
                condition="with_extrinsic",
                metrics=with_ext,
                sr_median=sr_median,
                sr_steepness=sr_steepness,
                ref_metrics=ref_with_ext,
                old_ref_metrics=old_ref_with_ext,
                ggm_params=None,
            )
        )

        ggm_params, no_ext = fit_ggm_without_extrinsic(hmd, year)
        rows.append(
            projection_row(
                country=country,
                year=year,
                condition="extrinsic_removed",
                metrics=no_ext,
                sr_median=sr_median,
                sr_steepness=sr_steepness,
                ref_metrics=ref_no_ext,
                old_ref_metrics=old_ref_no_ext,
                ggm_params=ggm_params,
            )
        )

    data = pd.DataFrame(rows)
    data.to_csv(country.data_path, index=False)
    return data


def add_country_columns_if_missing(data: pd.DataFrame, country: CountryConfig) -> pd.DataFrame:
    """Upgrade older per-country caches without rerunning expensive fits."""
    data = data.copy()
    if "country_key" not in data.columns:
        data["country_key"] = country.key
    if "country_name" not in data.columns:
        data["country_name"] = country.name
    if "hmd_country_code" not in data.columns:
        data["hmd_country_code"] = country.hmd_code
    return data


def projection_row(
    *,
    country: CountryConfig,
    year: int,
    condition: str,
    metrics: dict[str, float],
    sr_median: float,
    sr_steepness: float,
    ref_metrics: dict[str, float],
    old_ref_metrics: dict[str, float],
    ggm_params: dict[str, float] | None,
) -> dict[str, float | int | str]:
    """Build one saved historical-coordinate row."""
    row: dict[str, float | int | str] = {
        "country_key": country.key,
        "country_name": country.name,
        "hmd_country_code": country.hmd_code,
        "year": int(year),
        "condition": condition,
        "from_t": FROM_T,
        "ref_year": REF_YEAR,
        "old_fig4_ref_year": OLD_FIG4_REF_YEAR,
        "sr_baseline_median": sr_median,
        "sr_baseline_steepness": sr_steepness,
        "ref_median": ref_metrics["median_lifespan"],
        "ref_steepness": ref_metrics["steepness_iqr_absolute"],
        "old_fig4_ref_median": old_ref_metrics["median_lifespan"],
        "old_fig4_ref_steepness": old_ref_metrics["steepness_iqr_absolute"],
        **metrics,
        "x_relative_to_sr": metrics["median_lifespan"] / sr_median,
        "y_relative_to_sr": metrics["steepness_iqr_absolute"] / sr_steepness,
        "x_relative_to_ref_year": metrics["median_lifespan"] / ref_metrics["median_lifespan"],
        "y_relative_to_ref_year": metrics["steepness_iqr_absolute"] / ref_metrics["steepness_iqr_absolute"],
        "x_relative_to_old_fig4_ref": metrics["median_lifespan"] / old_ref_metrics["median_lifespan"],
        "y_relative_to_old_fig4_ref": (
            metrics["steepness_iqr_absolute"] / old_ref_metrics["steepness_iqr_absolute"]
        ),
    }

    for param in ("a", "b", "c", "m"):
        row[f"ggm_{param}"] = np.nan if ggm_params is None else ggm_params[param]

    return row


def apply_fig4_style() -> None:
    """Use the same publication-safe style as the new Fig1E panel."""
    fig1e.apply_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )


def draw_historical_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    condition: str,
    *,
    x_column: str,
    y_column: str,
    show_colorbar: bool,
):
    """Draw yearly country period coordinates on one steepness-longevity plane."""
    rows = data[data["condition"] == condition].copy()
    scatter = ax.scatter(
        rows[x_column],
        rows[y_column],
        c=rows["year"],
        cmap=HISTORICAL_CMAP,
        s=36,
        marker=CONDITION_MARKERS[condition],
        edgecolor="#111111",
        linewidth=0.45,
        alpha=0.92,
        zorder=20,
    )
    if show_colorbar:
        cbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("Year", fontsize=12)
        cbar.ax.tick_params(labelsize=10)
    return scatter


def draw_new_baseline(ax: plt.Axes) -> None:
    """Draw the new Fig1E SR response plane underlay."""
    data = fig1e.load_normalized_metrics()
    for param in fig1e.PLOT_PARAMS:
        fig1e.draw_parameter_curve(ax=ax, data=data, param=param)
    fig1e.draw_h_ext_curve(ax=ax, data=data)
    fig1e.finish_axes(ax)


def draw_old_baseline(ax: plt.Axes) -> None:
    """Draw the legacy SR response plane used by old Fig4 A/B."""
    th.plot_steepness_longevity(
        param_type="variation",
        from_t=FROM_T,
        h_ext=True,
        ax=ax,
        marker_size_range=(20, 300),
        linewidth=5,
        legend_fontsize=11,
    )


def render_old_recreation(data: pd.DataFrame) -> None:
    """Recreate old Fig4 A/B with legacy underlay and saved point coordinates."""
    apply_fig4_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.6))

    for ax, condition in zip(axes, CONDITION_LABELS):
        draw_old_baseline(ax)
        draw_historical_points(
            ax,
            data,
            condition,
            x_column="x_relative_to_old_fig4_ref",
            y_column="y_relative_to_old_fig4_ref",
            show_colorbar=condition == "extrinsic_removed",
        )
        xlim, ylim = OLD_AXIS_LIMITS[condition]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title("", loc="center")
        ax.set_title("", loc="right")
        ax.set_title(CONDITION_LABELS[condition], loc="left", pad=10)
        ax.set_xlabel(f"Median lifespan relative to Sweden {OLD_FIG4_REF_YEAR}")
        ax.set_ylabel(f"Steepness relative to Sweden {OLD_FIG4_REF_YEAR}")
        ax.grid(False)
        remove_legend(ax)

    add_panel_labels(axes)
    fig.subplots_adjust(left=0.08, right=0.94, bottom=0.13, top=0.90, wspace=0.30)
    fig.savefig(OLD_RECREATION_PNG, dpi=350, bbox_inches="tight")
    fig.savefig(OLD_RECREATION_PDF, bbox_inches="tight")
    plt.close(fig)


def render_new_projection(data: pd.DataFrame, country: CountryConfig) -> None:
    """Render new Fig4 A/B using the Fig1E baseline and saved country coordinates."""
    apply_fig4_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 10.8))
    fig.patch.set_facecolor("white")
    colorbar_scatter = None

    for ax, condition in zip(axes, CONDITION_LABELS):
        draw_new_baseline(ax)
        xlim, ylim = NEW_AXIS_LIMITS[condition]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Normalized median lifespan ({REF_YEAR} SR = 1)")
        ax.set_ylabel("")
        scatter = draw_historical_points(
            ax,
            data,
            condition,
            x_column="x_relative_to_sr",
            y_column="y_relative_to_sr",
            show_colorbar=False,
        )
        if condition == "extrinsic_removed":
            colorbar_scatter = scatter
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
        ax.set_title(CONDITION_LABELS[condition], loc="center", pad=12)
        remove_legend(ax)

    build_shared_legend(axes[0], country=country, data=data)
    fig.supylabel(f"Normalized steepness ({REF_YEAR} SR = 1)", fontsize=17, x=0.025)
    add_panel_labels(axes)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.12, top=0.88, wspace=0.08)
    add_full_height_colorbar(fig, axes, colorbar_scatter)
    fig.savefig(country.projection_png, dpi=350, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(country.projection_pdf, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def add_full_height_colorbar(fig: plt.Figure, axes: np.ndarray, scatter) -> None:
    """Add one grayscale year colorbar aligned to the full A/B panel height."""
    if scatter is None:
        return

    fig.canvas.draw()
    left_pos = axes[0].get_position()
    right_pos = axes[1].get_position()
    bottom = min(left_pos.y0, right_pos.y0)
    top = max(left_pos.y1, right_pos.y1)
    cax = fig.add_axes([right_pos.x1 + 0.025, bottom, 0.018, top - bottom])
    draw_vector_year_colorbar(cax, scatter)


def draw_vector_year_colorbar(ax: plt.Axes, scatter) -> None:
    """Draw a grayscale year colorbar as vector rectangles, not a raster image."""
    norm = scatter.norm
    cmap = scatter.cmap
    year_min, year_max = norm.vmin, norm.vmax
    edges = np.linspace(year_min, year_max, 120)

    for start, end in zip(edges[:-1], edges[1:]):
        midpoint = 0.5 * (start + end)
        ax.add_patch(
            Rectangle(
                (0.0, start),
                1.0,
                end - start,
                facecolor=cmap(norm(midpoint)),
                edgecolor="none",
                linewidth=0,
            )
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(year_min, year_max)
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("Year", fontsize=15)
    ax.tick_params(axis="y", labelsize=13, width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#555555")


def build_shared_legend(ax: plt.Axes, *, country: CountryConfig, data: pd.DataFrame) -> None:
    """Add a compact model-plus-data legend to the first panel."""
    first_year = int(data["year"].min())
    last_year = int(data["year"].max())
    handles = [
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["eta"], lw=3, label=fig1e.PARAM_LABELS["eta"]),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["beta"], lw=3, label=fig1e.PARAM_LABELS["beta"]),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["Xc"], lw=3, label=fig1e.PARAM_LABELS["Xc"]),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["epsilon"], lw=3, label=fig1e.PARAM_LABELS["epsilon"]),
        Line2D([0], [0], color=LEGEND_SEPARATOR_COLOR, lw=1.4, label=" "),
        Line2D([0], [0], color=fig1e.PARAM_COLORS["h_ext"], lw=3, label="Extrinsic mortality"),
        Line2D([], [], color="none", label=" "),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#B5B5B5",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            markersize=9,
            label=f"{country.name} period years ({first_year}-{last_year})",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=12,
        handlelength=2.0,
        labelspacing=0.38,
        borderpad=0.35,
    )
    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_fontweight("bold")


def remove_legend(ax: plt.Axes) -> None:
    """Remove whichever legend the underlay created."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def add_panel_labels(axes: np.ndarray) -> None:
    """Add manuscript-style A/B panel labels."""
    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.14,
            1.06,
            label,
            transform=ax.transAxes,
            fontsize=28,
            fontweight="normal",
            va="top",
            ha="left",
        )


def build_fig4c_data(period_data: pd.DataFrame) -> pd.DataFrame:
    """Save the Sweden fitted GGM Makeham term used for Fig4C."""
    rows = period_data[
        (period_data["country_key"] == SWEDEN_KEY)
        & (period_data["condition"] == "extrinsic_removed")
    ].copy()
    rows = rows[np.isfinite(rows["ggm_m"])].copy()
    makeham_ci = load_or_build_makeham_ci(rows["year"].to_numpy(dtype=int))
    rows = rows.merge(makeham_ci, on="year", how="left", suffixes=("", "_ci_fit"))
    rows["ggm_m_for_plot"] = rows["ggm_m_ci_fit"].fillna(rows["ggm_m"])

    fig4c_rows = pd.DataFrame(
        {
            "country_key": rows["country_key"],
            "country_name": rows["country_name"],
            "year": rows["year"].astype(int),
            "projection_name": "fitted_makeham_m",
            "source_condition": "extrinsic_removed",
            "focal_param": "m",
            "point_x": rows["x_relative_to_sr"],
            "point_y": rows["y_relative_to_sr"],
            "estimate": rows["ggm_m_for_plot"],
            "ci_low": rows["ggm_m_ci_low"],
            "ci_high": rows["ggm_m_ci_high"],
            "ggm_m_stderr": rows["ggm_m_stderr"],
            "ci_method": rows["ci_method"],
            "distance_to_mean_curve": np.nan,
            "n_scenario_fits": 0,
            "fit_min_value": rows["ggm_m_for_plot"].min(),
            "fit_max_value": rows["ggm_m_for_plot"].max(),
        }
    )
    fig4c_rows.to_csv(FIG4C_DATA_PATH, index=False)
    return fig4c_rows


def load_or_build_makeham_ci(years: np.ndarray) -> pd.DataFrame:
    """Load or calculate cached Sweden GGM Makeham-term fit intervals."""
    required = {
        "year",
        "ggm_m",
        "ggm_m_stderr",
        "ggm_m_ci_low",
        "ggm_m_ci_high",
        "ci_method",
    }
    if MAKEHAM_CI_DATA_PATH.exists():
        cached = pd.read_csv(MAKEHAM_CI_DATA_PATH)
        if required.issubset(cached.columns) and set(years).issubset(set(cached["year"])):
            return cached

    print(f"Building Makeham m CI cache: {MAKEHAM_CI_DATA_PATH}")
    hmd = HMD(country="swe", gender="both", data_type="period")
    rows = [fit_ggm_makeham_with_ci(hmd, year=int(year)) for year in sorted(years)]
    data = pd.DataFrame(rows)
    data = add_makeham_confidence_interval(data)
    data.to_csv(MAKEHAM_CI_DATA_PATH, index=False)
    return data


def add_makeham_confidence_interval(data: pd.DataFrame) -> pd.DataFrame:
    """Add a positive 95% interval for fitted Makeham mortality."""
    data = data.sort_values("year").reset_index(drop=True).copy()
    data["ggm_m_stderr"] = interpolate_nonfinite_by_year(data, "ggm_m_stderr")

    relative_se = data["ggm_m_stderr"] / data["ggm_m"]
    data["ggm_m_ci_low"] = data["ggm_m"] * np.exp(-NORMAL_CI_Z * relative_se)
    data["ggm_m_ci_high"] = data["ggm_m"] * np.exp(NORMAL_CI_Z * relative_se)
    return data


def interpolate_nonfinite_by_year(data: pd.DataFrame, column: str) -> pd.Series:
    """Fill nonfinite values in one yearly series by linear interpolation."""
    values = data[column].to_numpy(dtype=float)
    years = data["year"].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.all():
        return pd.Series(values, index=data.index)
    if not finite.any():
        return pd.Series(np.nan, index=data.index)

    filled = values.copy()
    filled[~finite] = np.interp(years[~finite], years[finite], values[finite])
    return pd.Series(filled, index=data.index)


def build_fig4d_data(period_data: pd.DataFrame) -> pd.DataFrame:
    """Fit each extrinsic-removed point to the Xc robustness response curve."""
    rows = fit_period_points_to_model_curve(
        period_data=period_data[period_data["country_key"] == SWEDEN_KEY].copy(),
        source_condition="extrinsic_removed",
        focal_param="Xc",
        projection_name="robustness",
    )
    rows = smooth_robustness_artifact(rows)
    rows.to_csv(FIG4D_DATA_PATH, index=False)
    return rows


def smooth_robustness_artifact(rows: pd.DataFrame) -> pd.DataFrame:
    """Replace the known 1918 robustness artifact by adjacent-year interpolation."""
    rows = rows.sort_values("year").reset_index(drop=True).copy()
    rows["artifact_adjustment"] = "none"
    target = rows["year"] == ROBUSTNESS_ARTIFACT_YEAR
    if not target.any():
        return rows

    previous_row = rows[rows["year"] == ROBUSTNESS_ARTIFACT_YEAR - 1]
    next_row = rows[rows["year"] == ROBUSTNESS_ARTIFACT_YEAR + 1]
    if previous_row.empty or next_row.empty:
        return rows

    for column in ("estimate", "ci_low", "ci_high"):
        replacement = 0.5 * (float(previous_row[column].iloc[0]) + float(next_row[column].iloc[0]))
        rows.loc[target, column] = replacement

    rows.loc[target, "artifact_adjustment"] = "mean_of_1917_and_1919"
    return rows


def build_fig4d_extrapolation_data(fig4d_data: pd.DataFrame) -> pd.DataFrame:
    """Build observed and extrapolated Fig4D robustness trajectories."""
    observed = fig4d_data[fig4d_data["year"] <= END_YEAR].copy()
    observed["series"] = "data"
    observed["fit_start_year"] = EXTRAPOLATION_START_YEAR
    observed["fit_end_year"] = END_YEAR

    fit_rows = observed[
        (observed["year"] >= EXTRAPOLATION_START_YEAR)
        & (observed["year"] <= END_YEAR)
    ].copy()
    fit_years = fit_rows["year"].to_numpy(dtype=float)
    fit_values = fit_rows["estimate"].to_numpy(dtype=float)
    linear_slope, _ = np.polyfit(fit_years, fit_values, 1)
    exponential_log_slope, _ = np.polyfit(fit_years, np.log(fit_values), 1)

    future_years = np.arange(END_YEAR, EXTRAPOLATION_END_YEAR + 1)
    last_value = float(observed.loc[observed["year"] == END_YEAR, "estimate"].iloc[0])
    projected = pd.concat(
        [
            extrapolation_rows(
                future_years,
                series="linear_extrapolation",
                values=last_value + linear_slope * (future_years - END_YEAR),
                slope=linear_slope,
            ),
            extrapolation_rows(
                future_years,
                series="exponential_extrapolation",
                values=last_value * np.exp(exponential_log_slope * (future_years - END_YEAR)),
                slope=exponential_log_slope,
            ),
        ],
        ignore_index=True,
    )

    result = pd.concat([observed, projected], ignore_index=True, sort=False)
    result.to_csv(FIG4D_EXTRAP_DATA_PATH, index=False)
    return result


def extrapolation_rows(
    years: np.ndarray,
    *,
    series: str,
    values: np.ndarray,
    slope: float,
) -> pd.DataFrame:
    """Return one extrapolated trajectory in the Fig4D schema."""
    return pd.DataFrame(
        {
            "country_key": SWEDEN_KEY,
            "country_name": "Sweden",
            "year": years.astype(int),
            "projection_name": "robustness_extrapolation",
            "source_condition": "extrinsic_removed",
            "focal_param": "Xc",
            "estimate": values,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "series": series,
            "fit_start_year": EXTRAPOLATION_START_YEAR,
            "fit_end_year": END_YEAR,
            "fit_slope": slope,
        }
    )


def fit_period_points_to_model_curve(
    *,
    period_data: pd.DataFrame,
    source_condition: str,
    focal_param: str,
    projection_name: str,
) -> pd.DataFrame:
    """Project country-year coordinates to one model curve and scenario CI."""
    model_rows = model_rows_for_fit(focal_param)
    mean_curve = mean_model_curve(model_rows)
    period_rows = period_data[period_data["condition"] == source_condition].copy()

    fitted_rows = []
    for _, period_row in period_rows.iterrows():
        point = np.array(
            [
                float(period_row["x_relative_to_sr"]),
                float(period_row["y_relative_to_sr"]),
            ]
        )
        central_fit = fit_point_to_curve(point, mean_curve, focal_param)
        scenario_fits = fit_point_to_scenario_curves(point, model_rows, focal_param)
        ci_low, ci_high = fitted_parameter_interval(scenario_fits, focal_param)

        fitted_rows.append(
            {
                "country_key": period_row["country_key"],
                "country_name": period_row["country_name"],
                "year": int(period_row["year"]),
                "projection_name": projection_name,
                "source_condition": source_condition,
                "focal_param": focal_param,
                "point_x": point[0],
                "point_y": point[1],
                "estimate": central_fit["estimate"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "distance_to_mean_curve": central_fit["distance"],
                "n_scenario_fits": len(scenario_fits),
                "fit_min_value": float(mean_curve["focal_value"].min()),
                "fit_max_value": float(mean_curve["focal_value"].max()),
            }
        )

    return pd.DataFrame(fitted_rows)


def model_rows_for_fit(focal_param: str) -> pd.DataFrame:
    """Return finite Fig1E model rows for one fitted parameter."""
    data = fig1e.load_normalized_metrics()
    if focal_param == "h_ext":
        mask = (data["curve_type"] == "h_ext_absolute") & (data["focal_param"] == "h_ext")
    else:
        mask = (data["curve_type"] == "parameter_factor") & (data["focal_param"] == focal_param)

    rows = data.loc[mask].copy()
    valid = (
        np.isfinite(rows["x_norm"])
        & np.isfinite(rows["y_norm"])
        & np.isfinite(rows["focal_value"])
        & (rows["t_median_absolute"] <= rows["tmax"] - FIT_TMAX_MEDIAN_MARGIN)
    )
    return rows.loc[valid].copy()


def mean_model_curve(model_rows: pd.DataFrame) -> pd.DataFrame:
    """Average raw scenario curves at each model parameter value."""
    return (
        model_rows.groupby("focal_value", as_index=False)
        .agg(x_norm=("x_norm", "mean"), y_norm=("y_norm", "mean"))
        .sort_values("focal_value")
        .reset_index(drop=True)
    )


def fit_point_to_scenario_curves(
    point: np.ndarray,
    model_rows: pd.DataFrame,
    focal_param: str,
) -> list[float]:
    """Fit one point to every scenario-specific curve that has enough values."""
    estimates = []
    for _, scenario_rows in model_rows.groupby("scenario_id"):
        if scenario_rows["focal_value"].nunique() < 2:
            continue
        fit = fit_point_to_curve(point, scenario_rows, focal_param)
        if np.isfinite(fit["estimate"]):
            estimates.append(float(fit["estimate"]))
    return estimates


def fit_point_to_curve(
    point: np.ndarray,
    curve: pd.DataFrame,
    focal_param: str,
) -> dict[str, float]:
    """Find the closest position on a parameterized x/y curve."""
    curve = curve.sort_values("focal_value").reset_index(drop=True)
    values = curve["focal_value"].to_numpy(dtype=float)
    value_positions = transformed_parameter_values(values, focal_param)
    coordinates = curve[["x_norm", "y_norm"]].to_numpy(dtype=float)

    best_estimate = np.nan
    best_distance = np.inf
    for index in range(len(coordinates) - 1):
        start = coordinates[index]
        end = coordinates[index + 1]
        segment = end - start
        segment_length_sq = float(np.dot(segment, segment))
        if segment_length_sq == 0:
            fraction = 0.0
            closest = start
        else:
            fraction = float(np.clip(np.dot(point - start, segment) / segment_length_sq, 0.0, 1.0))
            closest = start + fraction * segment

        distance = float(np.linalg.norm(point - closest))
        if distance < best_distance:
            value_position = value_positions[index] + fraction * (
                value_positions[index + 1] - value_positions[index]
            )
            best_estimate = inverse_transformed_parameter_value(value_position, focal_param)
            best_distance = distance

    return {"estimate": float(best_estimate), "distance": float(best_distance)}


def transformed_parameter_values(values: np.ndarray, focal_param: str) -> np.ndarray:
    """Transform parameter values before interpolation along a curve."""
    if focal_param == "h_ext":
        return np.log10(values)
    return values


def inverse_transformed_parameter_value(value: float, focal_param: str) -> float:
    """Undo the parameter transformation used for curve interpolation."""
    if focal_param == "h_ext":
        return float(10 ** value)
    return float(value)


def fitted_parameter_interval(estimates: list[float], focal_param: str) -> tuple[float, float]:
    """Return the 95% fitted-parameter interval across scenario curves."""
    if not estimates:
        return np.nan, np.nan

    values = np.array(estimates, dtype=float)
    if focal_param == "h_ext":
        log_values = np.log10(values[values > 0])
        if len(log_values) == 0:
            return np.nan, np.nan
        low, high = np.quantile(log_values, [CI_LOWER_Q, CI_UPPER_Q])
        return float(10**low), float(10**high)

    low, high = np.quantile(values, [CI_LOWER_Q, CI_UPPER_Q])
    return float(low), float(high)


def render_fig4c(data: pd.DataFrame) -> None:
    """Render Fig4C as the fitted GGM Makeham term over time."""
    apply_time_series_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    fig.patch.set_facecolor("white")

    plot_country_ci_series(ax=ax, data=data, color=EXTRINSIC_PANEL_COLOR)
    ax.set_yscale("log")
    ax.set_title("Extrinsic mortality over time", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Extrinsic mortality [year$^{-1}$]")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    remove_axis_legend(ax)
    finish_time_series_axes(ax, panel_label="c")

    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.16, top=0.86)
    fig.savefig(FIG4C_PNG, dpi=350, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG4C_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_fig4d(data: pd.DataFrame) -> None:
    """Render Fig4D robustness projection with fitted-curve CIs."""
    apply_time_series_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    fig.patch.set_facecolor("white")

    plot_country_ci_series(ax=ax, data=data)
    ax.axhline(1.0, color="#B8B8B8", linewidth=1.2, linestyle="--", zorder=0)
    ax.set_title("Robustness projection", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Threshold $X_c$ factor")
    finish_time_series_axes(ax, panel_label="d")

    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.16, top=0.86)
    fig.savefig(FIG4D_PNG, dpi=350, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG4D_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_fig4d_extrapolation(data: pd.DataFrame) -> None:
    """Render the 1800-2100 robustness extrapolation panel."""
    apply_time_series_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    fig.patch.set_facecolor("white")

    observed = data[data["series"] == "data"].sort_values("year")
    linear = data[data["series"] == "linear_extrapolation"].sort_values("year")
    exponential = data[data["series"] == "exponential_extrapolation"].sort_values("year")

    ax.axvspan(END_YEAR, EXTRAPOLATION_END_YEAR, color="#EDEDED", alpha=0.8, zorder=0)
    ax.axvline(END_YEAR, color="#B8B8B8", linewidth=1.5, linestyle="--", zorder=1)
    ax.axhline(1.0, color="#B8B8B8", linewidth=1.2, linestyle="--", zorder=1)
    ax.fill_between(
        observed["year"],
        observed["ci_low"],
        observed["ci_high"],
        color=ROBUSTNESS_PANEL_COLOR,
        alpha=0.13,
        linewidth=0,
        label="95% CI",
        zorder=2,
    )
    ax.plot(
        observed["year"],
        observed["estimate"],
        color=ROBUSTNESS_PANEL_COLOR,
        linewidth=2.1,
        marker="^",
        markersize=4.2,
        markerfacecolor=ROBUSTNESS_PANEL_COLOR,
        markeredgecolor=ROBUSTNESS_PANEL_COLOR,
        label="Data",
        zorder=4,
    )
    ax.plot(
        linear["year"],
        linear["estimate"],
        color=ROBUSTNESS_PANEL_COLOR,
        linewidth=2.3,
        linestyle="-",
        label="Linear extrapolation",
        zorder=3,
    )
    ax.plot(
        exponential["year"],
        exponential["estimate"],
        color=ROBUSTNESS_PANEL_COLOR,
        linewidth=2.3,
        linestyle="--",
        label="Exponential extrapolation",
        zorder=3,
    )
    add_extrapolation_label(ax)

    ax.set_xlim(1795, EXTRAPOLATION_END_YEAR)
    max_y = max(float(linear["estimate"].max()), float(exponential["estimate"].max()))
    min_y = min(float(observed["ci_low"].min()), float(observed["estimate"].min()))
    ax.set_ylim(max(0.45, min_y - 0.04), max_y + 0.08)
    ax.set_title("Projected robustness over time", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Threshold $X_c$ factor")
    finish_time_series_axes(ax, panel_label="d", preserve_xlim=True)
    remove_axis_legend(ax)

    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.16, top=0.86)
    fig.savefig(FIG4D_EXTRAP_PNG, dpi=350, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG4D_EXTRAP_PDF, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def add_extrapolation_label(ax: plt.Axes) -> None:
    """Label the future shaded region without covering the curves."""
    ax.text(
        2058,
        0.62,
        "extrapolation",
        color="#555555",
        fontsize=15,
        ha="center",
        va="center",
    )


def remove_axis_legend(ax: plt.Axes) -> None:
    """Remove an automatically added legend if one exists."""
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def apply_time_series_style() -> None:
    """Apply readable publication styling for Fig4 C/D."""
    fig1e.apply_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 20,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
        }
    )


def plot_country_ci_series(ax: plt.Axes, data: pd.DataFrame, *, color: str | None = None) -> None:
    """Plot one fitted parameter trajectory per country with CI bands."""
    countries = {country.key: country for country in COUNTRIES}
    drew_ci = False
    for country_key, country_rows in data.groupby("country_key", sort=False):
        country = countries[country_key]
        plot_color = color or country.color
        rows = country_rows.sort_values("year")
        has_visible_ci = rows_have_visible_ci(rows)
        if has_visible_ci:
            drew_ci = True
            ax.fill_between(
                rows["year"],
                rows["ci_low"],
                rows["ci_high"],
                color=plot_color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            rows["year"],
            rows["estimate"],
            color=plot_color,
            linewidth=2.3,
            marker=country.marker,
            markersize=3.5,
            markerfacecolor=plot_color,
            markeredgecolor=plot_color,
            label=country.name,
            zorder=3,
        )

    handles, labels = ax.get_legend_handles_labels()
    if drew_ci:
        ci_color = color or "#111111"
        handles.append(Patch(facecolor=ci_color, alpha=0.16, edgecolor="none", label="95% CI"))
        labels.append("95% CI")
    ax.legend(handles, labels, frameon=False, loc="best")


def rows_have_visible_ci(rows: pd.DataFrame) -> bool:
    """Return True when the fitted series has a nonzero saved interval."""
    if not {"ci_low", "ci_high", "estimate"}.issubset(rows.columns):
        return False

    finite = (
        np.isfinite(rows["ci_low"])
        & np.isfinite(rows["ci_high"])
        & np.isfinite(rows["estimate"])
    )
    if not finite.any():
        return False

    lower_width = np.abs(rows.loc[finite, "estimate"] - rows.loc[finite, "ci_low"])
    upper_width = np.abs(rows.loc[finite, "ci_high"] - rows.loc[finite, "estimate"])
    return bool((lower_width.gt(0).any()) or (upper_width.gt(0).any()))


def finish_time_series_axes(
    ax: plt.Axes,
    *,
    panel_label: str,
    preserve_xlim: bool = False,
) -> None:
    """Format Fig4 C/D axes and panel label."""
    if not preserve_xlim:
        ax.set_xlim(1795, 2025)
    ax.tick_params(length=6.5, width=1.25, color="#222222")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.set_facecolor("white")
    ax.text(
        -0.17,
        1.12,
        panel_label,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="normal",
        va="top",
        ha="left",
    )


def write_results_readme() -> None:
    """Write a compact methods note next to the reusable data."""
    README_PATH.write_text(
        f"""# Fig4 New Period Projections

This folder stores reusable historical period coordinates and fitted Fig4 C/D projections for Sweden and Denmark.

For each available country-year, the saved tables contain the median lifespan and IQR steepness from age {FROM_T}. Steepness is calculated as:

\\[
\\mathrm{{steepness}} = \\frac{{t_{{50}}}}{{t_{{25}} - t_{{75}}}}
\\]

Panel A uses the raw period HMD survival curve. Panel B fits a Gamma-Gompertz-Makeham model to the same period year, saves the fitted Makeham term \\(m\\), sets \\(m = 0\\), and recalculates the quantiles from the fitted survival curve.

The plotted new-panel coordinates are `x_relative_to_sr` and `y_relative_to_sr`, normalized to the central Sweden 2019 zero-\\(h_{{ext}}\\) SR baseline used in the new Figure 1E steepness-longevity plane.

Fig4C plots the fitted Sweden period Makeham term \\(m\\) from the Gamma-Gompertz-Makeham fit used to make Panel B. Its 95% interval is a lognormal delta-method interval from the covariance matrix returned by the GGM curve fit.

Fig4D fits each Sweden extrinsic-removed point to the Figure 1E threshold \\(X_c\\) response curve. The Fig4D central fitted value is the closest point on the mean response curve. The saved Fig4D 95% intervals repeat the same closest-curve fit across the raw scenario-specific curves underlying the shaded Figure 1E ribbons and take the 2.5th and 97.5th percentiles. The 1918 robustness artifact is replaced by the mean of the 1917 and 1919 fitted values.

Fig4D_extrap fits the 1980-2020 Sweden robustness trajectory and extends it to 2100 with two anchored trends:

\\[
X_c(t) = X_c(2020) + s(t - 2020)
\\]

and

\\[
X_c(t) = X_c(2020)\\exp\\left(k(t - 2020)\\right).
\\]
""",
        encoding="utf-8",
    )


def update_output_index() -> None:
    """Append generated Fig4_new artifacts to the project output index."""
    OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
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
    existing_paths = set()
    if OUTPUT_INDEX.exists():
        with OUTPUT_INDEX.open(newline="", encoding="utf-8") as handle:
            existing_paths = {row.get("path", "") for row in csv.DictReader(handle)}

    source_script = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    rows = []
    for country in COUNTRIES:
        rows.extend(
            [
                output_index_row(
                    artifact_type="csv",
                    path=country.data_path,
                    source_script=source_script,
                    input_paths=str(fig1e.METRICS_PATH.relative_to(PROJECT_ROOT)),
                    description=f"Saved per-year {country.name} period projection coordinates for Fig4 A/B.",
                    notes="Includes raw HMD and extrinsic-removed GGM coordinates.",
                ),
                output_index_row(
                    artifact_type="figure",
                    path=country.projection_png,
                    source_script=source_script,
                    input_paths=str(country.data_path.relative_to(PROJECT_ROOT)),
                    description=f"New Fig4 A/B PNG with {country.name} period coordinates over the Figure 1E SR baseline.",
                    notes="Matching PDF saved next to the PNG.",
                ),
            ]
        )

    rows.extend(
        [
            output_index_row(
                artifact_type="csv",
                path=COMBINED_PROJECTION_DATA_PATH,
                source_script=source_script,
                input_paths=str(fig1e.METRICS_PATH.relative_to(PROJECT_ROOT)),
                description="Combined Sweden and Denmark period coordinates for Fig4.",
                notes="Used as input to Fig4C and Fig4D fitted parameter projections.",
            ),
            output_index_row(
                artifact_type="csv",
                path=FIG4C_DATA_PATH,
                source_script=source_script,
                input_paths=str(COMBINED_PROJECTION_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="Fig4C fitted Sweden GGM Makeham m estimates with 95% CIs.",
                notes="CIs use a lognormal delta-method interval from the GGM curve_fit covariance.",
            ),
            output_index_row(
                artifact_type="csv",
                path=MAKEHAM_CI_DATA_PATH,
                source_script=source_script,
                input_paths=str(PROJECTION_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="Cached Sweden period GGM Makeham m fit covariance intervals.",
                notes="Used by Fig4C so style changes do not rerun yearly GGM covariance fits.",
            ),
            output_index_row(
                artifact_type="csv",
                path=FIG4D_DATA_PATH,
                source_script=source_script,
                input_paths=str(COMBINED_PROJECTION_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="Fig4D fitted Sweden robustness estimates and CIs.",
                notes="CIs are across Figure 1E scenario-specific Xc curves; 1918 artifact interpolated.",
            ),
            output_index_row(
                artifact_type="csv",
                path=FIG4D_EXTRAP_DATA_PATH,
                source_script=source_script,
                input_paths=str(FIG4D_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="Fig4D_extrap observed 1980-2020 robustness and linear/exponential extrapolations to 2100.",
                notes="Linear and log-linear slopes fit over 1980-2020 and anchored at the 2020 estimate.",
            ),
            output_index_row(
                artifact_type="figure",
                path=FIG4C_PNG,
                source_script=source_script,
                input_paths=str(FIG4C_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="New Fig4C PNG: fitted Sweden GGM Makeham m over time.",
                notes="Matching PDF saved next to the PNG.",
            ),
            output_index_row(
                artifact_type="figure",
                path=FIG4D_PNG,
                source_script=source_script,
                input_paths=str(FIG4D_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="New Fig4D PNG: Sweden robustness projection over time.",
                notes="Matching PDF saved next to the PNG.",
            ),
            output_index_row(
                artifact_type="figure",
                path=FIG4D_EXTRAP_PNG,
                source_script=source_script,
                input_paths=str(FIG4D_EXTRAP_DATA_PATH.relative_to(PROJECT_ROOT)),
                description="New Fig4D_extrap PNG: Sweden robustness extrapolation to 2100.",
                notes="Matching PDF saved next to the PNG.",
            ),
        ]
    )

    file_exists = OUTPUT_INDEX.exists()
    with OUTPUT_INDEX.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            if row["path"] not in existing_paths:
                writer.writerow(row)


def output_index_row(
    *,
    artifact_type: str,
    path: Path,
    source_script: str,
    input_paths: str,
    description: str,
    notes: str,
) -> dict[str, str]:
    """Build one standard output-index row."""
    return {
        "date": "2026-05-20",
        "task": "Fig4_new country period steepness-longevity projection",
        "artifact_type": artifact_type,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "source_script": source_script,
        "input_paths": input_paths,
        "description": description,
        "notes": notes,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = build_projection_data()
    for country in COUNTRIES:
        country_data = data[data["country_key"] == country.key].copy()
        render_new_projection(country_data, country)

    fig4c_data = build_fig4c_data(data)
    fig4d_data = build_fig4d_data(data)
    fig4d_extrap_data = build_fig4d_extrapolation_data(fig4d_data)
    render_fig4c(fig4c_data)
    render_fig4d(fig4d_data)
    render_fig4d_extrapolation(fig4d_extrap_data)
    update_output_index()

    print(f"Saved combined projection data: {COMBINED_PROJECTION_DATA_PATH}")
    print(f"Saved Fig4C data: {FIG4C_DATA_PATH}")
    print(f"Saved Fig4D data: {FIG4D_DATA_PATH}")
    print(f"Saved Fig4D extrapolation data: {FIG4D_EXTRAP_DATA_PATH}")
    for country in COUNTRIES:
        print(f"Saved {country.name} A/B PNG: {country.projection_png}")
        print(f"Saved {country.name} A/B PDF: {country.projection_pdf}")
    print(f"Saved Fig4C PNG: {FIG4C_PNG}")
    print(f"Saved Fig4C PDF: {FIG4C_PDF}")
    print(f"Saved Fig4D PNG: {FIG4D_PNG}")
    print(f"Saved Fig4D PDF: {FIG4D_PDF}")
    print(f"Saved Fig4D extrapolation PNG: {FIG4D_EXTRAP_PNG}")
    print(f"Saved Fig4D extrapolation PDF: {FIG4D_EXTRAP_PDF}")


if __name__ == "__main__":
    main()
