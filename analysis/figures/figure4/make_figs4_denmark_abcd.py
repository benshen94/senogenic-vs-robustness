#!/usr/bin/env python3
"""Make the Denmark supplementary Fig4 A-D panel."""

from __future__ import annotations

import csv
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.optimize import OptimizeWarning


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis import HMD
from analysis.figures.figure4 import make_fig4_ab_sweden_period_projection as fig4


DENMARK_KEY = "denmark"
DENMARK_CI_DATA_PATH = fig4.RESULTS_DIR / "denmark_period_makeham_m_fit_ci.csv"
DENMARK_FIG4C_DATA_PATH = fig4.RESULTS_DIR / "denmark_fig4c_extrinsic_mortality_projection.csv"
DENMARK_FIG4D_DATA_PATH = fig4.RESULTS_DIR / "denmark_fig4d_robustness_projection.csv"
DENMARK_FIG4D_EXTRAP_DATA_PATH = fig4.RESULTS_DIR / "denmark_fig4d_robustness_extrapolation.csv"
DENMARK_START_YEAR = 1835
DENMARK_END_YEAR = fig4.END_YEAR
DENMARK_SUPP_PNG = fig4.OUTPUT_DIR / "denmark Fig4.png"
DENMARK_SUPP_PDF = fig4.OUTPUT_DIR / "denmark Fig4.pdf"


def denmark_config() -> fig4.CountryConfig:
    """Return the Fig4 Denmark country configuration."""
    for country in fig4.COUNTRIES:
        if country.key == DENMARK_KEY:
            return country
    raise ValueError("No Denmark country configuration found.")


def load_denmark_projection_data() -> pd.DataFrame:
    """Load the cached Denmark A/B projection data."""
    country = denmark_config()
    if country.data_path.exists():
        data = pd.read_csv(country.data_path)
        data = fig4.add_country_columns_if_missing(data, country)
        return filter_denmark_years(data)

    combined = fig4.build_projection_data()
    return filter_denmark_years(combined[combined["country_key"] == DENMARK_KEY].copy())


def filter_denmark_years(data: pd.DataFrame) -> pd.DataFrame:
    """Restrict Denmark period data to the available 1835-2020 interval."""
    rows = data[
        (data["year"] >= DENMARK_START_YEAR)
        & (data["year"] <= DENMARK_END_YEAR)
        & (data["from_t"] == fig4.FROM_T)
    ].copy()
    return rows.sort_values(["year", "condition"]).reset_index(drop=True)


def build_denmark_fig4c_data(period_data: pd.DataFrame) -> pd.DataFrame:
    """Save the Denmark fitted GGM Makeham term with 95% intervals."""
    rows = period_data[period_data["condition"] == "extrinsic_removed"].copy()
    rows = rows[np.isfinite(rows["ggm_m"])].copy()
    makeham_ci = load_or_build_denmark_makeham_ci(rows["year"].to_numpy(dtype=int))
    rows = rows.merge(makeham_ci, on="year", how="left", suffixes=("", "_ci_fit"))
    rows["ggm_m_for_plot"] = rows["ggm_m_ci_fit"].fillna(rows["ggm_m"])

    result = pd.DataFrame(
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
    result.to_csv(DENMARK_FIG4C_DATA_PATH, index=False)
    return result


def load_or_build_denmark_makeham_ci(years: np.ndarray) -> pd.DataFrame:
    """Load or calculate cached Denmark GGM Makeham-term fit intervals."""
    required = {
        "year",
        "ggm_m",
        "ggm_m_stderr",
        "ggm_m_ci_low",
        "ggm_m_ci_high",
        "ci_method",
    }
    if DENMARK_CI_DATA_PATH.exists():
        cached = pd.read_csv(DENMARK_CI_DATA_PATH)
        if required.issubset(cached.columns) and set(years).issubset(set(cached["year"])):
            return cached

    print(f"Building Denmark Makeham m CI cache: {DENMARK_CI_DATA_PATH}")
    hmd = HMD(country="dan", gender="both", data_type="period")
    rows = [fit_denmark_makeham_with_ci(hmd, int(year)) for year in sorted(years)]
    data = pd.DataFrame(rows)
    data = fig4.add_makeham_confidence_interval(data)
    data.to_csv(DENMARK_CI_DATA_PATH, index=False)
    return data


def fit_denmark_makeham_with_ci(hmd: HMD, year: int) -> dict[str, float | int | str]:
    """Fit one Denmark period year and return the Makeham term interval inputs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        params, stderr = hmd.fit_ggm(
            year=year,
            age_start=fig4.GGM_FIT_AGE_START,
            age_end=fig4.GGM_FIT_AGE_END,
            return_cov=True,
        )

    return {
        "year": int(year),
        "ggm_m": float(params["m"]),
        "ggm_m_stderr": float(stderr["m"]),
        "ggm_fit_age_start": fig4.GGM_FIT_AGE_START,
        "ggm_fit_age_end": fig4.GGM_FIT_AGE_END,
        "ci_method": "lognormal delta-method from GGM curve_fit covariance",
    }


def build_denmark_fig4d_data(period_data: pd.DataFrame) -> pd.DataFrame:
    """Fit Denmark extrinsic-removed points to the Xc robustness response curve."""
    fit_data = period_data.copy()
    fit_data["x_relative_to_sr"] = fit_data["x_relative_to_old_fig4_ref"]
    fit_data["y_relative_to_sr"] = fit_data["y_relative_to_old_fig4_ref"]
    rows = fig4.fit_period_points_to_model_curve(
        period_data=fit_data,
        source_condition="extrinsic_removed",
        focal_param="Xc",
        projection_name="robustness",
    )
    rows = fig4.smooth_robustness_artifact(rows)
    rows.to_csv(DENMARK_FIG4D_DATA_PATH, index=False)
    return rows


def build_denmark_fig4d_extrapolation_data(fig4d_data: pd.DataFrame) -> pd.DataFrame:
    """Extend the Denmark robustness trajectory to 2100 from the 1980-2020 trend."""
    observed = fig4d_data[fig4d_data["year"] <= DENMARK_END_YEAR].copy()
    observed["series"] = "data"
    observed["fit_start_year"] = fig4.EXTRAPOLATION_START_YEAR
    observed["fit_end_year"] = DENMARK_END_YEAR

    fit_rows = observed[
        (observed["year"] >= fig4.EXTRAPOLATION_START_YEAR)
        & (observed["year"] <= DENMARK_END_YEAR)
    ].copy()
    fit_years = fit_rows["year"].to_numpy(dtype=float)
    fit_values = fit_rows["estimate"].to_numpy(dtype=float)
    linear_slope, _ = np.polyfit(fit_years, fit_values, 1)
    exponential_log_slope, _ = np.polyfit(fit_years, np.log(fit_values), 1)

    future_years = np.arange(DENMARK_END_YEAR, fig4.EXTRAPOLATION_END_YEAR + 1)
    last_value = float(observed.loc[observed["year"] == DENMARK_END_YEAR, "estimate"].iloc[0])
    projected = pd.concat(
        [
            denmark_extrapolation_rows(
                future_years,
                series="linear_extrapolation",
                values=last_value + linear_slope * (future_years - DENMARK_END_YEAR),
                slope=linear_slope,
            ),
            denmark_extrapolation_rows(
                future_years,
                series="exponential_extrapolation",
                values=last_value * np.exp(exponential_log_slope * (future_years - DENMARK_END_YEAR)),
                slope=exponential_log_slope,
            ),
        ],
        ignore_index=True,
    )

    result = pd.concat([observed, projected], ignore_index=True, sort=False)
    result.to_csv(DENMARK_FIG4D_EXTRAP_DATA_PATH, index=False)
    return result


def denmark_extrapolation_rows(
    years: np.ndarray,
    *,
    series: str,
    values: np.ndarray,
    slope: float,
) -> pd.DataFrame:
    """Return one Denmark projected robustness trajectory."""
    return pd.DataFrame(
        {
            "country_key": DENMARK_KEY,
            "country_name": "Denmark",
            "year": years.astype(int),
            "projection_name": "robustness_extrapolation",
            "source_condition": "extrinsic_removed",
            "focal_param": "Xc",
            "estimate": values,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "series": series,
            "fit_start_year": fig4.EXTRAPOLATION_START_YEAR,
            "fit_end_year": DENMARK_END_YEAR,
            "fit_slope": slope,
        }
    )


def render_denmark_supplement(
    period_data: pd.DataFrame,
    fig4c_data: pd.DataFrame,
    fig4d_data: pd.DataFrame,
) -> None:
    """Render Denmark supplement panels A-D."""
    apply_supplement_style()
    fig = plt.figure(figsize=(12.2, 12.6))
    fig.patch.set_facecolor("white")
    grid = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=(1.16, 0.84),
        hspace=0.32,
        wspace=0.12,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    top_axes = np.array([ax_a, ax_b])

    colorbar_scatter = draw_top_row(top_axes, period_data)
    draw_panel_c(ax_c, fig4c_data)
    draw_panel_d(ax_d, fig4d_data)
    add_panel_label(ax_a, "a", x=-0.18, y=1.08)
    add_panel_label(ax_b, "b", x=-0.21, y=1.08)
    add_panel_label(ax_c, "c", x=-0.17, y=1.14)
    add_panel_label(ax_d, "d", x=-0.17, y=1.14)

    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.94)
    fig4.add_full_height_colorbar(fig, top_axes, colorbar_scatter)
    fig.savefig(DENMARK_SUPP_PNG, dpi=350, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(DENMARK_SUPP_PDF, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def apply_supplement_style() -> None:
    """Use the Fig4 style, tuned for a compact four-panel supplement."""
    fig4.apply_fig4_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
        }
    )


def draw_top_row(axes: np.ndarray, period_data: pd.DataFrame):
    """Draw Denmark panels A/B on the Fig1E response plane."""
    colorbar_scatter = None
    country = denmark_config()
    for ax, condition in zip(axes, fig4.CONDITION_LABELS):
        fig4.draw_new_baseline(ax)
        xlim, ylim = fig4.NEW_AXIS_LIMITS[condition]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Normalized median lifespan (2020 = 1)")
        ax.set_ylabel("")
        scatter = fig4.draw_historical_points(
            ax,
            period_data,
            condition,
            x_column="x_relative_to_old_fig4_ref",
            y_column="y_relative_to_old_fig4_ref",
            show_colorbar=False,
        )
        if condition == "extrinsic_removed":
            colorbar_scatter = scatter
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
        ax.set_title(fig4.CONDITION_LABELS[condition], loc="center", pad=12)
        fig4.remove_legend(ax)

    build_compact_top_legend(axes[0], country=country, data=period_data)
    axes[0].set_ylabel("Normalized steepness (2020 = 1)")
    axes[1].set_ylabel("")
    return colorbar_scatter


def build_compact_top_legend(
    ax: plt.Axes,
    *,
    country: fig4.CountryConfig,
    data: pd.DataFrame,
) -> None:
    """Add a compact A/B legend sized for the four-panel supplement."""
    first_year = int(data["year"].min())
    last_year = int(data["year"].max())
    handles = [
        fig4.Line2D([], [], color="none", label="Senogenic parameters"),
        fig4.Line2D([0], [0], color=fig4.fig1e.PARAM_COLORS["eta"], lw=3, label=fig4.fig1e.PARAM_LABELS["eta"]),
        fig4.Line2D([0], [0], color=fig4.fig1e.PARAM_COLORS["beta"], lw=3, label=fig4.fig1e.PARAM_LABELS["beta"]),
        fig4.Line2D([], [], color="none", label="Robustness parameters"),
        fig4.Line2D([0], [0], color=fig4.fig1e.PARAM_COLORS["Xc"], lw=3, label=fig4.fig1e.PARAM_LABELS["Xc"]),
        fig4.Line2D([0], [0], color=fig4.fig1e.PARAM_COLORS["epsilon"], lw=3, label=fig4.fig1e.PARAM_LABELS["epsilon"]),
        fig4.Line2D([0], [0], color=fig4.LEGEND_SEPARATOR_COLOR, lw=1.2, label=" "),
        fig4.Line2D([0], [0], color=fig4.fig1e.PARAM_COLORS["h_ext"], lw=3, label="Extrinsic mortality"),
        fig4.Line2D([], [], color="none", label=" "),
        fig4.Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#B5B5B5",
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            markersize=8,
            label=f"{country.name} period years ({first_year}-{last_year})",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=12,
        handlelength=1.8,
        labelspacing=0.30,
        borderpad=0.25,
    )
    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_fontweight("bold")


def draw_panel_c(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw Denmark fitted Makeham mortality over time."""
    plot_denmark_ci_series(ax, data, color=fig4.EXTRINSIC_PANEL_COLOR, show_legend=False)
    ax.set_yscale("log")
    ax.set_title("Extrinsic mortality over time", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Extrinsic mortality [year$^{-1}$]")
    ax.set_xlim(DENMARK_START_YEAR - 5, DENMARK_END_YEAR + 5)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax.yaxis.set_minor_formatter(NullFormatter())
    fig4.finish_time_series_axes(ax, panel_label="", preserve_xlim=True)


def draw_panel_d(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Draw Denmark observed robustness over time."""
    observed = data.sort_values("year")

    ax.axhline(1.0, color="#B8B8B8", linewidth=1.2, linestyle="--", zorder=0)
    plot_denmark_ci_series(ax, observed, color=fig4.ROBUSTNESS_PANEL_COLOR, label="Denmark", show_legend=False)
    ax.set_xlim(DENMARK_START_YEAR - 5, DENMARK_END_YEAR + 5)
    max_y = float(observed["ci_high"].max())
    min_y = min(float(observed["ci_low"].min()), float(observed["estimate"].min()))
    ax.set_ylim(max(0.45, min_y - 0.04), max_y + 0.08)
    ax.set_title("Denmark robustness over time", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Threshold $X_c$ factor")
    fig4.finish_time_series_axes(ax, panel_label="", preserve_xlim=True)


def plot_denmark_ci_series(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    color: str | None = None,
    label: str = "Denmark",
    show_legend: bool = True,
) -> None:
    """Plot Denmark estimates with a 95% confidence band."""
    rows = data.sort_values("year")
    country = denmark_config()
    plot_color = color or country.color
    if fig4.rows_have_visible_ci(rows):
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
        marker="^",
        markersize=3.5,
        markerfacecolor=plot_color,
        markeredgecolor=plot_color,
        label=label,
        zorder=3,
    )
    if show_legend:
        ax.legend(frameon=False, loc="best")


def add_panel_label(ax: plt.Axes, label: str, *, x: float, y: float) -> None:
    """Add one lowercase supplement panel label."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=28,
        fontweight="bold",
        va="top",
        ha="left",
    )


def update_output_index() -> None:
    """Record Denmark supplement artifacts in the shared output index."""
    fig4.OUTPUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
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
    if fig4.OUTPUT_INDEX.exists():
        with fig4.OUTPUT_INDEX.open(newline="", encoding="utf-8") as handle:
            existing_paths = {row.get("path", "") for row in csv.DictReader(handle)}

    source_script = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    rows = [
        fig4.output_index_row(
            artifact_type="csv",
            path=DENMARK_CI_DATA_PATH,
            source_script=source_script,
            input_paths=str(denmark_config().data_path.relative_to(PROJECT_ROOT)),
            description="Cached Denmark period GGM Makeham m fit covariance intervals.",
            notes="Used by the Denmark supplemental Fig4C panel.",
        ),
        fig4.output_index_row(
            artifact_type="csv",
            path=DENMARK_FIG4C_DATA_PATH,
            source_script=source_script,
            input_paths=str(denmark_config().data_path.relative_to(PROJECT_ROOT)),
            description="Denmark supplemental Fig4C fitted Makeham m estimates with 95% CIs.",
            notes="CIs use a lognormal delta-method interval from the GGM curve_fit covariance.",
        ),
        fig4.output_index_row(
            artifact_type="csv",
            path=DENMARK_FIG4D_DATA_PATH,
            source_script=source_script,
            input_paths=str(denmark_config().data_path.relative_to(PROJECT_ROOT)),
            description="Denmark supplemental Fig4D fitted robustness estimates and CIs.",
            notes="CIs are across Figure 1E scenario-specific Xc curves.",
        ),
        fig4.output_index_row(
            artifact_type="figure",
            path=DENMARK_SUPP_PNG,
            source_script=source_script,
            input_paths=str(denmark_config().data_path.relative_to(PROJECT_ROOT)),
            description="Denmark Fig4 A-D PNG.",
            notes="Panels A/B use Fig1E baseline from age 20; panel C uses fitted Makeham m; panel D shows Xc through 2020.",
        ),
        fig4.output_index_row(
            artifact_type="figure",
            path=DENMARK_SUPP_PDF,
            source_script=source_script,
            input_paths=str(denmark_config().data_path.relative_to(PROJECT_ROOT)),
            description="Denmark Fig4 A-D vector PDF.",
            notes="Panels A/B use Fig1E baseline from age 20; panel C uses fitted Makeham m; panel D shows Xc through 2020.",
        ),
    ]

    file_exists = fig4.OUTPUT_INDEX.exists()
    with fig4.OUTPUT_INDEX.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            if row["path"] not in existing_paths:
                writer.writerow(row)


def main() -> None:
    fig4.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig4.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    period_data = load_denmark_projection_data()
    fig4c_data = build_denmark_fig4c_data(period_data)
    fig4d_data = build_denmark_fig4d_data(period_data)
    render_denmark_supplement(period_data, fig4c_data, fig4d_data)
    update_output_index()

    print(f"Saved Denmark Fig4 PNG: {DENMARK_SUPP_PNG}")
    print(f"Saved Denmark Fig4 PDF: {DENMARK_SUPP_PDF}")
    print(f"Saved Denmark Makeham CI data: {DENMARK_CI_DATA_PATH}")
    print(f"Saved Denmark Fig4C data: {DENMARK_FIG4C_DATA_PATH}")
    print(f"Saved Denmark Fig4D data: {DENMARK_FIG4D_DATA_PATH}")


if __name__ == "__main__":
    main()
