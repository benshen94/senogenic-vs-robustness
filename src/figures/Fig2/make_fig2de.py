from pathlib import Path
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from lifelines import NelsonAalenFitter
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import FIGURES_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_DIR
PDF_PATH = FIGURES_DIR / "fig2de.pdf"
PNG_PATH = FIGURES_DIR / "fig2de.png"
CACHE_PATH = SAVED_RESULTS_DIR / "cache" / "plots" / "fig2de_plot_cache.pkl"
CACHE_VERSION = 2

N_INDIVIDUALS = int(1e6)
ROBUSTNESS_PARAMETERS = ["Xc", "epsilon"]
SENOGENIC_PARAMETERS = ["eta", "beta"]
PARAMETERS = ROBUSTNESS_PARAMETERS + SENOGENIC_PARAMETERS
PARAMETER_STDS = {
    "Xc": 0.20,
    "epsilon": 0.30,
    "eta": 0.15,
    "beta": 0.10,
}
BOTTOM_DEATH_PERCENTILE = 10
TOP_SURVIVAL_FRACTION = 0.01

RANDOM_SEED = 1729
PLOT_AGE_MIN = 50
PLOT_AGE_MAX = 110

FONT_FAMILY = "Arial"
LABEL_FONT_SIZE = 26
TITLE_FONT_SIZE = 22
LEGEND_FONT_SIZE = 14
PANEL_LABEL_FONT_SIZE = 40
TICK_FONT_SIZE = 26

SUBGRID_LABEL_FONT_SIZE = 16
SUBGRID_TITLE_FONT_SIZE = 18
SUBGRID_LEGEND_FONT_SIZE = 13
SUBGRID_TICK_FONT_SIZE = 16
ROW_SUBTITLE_FONT_SIZE = 17

PARAMETER_TITLES = {
    "Xc": r"Threshold ($X_c$)",
    "epsilon": r"Noise ($\epsilon$)",
    "eta": r"Production ($\eta$)",
    "beta": r"Removal ($\beta$)",
}

PARAMETER_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
}

BASELINE_COLOR = "#2B2B2B"
BAD_SURVIVOR_LINESTYLE = (0, (2, 1.6))
GOOD_SURVIVOR_LINESTYLE = "-"


def configure_matplotlib():
    mpl.rcParams.update(
        {
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": False,
            "axes.linewidth": 1.2,
        }
    )


def gompertz_fit(t_array, alpha, h0):
    return h0 * np.exp(alpha * t_array)


def build_base_params():
    base_params = utils.load_baseline_human_params_dict()
    base_params["Xc"] = 1.08 * base_params["Xc"]
    base_params["eta"] = 1.26 * base_params["eta"]
    base_params["beta"] = 1.17 * base_params["beta"]
    return base_params


def create_dz_simulation(param_name, base_params, seed):
    np.random.seed(seed)
    param_dict = utils.create_param_distribution_dict(
        params=param_name,
        std=PARAMETER_STDS[param_name],
        n=N_INDIVIDUALS,
        dist_type="gaussian",
        params_dict=base_params,
        family="DZ",
    )
    return utils.create_sr_simulation(
        params_dict=param_dict,
        n=N_INDIVIDUALS,
        parallel=True,
        tmax=300,
        break_early=True,
        random_seed=seed,
    )


def create_dz_simulations():
    base_params = build_base_params()
    simulations = {}
    for offset, param_name in enumerate(PARAMETERS):
        print(f"Running DZ simulation for {param_name}...")
        simulations[param_name] = create_dz_simulation(
            param_name=param_name,
            base_params=base_params.copy(),
            seed=RANDOM_SEED + offset,
        )
    return simulations


def get_cache_metadata():
    return {
        "cache_version": CACHE_VERSION,
        "n_individuals": N_INDIVIDUALS,
        "parameters": PARAMETERS,
        "parameter_stds": PARAMETER_STDS,
        "bottom_death_percentile": BOTTOM_DEATH_PERCENTILE,
        "top_survival_fraction": TOP_SURVIVAL_FRACTION,
        "random_seed": RANDOM_SEED,
    }


def cache_metadata_matches(metadata):
    return metadata == get_cache_metadata()


def load_cached_plot_records():
    if not CACHE_PATH.exists():
        return None

    with CACHE_PATH.open("rb") as file:
        payload = pickle.load(file)

    if not cache_metadata_matches(payload.get("metadata")):
        return None

    print(f"Loaded cached fig2de plot data from {CACHE_PATH}")
    return payload["plot_records"]


def save_cached_plot_records(plot_records):
    payload = {
        "metadata": get_cache_metadata(),
        "plot_records": plot_records,
    }
    with CACHE_PATH.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cached fig2de plot data to {CACHE_PATH}")


def get_bottom_death_percentile_age(sim):
    finite_death_times = sim.death_times[np.isfinite(sim.death_times)]
    if finite_death_times.size == 0:
        raise ValueError("Simulation has no finite death times.")
    return float(np.percentile(finite_death_times, BOTTOM_DEATH_PERCENTILE))


def get_top_survival_threshold_age(sim):
    age = sim.find_time_at_survival(TOP_SURVIVAL_FRACTION)
    if age is None:
        raise ValueError("Simulation never reached 1% survivorship.")
    return float(age)


def get_sibling_death_times_for_probands(sim, proband_indices):
    if proband_indices.size == 0:
        return np.array([])

    sibling_indices = proband_indices + 1 - 2 * (proband_indices % 2)
    return sim.death_times[sibling_indices.astype(int)]


def calc_sibling_hazard_from_probands(sim, proband_indices):
    death_times = get_sibling_death_times_for_probands(sim, proband_indices)
    if death_times.size == 0:
        return np.full_like(sim.tspan_hazard, np.nan, dtype=float)

    censor_time = sim.params.tmax + sim.params.dt
    event_observed = death_times < censor_time

    fitter = NelsonAalenFitter()
    fitter.fit(death_times, event_observed=event_observed, timeline=sim.tspan_hazard)
    return fitter.smoothed_hazard_(bandwidth=3)


def calc_sibling_hazard_for_bottom_percentile(sim, percentile_age):
    proband_indices = np.where(sim.death_times <= percentile_age)[0]
    return calc_sibling_hazard_from_probands(sim, proband_indices)


def calc_sibling_hazard_for_top_survivors(sim, threshold_age):
    proband_indices = np.where(sim.death_times >= threshold_age)[0]
    return calc_sibling_hazard_from_probands(sim, proband_indices)


def as_1d_array(values):
    if hasattr(values, "iloc"):
        return values.iloc[:, 0].to_numpy(dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def create_plot_record(sim):
    bottom_age = get_bottom_death_percentile_age(sim)
    top_age = get_top_survival_threshold_age(sim)
    bottom_sibling_hazard = calc_sibling_hazard_for_bottom_percentile(sim, bottom_age)
    top_sibling_hazard = calc_sibling_hazard_for_top_survivors(sim, top_age)

    return {
        "tspan_hazard": np.asarray(sim.tspan_hazard, dtype=float),
        "full_hazard": as_1d_array(sim.hazard),
        "bottom_sibling_hazard": as_1d_array(bottom_sibling_hazard),
        "top_sibling_hazard": as_1d_array(top_sibling_hazard),
        "bottom_age": bottom_age,
        "top_age": top_age,
    }


def create_plot_records():
    simulations = create_dz_simulations()
    return {
        param_name: create_plot_record(simulations[param_name])
        for param_name in PARAMETERS
    }


def get_plot_records():
    cached_records = load_cached_plot_records()
    if cached_records is not None:
        return cached_records

    plot_records = create_plot_records()
    save_cached_plot_records(plot_records)
    return plot_records


def plot_positive_mortality(ax, ages, mortality, **kwargs):
    mortality = as_1d_array(mortality)
    ages = np.asarray(ages, dtype=float)
    keep = np.isfinite(mortality) & (mortality > 0)
    keep = keep & (ages >= PLOT_AGE_MIN) & (ages <= PLOT_AGE_MAX)
    if not np.any(keep):
        return
    ax.plot(ages[keep], mortality[keep], **kwargs)


def plot_sibling_mortality(ax, plot_record, param_name):
    color = PARAMETER_COLORS[param_name]
    plot_positive_mortality(
        ax,
        plot_record["tspan_hazard"],
        plot_record["full_hazard"],
        color=BASELINE_COLOR,
        linewidth=2.8,
    )
    plot_positive_mortality(
        ax,
        plot_record["tspan_hazard"],
        plot_record["top_sibling_hazard"],
        color=color,
        linewidth=3.2,
        linestyle=GOOD_SURVIVOR_LINESTYLE,
    )
    plot_positive_mortality(
        ax,
        plot_record["tspan_hazard"],
        plot_record["bottom_sibling_hazard"],
        color=color,
        linewidth=3.2,
        linestyle=BAD_SURVIVOR_LINESTYLE,
    )

    ax.set_yscale("log")
    ax.set_xlim(PLOT_AGE_MIN, PLOT_AGE_MAX)
    ax.set_ylim(5e-3, 1)
    ax.set_xlabel("Age [years]", fontsize=SUBGRID_LABEL_FONT_SIZE)
    ax.set_ylabel(r"Mortality rate [year$^{-1}$]", fontsize=SUBGRID_LABEL_FONT_SIZE)
    ax.set_title(
        PARAMETER_TITLES[param_name],
        fontsize=SUBGRID_TITLE_FONT_SIZE,
        pad=10,
    )
    ax.tick_params(labelsize=SUBGRID_TICK_FONT_SIZE)
    ax.tick_params(width=1.2, length=4.5)


def add_sibling_legend(fig, right_axes):
    handles = [
        Line2D(
            [0],
            [0],
            color=BASELINE_COLOR,
            linewidth=2.8,
            label="Full cohort",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=3.2,
            linestyle=GOOD_SURVIVOR_LINESTYLE,
            label="Good survivor siblings (top 1%)",
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=3.2,
            linestyle=BAD_SURVIVOR_LINESTYLE,
            label="Bad survivor siblings (bottom 10%)",
        ),
    ]
    first_position = right_axes[0].get_position()
    second_position = right_axes[1].get_position()
    x_center = (first_position.x0 + second_position.x1) / 2

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(x_center, 0.99),
        ncol=3,
        frameon=False,
        fontsize=SUBGRID_LEGEND_FONT_SIZE,
        handlelength=2.6,
        columnspacing=1.6,
    )


def add_row_subtitle(fig, axes, label):
    left_position = axes[0].get_position()
    right_position = axes[1].get_position()
    x_center = (left_position.x0 + right_position.x1) / 2
    y_top = max(left_position.y1, right_position.y1)

    fig.text(
        x_center,
        y_top + 0.035,
        label,
        ha="center",
        va="bottom",
        fontsize=ROW_SUBTITLE_FONT_SIZE,
        fontweight="bold",
        color="#333333",
    )


def simplify_sibling_axis_labels(right_axes):
    for index, ax in enumerate(right_axes):
        is_bottom_row = index >= 2
        is_left_column = index % 2 == 0

        if not is_bottom_row:
            ax.set_xlabel("")

        if not is_left_column:
            ax.set_ylabel("")


def add_convergence_annotation(ax, text, xy=(110, 0.5), xytext=(100, 1e-1), fontsize=16):
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        fontfamily=FONT_FAMILY,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=0.3",
            color="black",
            lw=1.5,
        ),
    )


def plot_gavrilov_panel(ax):
    gavrilov_data = {
        "male": {
            "centenarians": {"alpha": 0.09, "h0": 4.98e-5},
            "short_lived": {"alpha": 0.079, "h0": 2.033e-4},
        },
        "female": {
            "centenarians": {"alpha": 0.101, "h0": 1.32e-5},
            "short_lived": {"alpha": 0.085, "h0": 8.18e-5},
        },
    }
    colors = {"male": "blue", "female": "red"}
    line_styles = {"centenarians": "-", "short_lived": "--"}
    ages = np.arange(50, 110, 0.1)

    for gender, groups in gavrilov_data.items():
        for group, params in groups.items():
            label = "brothers of centenarians"
            if group == "short_lived":
                label = "brothers of short-lived persons"
            if gender == "female":
                label = label.replace("brothers", "sisters")

            ax.plot(
                ages,
                gompertz_fit(ages, params["alpha"], params["h0"]),
                color=colors[gender],
                linestyle=line_styles[group],
                linewidth=5,
                label=label.capitalize(),
            )

    add_convergence_annotation(
        ax,
        "mortality\nconvergence",
        fontsize=20,
    )

    ax.set_yscale("log")
    ax.set_xlim(50, 110)
    ax.set_ylim(5e-3, 1)
    ax.set_xlabel("Age [years]", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(r"Mortality rate [year$^{-1}$]", fontsize=LABEL_FONT_SIZE)
    ax.set_title(
        "Mortality convergence for siblings of centenarians\nand short-lived persons",
        fontsize=TITLE_FONT_SIZE,
        pad=10,
    )
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc="lower right", frameon=False)
    ax.tick_params(labelsize=TICK_FONT_SIZE)


def build_figure(plot_records):
    fig = plt.figure(figsize=(17, 8.8))
    grid = fig.add_gridspec(1, 2, wspace=0.3)

    ax_left = fig.add_subplot(grid[0, 0])
    right_grid = grid[0, 1].subgridspec(2, 2, wspace=0.45, hspace=0.85)
    right_axes = [
        fig.add_subplot(right_grid[0, 0]),
        fig.add_subplot(right_grid[0, 1]),
        fig.add_subplot(right_grid[1, 0]),
        fig.add_subplot(right_grid[1, 1]),
    ]

    plot_gavrilov_panel(ax_left)
    for ax, param_name in zip(right_axes, PARAMETERS):
        plot_sibling_mortality(ax, plot_records[param_name], param_name)

    simplify_sibling_axis_labels(right_axes)
    add_sibling_legend(fig, right_axes)
    add_row_subtitle(fig, right_axes[:2], "Robustness parameters")
    add_row_subtitle(fig, right_axes[2:], "Senogenic parameters")

    ax_left.text(
        -0.1,
        1.25,
        "d",
        transform=ax_left.transAxes,
        fontsize=PANEL_LABEL_FONT_SIZE,
        va="top",
        ha="right",
    )
    right_axes[0].text(
        -0.28,
        1.25,
        "e",
        transform=right_axes[0].transAxes,
        fontsize=PANEL_LABEL_FONT_SIZE,
        va="top",
        ha="right",
    )
    return fig


def main():
    configure_matplotlib()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plot_records = get_plot_records()
    fig = build_figure(plot_records)
    fig.savefig(PDF_PATH, bbox_inches="tight", dpi=300)
    fig.savefig(PNG_PATH, bbox_inches="tight", dpi=300)
    print(f"Saved {PDF_PATH}")
    print(f"Saved {PNG_PATH}")
    return fig, plot_records


if __name__ == "__main__":
    main()
