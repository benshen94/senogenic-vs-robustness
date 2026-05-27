#!/usr/bin/env python3
"""Make the new Fig2b/Fig2c maximum-lifespan parameter panels."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Fig2_new"
PANEL_B_PNG_PATH = OUTPUT_DIR / "fig2b_new.png"
PANEL_B_PDF_PATH = OUTPUT_DIR / "fig2b_new.pdf"
PANEL_C_PNG_PATH = OUTPUT_DIR / "fig2c_new.png"
PANEL_C_PDF_PATH = OUTPUT_DIR / "fig2c_new.pdf"

CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "Fig2_new"
CACHE_PATH = CACHE_DIR / "fig2bc_new_max_lifespan.csv"
METADATA_PATH = CACHE_DIR / "fig2bc_new_metadata.json"
PANEL_B_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "fig2b_new_max_lifespan_heterogeneity.csv"
PANEL_B_ENVELOPE_PATH = SAVED_RESULTS_DIR / "csv" / "fig2b_new_fit_ci_envelopes.csv"
PANEL_C_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "fig2c_new_max_lifespan_factor.csv"
PANEL_C_ENVELOPE_PATH = SAVED_RESULTS_DIR / "csv" / "fig2c_new_fit_ci_envelopes.csv"

BASELINE_FIT_PATH = (
    SAVED_RESULTS_DIR
    / "fit_archive"
    / "records"
    / "joint2019_tail90_sweden_emphasis.json"
)
BASELINE_CI_PATH = (
    SAVED_RESULTS_DIR
    / "fit_archive"
    / "ci"
    / "joint2019_tail90_sweden_emphasis_ci.csv"
)

PARAMS_TO_VARY = ("eta", "beta", "Xc", "epsilon")
SENOGENIC_PARAMS = ("eta", "beta")
ROBUSTNESS_PARAMS = ("Xc", "epsilon")
PARAMETER_LABELS = {
    "eta": r"Production $\eta$",
    "beta": r"Removal $\beta$",
    "Xc": r"Threshold $X_c$",
    "epsilon": r"Noise $\epsilon$",
}
PARAMETER_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
}

HETERO_VALUES = tuple(i / 100 for i in range(0, 21))
FACTOR_VALUES = tuple(round(0.85 + i * 0.05, 2) for i in range(7))
MAX_SURVIVAL = 1e-4
HETERO_N_SIM = 1_000_000
HETERO_CI_N_SIM = 1_000_000
FACTOR_N_SIM = 1_000_000
TMAX = 420.0
DT = 0.1
SAVE_TIMES = 420.0
RANDOM_SEED = 20260519
HETERO_SEED_METHOD = "common_random_numbers_for_heterogeneity_ci_scenarios"
FACTOR_SEED_METHOD = "common_random_numbers_for_factor_ci_scenarios"
CI_BASELINE_PARAMS = ("eta", "beta", "epsilon", "Xc")
CI_ROW_BY_PARAM = {
    "eta": "eta",
    "beta": "beta",
    "epsilon": "epsilon",
    "Xc": "SWE_Xc",
}


def load_sweden_baseline() -> dict[str, float]:
    """Load the Sweden 2019 baseline used by Fig1D_new and Fig2a_new."""
    record = json.loads(BASELINE_FIT_PATH.read_text())
    fitted = record["summary"]["fitted_parameters"]
    return {
        "eta": float(fitted["eta"]),
        "beta": float(fitted["beta"]),
        "kappa": 0.5,
        "epsilon": float(fitted["epsilon"]),
        "Xc": float(fitted["SWE_Xc"]),
    }


def load_sweden_ci_bounds() -> dict[str, tuple[float, float]]:
    """Load fit-local 95% CI bounds for the Sweden baseline parameters."""
    ci = pd.read_csv(BASELINE_CI_PATH).set_index("parameter")
    bounds = {}
    for param_name in CI_BASELINE_PARAMS:
        row_name = CI_ROW_BY_PARAM[param_name]
        row = ci.loc[row_name]
        bounds[param_name] = (float(row["ci95_lower"]), float(row["ci95_upper"]))
    return bounds


def build_focal_ci_baseline_scenarios(
    baseline: dict[str, float],
    focal_param: str,
) -> list[tuple[str, dict[str, float]]]:
    """Build central plus lower/upper fit-CI endpoint baselines for one curve."""
    lower, upper = load_sweden_ci_bounds()[focal_param]

    low = dict(baseline)
    low[focal_param] = lower

    high = dict(baseline)
    high[focal_param] = upper

    return [
        ("central", dict(baseline)),
        (f"{focal_param}_ci_lower", low),
        (f"{focal_param}_ci_upper", high),
    ]


def stable_seed(*parts: object) -> int:
    """Create a deterministic uint32 seed from readable metadata."""
    import hashlib

    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def sample_positive_gaussian(mean: float, cv: float, n: int, seed: int) -> np.ndarray:
    """Sample a positive Gaussian parameter distribution."""
    if cv == 0:
        return np.full(n, mean, dtype=float)

    rng = np.random.default_rng(seed)
    values = rng.normal(loc=mean, scale=mean * cv, size=n)

    while True:
        bad = values <= 0
        if not np.any(bad):
            return values
        values[bad] = rng.normal(loc=mean, scale=mean * cv, size=int(bad.sum()))


def build_params(
    panel: str,
    focal_param: str,
    value: float,
    baseline: dict[str, float],
    n: int,
    scenario_id: str,
) -> dict[str, np.ndarray]:
    """Build SR parameters for one panel point."""
    params = {key: np.full(n, baseline[key], dtype=float) for key in baseline}

    if panel == "heterogeneity":
        params[focal_param] = sample_positive_gaussian(
            mean=baseline[focal_param],
            cv=value,
            n=n,
            seed=heterogeneity_distribution_seed(focal_param, value, n),
        )
        return params

    if panel == "factor":
        params[focal_param] = np.full(n, baseline[focal_param] * value, dtype=float)
        return params

    raise ValueError(f"Unknown panel: {panel}")


def heterogeneity_distribution_seed(focal_param: str, value: float, n: int) -> int:
    """Share heterogeneity draws across central and CI endpoint baselines."""
    if np.isclose(value, 0.0):
        return stable_seed(RANDOM_SEED, "heterogeneity", "shared_zero_cv", value, n)

    return stable_seed(
        RANDOM_SEED,
        "heterogeneity",
        focal_param,
        value,
        "common_parameter_draws",
        n,
    )


def estimate_max_lifespan(
    panel: str,
    focal_param: str,
    value: float,
    baseline: dict[str, float],
    n: int,
    scenario_id: str,
) -> float:
    """Run one SR simulation and estimate the age where survival reaches 1e-4."""
    params = build_params(
        panel=panel,
        focal_param=focal_param,
        value=value,
        baseline=baseline,
        n=n,
        scenario_id=scenario_id,
    )
    sim = utils.create_sr_simulation(
        params_dict=params,
        n=n,
        h_ext=0.0,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        parallel=True,
        break_early=True,
        random_seed=simulation_seed(panel, focal_param, value, scenario_id, n),
    )
    return max_lifespan_from_death_times(sim.death_times, n)


def simulation_seed(panel: str, focal_param: str, value: float, scenario_id: str, n: int) -> int:
    """Return deterministic seeds, sharing stochastic streams across CI endpoints."""
    if panel == "heterogeneity" and np.isclose(value, 0.0):
        return stable_seed(RANDOM_SEED, panel, "shared_zero_cv", value, n)

    if panel == "heterogeneity":
        return stable_seed(RANDOM_SEED, panel, focal_param, value, "common_random_numbers", n)

    if panel == "factor" and np.isclose(value, 1.0):
        return stable_seed(RANDOM_SEED, panel, "shared_factor_one", value, n)

    if panel == "factor":
        return stable_seed(RANDOM_SEED, panel, focal_param, value, "common_random_numbers", n)

    return stable_seed(RANDOM_SEED, panel, focal_param, value, scenario_id, n)


def max_lifespan_from_death_times(death_times: np.ndarray, n: int) -> float:
    """Estimate the unconditional age where survival reaches MAX_SURVIVAL."""
    finite_deaths = np.sort(death_times[np.isfinite(death_times)])
    target_dead_count = (1.0 - MAX_SURVIVAL) * n
    if target_dead_count > len(finite_deaths):
        return np.nan

    quantile_level = target_dead_count / len(finite_deaths)
    return float(np.quantile(finite_deaths, quantile_level))


def build_metadata() -> dict[str, object]:
    """Return metadata that must match before reusing cached results."""
    return {
        "baseline_fit_path": str(BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)),
        "baseline_ci_path": str(BASELINE_CI_PATH.relative_to(PROJECT_ROOT)),
        "baseline": load_sweden_baseline(),
        "ci_method": "focal_parameter_95_ci_endpoint_envelope",
        "max_lifespan_definition": "age where unconditional SR survival reaches 1e-4",
        "params_to_vary": list(PARAMS_TO_VARY),
        "heterogeneity_values": list(HETERO_VALUES),
        "factor_values": list(FACTOR_VALUES),
        "hetero_n_sim": HETERO_N_SIM,
        "hetero_ci_n_sim": HETERO_CI_N_SIM,
        "factor_n_sim": FACTOR_N_SIM,
        "h_ext": 0.0,
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "random_seed": RANDOM_SEED,
        "hetero_seed_method": HETERO_SEED_METHOD,
        "factor_seed_method": FACTOR_SEED_METHOD,
    }


def load_cached_rows(metadata: dict[str, object]) -> pd.DataFrame:
    """Load reusable rows if the metadata matches this script version."""
    columns = ["panel", "param", "value", "scenario_id", "n_sim", "seed_method", "max_lifespan"]
    if not CACHE_PATH.exists() or not METADATA_PATH.exists():
        return pd.DataFrame(columns=columns)

    cached_metadata = json.loads(METADATA_PATH.read_text())
    if not cache_metadata_is_compatible(cached_metadata, metadata):
        return pd.DataFrame(columns=columns)

    rows = pd.read_csv(CACHE_PATH)
    return normalise_cache_columns(rows)


def normalise_cache_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Fill cache columns added after earlier Fig2c iterations."""
    if "seed_method" not in rows.columns:
        rows = rows.copy()
        rows["seed_method"] = rows["panel"].map(
            {
                "heterogeneity": HETERO_SEED_METHOD,
                "factor": FACTOR_SEED_METHOD,
            }
        )
    return rows


def cache_metadata_is_compatible(
    cached_metadata: dict[str, object],
    current_metadata: dict[str, object],
) -> bool:
    """Allow reuse of rows when inputs match, even if requested grids or N changed."""
    comparable_keys = [
        "baseline_fit_path",
        "baseline_ci_path",
        "baseline",
        "ci_method",
        "max_lifespan_definition",
        "params_to_vary",
        "h_ext",
        "tmax",
        "dt",
        "save_times",
        "random_seed",
    ]
    for key in comparable_keys:
        if cached_metadata.get(key) != current_metadata.get(key):
            return False

    return True


def save_cache(rows: pd.DataFrame, metadata: dict[str, object]) -> None:
    """Persist the simulation cache after each completed row."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(CACHE_PATH, index=False)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")


def required_jobs() -> list[dict[str, object]]:
    """List all central and CI endpoint simulation jobs."""
    baseline = load_sweden_baseline()
    jobs = []

    for panel, values in [("heterogeneity", HETERO_VALUES), ("factor", FACTOR_VALUES)]:
        for param_name in PARAMS_TO_VARY:
            scenarios = build_focal_ci_baseline_scenarios(baseline, param_name)
            for value in values:
                for scenario_id, scenario_baseline in scenarios:
                    jobs.append(
                        {
                            "panel": panel,
                            "param": param_name,
                            "value": float(value),
                            "scenario_id": scenario_id,
                            "baseline": scenario_baseline,
                            "n_sim": n_sim_for_job(panel, scenario_id),
                            "seed_method": seed_method_for_job(panel),
                        }
                    )

    return jobs


def n_sim_for_job(panel: str, scenario_id: str) -> int:
    """Return the simulation count for one requested row."""
    if panel == "factor":
        return FACTOR_N_SIM

    if scenario_id == "central":
        return HETERO_N_SIM

    return HETERO_CI_N_SIM


def seed_method_for_job(panel: str) -> str:
    """Return the seed method expected for a cached job."""
    if panel == "factor":
        return FACTOR_SEED_METHOD

    return HETERO_SEED_METHOD


def run_missing_simulations() -> pd.DataFrame:
    """Run all missing simulation rows and return the complete cache."""
    metadata = build_metadata()
    rows = load_cached_rows(metadata)
    required = required_jobs()
    required_keys = {
        (
            job["panel"],
            job["param"],
            float(job["value"]),
            job["scenario_id"],
            int(job["n_sim"]),
            job["seed_method"],
        )
        for job in required
    }
    rows = keep_current_required_rows(rows, required_keys)
    completed = {
        (row.panel, row.param, float(row.value), row.scenario_id, int(row.n_sim), row.seed_method)
        for row in rows.itertuples(index=False)
    }

    for job in required:
        key = (
            job["panel"],
            job["param"],
            float(job["value"]),
            job["scenario_id"],
            int(job["n_sim"]),
            job["seed_method"],
        )
        if key in completed:
            continue

        print(
            "Running Fig2bc_new:",
            job["panel"],
            job["param"],
            job["value"],
            job["scenario_id"],
        )
        max_lifespan = estimate_max_lifespan(
            panel=str(job["panel"]),
            focal_param=str(job["param"]),
            value=float(job["value"]),
            baseline=job["baseline"],
            n=int(job["n_sim"]),
            scenario_id=str(job["scenario_id"]),
        )
        new_row = pd.DataFrame(
            [
                {
                    "panel": job["panel"],
                    "param": job["param"],
                    "value": float(job["value"]),
                    "scenario_id": job["scenario_id"],
                    "n_sim": int(job["n_sim"]),
                    "seed_method": job["seed_method"],
                    "max_lifespan": max_lifespan,
                }
            ]
        )
        if rows.empty:
            rows = new_row
        else:
            rows = pd.concat([rows, new_row], ignore_index=True)
        completed.add(key)
        save_cache(rows, metadata)

    return rows


def keep_current_required_rows(rows: pd.DataFrame, required_keys: set[tuple[object, ...]]) -> pd.DataFrame:
    """Drop stale cache rows whose N no longer matches the requested grid."""
    if rows.empty:
        return rows

    rows = rows.copy()
    rows["n_sim"] = rows["n_sim"].astype(int)
    keep = [
        (row.panel, row.param, float(row.value), row.scenario_id, int(row.n_sim), row.seed_method) in required_keys
        for row in rows.itertuples(index=False)
    ]
    return rows.loc[keep].copy()


def split_panel_data(rows: pd.DataFrame, panel: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return central lines and fit-CI envelopes for one panel."""
    subset = rows[rows["panel"] == panel].copy()
    central = subset[subset["scenario_id"] == "central"].copy()
    envelopes = (
        subset.groupby(["panel", "param", "value"], as_index=False)["max_lifespan"]
        .agg(ci_lower="min", ci_upper="max")
        .copy()
    )
    return central, envelopes


def configure_matplotlib() -> None:
    """Apply publication-style figure defaults."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 11.5,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.35,
            "ytick.major.width": 1.35,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def draw_ci_envelopes(ax: plt.Axes, envelopes: pd.DataFrame, x_column: str) -> None:
    """Draw one shaded fit-CI envelope per parameter."""
    for param_name in PARAMS_TO_VARY:
        subset = envelopes[envelopes["param"] == param_name].sort_values("value")
        if subset.empty:
            continue

        ax.fill_between(
            subset[x_column],
            subset["ci_lower"],
            subset["ci_upper"],
            color=PARAMETER_COLORS[param_name],
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )


def draw_parameter_lines(ax: plt.Axes, central: pd.DataFrame, x_column: str) -> None:
    """Draw central model curves."""
    for param_name in PARAMS_TO_VARY:
        subset = central[central["param"] == param_name].sort_values("value")
        if subset.empty:
            continue

        ax.plot(
            subset[x_column],
            subset["max_lifespan"],
            color=PARAMETER_COLORS[param_name],
            lw=3.0,
            linestyle="-",
            marker="o",
            markersize=4.2,
            label=PARAMETER_LABELS[param_name],
            zorder=3,
        )


def label_curve_endpoints(ax: plt.Axes, central: pd.DataFrame, x_column: str) -> None:
    """Add direct labels at the right endpoints of the mean-shift curves."""
    for param_name in PARAMS_TO_VARY:
        subset = central[central["param"] == param_name].sort_values("value")
        if subset.empty:
            continue

        last = subset.iloc[-1]
        ax.text(
            float(last[x_column]) + 0.008,
            float(last["max_lifespan"]),
            PARAMETER_LABELS[param_name],
            color=PARAMETER_COLORS[param_name],
            fontsize=12.0,
            va="center",
            ha="left",
        )


def build_grouped_legend(ax: plt.Axes, title: str, loc: str = "upper left") -> None:
    """Build the grouped legend following the Fig2a_new convention."""
    handles = [
        Line2D([], [], color="none", label="Senogenic parameters"),
        Line2D([0], [0], color=PARAMETER_COLORS["eta"], lw=3.0, label=PARAMETER_LABELS["eta"]),
        Line2D([0], [0], color=PARAMETER_COLORS["beta"], lw=3.0, label=PARAMETER_LABELS["beta"]),
        Line2D([], [], color="none", label="Robustness parameters"),
        Line2D([0], [0], color=PARAMETER_COLORS["Xc"], lw=3.0, label=PARAMETER_LABELS["Xc"]),
        Line2D([0], [0], color=PARAMETER_COLORS["epsilon"], lw=3.0, label=PARAMETER_LABELS["epsilon"]),
    ]
    legend = ax.legend(
        handles=handles,
        loc=loc,
        frameon=False,
        title=title,
        title_fontsize=12.5,
        handlelength=2.0,
        borderpad=0.55,
        labelspacing=0.32,
    )
    for index, text in enumerate(legend.get_texts()):
        if index in (0, 3):
            text.set_color("#222222")
            text.set_fontweight("bold")


def style_axis(ax: plt.Axes) -> None:
    """Apply common axis styling."""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Top 0.01% survivors [years]")
    ax.tick_params(axis="both", which="major", pad=4)


def plot_panel_b(central: pd.DataFrame, envelopes: pd.DataFrame) -> None:
    """Draw and save Fig2b_new."""
    data = central.copy()
    env = envelopes.copy()
    data["heterogeneity_percent"] = 100 * data["value"]
    env["heterogeneity_percent"] = 100 * env["value"]

    data.to_csv(PANEL_B_DATA_PATH, index=False)
    env.to_csv(PANEL_B_ENVELOPE_PATH, index=False)

    fig, ax = plt.subplots(figsize=(6.7, 4.9))
    draw_ci_envelopes(ax, env, "heterogeneity_percent")
    draw_parameter_lines(ax, data, "heterogeneity_percent")
    ax.set_xlabel("Parameter heterogeneity (CV, %)")
    ax.set_title(
        "Upper lifespan tail is sensitive to\nheterogeneity in senogenic parameters",
        pad=8,
    )
    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(105, 150)
    style_axis(ax)
    build_grouped_legend(ax, "Heterogeneity in")
    fig.tight_layout(pad=0.8)
    fig.savefig(PANEL_B_PNG_PATH, dpi=300)
    fig.savefig(PANEL_B_PDF_PATH)
    plt.close(fig)


def plot_panel_c(central: pd.DataFrame, envelopes: pd.DataFrame) -> None:
    """Draw and save Fig2c_new."""
    central.to_csv(PANEL_C_DATA_PATH, index=False)
    envelopes.to_csv(PANEL_C_ENVELOPE_PATH, index=False)

    fig, ax = plt.subplots(figsize=(6.7, 4.9))
    draw_ci_envelopes(ax, envelopes, "value")
    draw_parameter_lines(ax, central, "value")
    label_curve_endpoints(ax, central, "value")
    ax.axvline(1.0, color="#777777", lw=1.5, linestyle=(0, (2.4, 1.8)), zorder=0)
    ax.set_xlabel("Parameter factor (relative to baseline)")
    ax.set_title(
        "Upper lifespan tail is sensitive to\nchanges in senogenic parameters",
        pad=8,
    )
    ax.set_xlim(0.84, 1.22)
    ax.set_xticks([0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
    ax.set_ylim(90, 130)
    style_axis(ax)
    fig.tight_layout(pad=0.8)
    fig.savefig(PANEL_C_PNG_PATH, dpi=300)
    fig.savefig(PANEL_C_PDF_PATH)
    plt.close(fig)


def update_output_index() -> None:
    """Upsert output-index rows for the new Fig2b/Fig2c artifacts."""
    index_path = SAVED_RESULTS_DIR / "index" / "outputs.csv"
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
    existing_rows = []
    if index_path.exists():
        with index_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = list(reader)

    figure_rows = []
    for panel_name, png_path, pdf_path, data_path, envelope_path in [
        ("fig2b_new_max_lifespan_heterogeneity", PANEL_B_PNG_PATH, PANEL_B_PDF_PATH, PANEL_B_DATA_PATH, PANEL_B_ENVELOPE_PATH),
        ("fig2c_new_max_lifespan_factor", PANEL_C_PNG_PATH, PANEL_C_PDF_PATH, PANEL_C_DATA_PATH, PANEL_C_ENVELOPE_PATH),
    ]:
        for artifact_type, path, description, notes in [
            ("figure", png_path, f"PNG preview of {panel_name}", "Sweden 2019 baseline; shaded bands are one-at-a-time fit-CI endpoint envelopes"),
            ("figure", pdf_path, f"Vector PDF of {panel_name}", "Matplotlib PDF with editable text where supported"),
            ("csv", data_path, f"Central source data for {panel_name}", "Central Sweden 2019 baseline rows"),
            ("csv", envelope_path, f"Fit-CI envelope source data for {panel_name}", "Envelope is min/max across central and lower/upper 95% CI endpoint perturbations"),
        ]:
            figure_rows.append(
                {
                    "date": "2026-05-19",
                    "task": panel_name,
                    "artifact_type": artifact_type,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "source_script": "src/figures/Fig2_new/make_fig2bc_new.py",
                    "input_paths": "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json; saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
                    "description": description,
                    "notes": notes,
                }
            )

    paths_to_replace = {row["path"] for row in figure_rows}
    kept_rows = [row for row in existing_rows if row.get("path") not in paths_to_replace]
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        writer.writerows(figure_rows)


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_B_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = run_missing_simulations()
    panel_b_data, panel_b_envelopes = split_panel_data(rows, "heterogeneity")
    panel_c_data, panel_c_envelopes = split_panel_data(rows, "factor")

    plot_panel_b(panel_b_data, panel_b_envelopes)
    plot_panel_c(panel_c_data, panel_c_envelopes)
    update_output_index()

    print(f"Saved {PANEL_B_PNG_PATH}")
    print(f"Saved {PANEL_B_PDF_PATH}")
    print(f"Saved {PANEL_C_PNG_PATH}")
    print(f"Saved {PANEL_C_PDF_PATH}")


if __name__ == "__main__":
    main()
