#!/usr/bin/env python3
"""Make the new Fig2a survival-tail heterogeneity panel."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGING_PYTHON_ROOT = PROJECT_ROOT.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(AGING_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(AGING_PYTHON_ROOT))

from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.utils import sr_utils as utils
from src.shared.thresholds.paths import FIGURES_NEW_DIR, SAVED_RESULTS_DIR


OUTPUT_DIR = FIGURES_NEW_DIR / "Fig2_new"
PNG_PATH = OUTPUT_DIR / "fig2a_new.png"
PDF_PATH = OUTPUT_DIR / "fig2a_new.pdf"

CACHE_DIR = SAVED_RESULTS_DIR / "cache" / "simulations" / "Fig2_new"
CACHE_PATH = CACHE_DIR / "fig2a_new_death_times.npz"
METADATA_PATH = CACHE_DIR / "fig2a_new_metadata.json"
CI_CACHE_PATH = CACHE_DIR / "fig2a_new_curve_fit_ci_survival.npz"
CI_METADATA_PATH = CACHE_DIR / "fig2a_new_curve_fit_ci_metadata.json"
PLOT_DATA_PATH = SAVED_RESULTS_DIR / "csv" / "fig2a_new_conditional_survival.csv"
CI_ENVELOPE_PATH = SAVED_RESULTS_DIR / "csv" / "fig2a_new_fit_ci_envelopes.csv"
LEGACY_OUTPUT_PATHS = {
    "Figures/Fig2_new/fig2a_new.png",
    "Figures/Fig2_new/fig2a_new.pdf",
}

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

PARAMETER_COLORS = {
    "eta": "#0B7F8C",
    "beta": "#173A6A",
    "Xc": "#D77A16",
    "epsilon": "#E5A100",
}
HMD_COLORS = {
    "SWE_2019": "#000000",
}

N_SIM = 1_000_000
CI_N_SIM = 500_000
HETERO_CV = 0.05
CURVE_SPECS = (
    ("Xc_15", "Xc", 0.15),
    ("epsilon_25", "epsilon", 0.25),
    ("eta_5", "eta", 0.05),
    ("beta_5", "beta", 0.05),
)
CURVE_LABELS = {
    "Xc_15": r"$X_c$ 15%",
    "epsilon_25": r"$\epsilon$ 25%",
    "eta_5": r"$\eta$ 5%",
    "beta_5": r"$\beta$ 5%",
}
CURVE_LABEL_ANCHOR_AGES = {
    "Xc_15": 112,
    "epsilon_25": 108,
    "eta_5": 121,
    "beta_5": 117,
}
CURVE_LABEL_OFFSETS = {
    "Xc_15": (0, 0, "center"),
    "epsilon_25": (0, 0, "center"),
    "eta_5": (0, 0, "center"),
    "beta_5": (0, 0, "center"),
}
CURVE_LABEL_BBOX = {
    "boxstyle": "round,pad=0.12,rounding_size=0.06",
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.74,
}
CONDITION_AGE = 90
AGE_MAX = 125
MIN_PLOTTED_SURVIVAL = 2e-5
CI_BAND_ALPHA = 0.14
TMAX = 150.0
DT = 0.025
SAVE_TIMES = 150.0
RANDOM_SEED = 20260519
CI_BASELINE_PARAMS = ("eta", "beta", "epsilon", "Xc")
CI_ROW_BY_PARAM = {
    "eta": "eta",
    "beta": "beta",
    "epsilon": "epsilon",
    "Xc": "SWE_Xc",
}


def load_sweden_baseline() -> dict[str, float]:
    """Load the Sweden 2019 baseline used by Fig1D_new."""
    record = json.loads(BASELINE_FIT_PATH.read_text())
    fitted = record["summary"]["fitted_parameters"]
    return {
        "eta": fitted["eta"],
        "beta": fitted["beta"],
        "kappa": 0.5,
        "epsilon": fitted["epsilon"],
        "Xc": fitted["SWE_Xc"],
    }


def sample_positive_gaussian(mean: float, cv: float, n: int, seed: int) -> np.ndarray:
    """Sample a positive Gaussian parameter distribution."""
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=mean, scale=mean * cv, size=n)

    while True:
        bad = values <= 0
        if not np.any(bad):
            return values

        values[bad] = rng.normal(loc=mean, scale=mean * cv, size=int(bad.sum()))


def build_params_for_heterogeneity(
    param_name: str,
    baseline: dict[str, float],
    n: int,
    heterogeneity_cv: float,
) -> dict[str, np.ndarray]:
    """Build a parameter dictionary with heterogeneity in one parameter."""
    params = {}
    for key, value in baseline.items():
        if key == param_name:
            params[key] = sample_positive_gaussian(
                mean=value,
                cv=heterogeneity_cv,
                n=n,
                seed=stable_seed(param_name, heterogeneity_cv, "heterogeneity"),
            )
        else:
            params[key] = np.full(n, value, dtype=float)
    return params


def stable_seed(*parts: object) -> int:
    """Create a deterministic uint32 seed from readable metadata."""
    import hashlib

    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run_simulation(
    param_name: str,
    baseline: dict[str, float],
    n: int,
    random_seed: int,
    heterogeneity_cv: float,
) -> np.ndarray:
    """Run one SR simulation and return death times."""
    params = build_params_for_heterogeneity(param_name, baseline, n, heterogeneity_cv)
    sim = utils.create_sr_simulation(
        params_dict=params,
        n=n,
        h_ext=0.0,
        tmax=TMAX,
        dt=DT,
        save_times=SAVE_TIMES,
        parallel=True,
        break_early=True,
        random_seed=random_seed,
    )
    return sim.death_times


def run_one_simulation(curve_id: str, param_name: str, heterogeneity_cv: float, baseline: dict[str, float]) -> np.ndarray:
    """Run one central SR simulation and return death times."""
    return run_simulation(
        param_name=param_name,
        baseline=baseline,
        n=N_SIM,
        random_seed=stable_seed(RANDOM_SEED, curve_id, param_name, heterogeneity_cv),
        heterogeneity_cv=heterogeneity_cv,
    )


def load_or_run_simulations() -> dict[str, np.ndarray]:
    """Load cached death times when possible, otherwise run missing simulations."""
    baseline = load_sweden_baseline()
    metadata = {
        "baseline_fit_path": str(BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)),
        "baseline": baseline,
        "curve_specs": [
            {"curve_id": curve_id, "param": param_name, "heterogeneity_cv": cv}
            for curve_id, param_name, cv in CURVE_SPECS
        ],
        "n_sim": N_SIM,
        "h_ext": 0.0,
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "random_seed": RANDOM_SEED,
    }

    results = {}
    if CACHE_PATH.exists() and METADATA_PATH.exists():
        cached_metadata = json.loads(METADATA_PATH.read_text())
        cached = np.load(CACHE_PATH)
        if cached_metadata == metadata:
            return {curve_id: cached[curve_id] for curve_id, _, _ in CURVE_SPECS}

        if cached_metadata.get("heterogeneity_cv") == HETERO_CV:
            for curve_id, param_name, cv in CURVE_SPECS:
                if np.isclose(cv, HETERO_CV) and param_name in cached.files:
                    results[curve_id] = cached[param_name]

        for curve_id, _, _ in CURVE_SPECS:
            if curve_id in cached.files:
                results[curve_id] = cached[curve_id]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for curve_id, param_name, cv in CURVE_SPECS:
        if curve_id in results:
            continue

        print(f"Running Fig2a_new simulation: {100 * cv:.0f}% heterogeneity in {param_name}")
        results[curve_id] = run_one_simulation(curve_id, param_name, cv, baseline)

    np.savez_compressed(CACHE_PATH, **results)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    return results


def load_sweden_ci_bounds() -> dict[str, tuple[float, float]]:
    """Load fit-local 95% CI bounds for the baseline parameters used here."""
    ci = pd.read_csv(BASELINE_CI_PATH).set_index("parameter")
    bounds = {}
    for param_name in CI_BASELINE_PARAMS:
        row_name = CI_ROW_BY_PARAM[param_name]
        row = ci.loc[row_name]
        bounds[param_name] = (float(row["ci95_lower"]), float(row["ci95_upper"]))
    return bounds


def build_curve_ci_scenarios(baseline: dict[str, float]) -> list[tuple[str, str, float, str, dict[str, float]]]:
    """Build broad lower/upper fit-CI scenarios for each drawn curve."""
    ci_bounds = load_sweden_ci_bounds()
    scenarios = []
    for curve_id, param_name, cv in CURVE_SPECS:
        for ci_param_name, (lower, upper) in ci_bounds.items():
            low = dict(baseline)
            low[ci_param_name] = lower
            scenarios.append((curve_id, param_name, cv, f"{ci_param_name}_ci_lower", low))

            high = dict(baseline)
            high[ci_param_name] = upper
            scenarios.append((curve_id, param_name, cv, f"{ci_param_name}_ci_upper", high))

    return scenarios


def load_or_run_ci_survival() -> pd.DataFrame:
    """Load or run curve-specific CI-endpoint simulations."""
    baseline = load_sweden_baseline()
    scenarios = build_curve_ci_scenarios(baseline)
    metadata = {
        "baseline_fit_path": str(BASELINE_FIT_PATH.relative_to(PROJECT_ROOT)),
        "baseline_ci_path": str(BASELINE_CI_PATH.relative_to(PROJECT_ROOT)),
        "baseline": baseline,
        "ci_method": "curve_all_parameter_95_ci_endpoint_envelope",
        "curve_specs": [
            {"curve_id": curve_id, "param": param_name, "heterogeneity_cv": cv}
            for curve_id, param_name, cv in CURVE_SPECS
        ],
        "scenario_ids": [scenario_id for _, _, _, scenario_id, _ in scenarios],
        "n_sim": CI_N_SIM,
        "condition_age": CONDITION_AGE,
        "age_max": AGE_MAX,
        "h_ext": 0.0,
        "tmax": TMAX,
        "dt": DT,
        "save_times": SAVE_TIMES,
        "random_seed": RANDOM_SEED,
    }

    if CI_CACHE_PATH.exists() and CI_METADATA_PATH.exists():
        cached_metadata = json.loads(CI_METADATA_PATH.read_text())
        if cached_metadata == metadata:
            cached = np.load(CI_CACHE_PATH)
            return ci_cache_to_frame(cached)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ages = np.arange(CONDITION_AGE, AGE_MAX + 1)
    arrays = {}

    for curve_id, param_name, cv, scenario_id, scenario_baseline in scenarios:
        cache_key = f"{curve_id}__{scenario_id}"
        print(f"Running Fig2a_new fit-CI simulation: {curve_id}, {scenario_id}")
        death_times = run_simulation(
            param_name=param_name,
            baseline=scenario_baseline,
            n=CI_N_SIM,
            random_seed=stable_seed(RANDOM_SEED, curve_id, scenario_id, "fit_ci"),
            heterogeneity_cv=cv,
        )
        arrays[cache_key] = conditional_survival_from_deaths(death_times, ages)

    np.savez_compressed(CI_CACHE_PATH, **arrays)
    CI_METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n")
    cached = np.load(CI_CACHE_PATH)
    return ci_cache_to_frame(cached)


def ci_cache_to_frame(cache: np.lib.npyio.NpzFile) -> pd.DataFrame:
    """Convert the CI survival cache into long-form rows."""
    ages = np.arange(CONDITION_AGE, AGE_MAX + 1)
    rows = []
    for key in cache.files:
        curve_id, scenario_id = key.split("__", maxsplit=1)
        param_name = next(param for candidate_id, param, _ in CURVE_SPECS if candidate_id == curve_id)
        for age, value in zip(ages, cache[key]):
            rows.append(
                {
                    "curve_id": curve_id,
                    "param": param_name,
                    "scenario_id": scenario_id,
                    "age": int(age),
                    "conditional_survival": float(value),
                }
            )
    return pd.DataFrame(rows)


def conditional_survival_from_deaths(death_times: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Return survival conditional on surviving to CONDITION_AGE."""
    at_risk = np.sum(death_times >= CONDITION_AGE)
    if at_risk == 0:
        raise ValueError("No simulated individuals survived to the conditioning age.")

    return np.array([np.sum(death_times >= age) / at_risk for age in ages], dtype=float)


def load_hmd_conditional_survival(country_code: str, country_label: str, year: int) -> pd.DataFrame:
    """Load HMD period survival conditional on the target age."""
    hmd = HMD(country_code, "both", "period")
    ages, survival = hmd.get_survival(year, strict=True)
    frame = pd.DataFrame({"age": ages, "survival": survival})
    frame = frame[(frame["age"] >= CONDITION_AGE) & (frame["age"] <= AGE_MAX)].copy()

    if frame.empty:
        raise ValueError(f"No {country_label} HMD survival rows for year {year}.")

    survival_at_condition_age = float(frame.loc[frame["age"] == CONDITION_AGE, "survival"].iloc[0])
    frame["conditional_survival"] = frame["survival"] / survival_at_condition_age
    frame["source"] = f"{country_label} {year} HMD"
    return frame[["source", "age", "conditional_survival"]]


def build_plot_data(death_times_by_curve: dict[str, np.ndarray]) -> pd.DataFrame:
    """Build all data drawn in the panel."""
    ages = np.arange(CONDITION_AGE, AGE_MAX + 1)
    rows = []

    for curve_id, param_name, cv in CURVE_SPECS:
        death_times = death_times_by_curve[curve_id]
        survival = conditional_survival_from_deaths(death_times, ages)
        for age, value in zip(ages, survival):
            rows.append(
                {
                    "source": f"SR {curve_id}",
                    "param": param_name,
                    "heterogeneity_cv": cv,
                    "curve_id": curve_id,
                    "age": int(age),
                    "conditional_survival": float(value),
                }
            )

    data = pd.DataFrame(rows)
    hmd_data = load_hmd_conditional_survival("SWE", "Sweden", 2019)
    return pd.concat([hmd_data, data], ignore_index=True)


def build_ci_envelopes(data: pd.DataFrame, ci_survival: pd.DataFrame) -> pd.DataFrame:
    """Build pointwise fit-CI envelopes around each central SR curve."""
    rows = []
    for curve_id, param_name, cv in CURVE_SPECS:
        central = data[data["source"] == f"SR {curve_id}"].copy()
        central["curve_id"] = curve_id
        central["param"] = param_name
        central["heterogeneity_cv"] = cv
        central["scenario_id"] = "central"

        ci_subset = ci_survival[ci_survival["curve_id"] == curve_id].copy()
        ci_subset["heterogeneity_cv"] = cv
        combined = pd.concat(
            [
                central[["curve_id", "param", "heterogeneity_cv", "scenario_id", "age", "conditional_survival"]],
                ci_subset[["curve_id", "param", "heterogeneity_cv", "scenario_id", "age", "conditional_survival"]],
            ],
            ignore_index=True,
        )

        envelope = (
            combined.groupby(["curve_id", "param", "heterogeneity_cv", "age"], as_index=False)["conditional_survival"]
            .agg(ci_lower="min", ci_upper="max")
            .copy()
        )
        rows.append(envelope)

    envelopes = pd.concat(rows, ignore_index=True)
    CI_ENVELOPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    envelopes.to_csv(CI_ENVELOPE_PATH, index=False)
    return envelopes


def configure_matplotlib() -> None:
    """Apply publication-style figure defaults."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
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


def plot_panel(data: pd.DataFrame, ci_envelopes: pd.DataFrame) -> None:
    """Draw and save Fig2a_new."""
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PLOT_DATA_PATH, index=False)

    fig, ax = plt.subplots(figsize=(6.7, 4.9))

    draw_hmd_curve(
        ax,
        data,
        "Sweden 2019 HMD",
        HMD_COLORS["SWE_2019"],
        "Sweden 2019 period data",
    )

    for curve_id, _, _ in CURVE_SPECS:
        draw_ci_envelope(ax, ci_envelopes, curve_id)

    for curve_id, param_name, _ in CURVE_SPECS:
        source = f"SR {curve_id}"
        subset = data[data["source"] == source]
        ax.plot(
            subset["age"],
            mask_plot_tail(subset["conditional_survival"]),
            color=PARAMETER_COLORS[param_name],
            lw=2.7,
            linestyle="-",
            alpha=1.0,
            zorder=4,
        )

    label_curve_endpoints(ax, data)
    ax.set_yscale("log")
    ax.set_xlim(CONDITION_AGE, AGE_MAX)
    ax.set_ylim(MIN_PLOTTED_SURVIVAL, 1.15)
    ax.set_xlabel("Age [years]")
    ax.set_ylabel(f"Conditional survival from age {CONDITION_AGE}")
    ax.set_title(
        "Late-life survival is consistent with heterogeneity\nin robustness parameters",
        pad=8,
    )
    ax.set_xticks([90, 100, 110, 120])
    ax.set_yticks([1, 1e-1, 1e-2, 1e-3, 1e-4])
    ax.tick_params(axis="both", which="major", pad=4)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_hmd_legend(ax)

    fig.tight_layout(pad=0.8)
    fig.savefig(PNG_PATH, dpi=300)
    fig.savefig(PDF_PATH)
    plt.close(fig)


def label_curve_endpoints(ax: plt.Axes, data: pd.DataFrame) -> None:
    """Directly label the survival curves since this panel has no legend."""
    for curve_id, param_name, _ in CURVE_SPECS:
        subset = data[data["source"] == f"SR {curve_id}"].sort_values("age")
        visible = subset[subset["conditional_survival"] >= MIN_PLOTTED_SURVIVAL]
        if visible.empty:
            continue

        anchor_age = CURVE_LABEL_ANCHOR_AGES[curve_id]
        anchor = visible.iloc[(visible["age"] - anchor_age).abs().argsort()[:1]].iloc[0]
        x_offset, y_offset, ha = CURVE_LABEL_OFFSETS[curve_id]
        ax.annotate(
            CURVE_LABELS[curve_id],
            xy=(float(anchor["age"]), float(anchor["conditional_survival"])),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            color=PARAMETER_COLORS[param_name],
            fontsize=13.5,
            ha=ha,
            va="center",
            alpha=1.0,
            bbox=CURVE_LABEL_BBOX,
            zorder=7,
        )


def draw_ci_envelope(ax: plt.Axes, envelopes: pd.DataFrame, curve_id: str) -> None:
    """Draw the fit-CI sensitivity band for one SR curve."""
    subset = envelopes[envelopes["curve_id"] == curve_id].copy()
    if subset.empty:
        return

    param_name = str(subset["param"].iloc[0])
    upper = subset["ci_upper"].to_numpy(dtype=float)
    lower = subset["ci_lower"].to_numpy(dtype=float)
    visible = upper >= MIN_PLOTTED_SURVIVAL
    lower = np.maximum(lower, MIN_PLOTTED_SURVIVAL)

    ax.fill_between(
        subset["age"],
        np.where(visible, lower, np.nan),
        np.where(visible, upper, np.nan),
        color=PARAMETER_COLORS[param_name],
        alpha=CI_BAND_ALPHA,
        linewidth=0,
        zorder=1,
    )


def draw_hmd_curve(
    ax: plt.Axes,
    data: pd.DataFrame,
    source: str,
    color: str,
    label: str,
) -> None:
    """Draw one empirical HMD curve."""
    subset = data[data["source"] == source]
    ax.plot(
        subset["age"],
        mask_plot_tail(subset["conditional_survival"]),
        color=color,
        lw=4.2,
        linestyle="-",
        label=label,
        zorder=6,
    )


def add_hmd_legend(ax: plt.Axes) -> None:
    """Add a compact legend for the empirical Sweden curve only."""
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        handlelength=2.0,
        borderpad=0.35,
        labelspacing=0.25,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("none")
    legend.get_frame().set_alpha(0.82)
    legend.set_zorder(8)


def mask_plot_tail(values: pd.Series) -> pd.Series:
    """Hide finite-sample tail points that are below the plotted range."""
    return values.where(values >= MIN_PLOTTED_SURVIVAL, np.nan)


def update_output_index() -> None:
    """Upsert output-index rows for this figure."""
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

    rows = [
        {
            "date": "2026-05-19",
            "task": "fig2a_new_survival_tail",
            "artifact_type": "figure",
            "path": "Figures_new/Fig2_new/fig2a_new.png",
            "source_script": "src/figures/Fig2_new/make_fig2a_new.py",
            "input_paths": "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json; saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
            "description": "PNG preview of new Fig2a survival-tail heterogeneity panel",
            "notes": "Sweden 2019 baseline; 15% Xc, 25% epsilon, 5% eta, and 5% beta heterogeneity; compared with Sweden 2019 HMD; conditional on age 90; shaded bands are all-parameter fit-CI endpoint envelopes",
        },
        {
            "date": "2026-05-19",
            "task": "fig2a_new_survival_tail",
            "artifact_type": "figure",
            "path": "Figures_new/Fig2_new/fig2a_new.pdf",
            "source_script": "src/figures/Fig2_new/make_fig2a_new.py",
            "input_paths": "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json; saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
            "description": "Vector PDF of new Fig2a survival-tail heterogeneity panel",
            "notes": "Matplotlib PDF with editable text where supported",
        },
        {
            "date": "2026-05-19",
            "task": "fig2a_new_survival_tail",
            "artifact_type": "csv",
            "path": "saved_results/csv/fig2a_new_conditional_survival.csv",
            "source_script": "src/figures/Fig2_new/make_fig2a_new.py",
            "input_paths": "saved_results/fit_archive/records/joint2019_tail90_sweden_emphasis.json",
            "description": "Conditional survival source data for Fig2a_new",
            "notes": "All curves normalized to survival at age 90",
        },
        {
            "date": "2026-05-19",
            "task": "fig2a_new_survival_tail",
            "artifact_type": "csv",
            "path": "saved_results/csv/fig2a_new_fit_ci_envelopes.csv",
            "source_script": "src/figures/Fig2_new/make_fig2a_new.py",
            "input_paths": "saved_results/fit_archive/ci/joint2019_tail90_sweden_emphasis_ci.csv",
            "description": "Pointwise fit-CI endpoint envelopes for Fig2a_new shaded bands",
            "notes": "Envelope is min/max across central and lower/upper 95% CI endpoint perturbations of eta, beta, epsilon, and Sweden Xc for each drawn heterogeneity curve",
        },
    ]

    replacement_by_path = {row["path"]: row for row in rows}
    paths_to_replace = set(replacement_by_path) | LEGACY_OUTPUT_PATHS
    kept_rows = [row for row in existing_rows if row.get("path") not in paths_to_replace]
    with index_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
        writer.writerows(rows)


def main() -> None:
    death_times_by_curve = load_or_run_simulations()
    data = build_plot_data(death_times_by_curve)
    ci_survival = load_or_run_ci_survival()
    ci_envelopes = build_ci_envelopes(data, ci_survival)
    plot_panel(data, ci_envelopes)
    update_output_index()
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")
    print(f"Saved {PLOT_DATA_PATH}")
    print(f"Saved {CI_ENVELOPE_PATH}")


if __name__ == "__main__":
    main()
