#!/usr/bin/env python3
"""Lightweight self-contained verification for the paper repository."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for path in (SRC_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from ageing_packages.hetero_analysis import nhanes_analysis as nhanes
from ageing_packages.utils.sr_utils import create_sr_simulation
from senogenic_vs_robustness.paths import FIGURES_DIR, HMD_DATA_DIR, NHANES_DATA_DIR, RESULTS_DIR


REQUIRED_FILES = [
    HMD_DATA_DIR / "mortality.org_File_GetDocument_hmd.v6_SWE_STATS_bltper_1x1.txt",
    HMD_DATA_DIR / "mortality.org_File_GetDocument_hmd.v6_USA_STATS_bltper_1x1.txt",
    HMD_DATA_DIR / "mortality.org_File_GetDocument_hmd.v6_DAN_STATS_bltper_1x1.txt",
    NHANES_DATA_DIR / "nhanes_mortality_all_years.csv",
    NHANES_DATA_DIR / "all_cohort_age_data.csv",
    RESULTS_DIR / "fits" / "records" / "joint2019_tail90_sweden_emphasis.json",
    RESULTS_DIR / "fits" / "records" / "hybrid2019_swe_tail90_usa_refit.json",
    RESULTS_DIR / "tables" / "fig6_progeria_fit_results.csv",
    RESULTS_DIR / "figure3_exposure_projection" / "exposure_coordinate_projection_paper_summary.csv",
    RESULTS_DIR / "figure3_exposure_projection" / "exposure_coordinate_projection_full_bootstrap_assignments.csv",
    RESULTS_DIR / "figure3_exposure_projection" / "exposure_coordinate_projection_full_bootstrap_failures.csv",
    RESULTS_DIR / "figure3_exposure_projection" / "exposure_xc_equivalent_projection_full_uncertainty.csv",
    RESULTS_DIR / "tables" / "extended_data_table1_fig3_projection.csv",
    RESULTS_DIR / "tau_wide_refit_profile" / "tau_wide_refit_profile_candidates.csv",
    RESULTS_DIR / "tau_wide_refit_profile" / "figS_tau_spread_constraint_panel_b_curves_n1000000_age55.csv",
    RESULTS_DIR / "tables" / "extended_data_figure1_tau_spread_constraint_source_data.csv",
    RESULTS_DIR / "tables" / "supplementary_figure1" / "sweden2019_decade_slopes.csv",
    RESULTS_DIR / "tables" / "supplementary_figure1" / "allowed_parameter_cv_vs_slope_distortion.csv",
    RESULTS_DIR / "tables" / "supplementary_figure1" / "survivor_parameter_means.csv",
    RESULTS_DIR / "tables" / "supplementary_figure1" / "senogenic_heterogeneity_hazards.csv",
    RESULTS_DIR / "tables" / "supp_figure3_fedichev_minimal_model_source.csv",
]

REQUIRED_FIGURE_PREVIEWS = [
    FIGURES_DIR / "Figure1" / "Fig1_alt.png",
    FIGURES_DIR / "Figure2" / "Figure2.png",
    FIGURES_DIR / "Figure3" / "fig3_exposure_projection.png",
    FIGURES_DIR / "Figure4" / "Figure4.png",
    FIGURES_DIR / "Figure5_progeria" / "fig6_progeria_composite.png",
    FIGURES_DIR / "ExtendedDataFigure1" / "extended_data_figure1_tau_spread_constraint.png",
    FIGURES_DIR / "ExtendedDataFigure2" / "extended_data_figure2_denmark_robustness.png",
    FIGURES_DIR / "ExtendedDataFigure3" / "extended_data_figure3_model_comparison.png",
    FIGURES_DIR / "Supplementary" / "supp_figure1_gompertz_constraint.png",
    FIGURES_DIR / "Supplementary" / "supp_figure3_fedichev_minimal_model.png",
    FIGURES_DIR / "Supplementary" / "supp_figure2_nhanes_exposure_groups.png",
    FIGURES_DIR / "Supplementary" / "supp_figure4_healthspan_morbidity.png",
]

STALE_PUBLIC_ARTIFACTS = [
    FIGURES_DIR / "Figure1" / "Fig1_new.png",
    FIGURES_DIR / "Figure3" / "fig3_new.png",
    FIGURES_DIR / "Figure3" / "fig3_usa_steepness_longevity.png",
    FIGURES_DIR / "Figure4" / "denmark Fig4.png",
    FIGURES_DIR / "Supplementary" / "supp_model_comparison.png",
    FIGURES_DIR / "Supplementary" / "supp_fig4_nhanes_exposure_groups.png",
    FIGURES_DIR / "Supplementary" / "supp_artificial_survival_composite.png",
    FIGURES_DIR / "Supplementary_Figure1" / "figs1_parameter_distributions_pretty.png",
]


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(path) for path in missing))
    print(f"ok required files: {len(REQUIRED_FILES)}")


def check_current_figure_previews() -> None:
    missing = [path for path in REQUIRED_FIGURE_PREVIEWS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing current manuscript figure previews:\n" + "\n".join(str(path) for path in missing)
        )
    print(f"ok current figure previews: {len(REQUIRED_FIGURE_PREVIEWS)}")


def check_no_stale_public_artifacts() -> None:
    stale = [path for path in STALE_PUBLIC_ARTIFACTS if path.exists()]
    if stale:
        raise RuntimeError("Stale figure artifacts still present:\n" + "\n".join(str(path) for path in stale))
    print("ok stale figure artifacts absent")


def check_no_svg_artifacts() -> None:
    skipped = {".git", ".venv", "venv", "__pycache__"}
    svg_paths = [
        path
        for path in PROJECT_ROOT.rglob("*.svg")
        if skipped.isdisjoint(path.relative_to(PROJECT_ROOT).parts)
    ]
    if svg_paths:
        raise RuntimeError("SVG artifacts should not remain in the public repo:\n" + "\n".join(str(path) for path in svg_paths))
    print("ok SVG artifacts absent")


def check_hmd() -> None:
    for country in ("SWE", "USA", "DAN"):
        hmd = HMD(country, "both", "period")
        ages, survival = hmd.get_survival(2019, strict=True)
        if len(ages) == 0 or not np.isfinite(survival).all():
            raise RuntimeError(f"HMD survival failed for {country}")
        print(f"ok HMD {country}: {len(ages)} ages, S(0)={survival[0]:.3f}")


def check_nhanes() -> None:
    core = nhanes.load_core(str(NHANES_DATA_DIR) + "/")
    expected = {"entry_age", "exit_age", "event"}
    if not expected.issubset(core.columns):
        raise RuntimeError(f"NHANES core missing columns: {expected - set(core.columns)}")
    deaths = int(pd.to_numeric(core["event"], errors="coerce").fillna(0).sum())
    print(f"ok NHANES core: n={len(core):,}, deaths={deaths:,}")


def check_fits() -> None:
    record_path = RESULTS_DIR / "fits" / "records" / "joint2019_tail90_sweden_emphasis.json"
    record = json.loads(record_path.read_text())
    params = record["summary"]["fitted_parameters"]
    for key in ("eta", "beta", "epsilon", "SWE_Xc"):
        float(params[key])
    print("ok fits: joint2019_tail90_sweden_emphasis")


def check_fig3_projection_table() -> None:
    table_path = RESULTS_DIR / "tables" / "extended_data_table1_fig3_projection.csv"
    table = pd.read_csv(table_path)
    expected_columns = {
        "Topic",
        "Group",
        "M/M₀ (bootstrap range)",
        "S/S₀ (bootstrap range)",
        "Robustness % (full uncertainty range)",
        "Senogenic % (full uncertainty range)",
        "mₑₓ % (full uncertainty range)",
        "Xc factor (full uncertainty range)",
    }
    if set(table.columns) != expected_columns:
        raise RuntimeError(f"Extended Data Table 1 columns changed: {list(table.columns)}")
    if len(table) != 23:
        raise RuntimeError(f"Expected 23 NHANES exposure rows, found {len(table)}")

    failures = pd.read_csv(
        RESULTS_DIR
        / "figure3_exposure_projection"
        / "exposure_coordinate_projection_full_bootstrap_failures.csv"
    )
    if len(failures) != 1:
        raise RuntimeError(f"Expected one logged bootstrap failure row, found {len(failures)}")
    failure = failures.iloc[0]
    if failure["topic"] != "income" or failure["group"] != "Q4 (Highest)":
        raise RuntimeError("Unexpected Fig. 3 projection bootstrap failure row")
    print("ok Fig. 3 projection table: 23 rows, one logged Q4-income bootstrap exclusion")


def check_extended_data_fig1_tau_profile() -> None:
    candidates = pd.read_csv(RESULTS_DIR / "tau_wide_refit_profile" / "tau_wide_refit_profile_candidates.csv")
    required = {"candidate_id", "source", "eval_score", "tau_cv", "tau_sd"}
    if not required.issubset(candidates.columns):
        raise RuntimeError(f"Tau-profile candidates missing columns: {required - set(candidates.columns)}")
    profile = candidates[candidates["source"] == "profile_tau_sd"].copy()
    if len(profile) != 9:
        raise RuntimeError(f"Expected 9 tau-profile grid rows, found {len(profile)}")
    if abs(float(profile["tau_cv"].min())) > 1e-12 or float(profile["tau_cv"].max()) < 0.24:
        raise RuntimeError("Tau-profile CV grid does not span 0 to about 25%")

    source = pd.read_csv(RESULTS_DIR / "tables" / "extended_data_figure1_tau_spread_constraint_source_data.csv")
    panels = set(source["panel"].dropna())
    expected_panels = {"A_fit_objective", "B_mortality_curves", "B_baseline_fit_ci_envelope"}
    if not expected_panels.issubset(panels):
        raise RuntimeError(f"Extended Data Fig. 1 source panels changed: {sorted(panels)}")
    print("ok Extended Data Fig. 1 tau profile: 9 grid rows and source data panels present")


def check_supplementary_fig1_sources() -> None:
    allowed = pd.read_csv(RESULTS_DIR / "tables" / "supplementary_figure1" / "allowed_parameter_cv_vs_slope_distortion.csv")
    if set(allowed["parameter"]) != {"eta", "beta", "Xc", "epsilon"}:
        raise RuntimeError("Supplementary Fig. 1 allowed-heterogeneity parameters changed")
    hazards = pd.read_csv(RESULTS_DIR / "tables" / "supplementary_figure1" / "senogenic_heterogeneity_hazards.csv")
    expected_hazard_cols = {"parameter", "age", "mortality_rate"}
    if set(hazards.columns) != expected_hazard_cols:
        raise RuntimeError("Supplementary Fig. 1 hazard source columns changed")
    if set(hazards["parameter"]) != {"eta", "beta"}:
        raise RuntimeError("Supplementary Fig. 1 hazard source should contain eta and beta scenarios")
    slopes = pd.read_csv(RESULTS_DIR / "tables" / "supplementary_figure1" / "sweden2019_decade_slopes.csv")
    if len(slopes) != 5 or "slope_ratio_to_mean" not in slopes.columns:
        raise RuntimeError("Supplementary Fig. 1 decade-slope source changed")
    print("ok Supplementary Fig. 1 Gompertz-constraint sources")


def check_supplementary_fig3_source() -> None:
    source = pd.read_csv(RESULTS_DIR / "tables" / "supp_figure3_fedichev_minimal_model_source.csv")
    if set(source["series"]) != {"survival", "mortality"}:
        raise RuntimeError("Supplementary Fig. 3 source should contain survival and mortality series")
    if source["value"].isna().any() or (source["value"] < 0).any():
        raise RuntimeError("Supplementary Fig. 3 source contains invalid values")
    print("ok Supplementary Fig. 3 Fedichev-Gruber source")


def check_tiny_sr_simulation() -> None:
    params = {
        "eta": np.full(64, 0.5868368258),
        "beta": np.full(64, 57.8717377207),
        "kappa": np.full(64, 0.5),
        "epsilon": np.full(64, 49.7186593046),
        "Xc": np.full(64, 21.7405634007),
    }
    sim = create_sr_simulation(
        n=64,
        params_dict=params,
        tmax=5,
        dt=0.25,
        save_times=1,
        random_seed=123,
        parallel=False,
        break_early=False,
    )
    death_times = sim.death_times
    if death_times.shape[0] != 64:
        raise RuntimeError("Tiny SR simulation returned an unexpected shape")
    print("ok tiny SR simulation")


def main() -> None:
    check_required_files()
    check_current_figure_previews()
    check_no_stale_public_artifacts()
    check_no_svg_artifacts()
    check_hmd()
    check_nhanes()
    check_fits()
    check_fig3_projection_table()
    check_extended_data_fig1_tau_profile()
    check_supplementary_fig1_sources()
    check_supplementary_fig3_source()
    check_tiny_sr_simulation()
    print("verification complete")


if __name__ == "__main__":
    main()
