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
from senogenic_vs_robustness.paths import HMD_DATA_DIR, NHANES_DATA_DIR, RESULTS_DIR


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
]


def check_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(path) for path in missing))
    print(f"ok required files: {len(REQUIRED_FILES)}")


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
    check_hmd()
    check_nhanes()
    check_fits()
    check_fig3_projection_table()
    check_tiny_sr_simulation()
    print("verification complete")


if __name__ == "__main__":
    main()
