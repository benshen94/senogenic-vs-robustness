#!/usr/bin/env python3
"""Run manuscript figure-generation scripts from the repository root."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_INDEX = PROJECT_ROOT / "results" / "index" / "outputs.csv"
PYTHON = sys.executable

SMOKE_COMMANDS = [
    ["analysis/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py"],
    ["analysis/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py"],
    ["analysis/figures/figure5_progeria/make_fig6_progeria.py"],
]

MAIN_COMMANDS = [
    ["analysis/figures/figure1_schematic/make_fig1_alt.py"],
    ["analysis/figures/figure2/make_fig2a_new.py"],
    ["analysis/figures/figure2/make_fig2bc_new.py"],
    ["analysis/figures/figure2/make_fig2de_new.py"],
    ["analysis/figures/figure2/assemble_figure2.py"],
    ["analysis/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py"],
    ["analysis/figures/steepness_longevity/make_fig3_exposure_projection.py"],
    ["analysis/figures/steepness_longevity/make_fig3_coordinate_projection_uncertainty.py", "--skip-point-mc"],
    ["analysis/figures/figure4/make_fig4_ab_sweden_period_projection.py"],
    ["analysis/figures/figure4/make_fig4_sr_contour_projection.py"],
    ["analysis/figures/figure4/make_fig4_age0_mean_lifespan_projection.py"],
    ["analysis/figures/figure4/assemble_figure4.py"],
    ["analysis/figures/figure5_progeria/make_fig6_progeria.py"],
]

EXTENDED_DATA_COMMANDS = [
    ["analysis/figures/extended_data/make_extended_data_figure1_tau_spread_constraint.py"],
    ["analysis/figures/extended_data/make_extended_data_figure2_denmark_robustness.py"],
    ["analysis/figures/extended_data/make_extended_data_figure3_model_comparison.py"],
]

SUPPLEMENT_COMMANDS = [
    ["analysis/figures/supplementary/make_supp_figure1_gompertz_constraint.py"],
    ["analysis/figures/supplementary/make_supp_figure2_fedichev_minimal_model.py"],
    ["analysis/figures/supplementary/make_supp_figure3_nhanes_exposure_groups.py"],
    ["analysis/figures/supplementary/make_supp_figure4_healthspan_morbidity.py"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("smoke", "main", "extended", "all"), default="smoke")
    return parser.parse_args()


def commands_for(command_set: str) -> list[list[str]]:
    if command_set == "smoke":
        return SMOKE_COMMANDS
    if command_set == "main":
        return MAIN_COMMANDS
    if command_set == "extended":
        return EXTENDED_DATA_COMMANDS
    return MAIN_COMMANDS + EXTENDED_DATA_COMMANDS + SUPPLEMENT_COMMANDS


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    env["PYTHONPATH"] = (
        str(SRC_DIR)
        + os.pathsep
        + str(PROJECT_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    original_index = OUTPUT_INDEX.read_bytes() if OUTPUT_INDEX.exists() else None
    try:
        for command in commands_for(args.set):
            print("$", " ".join([PYTHON] + command), flush=True)
            subprocess.run([PYTHON] + command, cwd=PROJECT_ROOT, env=env, check=True)
    finally:
        if original_index is not None:
            OUTPUT_INDEX.write_bytes(original_index)


if __name__ == "__main__":
    main()
