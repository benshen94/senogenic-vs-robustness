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
PYTHON = sys.executable

SMOKE_COMMANDS = [
    ["analysis/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py"],
    ["analysis/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py"],
    ["analysis/figures/figure5_progeria/make_fig6_progeria.py"],
]

MAIN_COMMANDS = [
    ["analysis/figures/steepness_longevity/make_fig1d_new_steepness_longevity.py"],
    ["analysis/figures/figure2/make_fig2a_new.py"],
    ["analysis/figures/figure2/make_fig2bc_new.py"],
    ["analysis/figures/figure2/make_fig2de_new.py"],
    ["analysis/figures/steepness_longevity/make_fig3_usa_steepness_longevity.py"],
    ["analysis/figures/steepness_longevity/make_fig3_exposure_projection.py"],
    ["analysis/figures/figure4/make_fig4_ab_sweden_period_projection.py"],
    ["analysis/figures/figure4/make_fig4_sr_contour_projection.py"],
    ["analysis/figures/figure5_progeria/make_fig6_progeria.py"],
]

SUPPLEMENT_COMMANDS = [
    ["analysis/figures/supplementary_parameter_distributions/make_parameter_distribution_supplement.py"],
    ["analysis/figures/supplementary/make_supp_artificial_survival_composite.py"],
    ["analysis/figures/supplementary/make_supp_model_comparison.py"],
    ["analysis/figures/supplementary/make_supp_fig4_nhanes_exposure_groups.py"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("smoke", "main", "all"), default="smoke")
    return parser.parse_args()


def commands_for(command_set: str) -> list[list[str]]:
    if command_set == "smoke":
        return SMOKE_COMMANDS
    if command_set == "main":
        return MAIN_COMMANDS
    return MAIN_COMMANDS + SUPPLEMENT_COMMANDS


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = (
        str(SRC_DIR)
        + os.pathsep
        + str(PROJECT_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    for command in commands_for(args.set):
        print("$", " ".join([PYTHON] + command), flush=True)
        subprocess.run([PYTHON] + command, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
