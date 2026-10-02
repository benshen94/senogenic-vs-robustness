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

# Figure entry points using saved analysis results.
CURRENT_COMMANDS = [
    ["analysis/figures/supplementary/render_parameter_shifts.py"],
    ["analysis/figures/supplementary/render_healthspan_morbidity.py"],
    ["analysis/figures/supplementary/render_gompertz_constraints.py"],
    ["analysis/figures/figure1_schematic/render_fp_panel.py"],
    ["analysis/figures/figure2/plot_fig2_fp.py", "--data-dir", "results/tables/fig2_fokker_planck", "--output-dir", "Figures/Figure2"],
    ["analysis/figures/figure3/render_nhanes.py"],
    ["analysis/figures/figure4/render_current_history.py"],
    ["analysis/figures/figure5/render_lifespan.py"],
    ["analysis/figures/figure6_progeria/plot_fig6.py"],
    ["analysis/figures/extended_data/render_senogenic_profile.py"],
    ["analysis/figures/extended_data/render_nhanes_aic.py"],
    ["analysis/figures/extended_data/render_nhanes_survival.py"],
    ["analysis/figures/extended_data/render_fedichev_constraints.py"],
    ["analysis/figures/supplementary/render_fedichev_minimal_model.py"],
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", choices=("current",), default="current",
                        help="Current manuscript saved-result renderers; no fitting or simulation")
    return parser.parse_args()


def main() -> None:
    parse_args()
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

    for command in CURRENT_COMMANDS:
        print("$", " ".join([PYTHON] + command), flush=True)
        subprocess.run([PYTHON] + command, cwd=PROJECT_ROOT, env=env, check=True)
    print("Saved-result rendering complete. Figure 1 artwork is preserved; "
          "its numerical preview is under tmp/.")


if __name__ == "__main__":
    main()
