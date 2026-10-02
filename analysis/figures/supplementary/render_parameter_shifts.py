#!/usr/bin/env python3
"""Supplementary Figure 1: SR upper-tail response to mean-parameter shifts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from analysis.figures.figure2.plot_fig2_fp import render_mean_shifts


if __name__ == '__main__':
    print(render_mean_shifts(ROOT / 'results/tables/fig2_fokker_planck',
                             ROOT / 'Figures/Supplementary'))
