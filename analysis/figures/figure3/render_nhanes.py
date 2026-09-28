"""Render revised NHANES Figure 3 from frozen FP and bootstrap tables."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from analysis.figures.figure3 import plot_helpers as fig3


def main():
    data = ROOT / 'results/nhanes/figure3'
    fig3.fig3_base.METRICS_PATH = data / 'metrics_long.csv'
    fig3.X_AXIS_SUMMARY = data / 'fig3_plane_point_intervals.csv'
    fig3.PANEL_B_RIBBON_ALPHA = 0.42
    display = pd.read_csv(data / 'exposure_km_points.csv')
    display['label'] = display['group'].map(fig3.DISPLAY_LABELS).fillna(display['group'])
    point = pd.read_csv(data / 'point_lifespan_gains.csv')
    ribbons = pd.read_csv(data / 'fig3b_bootstrap_ribbons.csv')
    fig3.OUTPUT_DIR = ROOT / 'Figures/Figure3'
    fig3.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig3.COMPOSITE_PNG_PATH = fig3.OUTPUT_DIR / 'Fig3.png'
    np.random.seed(20260924)
    fig3.save_composite(display, point, ribbons)


if __name__ == '__main__':
    main()
