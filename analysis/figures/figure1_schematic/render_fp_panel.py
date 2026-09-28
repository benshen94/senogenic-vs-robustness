"""Render quantitative Figure 1 panel g from saved FP response metrics."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from analysis.figures.steepness_longevity import response_plane as style


def main():
    style.METRICS_PATH = ROOT / 'results/figure1/metrics_long.csv'
    style.apply_style()
    data = style.load_normalized_metrics()
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor('white')
    for param in style.PLOT_PARAMS:
        style.draw_parameter_curve(ax=ax, data=data, param=param)
    style.draw_h_ext_curve(ax=ax, data=data)
    style.finish_axes(ax)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.92)
    style.build_grouped_legend(ax)
    style.build_factor_legend(ax)
    target = ROOT / 'tmp/figure1/panel_g.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=350, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
