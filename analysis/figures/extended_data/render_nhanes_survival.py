"""Extended Data Figure 2: NHANES exposure-group KM curves."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
HERE = PROJECT / 'results/nhanes'
GROUP_RUN = PROJECT / 'analysis/model_fits/nhanes'
OUT = PROJECT / 'Figures/ExtendedDataFigure2/ExtDataFig2.png'
TIMELINE = np.arange(20, 110.01, .25)
DOMAINS = [
    ('diet', 'Diet quality'), ('income', 'Income-poverty ratio'),
    ('number_of_friends', 'Number of friends'),
    ('sleep_duration', 'Sleep duration'),
    ('physical_activity', 'Physical activity'),
    ('alcohol', 'Alcohol consumption'),
    ('sleep_frailty', 'Sleep frailty'),
    ('church_frequency', 'Religious attendance'),
    ('education_level', 'Education'),
]
PALETTE = ('#cb5a3d', '#d9921e', '#237b68', '#375f9b')


def main():
    data = pd.read_csv(HERE/'survival_curves.csv')
    groups = json.loads((GROUP_RUN/'inputs'/'groups.json').read_text())
    base = data[data.group == 'all'].sort_values('age')
    n_total = int(base.n.iloc[0])
    plt.rcParams.update({'font.family':'Arial', 'font.size':17,
                         'axes.titlesize':22, 'axes.labelsize':19,
                         'xtick.labelsize':17, 'ytick.labelsize':17,
                         'legend.fontsize':16, 'pdf.fonttype':42,
                         'ps.fonttype':42})
    fig, axs = plt.subplots(3, 3, figsize=(16.2, 12.4), sharex=True, sharey=True)
    for ax, (domain, title) in zip(axs.flat, DOMAINS):
        ax.plot(TIMELINE, base.survival.to_numpy(), color='#222222', lw=1.8,
                label=f'All NHANES (n={n_total:,})')
        names = [name for name in groups if name.startswith(domain+'__')]
        for j, name in enumerate(names):
            subset = data[data.group == name].sort_values('age')
            ax.plot(TIMELINE, subset.survival.to_numpy(), color=PALETTE[j], lw=1.6,
                    label=f"{groups[name]['label']} (n={int(subset.n.iloc[0]):,})")
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlim(20, 110); ax.set_ylim(0, 1.02)
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.grid(axis='y', color='#e6e6e6', lw=.5)
        legend = ax.legend(loc='lower left', frameon=True, handlelength=2)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.86)
        legend.get_frame().set_edgecolor('none')
    fig.supxlabel('Age (years)', fontsize=20)
    fig.supylabel('Survival from age 20', fontsize=20)
    fig.tight_layout(rect=(.025,.025,1,1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, facecolor='white')
    plt.close(fig)
    print(OUT)


if __name__=='__main__': main()
