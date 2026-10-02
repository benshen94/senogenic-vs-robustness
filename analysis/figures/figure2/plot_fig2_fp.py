#!/usr/bin/env python3
"""Render Figure 2 from finite-volume FP CSV outputs; never run simulations.

Input tables: survival.csv (age, curve_id, param, cv, survival,
conditional_survival), tails_cv.csv (param, cv, tail_age), tails_factor.csv
(param, factor, tail_age), siblings.csv (param, cohort, age, mortality).
Tail ages must be computed at *unconditional* survival 1e-4 by the producer.
Conditional survival must be conditioned at age 90. Mortality is a hazard
per year, not its logarithm. NaN tail ages represent unreached crossings.
Optional ci_low/ci_high columns supply producer-computed uncertainty in
conditional survival, tail age, or mortality units, respectively. No old fit
uncertainty is read or recomputed. Missing bounds leave gaps in shading.

Empirical log_hazard is base 10. All four original digitized series are
preserved; panel c shows brothers, matching the original panel. Its shading
is the 95% OLS mean-fit CI (ages 50--100), extrapolated to the fitted
intersection. It does not include digitization or sampling uncertainty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

COLORS = {'eta': '#0B7F8C', 'beta': '#173A6A', 'Xc': '#D77A16', 'epsilon': '#E5A100'}
LABELS = {'eta': r'Production $\eta$', 'beta': r'Removal $\beta$',
          'Xc': r'Threshold $X_c$', 'epsilon': r'Noise $\epsilon$'}
PARAMETERS = ('Xc', 'epsilon', 'eta', 'beta')
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EMPIRICAL_PATH = PROJECT_ROOT / 'results/tables/fig2d_raw_digitized_points.csv'
SCHEMAS = {
    'survival': ('age', 'curve_id', 'param', 'cv', 'survival', 'conditional_survival'),
    'tails_cv': ('param', 'cv', 'tail_age'),
    'tails_factor': ('param', 'factor', 'tail_age'),
    'siblings': ('param', 'cohort', 'age', 'mortality'),
}


def _read_inputs(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for name, columns in SCHEMAS.items():
        frame = pd.read_csv(data_dir / f'{name}.csv')
        missing = set(columns) - set(frame.columns)
        if missing or frame.empty:
            raise ValueError(f'{name}.csv: empty table or missing columns {sorted(missing)}')
        models = frame.loc[frame.curve_id != 'hmd'] if name == 'survival' else frame
        unknown = set(models.param.dropna()) - set(PARAMETERS)
        if unknown:
            raise ValueError(f'{name}.csv: unknown parameters {sorted(unknown)}')
        if set(models.param.dropna()) != set(PARAMETERS):
            raise ValueError(f'{name}.csv must contain all four parameters')
        tables[name] = frame
    if 'hmd' not in set(tables['survival'].curve_id):
        raise ValueError('survival.csv must contain curve_id=hmd')
    for param, group in tables['siblings'].groupby('param'):
        if set(group.cohort) != {'full', 'good', 'bad'}:
            raise ValueError(f'siblings.csv: {param} needs full, good and bad cohorts')
    return tables


def _style(ax: plt.Axes) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=6, width=1.2)
    ax.grid(False)


def _letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.13, 1.12, letter, transform=ax.transAxes, fontsize=34,
            va='bottom', ha='left')


def _empirical_fit(points: pd.DataFrame, ages: np.ndarray):
    """Original age-window OLS and Student-t mean-fit band, without sim imports."""
    fit = points.loc[points.age.between(50, 100)]
    x, y = fit.age.to_numpy(float), fit.log_hazard.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    residual_se = np.sqrt(np.sum((y - (slope*x + intercept))**2) / (len(x)-2))
    se = residual_se * np.sqrt(1/len(x) + (ages-x.mean())**2 / np.sum((x-x.mean())**2))
    mean = slope*ages + intercept
    width = student_t.ppf(.975, len(x)-2)*se
    return slope, intercept, mean, width


def _draw_empirical(ax: plt.Axes) -> None:
    raw = pd.read_csv(EMPIRICAL_PATH).rename(columns={'series': 'cohort', 'log10_hazard': 'log_hazard'})
    raw = raw.loc[raw.plotted.astype(str).str.lower().eq('true')]
    groups = [raw.loc[raw.cohort == key] for key in ('brothers_short', 'brothers_cent')]
    fits = [_empirical_fit(g, np.array([50.])) for g in groups]
    intersection = (fits[1][1]-fits[0][1])/(fits[0][0]-fits[1][0])
    ages = np.linspace(50, intersection, 160)
    for group, filled, label in zip(groups, (True, False),
                                    ('siblings of short-lived', 'siblings of centenarians')):
        _, _, mean, width = _empirical_fit(group, ages)
        ax.fill_between(ages, mean-width, mean+width, color='#7A7A7A', alpha=.16, linewidth=0)
        ax.plot(ages, mean, color='#7A7A7A', lw=2.3, ls=(0, (3, 2)))
        points = group.loc[group.age <= 102]
        ax.scatter(points.age, points.log_hazard, s=43, facecolors='black' if filled else 'white',
                   edgecolors='black', linewidths=1.2, label=label, zorder=3)
    ax.set(xlim=(48, intersection+5), ylim=(-2.5, max(0, fits[0][0]*intersection+fits[0][1]+.28)),
           xlabel='Age [years]', ylabel=r'$\log_{10}$ mortality rate [year$^{-1}$]')
    ax.set_xticks([50, 70, 90, 110])
    ax.legend(loc='upper left', fontsize=26, frameon=False)
    ax.set_title('Mortality converges for siblings of\ncentenarians and short-lived persons\n(Gavrilova & Gavrilov)', pad=18)
    target = (intersection, fits[0][0]*intersection + fits[0][1])
    ax.annotate('convergence', xy=target, xytext=(102, -.8),
                fontsize=18, color='#333333', ha='center',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.24',
                                color='#555555', lw=1.3))


def _band(ax, group, x, color, *, floor=None, log10=False, alpha=.14):
    """Draw supplied bounds; clip nonpositive log bounds at the display floor."""
    if not {'ci_low', 'ci_high'}.issubset(group.columns):
        return
    low = group.ci_low.to_numpy(float)
    high = group.ci_high.to_numpy(float)
    valid = np.isfinite(low) & np.isfinite(high) & (low <= high)
    if floor is not None:
        valid &= high > 0
        low, high = np.maximum(low, floor), np.maximum(high, floor)
    if log10:
        low, high = np.log10(low), np.log10(high)
    ax.fill_between(np.asarray(x, dtype=float), low, high, where=valid,
                    color=color, alpha=alpha, linewidth=0, zorder=1)


def _draw_survival(a, tables):
    for curve_id, group in tables['survival'].groupby('curve_id', sort=False):
        group = group.sort_values('age')
        if curve_id == 'hmd':
            color, label = 'black', 'Sweden 2019 period data'
        else:
            param = group.param.iloc[0]
            color, label = COLORS[param], f'{LABELS[param]}, CV {100*group.cv.iloc[0]:g}%'
        y = group.conditional_survival.where(group.conditional_survival > 0)
        _band(a, group, group.age, color, floor=2e-5)
        a.plot(group.age, y, color=color, lw=3, label=label)
    entry = int(tables['survival'].age.min())
    a.set(yscale='log', xlim=(entry, 135), xticks=[90, 100, 110, 120, 130],
          ylim=(2e-5, 1.1), xlabel='Age [years]',
          ylabel=f'Conditional survival\nfrom age {entry}',
          title='Late-life survival is consistent with\nheterogeneity in robustness parameters')
    handles, labels = a.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: 0 if labels[i].startswith('Sweden') else 1)
    a.legend([handles[i] for i in order], [labels[i] for i in order],
             loc='upper right', frameon=True, facecolor='white',
             edgecolor='none', framealpha=.94, fontsize=26,
             labelspacing=.25, borderpad=.3, handlelength=1.7, handletextpad=.5)


def _draw_tail(ax, tables, name):
    xkey, scale = ('cv', 100) if name == 'tails_cv' else ('factor', 1)
    for param in COLORS:
        group = tables[name].loc[tables[name].param == param].sort_values(xkey)
        _band(ax, group, group[xkey]*scale, COLORS[param])
        ax.plot(group[xkey]*scale, group.tail_age, color=COLORS[param], lw=3,
                marker='o', ms=4, label=LABELS[param])
    ax.set_ylabel('Top 0.01% survivors [years]')
    ax.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=.92, loc='best')

    if name == 'tails_cv':
        ax.set(xlim=(0, 20), xticks=[0, 5, 10, 15, 20],
               xlabel='Parameter heterogeneity (CV, %)',
               title='Upper lifespan tail is sensitive to\nheterogeneity in senogenic parameters')
    else:
        ax.set(xlim=(.85, 1.15), xticks=[.85, .95, 1, 1.05, 1.15],
               xlabel='Parameter factor (relative to baseline)',
               title='Upper lifespan tail is sensitive to\nchanges in senogenic parameters')
        ax.axvline(1, color='#777777', lw=1.3, ls=':', zorder=0)


def _draw_siblings(d, axes, tables):
    sib = tables['siblings']
    columns = ['mortality'] + [c for c in ('ci_low', 'ci_high') if c in sib]
    values = sib.loc[sib.age.between(48, 110), columns].to_numpy(float).ravel()
    logs = np.log10(values[np.isfinite(values) & (values > 0)])
    lo, hi = (min(-2.5, logs.min()), max(0., logs.max())) if logs.size else (-2.5, 0.)
    _draw_empirical(d)
    for i, (ax, param) in enumerate(zip(axes, PARAMETERS)):
        for cohort, color, alpha in [('good', COLORS[param], 1),
                                     ('bad', COLORS[param], 1)]:
            group = tables['siblings'].loc[(tables['siblings'].param == param) &
                                           (tables['siblings'].cohort == cohort)].sort_values('age')
            _band(ax, group, group.age, color, floor=10**(lo-.08), log10=True, alpha=.14*alpha)
            points = group.loc[group.age.le(110) & group.age.mod(2).eq(0)]
            ax.scatter(points.age, np.log10(points.mortality), s=25,
                       facecolors='white' if cohort == 'good' else color,
                       edgecolors=color, linewidths=1.1, zorder=4)
        titles = {'Xc': r'Threshold ($X_c$)', 'epsilon': r'Noise ($\epsilon$)',
                  'eta': r'Production ($\eta$)', 'beta': r'Removal ($\beta$)'}
        ax.set(title=titles[param], xlim=(48, 112),
               xticks=[50, 70, 90, 110])
        if i >= 2:
            ax.set_xlabel('Age [years]')
        if i % 2 == 0:
            ax.set_ylabel(r'$\log_{10}$ mortality' + '\n' + r'[year$^{-1}$]')
    for ax in axes:
        ax.set_ylim(lo-.08, hi+.15)
    axes[0].text(1.06, 1.18, 'Robustness heterogeneity', transform=axes[0].transAxes,
                 fontsize=24, color=COLORS['Xc'], ha='center')
    axes[2].text(1.06, 1.18, 'Senogenic heterogeneity', transform=axes[2].transAxes,
                 fontsize=24, color=COLORS['eta'], ha='center')
    for i, (ax, param) in enumerate(zip(axes, PARAMETERS)):
        good = sib.loc[(sib.param == param) & (sib.cohort == 'good')].sort_values('age')
        bad = sib.loc[(sib.param == param) & (sib.cohort == 'bad')].sort_values('age')
        age = 106 if i < 2 else 101
        yg = float(np.interp(age, good.age, np.log10(good.mortality)))
        yb = float(np.interp(age, bad.age, np.log10(bad.mortality)))
        label = 'convergence' if i < 2 else 'no convergence'
        # Leave the arrowhead beneath both trajectories, as in the manuscript.
        text_y = -1.8 if i < 2 else -3.4
        ax.annotate(label, xy=(age, min(yg, yb)-.30), xytext=(94, text_y),
                    fontsize=16, ha='center', color='#333333',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.25',
                                    color='#555555', lw=1.2), zorder=6)
        if i in (1, 3):
            ax.tick_params(labelleft=False)
        if i < 2:
            ax.tick_params(labelbottom=False)
    ax = axes[0]
    for cohort, label, offset in [('bad', 'siblings of short-lived', .27),
                                  ('good', 'siblings of long-lived', -.26)]:
        group = sib.loc[(sib.param == 'Xc') & (sib.cohort == cohort)].sort_values('age')
        age = 62
        y = float(np.interp(age, group.age, np.log10(group.mortality)))
        slope = (float(np.interp(age+20, group.age, np.log10(group.mortality))) - y) / 20
        position = ax.get_position()
        fig_width, fig_height = ax.figure.get_size_inches()
        aspect = position.height*fig_height / (position.width*fig_width)
        angle = np.degrees(np.arctan(slope*64/(hi-lo+.23)*aspect))
        ax.text(age, y+offset, label, rotation=angle, rotation_mode='anchor',
                fontsize=16, color='#222222', ha='left', va='center', zorder=7)


def _sibling_axes(fig, slot):
    bottom = slot.subgridspec(1, 2, width_ratios=[1, 1.65], wspace=.25)
    d = fig.add_subplot(bottom[0, 0])
    right = bottom[0, 1].subgridspec(2, 2, hspace=.32, wspace=.12)
    axes = [fig.add_subplot(right[i//2, i%2]) for i in range(4)]
    for ax in [d, *axes]:
        _style(ax)
    _letter(d, 'c')
    _letter(axes[0], 'd')
    return d, axes


def _save(fig, output_dir, stem, pdf):
    fig.savefig(output_dir / f'{stem}.png', dpi=220)
    if pdf:
        fig.savefig(output_dir / f'{stem}.pdf')


def render(data_dir: Path, output_dir: Path, pdf: bool = False, panels: bool = False) -> Path:
    """Save the composite PNG, with separate panels only on explicit request.

    Composite a = CV/tail, b = survival, c = empirical siblings,
    d = modeled siblings. Mean-parameter shifts are Supplementary Figure 1.
    Optional PDF exports retain editable fonts. Returns the composite PNG path.
    ci_low/ci_high are optional producer-supplied bounds, never inferred here.
    Parameter identities are eta, beta, Xc, epsilon; CV is a fraction.
    """
    tables = _read_inputs(Path(data_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {'font.family': 'DejaVu Sans', 'font.size': 20, 'axes.labelsize': 32,
                'axes.titlesize': 26, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
                'legend.fontsize': 26, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                'svg.fonttype': 'none'}
    drawers = [lambda ax: _draw_tail(ax, tables, 'tails_cv'),
               lambda ax: _draw_survival(ax, tables)]
    with mpl.rc_context(settings):
        fig = plt.figure(figsize=(24, 21))
        try:
            outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.45], hspace=.40,
                                     left=.095, right=.985, bottom=.075, top=.935)
            top = outer[0].subgridspec(1, 2, wspace=.20)
            for i, (letter, draw) in enumerate(zip('ab', drawers)):
                ax = fig.add_subplot(top[0, i])
                _style(ax)
                _letter(ax, letter)
                draw(ax)
            d, axes = _sibling_axes(fig, outer[1])
            _draw_siblings(d, axes, tables)
            _save(fig, output_dir, 'Fig2', pdf)
        finally:
            plt.close(fig)
        if not panels:
            return output_dir / 'Fig2.png'
        for letter, stem, draw in zip('ab', ('fig2a', 'fig2b'), drawers):
            fig, ax = plt.subplots(figsize=(8.5, 6.5), layout='constrained')
            try:
                _style(ax)
                _letter(ax, letter)
                draw(ax)
                _save(fig, output_dir, stem, pdf)
            finally:
                plt.close(fig)
        fig = plt.figure(figsize=(24, 12), layout='constrained')
        try:
            d, axes = _sibling_axes(fig, fig.add_gridspec(1, 1)[0])
            _draw_siblings(d, axes, tables)
            _save(fig, output_dir, 'fig2cd', pdf)
        finally:
            plt.close(fig)
    return output_dir / 'Fig2.png'


def render_mean_shifts(data_dir: Path, output_dir: Path) -> Path:
    """Render Supplementary Figure 1 from the unchanged mean-shift table."""
    tables = _read_inputs(Path(data_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {'font.family': 'DejaVu Sans', 'font.size': 20,
                'axes.labelsize': 22, 'axes.titlesize': 23,
                'xtick.labelsize': 18, 'ytick.labelsize': 18,
                'legend.fontsize': 17}
    with mpl.rc_context(settings):
        fig, ax = plt.subplots(figsize=(10, 7.5), layout='constrained')
        try:
            _style(ax)
            _draw_tail(ax, tables, 'tails_factor')
            path = output_dir / 'SuppFig1.png'
            fig.savefig(path, dpi=220)
        finally:
            plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pdf', action='store_true', help='Also save editable-text PDF')
    parser.add_argument('--panels', action='store_true', help='Also export separate panel PNGs')
    args = parser.parse_args()
    print(render(args.data_dir, args.output_dir, pdf=args.pdf, panels=args.panels))


if __name__ == '__main__':
    main()
