#!/usr/bin/env python3
"""Readable original-sample and bootstrap summaries for NHANES likelihood fits."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
HERE = PROJECT / 'results' / 'nhanes'
OUT = PROJECT / 'Figures' / 'ExtendedDataFigure3'
PAIRS = ('eta+beta', 'Xc+eta', 'Xc+beta', 'epsilon+beta', 'epsilon+eta')
LABELS = {
'diet__0':'Diet: poor', 'diet__1':'Diet: good',
'number_of_friends__0':'Friends: none', 'number_of_friends__1':'Friends: one or more',
'income__0':'Income: Q1 (lowest)', 'income__1':'Income: Q2',
'income__2':'Income: Q3', 'income__3':'Income: Q4 (highest)',
'alcohol__0':'Alcohol: >4 drinks/day', 'alcohol__1':'Alcohol: 0–1 drink/day',
'physical_activity__0':'Activity: none', 'physical_activity__1':'Activity: some',
'sleep_duration__0':'Sleep: 1–<5 h', 'sleep_duration__1':'Sleep: 5–<7 h',
'sleep_duration__2':'Sleep: 7–<9 h', 'sleep_duration__3':'Sleep: ≥9 h',
'sleep_frailty__0':'Sleep frailty: Q4 (highest)', 'sleep_frailty__1':'Sleep frailty: Q1 (lowest)',
'church_frequency__0':'Attendance: never',
'church_frequency__1':'Attendance: sometimes',
'church_frequency__2':'Attendance: weekly',
'education_level__0':'Education: no high school',
'education_level__1':'Education: some college',
}
HEADERS = ['Exposure group', 'Participants\n(deaths)', 'Xc\nfactor', 'ε\nfactor', 'Best\nΔAIC',
           'Best senogenic\nΔAIC', 'Robustness – senogenic\nΔAIC',
           'Within\n2 AIC?', 'Bootstrap\ncompetitive (%)']


def main(table_only=False):
    raw = pd.read_csv(HERE / 'bootstrap_precision_group_raw.csv')
    order = list(json.loads((PROJECT/'analysis/model_fits/nhanes/inputs/groups.json').read_text()))
    assert len(raw) == 1150 and not raw.duplicated(['replicate','group']).any()
    assert set(raw.group) == set(order) and raw.groupby('replicate').size().eq(23).all()
    assert set(raw.replicate) == set(range(1,51))
    assert (raw.high_competitive_either == raw.high_competitive_either_vs_requested).all()
    counts = raw.groupby('replicate').high_competitive_either.sum().astype(int)
    singles = pd.read_csv(HERE / 'single_senogenic/comparisons_primary320.csv').set_index('group')
    assert singles.index.is_unique and set(singles.index) == set(order)
    rows=[]
    for group in order:
        result=json.loads((HERE/'results_high'/f'{group}.json').read_text())
        models=result['models']; key=min(('Xc','epsilon'), key=lambda k:models[k]['AIC'])
        chosen=models[key]; pair_aic=min(models[k]['AIC'] for k in PAIRS)
        frame=raw[raw.group.eq(group)]
        gap=float(chosen['AIC']-pair_aic)
        fold=chosen['params'][key]/result['baseline'][key]
        single = singles.loc[group]
        assert abs(gap - single.robustness_vs_pair) < 1e-8
        rows.append(dict(group=group,label=LABELS[group],n=result['n'],deaths=result['deaths'],
            preferred_single=key,fold=fold,mex=chosen['params']['mex'],delta_aic=gap,
            competitive=gap<=2,competitive_repeats=int(frame.high_competitive_either.sum()),
            bootstrap_percent=100*frame.high_competitive_either.mean(),
            Xc_fold=models['Xc']['params']['Xc']/result['baseline']['Xc'],
            epsilon_fold=models['epsilon']['params']['epsilon']/result['baseline']['epsilon'],
            delta_aic_Xc=models['Xc']['AIC']-pair_aic,
            delta_aic_epsilon=models['epsilon']['AIC']-pair_aic,
            delta_aic_best_senogenic=float(single.senogenic_vs_pair),
            delta_aic_robustness_minus_senogenic=-float(single.delta_senogenic_vs_robustness)))
    summary=pd.DataFrame(rows)
    assert summary.competitive.sum()==22
    assert abs(counts.mean()-18.84)<1e-12
    assert (summary.delta_aic_best_senogenic <= 2).sum() == 10
    assert (summary.delta_aic_robustness_minus_senogenic < 0).sum() == 17
    assert (summary.delta_aic_robustness_minus_senogenic < -2).sum() == 16
    table_summary = summary.sort_values('Xc_fold', ascending=False).copy()
    table_summary['Xc_percent_change'] = 100 * (table_summary.Xc_fold - 1)
    table_summary.to_csv(HERE/'extended_data_table1.csv',index=False)
    display=[[r.label,f'{r.n:,} ({r.deaths:,})',
              f'{r.Xc_fold:.2f}×', f'{r.epsilon_fold:.2f}×',
              f'{r.delta_aic:+.2f}', f'{r.delta_aic_best_senogenic:+.2f}',
              f'{r.delta_aic_robustness_minus_senogenic:+.2f}',
              'Yes' if r.competitive else 'No',
              f'{r.bootstrap_percent:.0f}'] for r in table_summary.itertuples()]
    (HERE/'extended_data_table1.json').write_text(json.dumps(dict(headers=HEADERS,rows=display),indent=2))
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.labelsize':15,
                        'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':13,
                        'pdf.fonttype':42,'ps.fonttype':42})
    OUT.mkdir(parents=True,exist_ok=True)
    if not table_only:
        fig=plt.figure(figsize=(8.5,10.0))
        ax=fig.add_axes([.12,.79,.81,.165])
        vals=counts.to_numpy();bins=np.arange(min(vals)-.5,24.5,1)
        hist,edges=np.histogram(vals,bins=bins)
        ax.bar((edges[:-1]+edges[1:])/2,hist,width=.78,color='#3D7E91',edgecolor='white',linewidth=.6)
        ax.axvline(22,color='#333333',linestyle='--',lw=1.8,label='Observed dataset: 22')
        ax.axvline(vals.mean(),color='#B45E2E',linestyle=':',lw=2.2,label=f'Bootstrap mean: {vals.mean():.1f}')
        ax.set_xlim(min(vals)-.8,23.5);ax.set_ylim(0,max(hist)*1.22)
        ax.set_xticks(np.arange(int(min(vals)),24,2));ax.yaxis.set_major_locator(MaxNLocator(integer=True,nbins=4))
        ax.set_xlabel('Groups with a competitive simple fit (out of 23)',labelpad=8)
        ax.set_ylabel('Number of\nbootstrap repeats',labelpad=8)
        ax.legend(loc='upper left',frameon=False,handlelength=2.2)
        ax.spines[['top','right']].set_visible(False)
        fig.text(.025,.97,'a',fontweight='normal',fontsize=21)
        ax=fig.add_axes([.445,.075,.47,.635])
        y=np.arange(len(order));pct=summary.bootstrap_percent.to_numpy()
        ax.barh(y,pct,height=.60,color='#3D7E91',zorder=2)
        for yy,value in zip(y,pct):ax.text(value+1.7,yy,f'{value:.0f}',va='center',fontsize=13)
        ax.set_yticks(y,summary.label,fontsize=13.5);ax.invert_yaxis()
        ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel('Bootstrap repeats with a\ncompetitive simple fit (%)',labelpad=8)
        ax.tick_params(axis='y',length=0,pad=9)
        ax.grid(axis='x',color='#E4E8EB',linewidth=.7,zorder=0)
        ax.spines[['top','right','left']].set_visible(False)
        fig.text(.025,.73,'b',fontweight='normal',fontsize=21)
        OUT.mkdir(parents=True,exist_ok=True)
        base=OUT/'ExtDataFig3'
        for ext in ['png']:fig.savefig(base.with_suffix('.'+ext),dpi=300,bbox_inches='tight',pad_inches=.12,facecolor='white')
        plt.close(fig)
    audit=dict(groups=23,original_competitive=int(summary.competitive.sum()),bootstrap_repeats=50,
        mean=float(counts.mean()),percentiles=np.quantile(counts,[.025,.975]).tolist(),
        all8_classifications_identical=True,
        table_order='Descending unrounded Xc factor',
        table_parameters='Separate Xc+mex and epsilon+mex factors relative to the full cohort',
        single_senogenic_comparisons=dict(competitive_vs_pairs=10,
            robustness_lower_AIC=17,robustness_advantage_over_2=16,
            senogenic_advantage_over_2=0,bootstrapped=False),
        new_column_definitions=dict(
            delta_aic_best_senogenic='min(AIC_eta,AIC_beta) minus best paired AIC',
            delta_aic_robustness_minus_senogenic='min(AIC_Xc,AIC_epsilon) minus min(AIC_eta,AIC_beta); negative favors robustness'),
        selection='Lower AIC of Xc+mex and epsilon+mex, separately selected in every bootstrap repeat',
        figure_note='Histogram sampling variation and groupwise repeat frequencies; no parameter confidence bars',
        table='extended_data_table1.csv')
    (HERE/'table_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--table-only', action='store_true')
    main(table_only=parser.parse_args().table_only)
