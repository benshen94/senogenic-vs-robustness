"""Supplementary Figure 1: observed slopes, local tolerance estimates and FP checks."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'results/supplementary1_fp'
COLORS = dict(eta='#008A99', beta='#244C7C', threshold='#C87816')


def main():
    plt.rcParams.update({'font.family':'Arial','font.size':16,'axes.titlesize':18,
                         'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':15,
                         'legend.fontsize':13,'axes.spines.top':False,'axes.spines.right':False})
    curves = pd.read_csv(HERE/'si_hazards.csv').query("grid == 'refined'")
    means = pd.read_csv(HERE/'si_death_bin_means.csv')
    observed = pd.read_csv(ROOT/'results/tables/supplementary_figure1/sweden2019_decade_slopes.csv')
    tolerance = pd.read_csv(HERE/'heterogeneity_tolerance.csv')
    fig, axs = plt.subplots(3,2,figsize=(12,15))
    fig.subplots_adjust(left=.11,right=.97,top=.95,bottom=.07,wspace=.38,hspace=.48)
    ax = axs[0,0]
    ax.axhspan(.8,1.2,color='#E9EEF1')
    ax.axhline(1,color='.3',ls='--',lw=1.4)
    ax.plot(observed.age_mid,observed.slope_ratio_to_mean,'o-',color=COLORS['eta'],lw=2.5,mfc='white')
    ax.set(title='Sweden 2019',xlabel='Age-window midpoint [years]',ylabel='Slope / mean slope',ylim=(.58,1.42),xlim=(50,100))
    ax = axs[0,1]
    ax.axvline(20,color='.35',ls=':',lw=1.5,zorder=0)
    for key,color,ls,label in [('eta',COLORS['eta'],'-',r'Production $\eta$'),
                              ('beta',COLORS['beta'],'--',r'Removal $\beta$'),
                              ('Xc',COLORS['threshold'],'-',r'Threshold $X_c$')]:
        ax.plot(100*tolerance.tolerance_fraction,100*tolerance[f'{key}_cv'],
                ls=ls,color=color,lw=2.5,label=label)
        at20 = tolerance.loc[np.isclose(tolerance.tolerance_fraction,.2),f'{key}_cv'].iloc[0]
        ax.plot(20,100*at20,'o',color=color,mfc='white',ms=5,zorder=4)
    ax.legend(frameon=False,loc='upper left')
    ax.set(title='Approximate allowed heterogeneity',
           xlabel='Allowed selection-induced\nslope reduction (%)',
           ylabel='Parameter CV among\nage-90 survivors (%)',
           ylim=(0,65),xlim=(0,50),xticks=np.arange(0,51,10),yticks=np.arange(0,61,10))
    for j,(name,symbol) in enumerate([('eta',r'$\eta$'),('beta',r'$\beta$')]):
        ax=axs[1,j]
        sub=means[means.parameter==name]
        ax.plot(sub.lifespan_midpoint,sub.mean_parameter,'o-',color=COLORS[name],lw=2.5,mec='white')
        ax.set(title=f'{symbol} among deaths in each interval',xlabel='Lifespan-interval midpoint [years]',ylabel=f'Mean {symbol}',xlim=(37,163))
        ax=axs[2,j]
        sub=curves[(curves.scenario==name)&(curves.age<=120)]
        ax.plot(sub.age,sub.mortality,color=COLORS[name],lw=2.7)
        ax.set_yscale('log')
        ax.set(title=f'20% heterogeneity in {symbol}',xlabel='Age [years]',ylabel=r'Mortality rate [year$^{-1}$]',xlim=(20,120),ylim=(1e-7,1.2))
    for label,ax in zip('abcdef',axs.flat):
        ax.text(-.16,1.06,label,transform=ax.transAxes,fontsize=24,fontweight='bold')
    path=ROOT/'Figures/Supplementary/SuppFig1.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path,dpi=300)
    plt.close(fig)
    print(path)


if __name__=='__main__':
    main()
