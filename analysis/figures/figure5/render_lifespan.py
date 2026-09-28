"""Render Figure 5 from archived covariance bands and observed HMD summaries."""
from pathlib import Path
import json, gzip
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'results/historical'
YEARS=[*range(1800,2016,5),2019]
FUTURE=[2019,*range(2020,2101,5)]
METRICS={'Mean':None,'Top 10%':'0.1','Top 1%':'0.01','Top 0.01%':'0.0001'}

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':20,'axes.labelsize':23,'xtick.labelsize':19,'ytick.labelsize':19,'legend.fontsize':15,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def bandline(ax,x,stats,color='black',linestyle='-',alpha=.12,marker=None,lw=2.2):
    point,low,high=stats
    if low is not None: ax.fill_between(x,low,high,color=color,alpha=alpha,linewidth=0,zorder=1)
    ax.plot(x,point,color=color,ls=linestyle,lw=lw,marker=marker,ms=5,zorder=3)
    return np.r_[point,low,high] if low is not None else point


def save(fig,out,name,dpi):
    # Direct Matplotlib vector/text export; no placed-page assembly or text outlines.
    fig.savefig(out/f'{name}.png',dpi=dpi)
    plt.close(fig)


def lifespan_panel(data,out,dpi):
    fig,ax=plt.subplots(figsize=(13,7.6))
    ax.axvspan(2019,2100,color='.88',alpha=.48,zorder=0)
    ax.axvline(2019,color='.45',ls=':',lw=1.6)
    years=data.hmd_years
    fit_years=[y for y in YEARS if y>=1900]
    extent=[]
    for label,key in METRICS.items():
        obs=np.array([data.hmd_annual[y]['mean_attained_age'] if key is None else
                      (np.nan if data.hmd_annual[y]['contours'][key] is None else data.hmd_annual[y]['contours'][key])
                      for y in years],dtype=float)
        ax.plot(years,obs,color='.55',lw=2.8,zorder=2);extent.append(obs)
        stats=data.series(fit_years,lambda r,y:data.metric(data.history[r,'Xc',y],label),'Fig5','historical',label)
        extent.append(bandline(ax,fit_years,stats))
        for scenario,ls,alpha in [('linear','-',.09),('exponential','--',.06)]:
            stats=data.series(FUTURE,lambda r,y:data.metric(data.future[r,scenario,y],label),'Fig5',scenario,label)
            extent.append(bandline(ax,FUTURE,stats,linestyle=ls,alpha=alpha))
        # A white textbox keeps whole editable text; no glyph-outlining path effects.
        available=np.isfinite(obs)
        if available.any():
            label_index=np.flatnonzero(available)[np.argmin(np.abs(np.asarray(years)[available]-1956))]
            ax.text(years[label_index],obs[label_index],label,fontweight='bold',fontsize=17,va='center',zorder=8,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1))
    grey=np.array([r['mean_attained_age'] for r in data.naive]);extent.append(grey)
    ax.plot(FUTURE,grey,color='.45',lw=4,zorder=4)
    bounds=np.concatenate(extent);bounds=bounds[np.isfinite(bounds)]
    if not len(bounds): raise ValueError('No finite observations or predictions for Fig5')
    ylim=(min(45,float(bounds.min())-2),max(120,float(bounds.max())+2))
    data.report['fig5_ylim']=list(ylim)
    ax.set(xlim=(1900,2100),ylim=ylim,xlabel='Year',ylabel='Age [years]\n(conditional on reaching age 20)')
    ax.set_xticks(np.arange(1900,2101,20));ax.tick_params(length=6,width=1.2)
    ax.grid(axis='y',color='.2',alpha=.08,lw=.8)
    ax.text(2022,ylim[1]-2,'forecast',rotation=90,va='top',fontsize=13,color='.35')
    first=ax.legend(handles=[Line2D([],[],color='.55',lw=2.8,label='HMD conditional mean / contours'),
                            Line2D([],[],color='black',lw=2.2,label='SR restricted fits')],loc='upper left',frameon=False)
    ax.add_artist(first)
    ax.legend(handles=[Line2D([],[],color='black',lw=2.2,label='Linear Xc'),
                       Line2D([],[],color='black',lw=2.2,ls='--',label='Exponential Xc'),
                       Line2D([],[],color='.45',lw=4,label='Linear observed-mean continuation')],
              loc='lower right',frameon=False,title='Projections')
    fig.subplots_adjust(left=.15,right=.955,bottom=.16,top=.97)
    save(fig,out,'Fig5_conditional20',dpi)


historical_critical=1.95996398454
base=json.loads((DATA/'joint_covariance.json').read_text())['baseline']
OUT=ROOT/'Figures/Figure5';OUT.mkdir(parents=True,exist_ok=True)
class SavedBands:
    def __init__(self):
        self.history={};self.future={};self.rows={}
        for row in pd.read_csv(DATA/'lifespan_bands.csv').to_dict('records'):
            self.rows[row['scenario'],row['year'],row['label']]=row
        self.hmd_annual={row['year']:row for row in json.loads((DATA/'hmd_observed_annual.json').read_text())}
        self.hmd_years=sorted(self.hmd_annual)
        with gzip.open(DATA/'naive_hmd_mean_projection.json.gz','rt') as f:self.naive=[row for row in json.load(f) if row['replicate']==0]
        self.naive.sort(key=lambda row:row['year'])
        self.report={'naive_mean_slope':self.naive[0]['slope'],'band_method':'Per-year Pearson-dispersion adjusted joint sandwich; approximate pointwise 95% delta sensitivity intervals','baseline':base,'status':'Dispersion-adjusted intervals from saved point fits; not empirically coverage calibrated','critical_value':historical_critical,'fig5_validation':'Prior Poisson recovery counts do not calibrate these adjusted bands'}
    def series(self,years,value,panel,scenario,metric,**kwargs):
        rows=[self.rows[scenario,y,metric] for y in years]
        return tuple(np.array([row[k] for row in rows]) for k in ['estimate','ci_low','ci_high'])
    def metric(self,*args):raise RuntimeError('Only saved metric bands should be used')
d=SavedBands()
style();plt.rcParams.update({'font.family':'Arial','font.size':18,'axes.labelsize':21,'axes.titlesize':22,'legend.fontsize':15,'legend.title_fontsize':15,'xtick.labelsize':18,'ytick.labelsize':18,'pdf.fonttype':42,'ps.fonttype':42})
original_save=save
def styled_save(fig,out,name,dpi):
    ax=fig.axes[0];ax.set_title('Extrapolating recent increase in robustness predicts diminishing longevity gains')
    ax.set_ylabel('Lifespan [years]\n(conditional on reaching age 20)');ax.set_ylim(min(45,ax.get_ylim()[0]),max(120,ax.get_ylim()[1]))
    for annotation in ax.texts:
        if annotation.get_text()=='forecast':
            annotation.remove();ax.annotate('forecast',xy=(2019,.995),xycoords=ax.get_xaxis_transform(),xytext=(4,0),textcoords='offset points',rotation=90,ha='left',va='top',fontsize=12,color='.35')
    for child in list(ax.get_children()):
        if isinstance(child,matplotlib.legend.Legend):child.remove()
    first=ax.legend(handles=[Line2D([],[],color='.55',lw=2.8,label='Sweden period contours'),Line2D([],[],color='black',lw=2.2,label=r'SR model: fitted $X_c$ and $m_{ex}$')],loc='upper left',frameon=False);ax.add_artist(first)
    ax.legend(handles=[Line2D([],[],color='black',lw=2.2,label='Linear increase in robustness'),Line2D([],[],color='black',lw=2.2,ls='--',label='Exponential increase in robustness')],title='Model forecasts',loc='lower right',frameon=False).get_title().set_fontweight('bold')
    slope=d.report['naive_mean_slope'];y0=d.naive[0]['mean_attained_age'];x=2042
    ax.text(x,y0+slope*(x-2019)+1,'Linear increase in mean lifespan',color='.55',fontsize=12,rotation=12)
    fig.subplots_adjust(left=.145,right=.98,bottom=.16,top=.88);original_save(fig,out,'Fig5',dpi)
save=styled_save;lifespan_panel(d,OUT,300)
