"""Render saved profile results only; no solver or cluster required."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'results/senogenic_heterogeneity'

def main():
 rows=[json.loads(p.read_text()) for p in sorted((HERE/'profiles').glob('[0-9][0-9].json'))]
 if len(rows)!=9:raise RuntimeError(f'Need all nine profiles, found {len(rows)}')
 rows.sort(key=lambda x:x['tau_cv'])
 for i,r in enumerate(rows):
  f=HERE/'selected'/f'{i:02d}.json'
  if f.exists():
   selected=json.loads(f.read_text());r.update(params=selected['params'],score=selected['score'],rates=selected['rates'],selected_source=selected['source'])
 zero_path=HERE/'zero_production_refit.json'
 if zero_path.exists():
  z=json.loads(zero_path.read_text())
  if z['success'] and z['score']<rows[0]['score']:
   rows[0].update(params=z['params'],score=z['score'],rates=z['rates'],production_refined=True)
 base=json.loads((HERE/'baseline_input.json').read_text())['params']
 bounds=[(-2.08,2.08),(-1.386,1.386),(-3.12,3.12),(-1.87,1.87),(0,.60),(0,20)]
 for r in rows:
  p=r['params'];x=[np.log(p['eta']/base['eta']),np.log((p['beta']/p['eta'])/(base['beta']/base['eta'])),np.log(p['epsilon']/base['epsilon']),np.log(p['Xc']/base['Xc']),p['CV'],1000*p['mex']]
  r['bound_hits']=[i for i,(v,b) in enumerate(zip(x,bounds)) if min(v-b[0],b[1]-v)<1e-4]
 out=ROOT/'Figures/ExtendedDataFigure1';out.mkdir(parents=True,exist_ok=True)
 mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':13,'axes.labelsize':19,'axes.titlesize':21,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':12,'axes.linewidth':1.15,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(13.2,6.7),constrained_layout=True)
 a,b=axes;x=np.array([r['tau_cv']*100 for r in rows]);y=np.array([r['score'] for r in rows])
 delta=y-y[0]
 a.plot(x,delta,color='#0B8793',lw=2.7,marker='o',ms=7.4,mec='white',mew=.8)
 a.axhline(0,color='#777777',ls='--',lw=1.2,zorder=0)
 a.set(xlim=(-1,26.5),ylim=(-.7,max(delta)*1.10),xlabel=r'Imposed $\tau_{\rm sen}$ heterogeneity (%)',ylabel=r'Increase in fitting error, $\Delta Q$')
 a.set_xticks([0,5,10,15,20,25]);a.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}%'))
 a.set_title('Higher values indicate a worse fit')
 with (HERE/'relative_fit_error.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['tau_cv','Q','delta_Q_from_zero_heterogeneity'])
  w.writerows((r['tau_cv'],float(q),float(dq)) for r,q,dq in zip(rows,y,delta))
 ages=np.array(rows[0]['ages'])+.5;d=np.array(rows[0]['deaths']);e=np.array(rows[0]['exposures'])
 b.plot(ages,d/e,color='black',lw=4,label='Sweden 2019 period',zorder=5)
 b.plot(ages,rows[0]['rates'],color='#D77A16',lw=4,label='Zero-spread refit',zorder=4)
 for cv,color in zip([.025,.05,.1,.15],mpl.colormaps['viridis'](np.linspace(.28,.82,4))):
  r=min(rows,key=lambda r:abs(r['tau_cv']-cv));b.plot(ages,r['rates'],color=color,lw=1.8,label=f'{100*cv:g}% heterogeneity')
 b.set(yscale='log',xlim=(55,100),ylim=(.0025,.60),xlabel='Age [years]',ylabel=r'Mortality rate [year$^{-1}$]')
 b.set_xticks(np.arange(55,101,5));b.set_title('Mortality after parameter refitting');b.legend(loc='upper left',frameon=False,handlelength=2.2,fontsize=14)
 for ax,label in zip(axes,['a','b']):ax.text(-.12,1.10,label,transform=ax.transAxes,fontsize=22,fontweight='normal',va='top',ha='left')
 fig.savefig(out/'ExtDataFig1.png',dpi=300,bbox_inches='tight');plt.close(fig)
 with (HERE/'profile_summary.csv').open('w') as f:
  fields=['tau_cv','score','refine_score','seconds','bound_hits','eta','beta','epsilon','Xc','CV','mex','kappa'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in rows:w.writerow({**{k:r[k] for k in fields[:5]},**r['params']})
 with (HERE/'mortality_source.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['tau_cv','age_start','deaths','exposures','HMD_rate','model_rate'])
  for r in rows:
   for age,dd,ee,rate in zip(r['ages'],r['deaths'],r['exposures'],r['rates']):w.writerow([r['tau_cv'],age,dd,ee,dd/ee,rate])
 print(out/'ExtDataFig1.png')
if __name__=='__main__':main()
