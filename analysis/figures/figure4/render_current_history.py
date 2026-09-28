"""One physical canvas for all four panels: shared text scale and aligned axes."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
H=R/'tmp/historical_layout';H.mkdir(parents=True,exist_ok=True)
from analysis.figures.figure4 import plot_helpers as old
old.fig1e.METRICS_PATH=R/'results/historical/age20_metrics_long.csv';old.fig1e.FROM_T=20
metrics=pd.read_csv(old.fig1e.METRICS_PATH);baseline=metrics[(metrics.scenario_id=='central')&(metrics.curve_type=='baseline')].iloc[0]

def curve_label(ax,f,label,x,offset):
 y=np.interp(x,f.year,f.estimate);dx=2.
 p0=ax.transData.transform((x-dx,np.interp(x-dx,f.year,f.estimate)));p1=ax.transData.transform((x+dx,np.interp(x+dx,f.year,f.estimate)))
 angle=np.degrees(np.arctan2(*(p1-p0)[::-1]));rad=np.radians(angle)
 # Perpendicular offset in display points; text rotation follows local tangent.
 delta=(-np.sin(rad)*offset,np.cos(rad)*offset)
 a=ax.annotate(label,(x,y),xytext=delta,textcoords='offset points',ha='center',va='center',rotation=angle,rotation_mode='anchor',fontsize=18,color=old.ROBUSTNESS_PANEL_COLOR,zorder=15)
 return a

def render(country):
 old.apply_fig4_style();plt.rcParams.update({'font.family':'Arial','font.size':20,'axes.labelsize':26,'axes.titlesize':24,'xtick.labelsize':20,'ytick.labelsize':20,'pdf.fonttype':42,'ps.fonttype':42})
 fig=plt.figure(figsize=(16,18.6));w=.335;left=.12;right=.575;abheight=(16*w*1.1/.6)/18.6;abtop=.94;bottom=.08;cdheight=.23
 axes=[fig.add_axes([x,abtop-abheight,w,abheight]) for x in [left,right]]+[fig.add_axes([x,bottom,w,cdheight]) for x in [left,right]]
 data=pd.read_csv(R/f'results/historical/{country.key}_observed_coordinates.csv');data['x_relative_to_sr']=data.median_lifespan/baseline.t_median_absolute;data['y_relative_to_sr']=data.steepness_iqr_absolute/baseline.steepness_iqr_absolute
 for ax,cond in zip(axes[:2],old.CONDITION_LABELS):
  old.draw_new_baseline(ax);ax.set_xlim(*old.NEW_AXIS_LIMITS[cond][0]);ax.set_ylim(*old.NEW_AXIS_LIMITS[cond][1]);ax.set_aspect('equal',adjustable='box')
  sc=old.draw_historical_points(ax,data,cond,x_column='x_relative_to_sr',y_column='y_relative_to_sr',show_colorbar=False)
  old.remove_legend(ax);ax.set_xlabel('Normalized median lifespan\n(2019 SR = 1)',fontsize=25,labelpad=12);ax.set_ylabel('');ax.set_title(old.CONDITION_LABELS[cond],fontsize=24,pad=22)
 axes[0].set_ylabel('Normalized steepness\n(2019 SR = 1)',fontsize=26,labelpad=16)
 old.build_shared_legend(axes[0],country=country,data=data)
 for tx in axes[0].get_legend().get_texts():tx.set_fontsize(15.5)
 cax=fig.add_axes([.927,abtop-abheight,.012,abheight]);old.draw_vector_year_colorbar(cax,sc);cax.set_ylabel('Year',fontsize=22,labelpad=4);cax.tick_params(labelsize=16)
 if country.key=='sweden':
  c=pd.read_csv(R/'results/historical/sweden_external_mortality.csv');d=pd.read_csv(R/'results/historical/sweden_threshold_fits.csv')
 else:
  c=pd.read_csv(R/'results/historical/denmark_external_mortality.csv');d=pd.read_csv(R/'results/historical/denmark_threshold_fits.csv');d['series']=d.series.replace({'linear':'linear_extrapolation','exponential':'exponential_extrapolation'})
 ax=axes[2];ax.plot(c.year,c.estimate,color='#B84A4F',lw=2.1,marker='^',ms=4);ax.fill_between(c.year,c.ci_low,c.ci_high,color='#B84A4F',alpha=.16,lw=0);ax.set_yscale('log');ax.set_xlim(1795 if country.key=='sweden' else 1830,2025);ax.set_ylabel(r'Extrinsic mortality [year$^{-1}$]',fontsize=26,labelpad=12);ax.set_title('Extrinsic mortality over time',fontsize=24,pad=22)
 ax=axes[3];ax.axvspan(2019,2100,color='#EDEDED',alpha=.8,zorder=0);ax.axvline(2019,color='#B8B8B8',ls='--',lw=1.4);ax.axhline(1,color='#B8B8B8',ls='--',lw=1.4)
 for series,ls in [('data','-'),('linear_extrapolation','-'),('exponential_extrapolation','--')]:
  f=d[d.series==series];ax.fill_between(f.year,f.ci_low,f.ci_high,color=old.ROBUSTNESS_PANEL_COLOR,alpha=.13 if series=='data' else .1,lw=0);ax.plot(f.year,f.estimate,color=old.ROBUSTNESS_PANEL_COLOR,lw=2.4,ls=ls,marker='^' if series=='data' else None,ms=4.5)
 ax.set_xlim(1795 if country.key=='sweden' else 1830,2100);ax.set_ylim(d.ci_low.min()-.04,d.ci_high.max()+.09)
 ax.set_ylabel(r'Threshold $X_c$ factor' if country.key=='sweden' else r'$X_c$ / Swedish 2019 $X_c$',fontsize=26,labelpad=12);ax.set_title('Fitted robustness over time',fontsize=24,pad=22);ax.text(2058,ax.get_ylim()[0]+.1,'extrapolation',ha='center',color='#666666',fontsize=17)
 for ax in axes:
  ax.tick_params(axis='both',labelsize=20,length=6,width=1.4);ax.spines[['top','right']].set_visible(False)
  for sp in ['left','bottom']:ax.spines[sp].set_linewidth(1.4)
 for ax in axes[2:]:ax.set_xlabel('Year',fontsize=26,labelpad=12);ax.xaxis.set_label_coords(.5,-.14)
 fig.canvas.draw()
 for letter,ax in zip('abcd',axes):
  pos=ax.get_position();fig.text(pos.x0-.055,pos.y1+.027,letter,fontsize=36,ha='left',va='center')
 curve_label(axes[3],d[d.series=='exponential_extrapolation'],'Exponential',2069,17)
 curve_label(axes[3],d[d.series=='linear_extrapolation'],'Linear',2070,-15)
 fig.canvas.draw()
 # A physical-layout assertion prevents the prior composite scaling error.
 assert abs(axes[2].get_position().y0-axes[3].get_position().y0)<1e-12
 assert abs(axes[2].get_position().height-axes[3].get_position().height)<1e-12
 name='Fig4' if country.key=='sweden' else 'Extended_Data_Fig3_Denmark'
 out=R/('Figures/Figure4/Fig4.png' if country.key=='sweden' else 'Figures/ExtendedDataFigure3/ExtDataFig3.png')
 out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=200)
 (H/(name+'_layout.json')).write_text(json.dumps({'canvas_inches':[16,18.6],'panel_font_pt':36,'axis_font_pt':26,'tick_font_pt':20,'positions':{k:list(ax.get_position().bounds) for k,ax in zip('abcd',axes)},'aligned_cd_axes':True,'source_values_unchanged':True},indent=2))
 plt.close(fig)
for c in old.COUNTRIES:render(c)
print('Rendered both aligned composites from saved curves and fits.')
