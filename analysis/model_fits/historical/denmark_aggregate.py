"""Rebuild Danish shared-baseline covariance and source tables from saved fits."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from inference import pearson_dispersion
ROOT=Path(__file__).resolve().parent;HERE=ROOT/'denmark'
Z=1.959963984540054
YEARS=list(range(1835,2016,5))+[2019]

def aggregate():
 b=json.loads((ROOT/'point/baseline.json').read_text())['params']
 swpath=ROOT/'joint_covariance.json'
 sw=json.loads(swpath.read_text());assert sw['baseline']==b
 if 'Pearson' not in sw['diagnostics']['method']:
  raise ValueError('Danish production intervals require the adjusted Swedish covariance')
 cb=np.asarray(sw['covariance'])[:4,:4]
 fits={y:json.loads((HERE/f'point/{y}.json').read_text()) for y in YEARS}
 n=4+2*len(YEARS);cov=np.zeros((n,n));cov[:4,:4]=cb;maps=[];conds=[]
 q=np.r_[[np.log(b['Xc']),np.log(b['epsilon']),b['CV'],1000*b['mex']],*[ [np.log(fits[y]['params']['Xc']),1000*fits[y]['params']['mex']] for y in YEARS]]
 for i,y in enumerate(YEARS):
  fit=fits[y];assert fit['success'],(y,'failed fit')
  j=np.array(fit['jacobian']);mu=np.array(fit['age_diagnostics']['expected_deaths']);keep=np.asarray(fit['age_diagnostics']['exposure'])>0;j=j[keep];mu=mu[keep]
  fit['dispersion']=pearson_dispersion(np.asarray(fit['age_diagnostics']['deaths'])[keep],mu,2)
  own=j[:,[0,3]];inherited=np.zeros((len(j),4));inherited[:,[1,2]]=j[:,[1,2]]
  a=own.T@own;ai=np.linalg.inv(a);g=-ai@own.T@inherited;noise=ai@((own.T/mu)@own)@ai.T
  sl=slice(4+2*i,6+2*i);cov[sl,sl]=fit['dispersion']['variance_multiplier']*noise;maps.append(g);conds.append(np.linalg.cond(a))
 for i,g in enumerate(maps):
  sl=slice(4+2*i,6+2*i);cov[sl,:4]=g@cb;cov[:4,sl]=cb@g.T
  for k,h in enumerate(maps):cov[sl,4+2*k:6+2*k]+=g@cb@h.T
 eig=np.linalg.eigvalsh(cov);assert eig.min()>-1e-10
 def logratio(q,y,scenario):
  vals=np.array([q[4+2*YEARS.index(t)] for t in YEARS if t>=1980]);ts=np.array([t for t in YEARS if t>=1980]);anchor=q[4+2*YEARS.index(2019)]
  if scenario=='data':val=q[4+2*YEARS.index(y)]
  elif scenario=='linear':val=np.log(np.exp(anchor)+np.polyfit(ts-2019,np.exp(vals),1)[0]*(y-2019))
  else:val=anchor+np.polyfit(ts-2019,vals,1)[0]*(y-2019)
  return val-q[0]
 rows=[]
 for sc,ys in [('data',YEARS),('linear',[2019,*range(2020,2101,5)]),('exponential',[2019,*range(2020,2101,5)])]:
  for y in ys:
   val=logratio(q,y,sc);grad=np.zeros(n)
   for i in range(n):
    hi=q.copy();lo=q.copy();hi[i]+=1e-5;lo[i]-=1e-5;grad[i]=(logratio(hi,y,sc)-logratio(lo,y,sc))/2e-5
   se=np.sqrt(max(0,grad@cov@grad));rows.append(dict(year=y,series=sc,estimate=np.exp(val),log_se=se,ci_low=np.exp(val-Z*se),ci_high=np.exp(val+Z*se)))
 out=HERE/'figures';out.mkdir(exist_ok=True);frame=pd.DataFrame(rows);frame.to_csv(out/'Fig4d_values.csv',index=False)
 pd.DataFrame([dict(year=y,**fits[y]['params'],score=fits[y]['score'],success=fits[y]['success'],bound_hits=str(fits[y]['bound_hits']),zero_exposure_bins=sum(e==0 for e in fits[y]['age_diagnostics']['exposure'])) for y in YEARS]).to_csv(HERE/'historical_fits.csv',index=False)
 c=pd.read_csv(ROOT/'inputs/denmark_panel_c.csv');c=c[c.year.between(1835,2020)].copy();c.to_csv(out/'Fig4c_values.csv',index=False)
 diagnostics=dict(smallest_covariance_eigenvalue=float(eig.min()),max_year_bread_condition=float(max(conds)),critical_value=Z,baseline_covariance_source=str(swpath.relative_to(ROOT)),baseline_covariance_source_sha256=hashlib.sha256(swpath.read_bytes()).hexdigest(),method='Per-year Pearson-dispersion adjusted expected-bread joint sandwich, Danish counts and adjusted Swedish reference',boundary_years=[y for y in YEARS if fits[y]['bound_hits']],validation='No Denmark coverage calibration; approximate pointwise sensitivity intervals, not validated by Swedish recovery; full local sandwich ignores active-boundary nonregularity')
 (HERE/'joint_covariance.json').write_text(json.dumps(dict(baseline=b,years=YEARS,q=q.tolist(),covariance=cov.tolist(),diagnostics=diagnostics),indent=2))
 (HERE/'dispersion.json').write_text(json.dumps([dict(year=y,**fits[y]['dispersion'],boundary=fits[y]['bound_hits']) for y in YEARS],indent=2))
 print(f'Wrote Danish covariance and tables for {len(YEARS)} saved fits; no refitting.')
if __name__=='__main__':aggregate()
