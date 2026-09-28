"""Optional Denmark Q point fits; run only in the staged rerun directory."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
from fitting import p,score,fit_history
from inference import jacobian
p.score=score
BASE=json.loads((HERE/'point/baseline.json').read_text())['params']
YEARS=list(range(1835,2016,5))+[2019]
RECENT=list(range(1980,2016,5))+[2019]
def dataset(country,year):
 f=pd.read_csv(HERE/'inputs/hmd.csv');f=f[(f.country==country)&(f.year==year)&f.age.between(20,109)].sort_values('age')
 assert np.array_equal(f.age.to_numpy(),p.AGES)
 d,e=f.deaths.to_numpy(float),f.exposure.to_numpy(float)
 assert np.isfinite(d).all() and np.isfinite(e).all() and (d>=0).all() and (e>=0).all() and not ((e==0)&(d>0)).any()
 return d,e

def run(index):
 if not 0 <= index < len(YEARS): raise ValueError('Denmark index must be 0..37')
 begin=time.monotonic();denmark=True;year=YEARS[index]
 folder=HERE/'denmark';dest=folder/f'point/{year}.json'
 if dest.exists(): print('existing',dest,flush=True);return
 model=p.Model();d,e=dataset('DNK' if denmark else 'SWE',year)
 if denmark:
  fit=fit_history(model,BASE,d,e)
  if year in [1835,1980,2019]:
   audit=p.fit(model,BASE,['Xc'],d,e,120,folder/f'point/{year}_starts.json')
   fit['multistart_audit']={k:audit[k] for k in ['score','success','start_score_range','multistart_disagreement']}
   fit['audit_score_difference']=float(fit['score']-audit['score'])
   if audit['score']<fit['score']:fit['params']=audit['params'];fit['score']=audit['score']

 fit.update(year=year,country='DNK' if denmark else 'SWE',objective='Q=sum(D/mu+log(mu)), data-only constant removed',seconds=time.monotonic()-begin)
 rates=model.rates(fit['params']);fit['age_diagnostics']=dict(age=p.AGES.tolist(),deaths=d.tolist(),exposure=e.tolist(),rates=rates.tolist(),expected_deaths=(e*rates).tolist())
 if denmark:fit['jacobian']=jacobian(model,fit['params'],['Xc','epsilon','CV','mex']).tolist()
 p.save(dest,fit);print(json.dumps({k:fit[k] for k in ['year','country','params','score','success','seconds']}),flush=True)
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else int(os.environ['LSB_JOBINDEX'])-1)
