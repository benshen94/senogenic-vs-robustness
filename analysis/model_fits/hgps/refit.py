import os,sys,json,csv,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from senogenic_vs_robustness.hgps_likelihood import HGPSLikelihood,NAMES,MODELS
DATA=ROOT/'results/progeria'
HERE=ROOT/'tmp/progeria_refit'
HERE.mkdir(parents=True,exist_ok=True)
old=json.loads((DATA/'initial_fits.json').read_text())['models']
base=json.loads((DATA/'baseline.json').read_text())['params']
records=list(csv.DictReader((ROOT/'data/hgps/records.csv').open()))
for r in records:r['age']=float(r['age']);r['death']=int(r['death'])
j=int(os.environ.get('LSB_JOBINDEX',sys.argv[1] if len(sys.argv)>1 else '1'))-1
if not 0 <= j < len(MODELS):
 raise ValueError('Task index must be from 1 to 10')
active=MODELS[j];ix=[NAMES.index(n) for n in active];prior=next(r for r in old if r['model']=='+'.join(active))
like=HGPSLikelihood(records,[base[k] for k in (*NAMES,'CV')],(120,.05,40))
fine=HGPSLikelihood(records,[base[k] for k in (*NAMES,'CV')],(240,.025,80))
def unpack(x):
 f=np.ones(4);f[ix]=np.exp(x);return f
oldbase=json.loads((DATA/'initial_baseline.json').read_text())['params']
start=np.clip(np.log([prior['factors'][n]*oldbase[n]/base[n] for n in active]),-np.log(20),np.log(20));starts=[start,np.zeros(len(ix)),np.full(len(ix),-2.8),np.full(len(ix),2.8)]
if len(ix)==2:starts.extend([np.array([-2.8,2.8]),np.array([2.8,-2.8])])
if 'eta' in active:
 for other in [-2.,0.,2.]:
  s=start.copy();s[active.index('eta')]=np.log(5)
  if len(ix)==2:s[1-active.index('eta')]=other
  starts.append(s)
traces=[];best=None;t0=time.time()
for s in starts:
 r=minimize(lambda x:like.nll(unpack(x)),s,method='L-BFGS-B',bounds=[(-np.log(20),np.log(20))]*len(ix),options={'maxiter':180,'ftol':1e-10,'gtol':.002,'maxls':30},jac='3-point')
 traces.append(dict(nll=float(r.fun),success=bool(r.success),x=r.x.tolist(),message=str(r.message)))
 if best is None or r.fun<best.fun:best=r
 print(active,round(r.fun,5),flush=True)
r=minimize(lambda x:fine.nll(unpack(x)),best.x,method='L-BFGS-B',bounds=[(-np.log(20),np.log(20))]*len(ix),options={'maxiter':150,'ftol':1e-10,'gtol':.002,'maxls':30},jac='3-point')
f=unpack(r.x);ls,h=fine.curve(f)
out=dict(model='+'.join(active),active=active,nll=float(r.fun),aic=float(2*r.fun+2*len(ix)),success=bool(r.success),message=str(r.message),factors=dict(zip(NAMES,f.tolist())),boundary=[n for n,v in zip(active,r.x) if abs(abs(v)-np.log(20))<.003],coarse_nll=float(best.fun),fine_at_coarse=fine.nll(unpack(best.x)),traces=traces,elapsed=time.time()-t0)
(HERE/(out['model']+'.json')).write_text(json.dumps(out,indent=2));np.savez(HERE/(out['model']+'.npz'),age=fine.t,survival=np.exp(ls));print('DONE',out,flush=True)
