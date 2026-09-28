"""Bounded conservative profiles; all identifiable mean parameters compensate."""
import os,json,time
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from model import Model,BASE,HERE,DATA,objective

def save(path, value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2))

def dataset():
 frame=pd.read_csv(DATA/'sweden_2019_input.csv')
 return frame.deaths.to_numpy(),frame.exposure.to_numpy()
CVS=[0.,.025,.04,.05,.075,.1,.15,.2,.25]
# Fix kappa as a damage-unit convention. Scaling x by c scales eta,beta,
# kappa,Xc by c and epsilon by c^2 without changing survival.
BOUNDS=[(-2.08,2.08),(-1.386,1.386),(-3.12,3.12),(-1.87,1.87),(0.,.60),(0.,20.)]
def unpack(x):
 p=dict(BASE);p.update(eta=BASE['eta']*np.exp(x[0]),epsilon=BASE['epsilon']*np.exp(x[2]),Xc=BASE['Xc']*np.exp(x[3]),CV=float(x[4]),mex=float(x[5])/1000)
 p['beta']=p['eta']*(BASE['beta']/BASE['eta'])*np.exp(x[1]);return p

def main():
 idx=int(os.environ.get('LSB_JOBINDEX','1'))-1;cv=CVS[idx];frame=pd.read_csv(DATA/'sweden_2019_input.csv');d=frame.deaths.to_numpy();e=frame.exposure.to_numpy()
 out=HERE/'profiles';out.mkdir(exist_ok=True);records=[];started=time.monotonic()
 x0=np.array([0,0,0,0,BASE['CV'],1000*BASE['mex']]);rng=np.random.default_rng(710+idx)
 starts=[x0,x0+np.array([.2,-.1,.1,0,-.06,0])]
 model=Model(cv,'screen')
 for j,x in enumerate(starts):
  opt=minimize(lambda z:objective(model,unpack(z),d,e),x,bounds=BOUNDS,method='L-BFGS-B',options={'maxiter':110,'ftol':1e-9,'gtol':2e-5,'finite_diff_rel_step':1e-4,'eps':1e-4})
  rec=dict(stage='screen',start=j,score=float(opt.fun),x=opt.x.tolist(),params=unpack(opt.x),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev))
  records.append(rec);save(out/f'{idx:02d}_progress.json',records);print(json.dumps(rec),flush=True)
 best=min(records,key=lambda r:r['score']);model=Model(cv,'refine')
 opt=minimize(lambda z:objective(model,unpack(z),d,e),best['x'],bounds=BOUNDS,method='L-BFGS-B',options={'maxiter':70,'ftol':2e-8,'gtol':1e-4,'eps':3e-5})
 p=unpack(opt.x);records.append(dict(stage='refine',score=float(opt.fun),x=opt.x.tolist(),params=p,success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev)))
 prod=Model(cv,'production');rates=prod.rates(p)
 result=dict(tau_cv=cv,params=p,score=objective(prod,p,d,e),refine_score=float(opt.fun),rates=rates.tolist(),ages=list(range(20,110)),deaths=d.tolist(),exposures=e.tolist(),records=records,seconds=time.monotonic()-started,bound_hits=[i for i,(v,b) in enumerate(zip(opt.x,BOUNDS)) if min(v-b[0],b[1]-v)<1e-4])
 save(out/f'{idx:02d}.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['records','rates','deaths','exposures','ages']}),flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--recompute',action='store_true')
 args=parser.parse_args()
 if not args.recompute:parser.error('Profile optimization requires --recompute.')
 main()
