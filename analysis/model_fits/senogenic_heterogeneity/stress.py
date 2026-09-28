from fit import save, dataset
import os,json,time
import numpy as np
from scipy.optimize import minimize
from model import Model,HERE,objective
from fit import unpack

def main():
 idx=[6,8][max(1,int(os.environ.get('LSB_JOBINDEX','1')))-1];cv=[0,.025,.04,.05,.075,.1,.15,.2,.25][idx]
 r=json.loads((HERE/'profiles'/f'{idx:02d}_progress.json').read_text());best=min(r,key=lambda z:z['score'])
 bounds=[(-5,5),(-4,4),(-8,8),(-5,5),(0,.6),(0,20)]
 d,e=dataset();m=Model(cv,'screen')
 opt=minimize(lambda z:objective(m,unpack(z),d,e),best['x'],bounds=bounds,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-8,'gtol':1e-4,'eps':1e-4})
 p=unpack(opt.x);prod=Model(cv,'production')
 res=dict(tau_cv=cv,params=p,screen_score=float(opt.fun),score=objective(prod,p,d,e),rates=prod.rates(p).tolist(),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev),x=opt.x.tolist(),bounds=bounds,bound_hits=[i for i,(v,b) in enumerate(zip(opt.x,bounds)) if min(v-b[0],b[1]-v)<1e-4])
 save(HERE/'stress'/f'{idx:02d}.json',res);print(json.dumps({k:v for k,v in res.items() if k!='rates'}),flush=True)

if __name__ == '__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Optional senogenic profile numerical worker')
 parser.add_argument('--recompute',action='store_true')
 args=parser.parse_args()
 if not args.recompute:parser.error('This numerical worker requires --recompute.')
 main()
