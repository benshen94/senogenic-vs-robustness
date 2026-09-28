from fit import save, dataset
import json,time
import numpy as np
from scipy.optimize import minimize
from model import Model,BASE,HERE,objective
from fit import unpack,BOUNDS

def main():
 r=json.loads((HERE/'profiles/00.json').read_text());m=Model(0,'production');d,e=dataset()
 x0=np.array([0.,0.,0.,0.,BASE['CV'],BASE['mex']*1000]);candidates=[]
 for x in [x0,np.array(r['records'][-1]['x'])]:
  candidates.append((objective(m,unpack(x),d,e),x))
 x=min(candidates,key=lambda z:z[0])[1]
 opt=minimize(lambda z:objective(m,unpack(z),d,e),x,bounds=BOUNDS,method='L-BFGS-B',options={'maxiter':70,'ftol':1e-9,'gtol':1e-4,'eps':1e-5})
 p=unpack(opt.x);fine=Model(0,'fine')
 save(HERE/'zero_production_refit.json',dict(params=p,score=float(opt.fun),fine_score=objective(fine,p,d,e),rates=m.rates(p).tolist(),fine_rates=fine.rates(p).tolist(),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev),x=opt.x.tolist()))
 print(opt.fun,opt.success,opt.message,flush=True)

if __name__ == '__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Optional senogenic profile numerical worker')
 parser.add_argument('--recompute',action='store_true')
 args=parser.parse_args()
 if not args.recompute:parser.error('This numerical worker requires --recompute.')
 main()
