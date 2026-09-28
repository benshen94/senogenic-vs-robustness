from fit import save, dataset
import json,time
import numpy as np
from scipy.optimize import least_squares
from model import Model,HERE,objective
from fit import unpack

def main():
 r=json.loads((HERE/'profiles/02.json').read_text());d,e=dataset()
 assert np.all(d>0)
 bounds=np.array([[-5,5],[-4,4],[-8,8],[-5,5],[0,.6],[0,20.]])
 m=Model(.04,'refine');calls=0;best=(np.inf,None);start=time.monotonic()
 def residual(x):
  nonlocal calls,best
  p=unpack(x);mu=e*m.rates(p);v=np.log(d/mu);res=np.sign(-v)*np.sqrt(np.maximum(2*(np.expm1(v)-v),0))
  calls+=1;score=float(np.sum(res**2)/2)
  if score<best[0]:
   best=(score,x.copy());save(HERE/'four_checkpoint.json',dict(score=score,x=x.tolist(),params=p,calls=calls))
  return res
 opt=least_squares(residual,r['records'][-1]['x'],bounds=(bounds[:,0],bounds[:,1]),x_scale='jac',max_nfev=45,ftol=1e-6,xtol=1e-5,gtol=1e-4,diff_step=3e-5)
 p=unpack(opt.x);prod=Model(.04,'production');fine=Model(.04,'fine')
 save(HERE/'four_wide_refit.json',dict(params=p,score=objective(prod,p,d,e),refine_score=objective(m,p,d,e),fine_score=objective(fine,p,d,e),rates=prod.rates(p).tolist(),fine_rates=fine.rates(p).tolist(),success=bool(opt.success),message=str(opt.message),nfev=int(opt.nfev),calls=calls,seconds=time.monotonic()-start,x=opt.x.tolist(),bound_hits=[i for i,(v,b) in enumerate(zip(opt.x,bounds)) if min(v-b[0],b[1]-v)<1e-4]))
 print('completed',opt.success,opt.message,calls,flush=True)

if __name__ == '__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Optional senogenic profile numerical worker')
 parser.add_argument('--recompute',action='store_true')
 args=parser.parse_args()
 if not args.recompute:parser.error('This numerical worker requires --recompute.')
 main()
