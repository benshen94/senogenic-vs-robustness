from fit import save, dataset
import os,json,time
import numpy as np
from model import Model,HERE,BASE,objective

def main():
 idx=max(1,int(os.environ.get('LSB_JOBINDEX','1')))-1
 r=json.loads((HERE/'profiles'/f'{idx:02d}.json').read_text());d,e=dataset();m=Model(r['tau_cv'],'production')
 candidates=[('stage_'+str(i),x['params']) for i,x in enumerate(r['records'])]+[('anchor',BASE)]
 for j in [idx-1,idx+1]:
  f=HERE/'profiles'/f'{j:02d}.json'
  if f.exists():candidates.append(('neighbor_'+str(j),json.loads(f.read_text())['params']))
 if idx==0:
  candidates.append(('production_refit',json.loads((HERE/'zero_production_refit.json').read_text())['params']))
 if idx==2 and (HERE/'four_wide_refit.json').exists():
  # Separate wider-bound diagnostic; never substitute into the common-bound line.
  pass
 scores=[]
 for name,p in candidates:
  scores.append(dict(source=name,params=p,score=objective(m,p,d,e),rates=m.rates(p).tolist()))
  print(name,scores[-1]['score'],flush=True)
 best=min(scores,key=lambda x:x['score']);result=dict(tau_cv=r['tau_cv'],**best,candidates=scores)
 if idx==4:
  fine=Model(r['tau_cv'],'fine');result.update(fine_score=objective(fine,best['params'],d,e),fine_rates=fine.rates(best['params']).tolist())
 save(HERE/'selected'/f'{idx:02d}.json',result)

if __name__ == '__main__':
 import argparse
 parser=argparse.ArgumentParser(description='Optional senogenic profile numerical worker')
 parser.add_argument('--recompute',action='store_true')
 args=parser.parse_args()
 if not args.recompute:parser.error('This numerical worker requires --recompute.')
 main()
