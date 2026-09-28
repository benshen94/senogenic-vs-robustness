"""Relative-error estimating-equation fitting; distinct from Poisson likelihood.

Q=sum(D/mu+log(mu)) has derivative sum((1-D/mu)*dlogmu).
Its expectation vanishes at the true Poisson mean, unlike random inverse-D
weighted deviance. Zero exposure bins carry no information and are omitted.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import sys,json,time,hashlib,argparse
import numpy as np
HERE=Path(__file__).resolve().parent
import pipeline as p
from scipy.optimize import minimize_scalar,minimize

def score(model,params,d,e):
    keep=e>0
    mu=np.maximum(e[keep]*model.rates(params)[keep],1e-250)
    # Data-only constant improves optimizer scaling without changing derivatives.
    reference=np.maximum(d[keep],1.)
    return float(np.sum(d[keep]/mu+np.log(mu/reference)-d[keep]/reference))

def fit_history(model,anchor,d,e,local=False):
    """Profile mex and bracket all interior minima on a broad log-Xc grid.

    This is for the single intrinsic parameter only. Compare with the original
    multistart optimizer on representative years before production use.
    """
    cache={}
    def objective(x):
        if x in cache:return cache[x][0]
        params=dict(anchor,Xc=float(np.exp(x)))
        def fun(m):return score(model,dict(params,mex=float(m)),d,e)
        r=minimize_scalar(fun,bounds=p.BOUNDS['mex'],method='bounded',options={'xatol':1e-9})
        value,mex=min([(r.fun,r.x),(fun(0),0.),(fun(.2),.2)])
        cache[x]=(value,dict(params,mex=float(mex)))
        return value
    # Wide fixed domain plus dense coverage around the anchored scale.
    if local:
        width=.22
        for expansion in range(8):
            grid=np.clip(np.log(anchor['Xc'])+np.linspace(-width,width,5),np.log(.02),np.log(3000))
            values=np.array([objective(float(x)) for x in grid])
            if 0<int(np.argmin(values))<len(grid)-1:break
            width*=2
        else:local=False
    if not local:
        grid=np.unique(np.r_[np.linspace(np.log(.02),np.log(3000),13),
            np.log(anchor['Xc'])+np.log([.35,.5,.65,.8,1.,1.25,1.6,2.])])
    grid=grid[(grid>=np.log(.02))&(grid<=np.log(3000))]
    values=np.array([objective(float(x)) for x in grid]);results=[]
    for i in range(1,len(grid)-1):
        if values[i]<=values[i-1] and values[i]<=values[i+1]:
            r=minimize_scalar(objective,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':2e-6})
            results.append(r)
    x=min(cache,key=lambda x:cache[x][0]);value,params=cache[x]
    return dict(params=params,score=float(value),success=bool(results and all(r.success for r in results)),
        evaluations=len(cache),method='adaptive local bracket with expansion' if local else 'broad log-grid bracketing and scalar profiled refinement',
        bound_hits=[k for k in ['Xc','mex'] if params[k] in p.BOUNDS[k]])

def fit_baseline(model,anchor,d,e,checkpoint):
    """Two nearby independent starts for bootstrap samples; broad fallback.

    The real-data baseline is still audited with the original broad starts.
    Bootstrap optima should be near that generating solution. Disagreement or
    unsuccessful convergence triggers the original broad multistart procedure.
    """
    keys=['Xc','epsilon','CV'];bounds=[tuple(np.log(p.BOUNDS[k])) if k!='CV' else p.BOUNDS[k] for k in keys]
    def profile(q):
        params=dict(anchor,Xc=float(np.exp(q[0])),epsilon=float(np.exp(q[1])),CV=float(q[2]))
        def fun(m):return score(model,dict(params,mex=float(m)),d,e)
        r=minimize_scalar(fun,bounds=p.BOUNDS['mex'],method='bounded',options={'xatol':1e-9})
        value,mex=min([(r.fun,r.x),(fun(0),0.),(fun(.2),.2)])
        return value,dict(params,mex=float(mex))
    rows=[]
    for factor,cvdelta in [(1.,0.),(1.08,.02)]:
        q=[np.log(anchor['Xc']*factor),np.log(anchor['epsilon']*factor),np.clip(anchor['CV']+cvdelta,0,.5)]
        r=minimize(lambda q:profile(q)[0],q,method='L-BFGS-B',jac='3-point',bounds=bounds,
            options={'maxiter':120,'ftol':1e-10,'gtol':1e-3,'maxls':30,'finite_diff_rel_step':2e-5})
        value,params=profile(r.x)
        rows.append(dict(params=params,score=float(value),success=bool(r.success),
            max_abs_profile_gradient=float(np.abs(r.jac).max()),evaluations=int(r.nfev),message=str(r.message)))
        p.save(checkpoint,rows)
    best=dict(min(rows,key=lambda x:x['score']))
    best['bound_hits']=[k for k in [*keys,'mex'] if min(abs(best['params'][k]-v) for v in p.BOUNDS[k])<1e-8]
    best['starts']=rows
    if not all(r['success'] for r in rows) or abs(rows[0]['score']-rows[1]['score'])>1e-5:
        broad=p.fit(model,anchor,keys,d,e,120,checkpoint.with_name(checkpoint.stem+'_broad.json'))
        if broad['score']<best['score']:best=broad
        best['broad_fallback_used']=True
    return best

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--years',nargs='+',type=int,default=[1900,1980,2019])
    args=parser.parse_args()
    p.score=score
    folder=HERE/'point';folder.mkdir(parents=True,exist_ok=True)
    model=p.Model();d,e=p.dataset(2019,0,1)
    before=score(model,p.BASE,d,e)
    dest=folder/'baseline.json'
    if dest.exists(): baseline=json.loads(dest.read_text())
    else:
        baseline=p.fit(model,p.BASE,['Xc','epsilon','CV'],d,e,120,folder/'baseline_starts.json')
        baseline['objective']='relative-error estimating equation Q=sum(D/mu+log(mu)); not a sampling likelihood'
        baseline['old_baseline_new_score']=before
        baseline['prediction']=p.prediction(baseline['params'])
        baseline['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        p.save(dest,baseline)
    print(json.dumps({'baseline':baseline['params'],'score':baseline['score'],'old_score':before}),flush=True)
    for year in args.years:
        dest=folder/f'{year}_Xc.json'
        if dest.exists():continue
        d,e=p.dataset(year,0,1)
        fit=p.fit(model,baseline['params'],['Xc'],d,e,120,folder/f'{year}_starts.json')
        fit['prediction']=p.prediction(fit['params'])
        fit['ratio']=fit['params']['Xc']/baseline['params']['Xc']
        p.save(dest,fit)
        print(json.dumps({'year':year,'params':fit['params'],'score':fit['score'],'ratio':fit['ratio']}),flush=True)

if __name__=='__main__':main()
