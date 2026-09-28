#!/usr/bin/env python3
"""Selected Sweden equal-age parameters, deterministic FP sensitivity; original Figure 1 style."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'tmp/response_curves_recomputed'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'numba_cache')
import sys,json,time,hashlib,argparse
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from scipy.special import roots_legendre,ndtr
from concurrent.futures import ProcessPoolExecutor,as_completed
from senogenic_vs_robustness.sr_finite_volume import forward
from analysis.figures.steepness_longevity import response_plane as style
BASE=dict(json.loads((ROOT/'results/historical/fit_records/point/baseline.json').read_text())['params'])
BASE['mex']=0.
NUISANCE=('eta','beta','epsilon','Xc','CV')

def curve(p,n=320,dt=.025,nq=61):
    t=np.arange(round(240/dt)+1)*dt
    nodes,weights=roots_legendre(nq)
    lo=max(-8.,-1/p['CV']);hi=8.
    z=lo+(nodes+1)*(hi-lo)/2
    w=weights*(hi-lo)/2*np.exp(-z*z/2)/np.sqrt(2*np.pi)/(ndtr(hi)-ndtr(lo))
    w/=w.sum()
    s=np.zeros_like(t)
    for zi,wi in zip(z,w):
        s+=wi*forward(p['eta'],p['beta'],p['epsilon'],p['Xc']*(1+p['CV']*zi),n,dt,len(t)-1,kappa=p['kappa'])
    s*=np.exp(-p.get('mex',0)*t)
    assert np.all(np.isfinite(s)) and np.max(np.diff(s))<1e-10 and abs(s[0]-1)<1e-10
    return t,s

def metrics(t,s):
    assert s[-1]<.25,'Unresolved interquartile survival range'
    q=np.interp([.75,.5,.25],s[::-1],t[::-1])
    return dict(t_median_absolute=float(q[1]),steepness_iqr_absolute=float(q[1]/(q[2]-q[0])),q75=float(q[0]),q25=float(q[2]),terminal_survival=float(s[-1]))

def task(item,n=320,dt=.025,nq=61):
    key,p=item;path=HERE/'curves'/f'{key}_n{n}_dt{dt}_q{nq}.npz'
    if path.exists():
        d=np.load(path);return key,metrics(d['t'],d['s'])
    t,s=curve(p,n,dt,nq)
    path.parent.mkdir(exist_ok=True);np.savez_compressed(path,t=t,s=s)
    return key,metrics(t,s)

def key(p):return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()[:20]

def design():
    scenarios=[('central','central',1.,BASE)]
    for name in NUISANCE:
        for f in [.8,1.2]:
            p=dict(BASE);p[name]*=f;scenarios.append((name+('_low' if f<1 else '_high'),name,f,p))
    jobs={};rows=[]
    for sid,nuis,f,base in scenarios:
        k=key(base);jobs[k]=base
        rows.append(dict(run_id=sid+'__baseline',scenario_id=sid,nuisance_param=nuis,nuisance_factor=f,curve_type='baseline',focal_param='baseline',focal_value=1.,h_ext=0.,from_t=0,tmax=240.,key=k))
        for focal in style.PLOT_PARAMS:
            for factor in np.round(np.arange(style.MIN_FACTOR_BY_PARAM[focal],1.40001,.1),10):
                p=dict(base);p[focal]*=float(factor);k=key(p);jobs[k]=p
                rows.append(dict(run_id=f'{sid}__{focal}_{factor}',scenario_id=sid,nuisance_param=nuis,nuisance_factor=f,curve_type='parameter_factor',focal_param=focal,focal_value=float(factor),h_ext=0.,from_t=0,tmax=240.,key=k))
        for rate in np.logspace(-4,-2,10):
            if rate<=style.H_EXT_MAX_VISIBLE:
                rows.append(dict(run_id=f'{sid}__mex_{rate}',scenario_id=sid,nuisance_param=nuis,nuisance_factor=f,curve_type='h_ext_absolute',focal_param='h_ext',focal_value=float(rate),h_ext=float(rate),from_t=0,tmax=240.,key=key(base)))
    return jobs,rows

def run(workers):
    HERE.mkdir(parents=True, exist_ok=True)
    start=time.time();jobs,rows=design();print(f'{len(jobs)} distinct deterministic FP mixtures; {len(rows)} response records',flush=True)
    results={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs={pool.submit(task,it):it[0] for it in jobs.items()}
        for i,f in enumerate(as_completed(futs),1):
            k,m=f.result();results[k]=m
            if i%25==0:print(f'{i}/{len(jobs)} mixtures, {time.time()-start:.1f}s',flush=True)
    for row in rows:
        if row['h_ext']:
            z=np.load(HERE/'curves'/f"{row['key']}_n320_dt0.025_q61.npz")
            m=metrics(z['t'],z['s']*np.exp(-row['h_ext']*z['t']))
        else:m=results[row['key']]
        row.update(m)
    pd.DataFrame(rows).to_csv(HERE/'metrics_long.csv',index=False)
    manifest=dict(baseline=BASE,age_conditioning=0,solver='Conservative nonuniform finite-volume Fokker-Planck; reflecting zero and absorbing Xc; implicit time integration',diffusion='epsilon in dX=(eta*t-beta*X/(kappa+X))dt+sqrt(2*epsilon)dW',threshold_distribution='Positive-truncated Gaussian; CV parameter is sigma/mean before truncation',n_cells=320,dt=.025,tmax=240,threshold_nodes=61,nuisance_parameters=NUISANCE,nuisance_factors=[.8,1.2],shading='Original pointwise 2.5/97.5% quantiles across 11 one-at-a-time sensitivity scenarios; not statistical confidence intervals',center='Original style: scenario mean of within-scenario-normalized response metrics',extrinsic='Baseline mex=0, intrinsic sweeps mex=0; original red curve varies mex from zero baseline',distinct_mixtures=len(jobs),elapsed_seconds=time.time()-start)
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Saved numerical response tables under {HERE}',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recompute',action='store_true')
    ap.add_argument('--workers',type=int,default=1)
    args=ap.parse_args()
    if not args.recompute:ap.error('Use --recompute for the full calculation; normal rendering reads saved metrics.')
    if not 1<=args.workers<=8:ap.error('Workers must be between 1 and 8.')
    run(args.workers)
