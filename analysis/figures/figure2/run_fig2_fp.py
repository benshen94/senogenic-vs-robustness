#!/usr/bin/env python3
"""Reproduce Figure 2 sensitivity results with deterministic finite-volume SR."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
from ageing_packages.mortality_data_analysis.HMD_lifetables import HMD
from senogenic_vs_robustness.sr_population import Grid, population, sibling_populations

BASELINE = ROOT/'results/fits/records/sweden_2019_fig2_fp_baseline.json'
DATA = ROOT/'tmp/fig2_fokker_planck_recomputed'
OUT = ROOT/'tmp/fig2_fokker_planck_preview'
PARAMETERS = ('eta','beta','Xc','epsilon')
SURVIVAL_CVS = {'Xc':.15,'epsilon':.25,'eta':.05,'beta':.05}
SIBLING_CVS = {'Xc':.20,'epsilon':.30,'eta':.15,'beta':.10}


def jobs():
    result = []
    for par in PARAMETERS:
        result.extend(dict(kind='tails_cv',param=par,cv=float(cv),factor=1.)
                      for cv in np.round(np.arange(0,.201,.01),2))
        result.extend(dict(kind='tails_factor',param=par,cv=0.,factor=float(f))
                      for f in np.round(np.arange(.85,1.151,.05),2))
        result.append(dict(kind='survival',param=par,cv=SURVIVAL_CVS[par],factor=1.))
        result.append(dict(kind='siblings',param=par,cv=SIBLING_CVS[par],factor=1.))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute(job, params, grid, condition_age):
    # Sibling rates and conditional tail curves need only 151 years. Extreme
    # unconditional quantiles retain the full horizon, with an explicit failure
    # if the requested tail has not been reached.
    horizon = grid.horizon if job['kind'].startswith('tails') else 151.
    local_grid = Grid(grid.cells,grid.dt,horizon,grid.nodes)
    pop = population(params,job['param'],job['cv'],job['factor'],local_grid,mex=0.)
    if job['kind'].startswith('tails'):
        return np.array([pop.quantile(1e-4)]), {}
    if job['kind']=='survival':
        ages = np.arange(condition_age,126)
        return pop.conditional(ages,condition_age), {'ages':ages.tolist(), 'survival':np.exp(np.interp(ages,pop.time,pop.log_survival())).tolist()}
    weights, selection = sibling_populations(pop,job['cv'],rho=.5)
    ages = np.arange(50,126)
    values = np.concatenate([pop.annual_mortality(ages,weights[k]) for k in ('full','good','bad')])
    return values, dict(ages=ages.tolist(),**selection)


def run_job(payload):
    job, record, grid_values, cache_dir, uncertainty, condition_age = payload
    grid = Grid(**grid_values)
    key = f"{job['kind']}_{job['param']}_{job['cv']:.3f}_{job['factor']:.3f}"
    path = Path(cache_dir)/f'{key}.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.monotonic()
    params = record['params']
    values, info = compute(job,params,grid,condition_age)
    if not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError(f'Invalid prediction for {key}')
    result = dict(job=job,values=values.tolist(),info=info)
    if uncertainty:
        # Marginal covariance of log Xc and log epsilon retains their coupling
        # from the full baseline fit. Other baseline parameters are fixed or
        # deliberately replaced in the controlled Figure 2 experiments.
        covariance = np.asarray(record['log_Xc_epsilon_covariance'])
        h = .002
        derivatives = []
        for par in ('Xc','epsilon'):
            lo,hi = dict(params),dict(params)
            lo[par] *= np.exp(-h);hi[par] *= np.exp(h)
            low,_ = compute(job,lo,grid,condition_age)
            high,_ = compute(job,hi,grid,condition_age)
            derivatives.append((np.log(high)-np.log(low))/(2*h))
        jac = np.asarray(derivatives).T
        se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',jac,covariance,jac),0))
        result['ci_low'] = (values*np.exp(-1.95996398454*se)).tolist()
        result['ci_high'] = (values*np.exp(1.95996398454*se)).tolist()
        if job['kind']=='survival':
            result['ci_high'] = np.minimum(result['ci_high'],1.).tolist()
    result['seconds'] = time.monotonic()-started
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(path)
    return result


def write_tables(results, record, data_dir, condition_age):
    tables = {k:[] for k in ('survival','tails_cv','tails_factor','siblings')}
    for r in results:
        j = r['job'];kind = j['kind']
        if kind.startswith('tails'):
            rows=[dict(param=j['param'],cv=j['cv'],factor=j['factor'],tail_age=r['values'][0])]
        elif kind=='survival':
            rows=[dict(age=a,curve_id=f"{j['param']}_{j['cv']:g}",param=j['param'],cv=j['cv'],
                       survival=s,conditional_survival=y)
                  for a,y,s in zip(r['info']['ages'],r['values'],r['info']['survival'])]
        else:
            n=len(r['info']['ages'])
            rows=[dict(param=j['param'],cohort=k,age=a,mortality=r['values'][ci*n+i],
                       top_age=r['info']['top_age'],bottom_age=r['info']['bottom_age'])
                  for ci,k in enumerate(('full','good','bad')) for i,a in enumerate(r['info']['ages'])]
        if 'ci_low' in r:
            for row,low,high in zip(rows,r['ci_low'],r['ci_high']):
                row.update(ci_low=low,ci_high=high)
        tables[kind].extend(rows)
    os.environ['SENOGENIC_HMD_DATA_DIR'] = str(ROOT/'data/hmd')
    ages, survival = HMD('SWE', 'both', 'period').get_survival(2019, strict=True)
    hmd = pd.DataFrame({'age': ages, 'survival': survival})
    denominator=float(hmd.loc[hmd.age.eq(condition_age),'survival'].iloc[0])
    for row in hmd[hmd.age.ge(condition_age)].itertuples():
        tables['survival'].append(dict(age=row.age,curve_id='hmd',param=None,cv=None,
            survival=row.survival,conditional_survival=row.survival/denominator))
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(data_dir/f'{name}.csv',index=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true',
                        help='Explicitly opt into the expensive 120-task calculation')
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'tmp/fig2_fokker_planck_cache')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cells',type=int,default=320)
    parser.add_argument('--dt',type=float,default=.025)
    parser.add_argument('--nodes',type=int,default=64)
    parser.add_argument('--horizon',type=float,default=420.)
    parser.add_argument('--condition-age',type=int,choices=(90,100),default=90)
    parser.add_argument('--no-ci',action='store_true')
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args(argv)
    if not args.recompute:
        parser.error('Full calculation requires --recompute; use plot_fig2_fp.py for saved-output rendering.')
    if not 1<=args.workers<=8:parser.error('workers must be 1–8')
    grid=Grid(args.cells,args.dt,args.horizon,args.nodes)
    record=json.loads(args.baseline.read_text())
    sources=[Path(__file__),Path(__file__).with_name('plot_fig2_fp.py'),
             ROOT/'src/senogenic_vs_robustness/sr_population.py',
             ROOT/'src/senogenic_vs_robustness/sr_finite_volume.py',args.baseline,
             ROOT/'data/hmd/mortality.org_File_GetDocument_hmd.v6_SWE_STATS_bltper_1x1.txt',
             ROOT/'results/tables/fig2d_raw_digitized_points.csv']
    contract=dict(source_hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha(p) for p in sources},
        baseline=record,grid=asdict(grid),condition_age=args.condition_age,
        uncertainty=not args.no_ci,heterogeneity='single focal positive Gaussian; all other CV=0',
        mex=0.,sibling_correlation_before_positivity_repair=.5,
        uncertainty_method='Approximate pointwise log-delta intervals from the joint baseline covariance of log Xc and log epsilon; no old CI endpoints')
    fingerprint=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:16]
    cache=args.cache_dir/fingerprint;cache.mkdir(parents=True,exist_ok=True)
    args.data_dir.mkdir(parents=True,exist_ok=True)
    payloads=[(j,record,asdict(grid),str(cache),not args.no_ci,args.condition_age) for j in jobs()]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_job,p) for p in payloads]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result();results.append(result)
            if i%10==0 or i==len(futures):print(f'{i}/{len(futures)} completed',flush=True)
    results.sort(key=lambda r:(r['job']['kind'],r['job']['param'],r['job']['cv'],r['job']['factor']))
    write_tables(results,record,args.data_dir,args.condition_age)
    contract.update(cache_fingerprint=fingerprint,tasks=len(results),complete=True)
    (args.data_dir/'manifest.json').write_text(json.dumps(contract,indent=2))
    if not args.no_render:
        from analysis.figures.figure2.plot_fig2_fp import render
        render(args.data_dir,args.output_dir,pdf=args.pdf)
    print(f'Figure 2 data: {args.data_dir}',flush=True)


if __name__=='__main__':main()
