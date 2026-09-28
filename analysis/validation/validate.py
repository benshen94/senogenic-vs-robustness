"""Replay historical SR Monte Carlo evidence or isolate SI quadrature refinement.

No refitting, cluster submission, or new Monte Carlo generation. Results are
new records, not replacements for the archived experiments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.special import logsumexp, roots_legendre

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from senogenic_vs_robustness.sr_finite_volume import forward
from senogenic_vs_robustness.sr_population import Grid, population


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_record(path):
    record = json.loads(path.read_text())
    arrays = path.with_suffix('.npz')
    if sha(arrays) != record['arrays_sha256']:
        raise ValueError('Archived arrays hash mismatch')
    with np.load(arrays, allow_pickle=False) as data:
        ages = data['ages']
        mc = np.array([(data['death_times'] > age).mean() for age in ages])
        np.testing.assert_array_equal(mc, data['mc_survival'])
        error = float(np.max(abs(mc-data['fp_survival'])))
        np.testing.assert_allclose(error, record['maximum_survival_difference'], atol=1e-14)
    np.testing.assert_allclose(record['mc_dkw95_half_width'],
                               np.sqrt(np.log(40)/(2*record['n'])))
    return record


def replay(path, smoke=False):
    record = audit_record(path)
    eta, beta, eps, xc, cv, mex = record['params']
    cells, dt, nodes = (80, .1, 24) if smoke else record['fp_grids'][-1]
    z, w = roots_legendre(nodes)
    lo, hi = max(-8., -1/cv+1e-10), 8.
    z = lo+(z+1)*(hi-lo)/2
    lw = np.log(w)-z*z/2
    lw -= logsumexp(lw)
    time = np.arange(round(110/dt)+1)*dt
    logs = [weight+forward(eta,beta,eps,xc*(1+cv*zi),cells,dt,len(time)-1,
                           log_output=True) for zi,weight in zip(z,lw)]
    survival = np.exp(logsumexp(logs,axis=0)-mex*time)
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        prediction = np.interp(data['ages'],time,survival)
        return dict(case=record['case'], record=path.name, smoke=smoke,
                    grid=[cells,dt,nodes],
                    current_vs_archived_fp_max=float(np.max(abs(prediction-data['fp_survival']))),
                    current_vs_archived_mc_max=float(np.max(abs(prediction-data['mc_survival']))),
                    mc_dkw95_half_width=record['mc_dkw95_half_width'])


def quadrature(smoke=False):
    baseline_path = ROOT/'results/supplementary1_fp/si_checks.json'
    params = json.loads(baseline_path.read_text())['baseline']
    cells, dt, horizon = (40,.1,100.) if smoke else (480,1/60,260.)
    rows = []
    for focal,cv in [('Xc',params['CV']),('eta',.2),('beta',.2)]:
        curves = []
        for nodes in (160,256):
            pop = population(params,focal,cv,grid=Grid(cells,dt,horizon,nodes),mex=0)
            ages = np.arange(20,min(255,horizon))
            curves.append((np.exp(pop.log_survival()),pop.annual_mortality(ages)))
        a,b = curves
        rows.append(dict(parameter=focal,cv=cv,
                         max_survival_difference=float(np.max(abs(a[0]-b[0]))),
                         max_relative_annual_mortality_difference=float(np.max(abs(a[1]/b[1]-1)))))
    return dict(smoke=smoke,baseline=params,baseline_sha256=sha(baseline_path),
                grid=dict(cells=cells,dt=dt,horizon=horizon,nodes=[160,256]),checks=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['audit','replay','quadrature'])
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    records = sorted((ROOT/'results/validation/records').glob('*.json'))
    if not records:
        raise ValueError('Missing archived validation records')
    if args.mode == 'audit':
        result = [dict(record=p.name,maximum_survival_difference=audit_record(p)[
            'maximum_survival_difference']) for p in records]
    elif args.mode == 'replay':
        result = [replay(p,args.smoke) for p in records if 'paper_sweden_' in p.name
                  and 'dt0.0015625_' in p.name]
    else:
        result = quadrature(args.smoke)
    payload = dict(mode=args.mode,python=platform.python_version(),numpy=np.__version__,
                   scipy=scipy.__version__,result=result,
                   code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
                       Path(__file__),ROOT/'src/senogenic_vs_robustness/sr_finite_volume.py',
                       ROOT/'src/senogenic_vs_robustness/sr_population.py']})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(payload,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
