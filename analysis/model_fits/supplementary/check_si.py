"""Bounded, unfitted FP checks for the SI revision; no archived outputs modified."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from senogenic_vs_robustness.sr_population import Grid, population, positive_normal

P = json.loads((ROOT/'results/historical/joint_covariance.json').read_text())['baseline']


def slopes(ages, hazard, centers, log_age=False):
    result = []
    for age in centers:
        keep = abs(ages-age) <= 5
        x = np.log(ages[keep]) if log_age else ages[keep]
        result.append(np.polyfit(x, np.log(hazard[keep]), 1)[0])
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true', help='Explicitly run the numerical grid checks; no fitting')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'tmp/si_checks')
    args = parser.parse_args()
    if not args.recompute:
        parser.error('Numerical calculations require --recompute. Rendering uses saved results instead.')
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    rows, means, checks, comparisons = [], [], [], []
    for label, grid in [('production', Grid(320, .025, 260, 64)),
                        ('quadrature', Grid(320, .025, 260, 160)),
                        ('refined', Grid(480, 1/60, 260, 160))]:
        for name, cv in [('homogeneous', 0), ('Xc', P['CV']), ('eta', .2), ('beta', .2)]:
            print(label, name, flush=True)
            pop = population(P, 'Xc' if name=='homogeneous' else name, cv, grid=grid, mex=0)
            ages = np.arange(20, 255) + .5
            hazard = pop.annual_mortality(ages-.5)
            centers = np.arange(50, 91, 10)
            b = slopes(ages, hazard, centers)
            checks.append(dict(grid=label, scenario=name, slopes=b.tolist(),
                               normalized=(b/b.mean()).tolist(),
                               exponents=slopes(ages, hazard, [80,100,120,150,200,240],True).tolist()))
            rows.extend(dict(grid=label, scenario=name, age=a, mortality=h)
                        for a,h in zip(ages,hazard))
            if name == 'homogeneous':
                for t in [30,50,70,90]:
                    numerical = -np.gradient(pop.log_survival(),grid.dt)[round(t/grid.dt)]
                    eta,beta,eps,xc,kappa = [P[k] for k in ['eta','beta','epsilon','Xc','kappa']]
                    xst = eta*kappa*t/(beta-eta*t)
                    U = lambda x: (beta-eta*t)*x-beta*kappa*np.log1p(x/kappa)
                    barrier = U(xc)-U(xst)
                    endpoint = beta-eta*t-beta*kappa/(kappa+xc)
                    curvature = (beta-eta*t)**2/(beta*kappa)
                    absorbing = endpoint*np.sqrt(curvature/(2*np.pi*eps))*np.exp(-barrier/eps)
                    legacy = beta**2/eps*np.exp(xc/eps*(eta*t-beta))
                    comparisons.append(dict(grid=label,age=t,numerical=numerical,
                                            absorbing=absorbing,legacy=legacy,barrier_over_noise=barrier/eps))
            if label=='refined' and name in ('eta','beta'):
                factors,_,_ = positive_normal(.2,grid.nodes)
                for start in range(40,160,10):
                    left = np.array([np.interp(start,pop.time,s) for s in pop.log_components])
                    right = np.array([np.interp(start+10,pop.time,s) for s in pop.log_components])
                    weights = pop.weights*np.exp(left)*(-np.expm1(right-left))
                    mass = weights.sum()
                    if mass>=25/1e6:
                        means.append(dict(parameter=name,lifespan_midpoint=start+5,
                                          mean_parameter=float(weights@(P[name]*factors)/mass),death_probability=mass))
    pd.DataFrame(rows).to_csv(OUT/'si_hazards.csv',index=False)
    pd.DataFrame(means).to_csv(OUT/'si_death_bin_means.csv',index=False)
    pd.DataFrame(comparisons).to_csv(OUT/'si_approximation_check.csv',index=False)
    (OUT/'si_checks.json').write_text(json.dumps(dict(baseline=P,checks=checks),indent=2))
    print('Complete: no fits or bootstraps',flush=True)


if __name__=='__main__':
    main()
