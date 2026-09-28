"""Finite-volume historical model and optimizer; called by the Q-fit runners."""
from __future__ import annotations
from functools import lru_cache
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.integrate import trapezoid
from scipy.special import roots_hermitenorm, logsumexp, xlogy
from inputs.nonuniform_solver import forward

HERE = Path(__file__).resolve().parent
YEARS = sorted(set(range(1800, 2016, 5)) | {2019})
BOUNDS = {'Xc': (.02, 3000.), 'epsilon': (.002, 20_000_000.), 'CV': (0., .5), 'mex': (0., .2)}
BASE = json.loads((HERE/'inputs/baseline.json').read_text())['params']
AGES = np.arange(20, 110)
PROBS = [.75, .5, .25, .1, .01, .001, .0001]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def dataset(year, rep, seed):
    frame = pd.read_csv(HERE/'inputs/hmd.csv')
    d = frame[(frame.country == 'SWE') & (frame.year == year) & frame.age.between(20, 109)].sort_values('age')
    if not np.array_equal(d.age.to_numpy(), AGES):
        raise ValueError(f'Missing/duplicate ages for {year}')
    deaths, exposure = d.deaths.to_numpy(float), d.exposure.to_numpy(float)
    if not (np.isfinite(deaths).all() and np.isfinite(exposure).all() and (deaths >= 0).all() and (exposure >= 0).all() and not ((exposure == 0) & (deaths > 0)).any()):
        raise ValueError(f'Invalid HMD values {year}')
    if rep:
        deaths = np.random.default_rng(np.random.SeedSequence([seed, rep, year])).poisson(deaths).astype(float)
    return deaths, exposure


class Model:
    def __init__(self, horizon=110):
        self.dt = .025
        self.steps = round(horizon/self.dt)
        self.grid = np.arange(self.steps+1)*self.dt
        z, w = roots_hermitenorm(31)
        self.z, self.w = z, w/np.sqrt(2*np.pi)
        self.bins = AGES[:, None]*40 + np.arange(40)[None, :]

    @lru_cache(maxsize=3)
    def components(self, xc, eps, cv):
        thresholds = xc*(1+cv*self.z) if cv > 1e-10 else np.array([xc])
        weights = self.w.copy() if cv > 1e-10 else np.ones(1)
        keep = thresholds > 0
        thresholds, weights = thresholds[keep], weights[keep]
        weights /= weights.sum()
        out = [forward(BASE['eta'], BASE['beta'], eps, float(x), 320, self.dt,
                       self.steps, kappa=BASE['kappa'], log_output=True, return_deaths=True) for x in thresholds]
        logs, q = np.stack([r[0] for r in out]), np.stack([r[1] for r in out])
        return logs, q, weights, logsumexp(logs+np.log(weights)[:, None], axis=0)

    def rates(self, p):
        logs, q, weights, _ = self.components(p['Xc'], p['epsilon'], p['CV'])
        risk = weights[:, None]*np.exp(np.minimum(logs[:, :-1]-p['mex']*self.grid[:-1], 0))
        qt = q+(1-q)*(-np.expm1(-p['mex']*self.dt))
        return (risk*qt)[:, self.bins].sum(axis=(0, 2))/np.maximum((risk*self.dt*(1-.5*qt))[:, self.bins].sum(axis=(0, 2)), 1e-250)

    def log_survival(self, p):
        return self.components(p['Xc'], p['epsilon'], p['CV'])[3]-p['mex']*self.grid


def deviance(d, mu):
    mu = np.maximum(mu, 1e-250)
    return np.maximum(2*(mu-d+xlogy(d, d/mu)), 0)


def score(model, p, d, e):
    keep = e > 0
    mu = np.maximum(e[keep]*model.rates(p)[keep], 1e-250)
    reference = np.maximum(d[keep], 1.)
    v = np.sum(d[keep]/mu + np.log(mu/reference) - d[keep]/reference)
    if not np.isfinite(v):
        raise FloatingPointError('Nonfinite objective')
    return float(v)


def fit(model, anchor, free, d, e, maxiter=120, checkpoint=None):
    # Profile mex numerically using the exact competing-risk rate; it is not
    # simply added to an intrinsic central death rate.
    def profile(q):
        p = dict(anchor)
        for key, val in zip(free, q):
            p[key] = float(np.exp(val)) if key != 'CV' else float(val)
        def objective(m):
            return score(model, dict(p, mex=float(m)), d, e)
        r = minimize_scalar(objective, bounds=BOUNDS['mex'], method='bounded', options={'xatol': 1e-9, 'maxiter': 60})
        choices = [(r.fun, r.x), (objective(0), 0.), (objective(.2), .2)]
        value, mex = min(choices)
        return float(value), dict(p, mex=float(mex)), bool(r.success or mex in BOUNDS['mex'])
    bounds = [(np.log(BOUNDS[k][0]), np.log(BOUNDS[k][1])) if k != 'CV' else BOUNDS[k] for k in free]
    starts = []
    for factor in [1., .55, 1.8]:
        starts.append(np.array([np.clip(np.log(anchor[k]*factor), *b) if k != 'CV' else np.clip(anchor[k]+(.08 if factor>1 else -.08 if factor<1 else 0), *b) for k,b in zip(free,bounds)]))
    rows = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else []
    for start, q in enumerate(starts[len(rows):], start=len(rows)):
        begin = time.monotonic()
        r = minimize(lambda q: profile(q)[0], q, method='L-BFGS-B', jac='3-point', bounds=bounds,
                     options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-3, 'maxls': 30, 'finite_diff_rel_step': 2e-5})
        value, p, profile_success = profile(r.x)
        rows.append(dict(start=start, params=p, score=value, success=bool(r.success and profile_success), outer_success=bool(r.success), mex_profile_success=profile_success, message=str(r.message), iterations=int(r.nit), evaluations=int(r.nfev), seconds=time.monotonic()-begin))
        if checkpoint: save(checkpoint, rows)
    best = dict(min(rows, key=lambda r:r['score']))
    p = best['params']
    best['bound_hits'] = [k for k in [*free, 'mex'] if (min(abs(np.log(p[k]/v)) for v in BOUNDS[k]) < 1e-3 if k in ['Xc','epsilon'] else min(abs(p[k]-v) for v in BOUNDS[k]) <= 1e-4*(BOUNDS[k][1]-BOUNDS[k][0]))]
    best['starts'] = rows
    best['start_score_range'] = [min(r['score'] for r in rows), max(r['score'] for r in rows)]
    best['multistart_disagreement'] = best['start_score_range'][1]-best['score'] > max(.001,.001*abs(best['score']))
    best['review_required'] = not best['success'] or bool(best['bound_hits']) or best['multistart_disagreement']
    rates = model.rates(p)
    best['age_diagnostics'] = dict(age=AGES.tolist(), zero_exposure_ages=AGES[e==0].tolist(), deaths=d.tolist(), exposure=e.tolist(), rates=rates.tolist(), expected_deaths=(e*rates).tolist(), weighted_deviance=(deviance(d,e*rates)/np.maximum(d,1)).tolist())
    best['ordinary_deviance_diagnostic'] = float(deviance(d,e*rates).sum())
    return best


def prediction(p):
    model = Model(180)
    for horizon in [180, 270, 405, 610]:
        if horizon != 180: model = Model(horizon)
        ls = model.log_survival(p)
        if not np.isfinite(ls).all() or (np.diff(ls)>1e-8).any():
            raise ValueError('Invalid survival')
        if ls[-1]-np.interp(20,model.grid,ls) < np.log(1e-10): break
    metrics = {}
    for entry in [20]:
        cond = ls-np.interp(entry,model.grid,ls)
        contours = {}
        for prob in PROBS:
            ix = np.flatnonzero((model.grid>=entry) & (cond<=np.log(prob)))
            contours[str(prob)] = float(np.interp(np.log(prob), cond[ix[0]-1:ix[0]+1][::-1], model.grid[ix[0]-1:ix[0]+1][::-1])) if len(ix) else None
        mask=model.grid>=entry
        integral=float(trapezoid(np.exp(cond[mask]),model.grid[mask]))
        terminal=float(np.exp(cond[-1]))
        metrics[str(entry)] = dict(contours=contours, mean_attained_age=entry+integral,
                                  mean_horizon=float(model.grid[-1]), terminal_survival=terminal,
                                  mean_tail_upper_bound=terminal/p['mex'] if p['mex']>0 else None,
                                  mean_is_restricted=bool(terminal>=1e-10))
    age = np.arange(20, int(model.grid[-1])+1)
    return dict(age=age.tolist(), survival_age20=[float(np.exp(v-np.interp(20,model.grid,ls))) if a>=20 else None for a,v in zip(age,np.interp(age,model.grid,ls))], contours=metrics, tail_resolved=bool(ls[-1]-np.interp(20,model.grid,ls)<np.log(1e-10)))
