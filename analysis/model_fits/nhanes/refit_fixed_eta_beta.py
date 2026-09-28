"""Refit NHANES with paper eta/beta fixed; optimize epsilon, Xc, CV and mex."""
from __future__ import annotations
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp, roots_hermitenorm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'inputs'))
from nonuniform_solver import forward

# Point estimates from the manuscript table, USA 2019 HMD.
ETA_FIXED = 0.59
BETA_FIXED = 57.9
KAPPA = 0.5
PAPER_EPSILON = 49.7
PAPER_XC = 20.85
PAPER_CV = 0.19
MEX_MAX = 0.2
# Expanded bounds match the current likelihood baseline search.
XC_BOUNDS = (0.02, 3000.0)
EPS_BOUNDS = (0.002, 20_000_000.0)
CV_BOUNDS = (0.0, 0.5)

class DelayedEntryLikelihood:
    def __init__(self, data: pd.DataFrame, n: int, dt: float, quadrature: int):
        self.entry = data.entry.to_numpy(float)
        self.exit = data.exit.to_numpy(float)
        self.event = data.event.to_numpy(int)
        self.event_exit = self.exit[self.event == 1]
        self.followup = float(np.sum(self.exit - self.entry))
        self.n, self.dt, self.quadrature = n, dt, quadrature
        self.steps = round(110.0 / dt)
        self.grid = np.arange(self.steps + 1) * dt
        self.midpoints = self.grid[1:] - dt / 2
        z, w = roots_hermitenorm(quadrature)
        self.z = z
        self.w = w / np.sqrt(2 * np.pi)
        self.evaluations = 0

    @lru_cache(maxsize=8)
    def curve(self, xc: float, epsilon: float, cv: float):
        if cv <= 1e-10:
            thresholds = np.array([xc])
            weights = np.array([1.0])
        else:
            thresholds = xc * (1 + cv * self.z)
            keep = thresholds > 0
            thresholds = thresholds[keep]
            weights = self.w[keep]
            weights = weights / weights.sum()
        outputs = [
            forward(ETA_FIXED, BETA_FIXED, epsilon, float(x), self.n, self.dt,
                    self.steps, kappa=KAPPA, log_output=True,
                    return_deaths=True)
            for x in thresholds
        ]
        log_components = np.stack([out[0] for out in outputs])
        component_death_fraction = np.stack([out[1] for out in outputs])
        log_s = logsumexp(log_components + np.log(weights)[:, None], axis=0)
        # Form interval deaths and exposure directly from the FP boundary flux.
        # Trapezoidal person-time integrates the alive mass over each interval.
        component_start_survival = np.exp(log_components[:, :-1])
        weighted_start_survival = weights[:, None] * component_start_survival
        interval_deaths = np.sum(weighted_start_survival * component_death_fraction, axis=0)
        interval_exposure = np.sum(
            weighted_start_survival * self.dt * (1.0 - 0.5 * component_death_fraction),
            axis=0)
        hazard = np.divide(interval_deaths, interval_exposure,
                            out=np.zeros_like(interval_deaths),
                            where=interval_exposure > 0)
        return log_s, hazard

    def likelihood(self, xc: float, epsilon: float, cv: float, mex: float):
        log_s, hazard = self.curve(float(xc), float(epsilon), float(cv))
        log_survival = (np.interp(self.exit, self.grid, log_s)
                        - np.interp(self.entry, self.grid, log_s)
                        - mex * (self.exit - self.entry))
        h_death = np.interp(self.event_exit, self.midpoints, hazard)
        log_h = np.log(np.maximum(h_death + mex, 1e-250))
        return float(-np.sum(log_survival) - np.sum(log_h))

    def profile_mex(self, xc: float, epsilon: float, cv: float):
        log_s, hazard = self.curve(float(xc), float(epsilon), float(cv))
        survival_term = (np.interp(self.exit, self.grid, log_s)
                         - np.interp(self.entry, self.grid, log_s))
        h_death = np.interp(self.event_exit, self.midpoints, hazard)
        def derivative(m):
            return self.followup - np.sum(1.0 / np.maximum(h_death + m, 1e-250))
        if derivative(0.0) >= 0:
            mex = 0.0
        elif derivative(MEX_MAX) <= 0:
            mex = MEX_MAX
        else:
            mex = brentq(derivative, 0.0, MEX_MAX, xtol=1e-12)
        nll = (-np.sum(survival_term) + mex * self.followup
               - np.sum(np.log(np.maximum(h_death + mex, 1e-250))))
        self.evaluations += 1
        return float(nll), float(mex)

    def decode(self, q):
        return float(np.exp(q[0])), float(np.exp(q[1])), float(q[2])

    def optimize(self, starts, maxiter=180):
        bounds = [(np.log(XC_BOUNDS[0]), np.log(XC_BOUNDS[1])),
                  (np.log(EPS_BOUNDS[0]), np.log(EPS_BOUNDS[1])), CV_BOUNDS]
        results = []
        for i, start in enumerate(starts):
            xc, epsilon, cv = start
            q0 = np.array([np.log(xc), np.log(epsilon), cv])
            def objective(q):
                x, e, c = self.decode(q)
                return self.profile_mex(x, e, c)[0]
            r = minimize(objective, q0, method='L-BFGS-B', jac='3-point',
                         bounds=bounds,
                         options={'maxiter': maxiter, 'ftol': 1e-9,
                                  'gtol': 0.002, 'maxls': 30,
                                  'finite_diff_rel_step': 2e-5})
            x, e, c = self.decode(r.x)
            nll, mex = self.profile_mex(x, e, c)
            results.append({'start_index': i, 'params': {'eta': ETA_FIXED,
                'beta': BETA_FIXED, 'epsilon': e, 'Xc': x, 'CV': c,
                'kappa': KAPPA, 'mex': mex}, 'nll': nll,
                'success': bool(r.success), 'message': str(r.message),
                'iterations': int(r.nit), 'evaluations': int(r.nfev),
                'bounds': [name for name, value, bounds_ in
                    [('Xc', x, XC_BOUNDS), ('epsilon', e, EPS_BOUNDS),
                     ('CV', c, CV_BOUNDS), ('mex', mex, (0.0, MEX_MAX))]
                    if min(abs(value - bounds_[0]), abs(value - bounds_[1]))
                       < 1e-5 * max(1.0, abs(value))]})
        return sorted(results, key=lambda r: r['nll'])


def get_starts():
    # Stable hand-picked starts spanning the paper fit and nearby scale/diffusion regimes.
    starts = [
        (PAPER_XC, PAPER_EPSILON, PAPER_CV),
        (PAPER_XC, 20.0, 0.10),
        (PAPER_XC, 100.0, 0.30),
        (10.0, PAPER_EPSILON, 0.05),
        (40.0, PAPER_EPSILON, 0.35),
        (10.0, 20.0, 0.30),
        (40.0, 100.0, 0.10),
        (60.0, 500.0, 0.20),
        (5.0, 5.0, 0.45),
        (100.0, 1000.0, 0.05),
    ]
    return starts


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def main():
    data = pd.read_csv(HERE / 'nhanes.csv')
    # The comparison anchor uses paper values for epsilon/Xc/CV and profiles mex.
    paper_fit = DelayedEntryLikelihood(data, n=320, dt=0.025, quadrature=31)
    paper_nll_mex0 = paper_fit.likelihood(PAPER_XC, PAPER_EPSILON, PAPER_CV, 0.0)
    paper_nll_profiled, paper_mex_profiled = paper_fit.profile_mex(
        PAPER_XC, PAPER_EPSILON, PAPER_CV)
    paper_anchor = {'eta': ETA_FIXED, 'beta': BETA_FIXED,
        'epsilon': PAPER_EPSILON, 'Xc': PAPER_XC, 'CV': PAPER_CV,
        'kappa': KAPPA, 'mex': 0.0, 'nll': paper_nll_mex0,
        'profiled_mex_nll': paper_nll_profiled,
        'profiled_mex': paper_mex_profiled, 'k_mex_profiled': 1}
    save_json(HERE / 'paper_anchor_fit_grid.json', paper_anchor)
    print('PAPER_ANCHOR', json.dumps(paper_anchor), flush=True)

    coarse = DelayedEntryLikelihood(data, n=96, dt=0.1, quadrature=15)
    coarse_results = coarse.optimize(get_starts(), maxiter=150)
    save_json(HERE / 'coarse_multistart.json', coarse_results)
    print('COARSE_BEST', json.dumps(coarse_results[0]), flush=True)

    fit_grid = DelayedEntryLikelihood(data, n=320, dt=0.025, quadrature=31)
    fine_starts = [tuple(r['params'][k] for k in ('Xc','epsilon','CV'))
                   for r in coarse_results[:4]]
    fine_starts += [(PAPER_XC, PAPER_EPSILON, PAPER_CV)]
    fine_results = fit_grid.optimize(fine_starts, maxiter=220)
    save_json(HERE / 'fit_grid_multistart.json', fine_results)
    print('FIT_GRID_BEST', json.dumps(fine_results[0]), flush=True)

    check_grid = DelayedEntryLikelihood(data, n=480, dt=1/60, quadrature=41)
    check_starts = [tuple(r['params'][k] for k in ('Xc','epsilon','CV'))
                    for r in fine_results[:3]]
    check_starts += [(PAPER_XC, PAPER_EPSILON, PAPER_CV)]
    check_results = check_grid.optimize(check_starts, maxiter=220)
    best = check_results[0]
    best_params = best['params']
    check_nll = best['nll']
    paper_check = DelayedEntryLikelihood(data, n=480, dt=1/60, quadrature=41)
    paper_nll_check_mex0 = paper_check.likelihood(PAPER_XC, PAPER_EPSILON,
                                                   PAPER_CV, 0.0)
    paper_nll_check_profiled, paper_mex_check = paper_check.profile_mex(
        PAPER_XC, PAPER_EPSILON, PAPER_CV)
    # AIC is conditional on paper-fixed eta/beta. The anchor frees only mex;
    # the constrained refit frees Xc, epsilon, CV and mex.
    best['k'] = 4
    best['aic'] = 2 * check_nll + 2 * best['k']
    best['check_nll'] = check_nll
    best['aic_anchor_mex_profiled'] = 2 * paper_nll_check_profiled + 2
    best['delta_aic_vs_paper_anchor_mex_profiled'] = (
        best['aic'] - best['aic_anchor_mex_profiled'])
    best['nll_paper_point_mex0'] = paper_nll_check_mex0
    best['nll_paper_point_mex_profiled'] = paper_nll_check_profiled
    best['paper_point_profiled_mex'] = paper_mex_check
    best['nll_improvement_vs_paper_point_mex0'] = paper_nll_check_mex0 - check_nll
    best['cohort'] = {'n': int(len(data)), 'deaths': int(data.event.sum()),
                      'censored': int((data.event == 0).sum())}
    best['fixed'] = {'eta': ETA_FIXED, 'beta': BETA_FIXED, 'kappa': KAPPA}
    best['fit_grid'] = {'n_cells': 320, 'dt': 0.025, 'threshold_quadrature': 31}
    best['check_grid'] = {'n_cells': 480, 'dt': 1/60, 'threshold_quadrature': 41}
    best['bootstrap'] = False
    save_json(HERE / 'best_constrained_fit.json', best)
    save_json(HERE / 'check_grid_multistart.json', check_results)
    print('FINAL', json.dumps(best), flush=True)

if __name__ == '__main__':
    main()
