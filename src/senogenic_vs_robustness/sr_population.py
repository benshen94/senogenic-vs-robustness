"""Population and sibling mixtures of the production SR first-passage solver.

Parameters are assigned at birth and remain fixed. Mixture weights update through
survival; hazards are never averaged with the original birth weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, roots_legendre, logsumexp

from .sr_finite_volume import forward


@dataclass(frozen=True)
class Grid:
    cells: int = 320
    dt: float = .025
    horizon: float = 420.
    nodes: int = 64


@lru_cache(maxsize=256)
def positive_normal(cv: float, nodes: int):
    """Quadrature for N(1,cv²) conditional on positivity (CV before truncation).

    Normal tails outside ±9 SD have total probability <3e-19. Gauss–Legendre
    integration in the standardized coordinate resolves the extreme-age mixture.
    """
    if cv < 0:
        raise ValueError('CV must be nonnegative')
    if cv == 0:
        return np.ones(1), np.ones(1), np.zeros(1)
    lower, upper = max(-1 / cv, -9.), 9.
    x, w = roots_legendre(nodes)
    z = lower + (x + 1) * (upper - lower) / 2
    w = w * (upper - lower) / 2 * np.exp(-z*z/2) / np.sqrt(2*np.pi)
    w /= w.sum()
    return 1 + cv*z, w, z


@dataclass
class Population:
    time: np.ndarray
    log_components: np.ndarray
    death_fractions: np.ndarray
    weights: np.ndarray
    grid: Grid

    def log_survival(self, weights=None):
        w = self.weights if weights is None else np.asarray(weights)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError('Invalid mixture weights')
        w = w/w.sum()
        with np.errstate(divide='ignore'):
            return logsumexp(self.log_components + np.log(w)[:, None], axis=0)

    def quantile(self, surviving_fraction, weights=None):
        if not 0 < surviving_fraction < 1:
            raise ValueError('Surviving fraction must be in (0,1)')
        log_s = self.log_survival(weights)
        target = np.log(surviving_fraction)
        if log_s[-1] > target:
            raise ValueError(f'Survival {surviving_fraction:g} not reached by {self.time[-1]:g}')
        return float(np.interp(target, log_s[::-1], self.time[::-1]))

    def conditional(self, ages, entry=90., weights=None):
        log_s = self.log_survival(weights)
        return np.exp(np.interp(ages, self.time, log_s)-np.interp(entry, self.time, log_s))

    def annual_mortality(self, ages, weights=None):
        """Flux deaths / trapezoidal person-years within [age, age+1)."""
        w = self.weights if weights is None else np.asarray(weights)
        with np.errstate(divide='ignore'):
            log_risk = self.log_components[:, :-1] + np.log(w/w.sum())[:, None]
        per_year = round(1/self.grid.dt)
        if abs(per_year*self.grid.dt-1) > 1e-10:
            raise ValueError('Time step must divide one year')
        out = []
        for age in ages:
            start = round(float(age)/self.grid.dt)
            sl = slice(start, start+per_year)
            if start < 0 or start+per_year > self.death_fractions.shape[1]:
                raise ValueError('Requested rate outside solver horizon')
            lr = log_risk[:, sl]
            risk = np.exp(lr-np.max(lr))
            q = self.death_fractions[:, sl]
            out.append(float(np.sum(risk*q)/np.sum(risk*self.grid.dt*(1-.5*q))))
        return np.asarray(out)


def population(params, focal='Xc', cv=0., factor=1., grid=Grid(), mex=0.):
    """Vary one intrinsic parameter; other intrinsic parameters are homogeneous.

    A fitted CV field in params is intentionally NOT implicitly applied: callers
    specify the controlled experiment's CV explicitly, including for Xc.
    """
    if focal not in ('eta','beta','epsilon','Xc') or factor <= 0 or mex < 0:
        raise ValueError('Invalid experiment')
    factors, weights, _ = positive_normal(float(cv), grid.nodes)
    steps = round(grid.horizon/grid.dt)
    time = np.arange(steps+1)*grid.dt
    logs, deaths = [], []
    for f in factors:
        p = {k:float(params[k]) for k in ('eta','beta','epsilon','Xc','kappa')}
        p[focal] *= factor*float(f)
        s,q = forward(p['eta'],p['beta'],p['epsilon'],p['Xc'],grid.cells,
                      grid.dt,steps,kappa=p['kappa'],log_output=True,return_deaths=True)
        logs.append(s-mex*time)
        deaths.append(q+(1-q)*(-np.expm1(-mex*grid.dt)))
    return Population(time,np.stack(logs),np.stack(deaths),weights,grid)


def sibling_joint_weights(cv, nodes, rho=.5):
    """Joint quadrature reproducing legacy DZ Gaussian draws and positivity repair.

    Original correlated Gaussian draws have rho=.5. A nonpositive member is
    independently redrawn from its positive marginal, as in sr_utils. This
    preserves each positive marginal but slightly changes the final correlation.
    """
    _, w, z = positive_normal(float(cv), nodes)
    if cv == 0 or rho == 0:
        return np.outer(w,w)
    if not -1 < rho < 1:
        raise ValueError('Correlation must lie strictly between -1 and 1')
    a = -1/cv
    positive = ndtr(-a)
    sigma = np.sqrt(1-rho*rho)
    both_negative = quad(lambda u: np.exp(-u*u/2)/np.sqrt(2*np.pi)*
                         ndtr((a-rho*u)/sigma),-np.inf,a,epsabs=1e-13)[0]
    zi,zj = z[:,None],z[None,:]
    ratio = np.exp((2*rho*zi*zj-rho*rho*(zi*zi+zj*zj))/(2*(1-rho*rho)))/sigma
    repaired_ratio = positive**2*ratio + positive*(ndtr((a-rho*zi)/sigma)+
                        ndtr((a-rho*zj)/sigma)) + both_negative
    joint = w[:,None]*w[None,:]*repaired_ratio
    return joint/joint.sum()


def sibling_populations(pop, cv, rho=.5):
    """Condition sibling weights on an independently evolving proband's lifespan."""
    joint = sibling_joint_weights(cv,pop.grid.nodes,rho)
    marginal = joint.sum(axis=1)
    top_age = pop.quantile(.01,marginal)
    bottom_age = pop.quantile(.9,marginal)
    survival_top = np.exp([np.interp(top_age,pop.time,s) for s in pop.log_components])
    death_bottom = -np.expm1([np.interp(bottom_age,pop.time,s) for s in pop.log_components])
    good = joint.T@survival_top
    bad = joint.T@death_bottom
    good /= good.sum()
    bad /= bad.sum()
    return dict(full=marginal,good=good,bad=bad), dict(top_age=top_age,bottom_age=bottom_age)
