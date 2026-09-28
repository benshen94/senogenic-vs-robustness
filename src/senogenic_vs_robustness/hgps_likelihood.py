"""HGPS right-censored likelihood used by the archived Q-reference fits."""
import numpy as np
from scipy.special import logsumexp, roots_legendre
from .sr_finite_volume import forward

NAMES = ("eta", "beta", "epsilon", "Xc")
MODELS = (("eta",), ("beta",), ("Xc",), ("epsilon",),
          ("Xc", "epsilon"), ("Xc", "eta"), ("Xc", "beta"),
          ("eta", "beta"), ("epsilon", "eta"), ("epsilon", "beta"))

class HGPSLikelihood:
    def __init__(self, records, baseline, grid):
        self.n, self.dt, quad = grid
        self.steps = round(30/self.dt)
        self.t = np.arange(self.steps+1)*self.dt
        self.age = np.asarray([r['age'] for r in records])
        self.death = np.asarray([r['death'] for r in records], dtype=bool)
        self.base = np.asarray(baseline[:4], dtype=float)
        self.cv = float(baseline[4])
        if self.cv == 0:
            self.z = np.array([0.])
            self.lw = np.array([0.])
        else:
            z, w = roots_legendre(quad)
            lo = max(-8., -1/self.cv+1e-10)
            self.z = lo+(z+1)*(8-lo)/2
            self.lw = np.log(w)-self.z**2/2
            self.lw -= logsumexp(self.lw)
        self.calls = 0

    def curve(self, factors):
        eta, beta, eps, xc = self.base*np.asarray(factors)
        components = [lw+forward(eta,beta,eps,xc*(1+self.cv*z),self.n,self.dt,self.steps,log_output=True)
                      for z,lw in zip(self.z,self.lw)]
        ls = logsumexp(np.asarray(components), axis=0)
        loss = -np.expm1(np.minimum(np.diff(ls), 0.))
        h = loss/(self.dt*(1-.5*loss))
        self.calls += 1
        return ls, np.maximum(h, 1e-300)

    def nll(self, factors):
        ls, h = self.curve(factors)
        log_s = np.interp(self.age, self.t, ls)
        event_h = np.interp(self.age[self.death], self.t[:-1]+self.dt/2, h)
        return float(-np.sum(log_s)-np.sum(np.log(np.maximum(event_h,1e-300))))
