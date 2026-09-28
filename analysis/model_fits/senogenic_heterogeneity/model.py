"""Deterministic two-dimensional mixture using the existing validated SR solver.

Preserves old tau-spread construction: v~N(0, sigma^2), eta_i=eta exp(-v/2),
beta_i=beta exp(v/2). Threshold variation is independent positive Gaussian.
Rates use direct boundary deaths and model person-time, never log-S differences.
"""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMBA_NUM_THREADS']=os.getenv('SR_THREADS','1')
from pathlib import Path
import sys,json,time
import numpy as np
from functools import lru_cache
from scipy.special import roots_hermitenorm,logsumexp
from numba import njit,prange
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from senogenic_vs_robustness.sr_finite_volume import _forward_core
DATA=ROOT/'results/senogenic_heterogeneity'
HERE=ROOT/'tmp/senogenic_refit'
HERE.mkdir(parents=True,exist_ok=True)
BASE=json.loads((DATA/'baseline_input.json').read_text())['params']

AGES=np.arange(20,110)
LEVELS={'screen':(120,.1,11,7),'refine':(200,.05,21,11),'production':(320,.025,31,17),'fine':(640,.0125,61,25)}

@njit(parallel=True,cache=True)
def mixture_components(etas,betas,thresholds,epsilon,kappa,n,dt,steps):
    logs=np.empty((len(etas),steps+1));deaths=np.empty((len(etas),steps))
    for i in prange(len(etas)):
        logs[i],deaths[i]=_forward_core(etas[i],betas[i],epsilon,thresholds[i],n,dt,steps,False,kappa,True)
    return logs,deaths

class Model:
    def __init__(self,tau_cv=0.,level='screen',horizon=110):
        self.tau_cv=float(tau_cv);self.level=level
        self.n,self.dt,self.nx,self.nt=LEVELS[level];self.steps=round(horizon/self.dt)
        self.grid=np.arange(self.steps+1)*self.dt;self.peryear=round(1/self.dt)
        self.bins=AGES[:,None]*self.peryear+np.arange(self.peryear)[None,:]
        self.z,self.w=roots_hermitenorm(self.nx);self.w/=np.sqrt(2*np.pi)
        if tau_cv:
            z,w=roots_hermitenorm(self.nt);self.v=z*np.sqrt(np.log1p(tau_cv*tau_cv));self.tw=w/np.sqrt(2*np.pi)
        else:self.v=np.array([0.]);self.tw=np.array([1.])
    @lru_cache(maxsize=3)
    def components(self,eta,beta,epsilon,xc,cv,kappa):
        xs=xc*(1+cv*self.z) if cv>1e-10 else np.array([xc])
        w=self.w.copy() if cv>1e-10 else np.array([1.]);keep=xs>0;xs=xs[keep];w=w[keep];w/=w.sum()
        etas=np.repeat(eta*np.exp(-self.v/2),len(xs));betas=np.repeat(beta*np.exp(self.v/2),len(xs))
        thresholds=np.tile(xs,len(self.v));weights=(self.tw[:,None]*w[None,:]).ravel()
        logs,deaths=mixture_components(etas,betas,thresholds,epsilon,kappa,self.n,self.dt,self.steps)
        logweights=np.log(weights)[:,None];ls=logsumexp(logs+logweights,axis=0)
        q=(np.exp(logs[:,:-1]+logweights-ls[:-1])*deaths).sum(axis=0)
        return ls,q
    def intrinsic(self,p):return self.components(*(float(p[k]) for k in ['eta','beta','epsilon','Xc','CV','kappa']))
    def rates(self,p):
        ls,q=self.intrinsic(p);mex=p['mex'];bins=self.bins
        # Per-year normalization cancels exactly from deaths/person-time and
        # avoids underflow when exploring poor fits with very short lifetimes.
        risk=np.exp(ls[bins]-ls[bins[:,0]][:,None]-mex*self.dt*np.arange(self.peryear)[None,:])
        qt=q[bins]+(1-q[bins])*(-np.expm1(-mex*self.dt))
        return (risk*qt).sum(axis=1)/(risk*self.dt*(1-.5*qt)).sum(axis=1)
    def survival(self,p):
        ls,_=self.intrinsic(p);return np.exp(ls-np.interp(20,self.grid,ls)-p['mex']*(self.grid-20))

def objective(model,p,d,e):
    keep=e>0;mu=e[keep]*model.rates(p)[keep]
    if not np.isfinite(mu).all() or (mu<=0).any():return 1e100
    ref=np.maximum(d[keep],1.)
    return float(np.sum(d[keep]/mu+np.log(mu/ref)-d[keep]/ref))
