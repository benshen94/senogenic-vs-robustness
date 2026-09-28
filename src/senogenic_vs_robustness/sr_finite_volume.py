"""Conservative finite-volume SR on a grid resolving the kappa boundary layer.

Cell probabilities; exponentially fitted face flux; exactly reflecting x=0
and absorbing x=Xc at the actual boundary (half-cell distance at uniform grid).
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _forward_core(eta,beta,eps,xc,n,dt,steps,coupled=False,kappa=.5,log_output=False):
    edges=np.empty(n+1)
    scale=np.log1p(xc/kappa)
    for j in range(n+1):edges[j]=kappa*np.expm1(scale*j/n)
    centers=(edges[:-1]+edges[1:])/2
    widths=edges[1:]-edges[:-1]
    up=np.empty(n);down=np.empty(n);diag=np.empty(n);upper=np.empty(n);rhs=np.empty(n)
    p=np.zeros(n);p[0]=1.;s=np.zeros(steps+1);q_steps=np.zeros(steps);log_survival=0.
    for k in range(1,steps+1):
        for j in range(n):
            if j<n-1:
                dist=centers[j+1]-centers[j];x=(centers[j+1]+centers[j])/2
            else:dist=xc-centers[j];x=(xc+centers[j])/2
            prod=eta*k*dt;rem=beta*x/(kappa+x);drift=prod-rem
            diffusion=eps
            if coupled:
                diffusion=eps*(prod+rem)/2
                drift-=eps*beta*kappa/(kappa+x)**2/2
            z=min(max(drift*dist/diffusion,-500.),500.)
            if abs(z)<1e-7:bp=1-z/2+z*z/12;bm=1+z/2+z*z/12
            else:bp=z/np.expm1(z);bm=-z/np.expm1(-z)
            up[j]=diffusion/dist*bm/widths[j]
            down[j]=diffusion/dist*bp/widths[j+1] if j<n-1 else 0.
        for j in range(n):
            diag[j]=1+dt*(up[j]+(down[j-1] if j else 0.));upper[j]=-dt*down[j] if j<n-1 else 0.;rhs[j]=p[j]
        for j in range(1,n):
            w=-dt*up[j-1]/diag[j-1];diag[j]-=w*upper[j-1];rhs[j]-=w*rhs[j-1]
        p[-1]=rhs[-1]/diag[-1]
        for j in range(n-2,-1,-1):p[j]=(rhs[j]-upper[j]*p[j+1])/diag[j]
        # Renormalize the conditional state. Boundary flux gives the mass loss
        # without subtracting nearly equal numbers or underflowing survival.
        mass=p.sum()
        loss=dt*up[-1]*p[-1]
        # Direct boundary-flux deaths divided by total mass at the interval
        # start. The implicit FV balance is old mass = surviving mass + flux.
        q_steps[k-1]=loss/(mass+loss)
        log_survival-=np.log1p(loss/mass)
        p/=mass
        s[k]=log_survival
    curve=s if log_output else np.exp(s)
    return curve,q_steps

def forward(eta,beta,eps,xc,n,dt,steps,coupled=False,kappa=.5,log_output=False,
            return_deaths=False):
    """Finite-volume SR curve; optionally return direct interval death fractions."""
    curve,q_steps=_forward_core(eta,beta,eps,xc,n,dt,steps,coupled,kappa,log_output)
    if return_deaths:
        return curve,q_steps
    return curve
