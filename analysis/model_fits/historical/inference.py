"""Model-conditional sandwich covariance for shared-baseline two-stage fits.

The baseline is fitted to 2019. Each historical fit estimates Xc and mex,
inheriting epsilon and CV from that same baseline. Off-diagonal bread blocks
propagate baseline uncertainty; treating yearly fits as independent is wrong.
Expected bread is used, appropriate under the stated model for recovery tests.
"""
import numpy as np

def encode(p,keys):
    return np.array([np.log(p[k]) if k in ('Xc','epsilon') else p[k]*1000 if k=='mex' else p[k] for k in keys])

def decode(q,anchor,keys):
    p=dict(anchor)
    for k,v in zip(keys,q):p[k]=float(np.exp(v) if k in ('Xc','epsilon') else v/1000 if k=='mex' else v)
    return p

def jacobian(model,params,keys):
    q=encode(params,keys);cols=[]
    for i,key in enumerate(keys):
        h=2e-4 if key!='CV' else 2e-5
        lo=q.copy();hi=q.copy();hi[i]+=h;lo[i]-=h
        if key in ('CV','mex') and lo[i]<0:lo[i]=q[i]
        a=np.log(np.maximum(model.rates(decode(lo,params,keys)),1e-250))
        b=np.log(np.maximum(model.rates(decode(hi,params,keys)),1e-250))
        cols.append((b-a)/(hi[i]-lo[i]))
    return np.array(cols).T

def pearson_dispersion(deaths,mu,k):
    """Estimate per-year dispersion on the positive-exposure age bins."""
    deaths=np.asarray(deaths);mu=np.asarray(mu)
    if len(mu)<=k or np.any(mu<=0):
        raise ValueError('Dispersion requires positive expected counts and n > k')
    pearson=float(np.sum((deaths-mu)**2/mu));phi=pearson/(len(mu)-k)
    return dict(pearson=pearson,n=len(mu),k=k,df=len(mu)-k,
                phi=phi,variance_multiplier=max(1.,phi))

def joint_covariance(model,baseline,histories,exposures,deaths=None):
    """Use per-year Pearson scaling when counts are supplied; otherwise Poisson.

    The unadjusted default is retained for the original recovery diagnostic.
    Scaling independent meat blocks before inversion preserves cross-year terms.
    """
    keys=['Xc','epsilon','CV','mex'];years=sorted(histories)
    n=4+2*len(years);bread=np.zeros((n,n));meat=np.zeros((n,n))
    dispersion=[]
    j=jacobian(model,baseline,keys);mu=exposures[2019]*model.rates(baseline);keep=exposures[2019]>0
    j=j[keep];mu=mu[keep]
    bread[:4,:4]=j.T@j;meat[:4,:4]=(j.T/mu)@j
    if deaths is not None:
        st=pearson_dispersion(np.asarray(deaths[2019])[keep],mu,4)
        meat[:4,:4]*=st['variance_multiplier'];dispersion.append(dict(year=2019,**st))
    for i,year in enumerate(years):
        sl=slice(4+2*i,6+2*i);params=histories[year]
        j=jacobian(model,params,keys);mu=exposures[year]*model.rates(params);keep=exposures[year]>0
        j=j[keep];mu=mu[keep];own=j[:,[0,3]]
        inherited=np.zeros((len(mu),4));inherited[:,[1,2]]=j[:,[1,2]]
        bread[sl,sl]=own.T@own;bread[sl,:4]=own.T@inherited
        meat[sl,sl]=(own.T/mu)@own
        if deaths is not None:
            st=pearson_dispersion(np.asarray(deaths[year])[keep],mu,2)
            meat[sl,sl]*=st['variance_multiplier'];dispersion.append(dict(year=year,**st))
    inv=np.linalg.inv(bread);cov=inv@meat@inv.T
    diagnostics={'bread_condition':float(np.linalg.cond(bread)),
                 'smallest_covariance_eigenvalue':float(np.linalg.eigvalsh(cov).min()),
                 'method':'expected-bread Poisson model-conditional sandwich; shared baseline propagated'}
    if deaths is not None:
        diagnostics.update(method='Per-year Pearson-dispersion adjusted expected-bread joint sandwich',
                           dispersion=dispersion,critical_value=1.95996398454,
                           coverage='Approximate pointwise sensitivity intervals; not coverage calibrated')
    ratios={}
    for i,year in enumerate(years):
        g=np.zeros(n);g[0]=-1;g[4+2*i]=1
        logratio=np.log(histories[year]['Xc']/baseline['Xc'])
        se=float(np.sqrt(max(0,g@cov@g)))
        ratios[year]={'estimate':float(np.exp(logratio)),'log_se':se,
                      'low':float(np.exp(logratio-1.95996398454*se)),
                      'high':float(np.exp(logratio+1.95996398454*se))}
    return cov,ratios,diagnostics

def joint_vector(baseline,histories):
    return np.r_[encode(baseline,['Xc','epsilon','CV','mex']),
        *[encode(histories[y],['Xc','mex']) for y in sorted(histories)]]

def projected_params(q,baseline,years,year,scenario):
    result=decode(q[:4],baseline,['Xc','epsilon','CV','mex'])
    if scenario=='historical':
        if year==2019:return result
        i=years.index(year);return decode(q[4+2*i:6+2*i],result,['Xc','mex'])
    recent=[y for y in years if 1980<=y<2019]+[2019]
    xs=np.array([q[4+2*years.index(y)] if y!=2019 else q[0] for y in recent])
    if scenario=='linear':xs=np.exp(xs)
    slope=np.polyfit(np.array(recent)-2019,xs,1)[0]
    result['Xc']=float(np.exp(q[0])+slope*(year-2019) if scenario=='linear'
                       else np.exp(q[0]+slope*(year-2019)))
    if result['Xc']<=0:raise ValueError('Nonpositive projected Xc')
    return result

def parameter_mapping(q,baseline,years,year,scenario):
    keys=['Xc','epsilon','CV','mex'];cols=[]
    for i in range(len(q)):
        h=1e-5;lo=q.copy();hi=q.copy();lo[i]-=h;hi[i]+=h
        cols.append((encode(projected_params(hi,baseline,years,year,scenario),keys)-
            encode(projected_params(lo,baseline,years,year,scenario),keys))/(2*h))
    return np.array(cols).T
