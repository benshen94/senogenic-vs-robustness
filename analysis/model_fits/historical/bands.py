"""Propagate shared-baseline covariance through the actual SR solver.

Produces per-year dispersion-adjusted pointwise sensitivity bands. The older
Poisson recovery experiment does not calibrate these intervals.
"""
from fitting import p,HERE,json,np,os
from inference import joint_covariance,joint_vector,projected_params,parameter_mapping,encode,decode
import argparse
KEYS=['Xc','epsilon','CV','mex']
LABELS=['Mean','Top 10%','Top 1%','Top 0.01%']
def metric(params):
    result=p.prediction(params);prediction=result['contours']['20']
    bound=prediction['mean_tail_upper_bound']
    if not result['tail_resolved'] or prediction['mean_is_restricted'] or bound is None or bound>1e-4:
        raise ValueError('Unresolved unrestricted mean or insufficient tail bound; no nominal interval produced')
    values=np.array([prediction['mean_attained_age'],
        *[prediction['contours'][k] for k in ['0.1','0.01','0.0001']]],float)
    if not np.isfinite(values).all():raise ValueError('Unresolved or nonfinite contour metric')
    return values

def metric_jacobian(params):
    q=encode(params,KEYS);cols=[]
    for i,k in enumerate(KEYS):
        h=2e-4 if k!='CV' else 2e-5
        lo=q.copy();hi=q.copy();lo[i]-=h;hi[i]+=h
        if k in ['CV','mex'] and lo[i]<0:lo[i]=q[i]
        cols.append((metric(decode(hi,params,KEYS))-metric(decode(lo,params,KEYS)))/(hi[i]-lo[i]))
    return np.array(cols).T

def prepare():
    b=json.loads((HERE/'point/baseline.json').read_text())['params']
    histories={y:json.loads((HERE/f'point_full/{y}_Xc.json').read_text())['params'] for y in p.YEARS if y!=2019}
    datasets={y:p.dataset(y,0,1) for y in p.YEARS}
    deaths={y:values[0] for y,values in datasets.items()}
    e={y:values[1] for y,values in datasets.items()}
    cov,ratios,diagnostics=joint_covariance(p.Model(),b,histories,e,deaths=deaths)
    p.save(HERE/'dispersion.json',diagnostics['dispersion'])
    q=joint_vector(b,histories);years=sorted(histories)
    targets=[dict(year=y,scenario='historical') for y in p.YEARS if y>=1900]
    targets += [dict(year=y,scenario=sc) for sc in ['linear','exponential'] for y in [2019,*range(2020,2101,5)]]
    p.save(HERE/'joint_covariance.json',dict(baseline=b,years=years,q=q.tolist(),covariance=cov.tolist(),ratios=ratios,diagnostics=diagnostics,targets=targets))

def target(index):
    data=json.loads((HERE/'joint_covariance.json').read_text());t=data['targets'][index]
    q=np.array(data['q']);cov=np.array(data['covariance']);b=data['baseline'];years=data['years']
    params=projected_params(q,b,years,**t);mapping=parameter_mapping(q,b,years,**t)
    pcov=mapping@cov@mapping.T
    values=metric(params);j=metric_jacobian(params);v=j@pcov@j.T
    se=np.sqrt(np.maximum(np.diag(v),0));rows=[]
    for label,est,s in zip(LABELS,values,se):
        rows.append(dict(label=label,estimate=float(est),se=float(s),ci_low=float(est-1.95996398454*s),ci_high=float(est+1.95996398454*s)))
    p.save(HERE/f'bands/{index:03d}.json',dict(**t,params=params,metrics=rows,parameter_covariance=pcov.tolist(),metric_covariance=v.tolist(),method='Per-year Pearson-dispersion adjusted expected-bread sandwich, shared baseline propagated, pointwise normal delta intervals'))
    print(json.dumps(dict(**t,metrics=rows)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--index',type=int,default=int(os.getenv('LSB_JOBINDEX','1'))-1);a=ap.parse_args()
    if a.prepare:prepare()
    else:target(a.index)
