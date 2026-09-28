"""Dataset-specific coverage check for Fig5, including the extreme contour."""
from fitting import p,HERE,json,np,os
from inference import joint_vector,projected_params,parameter_mapping
from bands import metric,metric_jacobian,LABELS

if __name__=='__main__':
    rep=int(os.environ['LSB_JOBINDEX']);folder=HERE/'recovery'/f'r{rep:04d}'
    if (folder/'contour_coverage.json').exists():
        print(json.dumps({'rep':rep,'cached':True}),flush=True)
        raise SystemExit(0)
    data=json.loads((folder/'uncertainty.json').read_text());years=data['years'];base=data['baseline']
    required=list(range(1980,2016,5))
    if [y for y in years if 1980<=y<2019]!=required:
        raise ValueError('Recovery and production trend-year sets differ; contour truth would be mismatched')
    histories={y:json.loads((folder/f'{y}_Xc.json').read_text())['params'] for y in years}
    q=joint_vector(base,histories);cov=np.array(data['covariance']);rows=[]
    targets=json.loads((HERE/'joint_covariance.json').read_text())['targets']
    for scenario in ['linear','exponential']:
        for year in [2050,2100]:
            index=targets.index(dict(year=year,scenario=scenario))
            truth=json.loads((HERE/f'bands/{index:03d}.json').read_text())['metrics']
            params=projected_params(q,base,years,year,scenario)
            mapping=parameter_mapping(q,base,years,year,scenario)
            jac=metric_jacobian(params);v=jac@mapping@cov@mapping.T@jac.T
            se=np.sqrt(np.maximum(np.diag(v),0));estimates=metric(params)
            for i,label in enumerate(LABELS):
                value=float(estimates[i]);target=truth[i]['estimate'];s=float(se[i])
                lo=value-1.95996398454*s;hi=value+1.95996398454*s
                rows.append(dict(year=year,scenario=scenario,label=label,estimate=value,truth=target,
                    se=s,low=lo,high=hi,covered=bool(lo<=target<=hi),standardized_error=(value-target)/s))
    p.save(folder/'contour_coverage.json',dict(rep=rep,rows=rows))
    print(json.dumps({'rep':rep,'complete':True}),flush=True)
