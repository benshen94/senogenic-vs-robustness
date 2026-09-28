"""One reproducible joint parametric draw with dataset-specific uncertainty."""
from fitting import p,score,HERE,np,json,os,fit_baseline,fit_history
from inference import joint_covariance,joint_vector,projected_params,parameter_mapping
import argparse,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rep',type=int,default=int(os.getenv('LSB_JOBINDEX','0')))
    ap.add_argument('--years',nargs='+',type=int,default=[1900,1980,1985,1990,1995,2000,2005,2010,2015])
    a=ap.parse_args();p.score=score
    out=HERE/'recovery'/f'r{a.rep:04d}';out.mkdir(parents=True,exist_ok=True)
    truth=json.loads((HERE/'point/baseline.json').read_text())['params']
    model=p.Model();exposures={};histories={};begin=time.monotonic()
    def dataset(year,params):
        _,e=p.dataset(year,0,1);exposures[year]=e
        mu=e*model.rates(params)
        rng=np.random.default_rng(np.random.SeedSequence([2026092701,a.rep,year]))
        return rng.poisson(mu).astype(float),e
    d,e=dataset(2019,truth)
    dest=out/'baseline.json'
    if dest.exists():b=json.loads(dest.read_text())
    else:
        b=fit_baseline(model,truth,d,e,out/'baseline_starts.json');p.save(dest,b)
    truth_ratios={};truth_histories={}
    for year in a.years:
        generating=json.loads((HERE/f'point_full/{year}_Xc.json').read_text())['params']
        truth_histories[year]=generating
        truth_ratios[year]=generating['Xc']/truth['Xc']
        d,e=dataset(year,generating);dest=out/f'{year}_Xc.json'
        if dest.exists():h=json.loads(dest.read_text())
        else:
            anchor=dict(b['params'],Xc=generating['Xc']*b['params']['Xc']/truth['Xc'])
            h=fit_history(model,anchor,d,e,local=True)
            if a.rep<=5 and year in [1900,2015]:
                audit=fit_history(model,anchor,d,e)
                h['broad_audit_score_difference']=h['score']-audit['score']
                if audit['score']<h['score']-1e-5:
                    h['local_search_missed_better_solution']=True
                    h['params']=audit['params'];h['score']=audit['score']
            p.save(dest,h)
        histories[year]=h['params']
    cov,ratios,diagnostics=joint_covariance(model,b['params'],histories,exposures)
    for year,r in ratios.items():
        r['truth']=truth_ratios[year];r['covered']=bool(r['low']<=r['truth']<=r['high'])
        r['standardized_error']=float(np.log(r['estimate']/r['truth'])/r['log_se'])
    projections=[];years=sorted(histories);q=joint_vector(b['params'],histories);true_q=joint_vector(truth,truth_histories)
    for scenario in ['linear','exponential']:
        for year in [2050,2100]:
            fitted=projected_params(q,b['params'],years,year,scenario)
            generating=projected_params(true_q,truth,years,year,scenario)
            g=parameter_mapping(q,b['params'],years,year,scenario)[0];g[0]-=1
            se=float(np.sqrt(max(0,g@cov@g)))
            estimate=fitted['Xc']/b['params']['Xc'];target=generating['Xc']/truth['Xc']
            lo=float(estimate*np.exp(-1.95996398454*se));hi=float(estimate*np.exp(1.95996398454*se))
            projections.append(dict(year=year,scenario=scenario,estimate=estimate,truth=target,
                low=lo,high=hi,log_se=se,covered=bool(lo<=target<=hi),
                standardized_error=float(np.log(estimate/target)/se)))
    p.save(out/'uncertainty.json',dict(rep=a.rep,baseline=b['params'],ratios=ratios,
        covariance=cov.tolist(),years=sorted(histories),diagnostics=diagnostics,
        projections=projections,baseline_success=b['success'],baseline_bounds=b['bound_hits'],seconds=time.monotonic()-begin))
    print(json.dumps({'rep':a.rep,'ratios':ratios,'seconds':time.monotonic()-begin}),flush=True)
if __name__=='__main__':main()
