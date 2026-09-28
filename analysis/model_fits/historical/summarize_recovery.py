"""Read-only recovery audit. Never drops a completed simulation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
BOUNDS={'Xc':(.02,3000.),'epsilon':(.002,20000000.),'CV':(0.,.5),'mex':(0.,.2)}
def near_bounds(params):
    return [k for k,limits in BOUNDS.items() if
        (min(abs(np.log(params[k]/v)) for v in limits)<1e-5 if k in ['Xc','epsilon']
         else min(abs(params[k]-v) for v in limits)<1e-8)]
def summarize(rows):
    frame=pd.DataFrame(rows);out=[]
    if frame.empty:return out
    for key,group in frame.groupby(['kind','year','scenario','label'],dropna=False):
        n=len(group);count=int(group.covered.sum());phat=count/n;z=1.95996398454
        center=(phat+z*z/(2*n))/(1+z*z/n)
        half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
        key=tuple(v.item() if isinstance(v,np.generic) else v for v in key)
        out.append(dict(zip(['kind','year','scenario','label'],key),n=n,covered=count,
            coverage=phat,coverage_mc_low=float(center-half),coverage_mc_high=float(center+half),
            mean_standardized_error=float(group.standardized_error.mean()),
            sd_standardized_error=float(group.standardized_error.std(ddof=1)) if n>1 else None,
            misses_above=int((group.low>group.truth).sum()),misses_below=int((group.high<group.truth).sum()),
            median_estimate=float(group.estimate.median()),truth=float(group.truth.iloc[0])))
    return out

if __name__=='__main__':
    rows=[];complete=[];failures=[];audits=[];contour_complete=[];constrained=[]
    for folder in sorted((HERE/'recovery').glob('r*')):
        path=folder/'uncertainty.json'
        if not path.exists():continue
        x=json.loads(path.read_text());complete.append(x['rep'])
        if not x['baseline_success']:failures.append(dict(rep=x['rep'],stage='baseline'))
        hits=near_bounds(x['baseline'])
        if hits:constrained.append(dict(rep=x['rep'],stage='baseline',bounds=hits))
        for year,row in x['ratios'].items():rows.append(dict(row,kind='historical',year=int(year),scenario='historical',label='Xc ratio',rep=x['rep']))
        for row in x['projections']:rows.append(dict(row,kind='projection',label='Xc ratio',rep=x['rep']))
        for f in folder.glob('*_Xc.json'):
            h=json.loads(f.read_text())
            if not h['success']:failures.append(dict(rep=x['rep'],stage=f.stem))
            hits=near_bounds(h['params'])
            if hits:constrained.append(dict(rep=x['rep'],stage=f.stem,bounds=hits))
            if 'broad_audit_score_difference' in h:audits.append(dict(rep=x['rep'],year=int(f.stem.split('_')[0]),score_difference=h['broad_audit_score_difference']))
        if (folder/'contour_coverage.json').exists():
            c=json.loads((folder/'contour_coverage.json').read_text());contour_complete.append(x['rep'])
            rows.extend(dict(row,kind='contour',rep=x['rep']) for row in c['rows'])
    summary=summarize(rows)
    pd.DataFrame(rows).to_csv(HERE/'recovery_draws.csv',index=False)
    pd.DataFrame(summary).to_csv(HERE/'recovery_coverage.csv',index=False)
    report=dict(complete_replicates=complete,contour_complete_replicates=contour_complete,
        optimizer_failures=failures,constrained_fits=constrained,broad_search_audits=audits,coverage=summary,
        inference='model-conditional independent Poisson simulation; each dataset gets its own refitted baseline and joint sandwich intervals',
        caveat='100 simulations give about 2.2 percentage points Monte Carlo SE near 95%; this is a calibration screen, not proof of coverage')
    (HERE/'recovery_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(completed=len(complete),contour_completed=len(contour_complete),failures=len(failures),coverage=summary),indent=2))
