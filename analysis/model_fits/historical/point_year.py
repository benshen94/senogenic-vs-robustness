from fitting import p,HERE,json,np,score,fit_history,os
import time

if __name__=='__main__':
    year=p.YEARS[int(os.environ['LSB_JOBINDEX'])-1];p.score=score
    baseline=json.loads((HERE/'point/baseline.json').read_text())
    model=p.Model();d,e=p.dataset(year,0,1);begin=time.monotonic()
    if year==2019:result=dict(baseline)
    else:result=fit_history(model,baseline['params'],d,e)
    original=HERE/f'point/{year}_Xc.json'
    if original.exists():
        old=json.loads(original.read_text());result['multistart_score_difference']=result['score']-old['score']
        if result['score']>old['score']+1e-5:
            result['scalar_search_missed_better_solution']=True
            result['scalar_params']=result['params'];result['scalar_score']=result['score']
            result['params']=old['params'];result['score']=old['score']
    result['prediction']=p.prediction(result['params'])
    result['ratio']=result['params']['Xc']/baseline['params']['Xc']
    result['year']=year;result['seconds']=time.monotonic()-begin
    p.save(HERE/f'point_full/{year}_Xc.json',result)
    print(json.dumps({k:result[k] for k in ['year','params','score','seconds']}),flush=True)
