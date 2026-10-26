#!/usr/bin/env python3
"""Independent seeded synthetic test set, generated with SciPy rather than model RK4.

This evaluates inverse dynamics given trajectories, not video tracking. No numbers
from this script establish real-footage accuracy or a reproduced published baseline.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from scipy.integrate import solve_ivp
from realityforge.inference import infer_track
from realityforge.tracking import Track


def make_cases(seed=2026,count=24):
    rng=np.random.default_rng(seed)
    cases=[]
    for mode in ['pendulum','sliding']:
        for index in range(count):
            duration=float(rng.uniform(4.5,6.5));t=np.arange(0,duration,1/30)
            noise=[0.,.5,2.][index%3]
            if mode=='pendulum':
                truth={'k':float(rng.uniform(8,30)),'beta':float(rng.uniform(.05,.5)),'theta0':float(rng.uniform(.25,.9)),'omega0':float(rng.uniform(-.6,.6))}
                angle=solve_ivp(lambda _,z:[z[1],-truth['k']*np.sin(z[0])-truth['beta']*z[1]],(0,float(t[-1])),[truth['theta0'],truth['omega0']],t_eval=t,rtol=1e-11,atol=1e-12).y[0]
                radius=float(rng.uniform(180,240));pivot=np.array([320.,40.])
                clean=pivot+radius*np.c_[np.sin(angle),np.cos(angle)]
            else:
                truth={'acceleration':float(rng.uniform(4,16)),'v0':float(rng.uniform(15,35)),'x0':0.}
                direction=np.array([1.,.15]);direction/=np.linalg.norm(direction)
                scalar=truth['v0']*t+.5*truth['acceleration']*t*t
                clean=np.array([60.,100.])+scalar[:,None]*direction
            xy=clean+rng.normal(0,noise,clean.shape)
            confidence=np.ones(len(t))
            outliers=(index==count-1)
            if outliers:
                corrupt=rng.choice(len(t),max(1,int(.05*len(t))),replace=False)
                xy[corrupt]+=rng.normal(0,15,(len(corrupt),2));confidence[corrupt]=.2
            track=Track(t,xy,confidence,640,480,30.,len(t),[])
            cases.append({'id':f'{mode}-{index:02d}','mode':mode,'track':track,'clean':clean,'truth':truth,'noise_px':noise,'outlier_fraction':.05 if outliers else 0.})
    return cases


def evaluate(seed=2026,count=24,output='benchmarks/synthetic.json'):
    source_files=['realityforge/inference.py','realityforge/physics.py','realityforge/estimator.py','models/pendulum.pt','models/sliding.pt']
    artifact_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_files}
    methods=['hybrid','neural','analytical','analytical_refine','constant_velocity','iris_recurrence']
    rows=[]
    for case in make_cases(seed,count):
        track=case['track'];t=track.time;n=int(len(t)*.65)
        for method in methods:
            if method=='iris_recurrence' and case['mode']!='pendulum': continue
            row={k:case[k] for k in ['id','mode','truth','noise_px','outlier_fraction']};row['method']=method
            start=time.perf_counter()
            try:
                if method=='constant_velocity':
                    coefficients=np.polyfit(t[:n],track.xy[:n],1)
                    predicted=t[:,None]*coefficients[0]+coefficients[1]
                    parameters={};ci={}
                else:
                    result=infer_track(track,{'mode':case['mode'],'fit_fraction':.65},method=method)
                    predicted=np.array(result['predicted_xy'])
                    parameters=result['simulation']['parameters']
                    ci={p['name']:p.get('ci95') for p in result['parameters']}
                    row['fit_rmse_px']=result['metrics']['fit_rmse_px']
                row['runtime_s']=time.perf_counter()-start
                row['parameters']=parameters
                row['forecast_rmse_px']=float(np.sqrt(np.mean(np.sum((predicted[n:]-case['clean'][n:])**2,axis=1))))
                row['forecast_nrmse']=row['forecast_rmse_px']/max(float(np.ptp(case['clean'],axis=0).max()),1)
                if method!='constant_velocity':
                    key='k' if case['mode']=='pendulum' else 'acceleration'
                    row['primary_parameter_relative_error']=abs(parameters[key]-case['truth'][key])/abs(case['truth'][key])
                    if case['mode']=='pendulum': row['damping_absolute_error']=abs(parameters['beta']-case['truth']['beta'])
                    row['primary_ci_covers_truth']=None if ci[key] is None else bool(ci[key][0]<=case['truth'][key]<=ci[key][1])
            except Exception as error:
                row['runtime_s']=time.perf_counter()-start;row['error']=f'{type(error).__name__}: {error}'
            rows.append(row)
            print(case['id'],method,round(row.get('forecast_rmse_px',float('nan')),3),flush=True)
    summary=[]
    for mode in ['pendulum','sliding']:
        for method in methods:
            group=[r for r in rows if r['mode']==mode and r['method']==method]
            if not group: continue
            success=[r for r in group if 'error' not in r]
            item={'mode':mode,'method':method,'count':len(group),'failures':len(group)-len(success)}
            for metric in ['forecast_rmse_px','forecast_nrmse','runtime_s','primary_parameter_relative_error','damping_absolute_error']:
                values=[r[metric] for r in success if metric in r]
                if values:
                    item[metric+'_median']=float(np.median(values));item[metric+'_mean']=float(np.mean(values));item[metric+'_max']=float(np.max(values))
            coverage=[r['primary_ci_covers_truth'] for r in success if r.get('primary_ci_covers_truth') is not None]
            if coverage: item['primary_ci_coverage']=float(np.mean(coverage))
            summary.append(item)
    report={'artifact_sha256':artifact_hashes,'protocol':{'seed':seed,'cases_per_family':count,'fit_fraction':.65,'forecast_fraction':.35,'frame_rate':30,'noise_px':[0,.5,2],'outlier_note':'Last case per family receives 5% positional outliers with confidence 0.2.','ground_truth_solver':'scipy.integrate.solve_ivp rtol=1e-11 atol=1e-12 for pendulum; exact polynomial for sliding','training_overlap':'none: seed 2026 held out from training seed47','scope':'trajectory-level synthetic evaluation; excludes decoding and tracking','baseline_note':'analytical uses smoothed finite-difference angular least squares for pendulum and quadratic least squares for sliding. It is an IRIS-inspired classical baseline, not a reproduction of reported IRIS published scores.'},'environment':{'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads()},'summary':summary,'cases':rows}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=2026);parser.add_argument('--cases',type=int,default=24);parser.add_argument('--output',default='benchmarks/synthetic.json');args=parser.parse_args();evaluate(args.seed,args.cases,args.output)
