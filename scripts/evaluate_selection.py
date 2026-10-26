"""Evaluate supported-family selection separately from parameter fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from realityforge.inference import infer_track
from realityforge.tracking import Track,track_video
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for amplitude in [.05,.1,.2,.5,.9]:
    t=np.arange(0,6,1/30);k=19.62
    a=solve_ivp(lambda _,z:[z[1],-k*np.sin(z[0])-.1*z[1]],(0,float(t[-1])),[amplitude,0],t_eval=t,rtol=1e-10,atol=1e-11).y[0]
    xy=np.array([320,40])+230*np.c_[np.sin(a),np.cos(a)]
    track=Track(t,xy,np.ones(len(t)),640,480,30,len(t),[])
    r=infer_track(track,{'mode':'auto'},method='analytical')
    rows.append({'id':f'synthetic-pendulum-{amplitude}','kind':'synthetic','expected':'pendulum','selected':r['model'],'selection':r['selection']})
for accel in [-10,0,10]:
    t=np.arange(0,6,1/30);s=50*t+.5*accel*t*t
    track=Track(t,np.c_[100+s,np.full(len(t),200)],np.ones(len(t)),640,480,30,len(t),[])
    r=infer_track(track,{'mode':'auto'},method='analytical')
    rows.append({'id':f'synthetic-sliding-{accel}','kind':'synthetic','expected':'sliding','selected':r['model'],'selection':r['selection']})
for sample in json.loads((ROOT/'samples/manifest.json').read_text())['samples']:
    if sample['kind']!='real':continue
    settings={**sample['settings'],'mode':'auto'}
    track=track_video(ROOT/sample['video_url'].lstrip('/'),settings.get('roi'))
    try:
        r=infer_track(track,settings,method='analytical')
        rows.append({'id':sample['id'],'kind':'real','expected':sample['settings']['mode'],'selected':r['model'],'selection':r['selection']})
    except ValueError as e:rows.append({'id':sample['id'],'kind':'real','expected':sample['settings']['mode'],'selected':'abstain','reason':str(e)})
report={'scope':'Small supported-family geometry/reversal screen. No arbitrary equation discovery or blind-test claim. Parameter/trajectory benchmarks select the known family explicitly.','inference_sha256':hashlib.sha256((ROOT/'realityforge/inference.py').read_bytes()).hexdigest(),'correct':sum(r['expected']==r['selected'] for r in rows),'count':len(rows),'rows':rows}
(ROOT/'benchmarks/model-selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
