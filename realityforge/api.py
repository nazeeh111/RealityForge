"""Local-only web service. Uploaded video bytes never leave this process."""
from pathlib import Path
import json
import uuid
import shutil
import threading
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from .inference import analyze
from .physics import simulate

from .resources import ROOT, RUNTIME
RUNTIME.mkdir(parents=True,exist_ok=True)
app=FastAPI(title='RealityForge',version='0.1.0')
# Bound concurrent optimization on small CPU hosts.
lock=threading.Lock()


def samples():
    path=ROOT/'samples'/'manifest.json'
    return json.loads(path.read_text())['samples'] if path.exists() else []


class AnalysisSettings(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    mode:Literal['auto','pendulum','sliding']='auto'
    roi:tuple[float,float,float,float]|None=None
    pivot:tuple[float,float]|None=None
    meters_per_pixel:float|None=Field(default=None,gt=0,lt=10)
    fit_fraction:float=Field(default=.65,ge=.3,le=.85)


@app.get('/api/samples')
def get_samples(): return {'samples':samples()}


@app.post('/api/analyze')
def analyze_video(file: UploadFile|None=File(None),sample_id:str|None=Form(None),settings:str=Form('{}')):
    try:
        config=json.loads(settings)
        if not isinstance(config,dict): raise ValueError('Settings must be an object.')
        config=AnalysisSettings.model_validate(config).model_dump(exclude_unset=True)
    except (ValueError,TypeError) as e: raise HTTPException(422,str(e))
    id=uuid.uuid4().hex
    directory=RUNTIME/id;directory.mkdir()
    if file is not None:
        path=directory/'input.mp4'
        with path.open('wb') as out:
            total=0
            while chunk:=file.file.read(1024*1024):
                total+=len(chunk)
                if total>100*1024*1024:
                    shutil.rmtree(directory);raise HTTPException(413,'Upload limit is 100 MB. Trim or compress the clip.')
                out.write(chunk)
        url=f'/runtime/{id}/input.mp4'
    else:
        sample=next((s for s in samples() if s['id']==sample_id),None)
        if sample is None:
            shutil.rmtree(directory);raise HTTPException(422,'Choose a sample or upload a video.')
        path=ROOT/sample['video_url'].lstrip('/')
        url=sample['video_url']
        config={**sample.get('settings',{}),**config}
    if not lock.acquire(blocking=False):
        shutil.rmtree(directory);raise HTTPException(409,'Another analysis is running. Please wait and try again.')
    try:
        result=analyze(path,config)
        result.update(id=id,video_url=url)
        result['provenance']['settings']=config
        result['provenance']['sample_id']=sample_id
        (directory/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        return result
    except (ValueError,TypeError,KeyError,FileNotFoundError,RuntimeError) as e:
        shutil.rmtree(directory);raise HTTPException(422,str(e))
    finally: lock.release()


class SimulationRequest(BaseModel):
    simulation:dict
    changes:dict[str,float]=Field(default_factory=dict)


@app.post('/api/simulate')
def counterfactual(body:SimulationRequest):
    import numpy as np
    sim=body.simulation
    try:
        mode=sim['model'];p=dict(sim['parameters']);t=np.array(sim['time'],dtype=float)
        if mode not in ('pendulum','sliding'): raise ValueError('Unsupported model.')
        if len(t)<2 or len(t)>1800 or t[0]!=0 or t[-1]>60 or np.any(np.diff(t)<=0) or not np.isfinite(t).all(): raise ValueError('Time must increase from zero and end within 60 seconds.')
        allowed={'k','beta','theta0','omega0'} if mode=='pendulum' else {'acceleration','v0','x0'}
        if set(body.changes)-allowed: raise ValueError('Unknown physical parameter.')
        p.update(body.changes)
        if set(p)!=allowed or not np.isfinite(list(p.values())).all(): raise ValueError('Invalid parameters.')
        if mode=='pendulum' and not (.1<=p['k']<=100 and 0<=p['beta']<=5 and abs(p['theta0'])<=3.14 and abs(p['omega0'])<=20): raise ValueError('Pendulum parameters exceed supported ranges.')
        if mode=='sliding' and max(abs(v) for v in p.values())>1e5: raise ValueError('Sliding parameters exceed supported ranges.')
        xy=simulate(mode,p,sim['geometry'],t)
        if not np.isfinite(xy).all(): raise ValueError('Simulation diverged.')
        return {'time':t.tolist(),'xy':xy.tolist()}
    except (KeyError,TypeError,ValueError,OverflowError) as e: raise HTTPException(422,str(e))


@app.get('/api/export/{id}')
def export(id:str):
    if len(id)!=32 or any(c not in '0123456789abcdef' for c in id): raise HTTPException(404)
    path=RUNTIME/id/'analysis.json'
    if not path.exists(): raise HTTPException(404)
    return FileResponse(path,filename=f'realityforge-{id[:8]}.json',media_type='application/json')


app.mount('/samples',StaticFiles(directory=ROOT/'samples'),name='samples')
app.mount('/runtime',StaticFiles(directory=RUNTIME),name='runtime')
app.mount('/',StaticFiles(directory=ROOT/'web',html=True),name='web')
