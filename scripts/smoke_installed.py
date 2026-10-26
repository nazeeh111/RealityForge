"""Exercise a packaged install, including bundled assets and actual inference."""
import json
from fastapi.testclient import TestClient
from realityforge.api import app
from realityforge.resources import ROOT

client=TestClient(app)
assert client.get('/').status_code==200
samples=client.get('/api/samples').json()['samples']
sample=next(s for s in samples if s['id']=='generated-pendulum')
r=client.post('/api/analyze',data={'sample_id':sample['id'],'settings':json.dumps(sample['settings'])})
assert r.status_code==200,r.text
result=r.json()
assert result['metrics']['forecast_rmse_px']<2
r=client.post('/api/simulate',json={'simulation':result['simulation'],'changes':{'k':10.}})
assert r.status_code==200,r.text
assert client.get('/api/export/'+result['id']).status_code==200
assert (ROOT/'models/pendulum.pt2').is_file()
sliding=ROOT/'samples/generated/sliding.mp4'
with sliding.open('rb') as video:
    upload=client.post('/api/analyze',files={'file':('sliding.mp4',video,'video/mp4')},data={'settings':json.dumps({'mode':'sliding','roi':[111,285,48,50]})})
assert upload.status_code==200,upload.text
assert upload.json()['metrics']['forecast_rmse_px']<3
print(json.dumps({'assets_bundled':True,'forecast_rmse_px':result['metrics']['forecast_rmse_px'],'inference_counterfactual_export':'pass'},indent=2))
