"""Input boundaries and artifact contracts for the local FastAPI service."""
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from realityforge.api import app

client=TestClient(app)


def valid_simulation():
    return {'model':'sliding','parameters':{'acceleration':8.,'v0':20.,'x0':0.},'geometry':{'origin':[30.,100.],'direction':[1.,0.]},'time':[0.,.5,1.,1.5]}


def test_samples_schema():
    response=client.get('/api/samples')
    assert response.status_code==200
    assert isinstance(response.json()['samples'],list)


@pytest.mark.parametrize('settings',['not json','[]','null'])
def test_malformed_settings_are_rejected(settings):
    response=client.post('/api/analyze',data={'settings':settings})
    assert response.status_code==422


def test_missing_video_is_rejected():
    assert client.post('/api/analyze').status_code==422


def test_undecodable_upload_is_rejected():
    result=client.post('/api/analyze',files={'file':('invalid.mp4',b'not a video','video/mp4')})
    assert result.status_code==422
    assert 'frame' in result.json()['detail'].lower()


def test_simulation_and_edits_are_real():
    body={'simulation':valid_simulation(),'changes':{'acceleration':16.}}
    response=client.post('/api/simulate',json=body)
    assert response.status_code==200
    np.testing.assert_allclose(response.json()['xy'][-1],[78.,100.])


@pytest.mark.parametrize('changes',[{'mass':1},{'v0':1e6}])
def test_unsupported_counterfactual_is_rejected(changes):
    assert client.post('/api/simulate',json={'simulation':valid_simulation(),'changes':changes}).status_code==422


@pytest.mark.parametrize('time',[[1.,2.],[0.,0.],[0.,1.,.5],[0.,61.]])
def test_invalid_counterfactual_times_are_rejected(time):
    sim=valid_simulation();sim['time']=time
    assert client.post('/api/simulate',json={'simulation':sim}).status_code==422


def test_nonfinite_simulation_is_rejected():
    # Raw JSON exercises parser boundary; standard HTTP clients refuse to encode NaN.
    body=json.dumps({'simulation':valid_simulation(),'changes':{'v0':float('nan')}})
    response=client.post('/api/simulate',content=body,headers={'content-type':'application/json'})
    assert response.status_code==422


def test_geometry_nan_is_rejected():
    sim=valid_simulation();sim['geometry']['origin']=[float('nan'),0]
    response=client.post('/api/simulate',content=json.dumps({'simulation':sim}),headers={'content-type':'application/json'})
    assert response.status_code==422


def test_export_identifiers_cannot_escape_runtime():
    assert client.get('/api/export/not-an-id').status_code==404
    assert client.get('/api/export/'+'a'*32).status_code==404


@pytest.mark.parametrize('settings',[{'fit_fraction':{}},{'roi':{}},{'pivot':[1,2,3]},{'meters_per_pixel':-1},{'mode':'arbitrary'},{'unknown':True}])
def test_typed_settings_reject_bad_values_before_video_access(settings):
    response=client.post('/api/analyze',data={'settings':json.dumps(settings)})
    assert response.status_code==422
    assert 'validation' in response.json()['detail'].lower()


def test_explicitly_cleared_sample_settings_override_defaults(monkeypatch,tmp_path):
    from realityforge import api
    captured={}
    monkeypatch.setattr(api,'RUNTIME',tmp_path)
    monkeypatch.setattr(api,'samples',lambda:[{'id':'calibrated','video_url':'/samples/example.mp4','settings':{'mode':'pendulum','roi':[10,20,30,40],'pivot':[100,10],'meters_per_pixel':.002}}])
    def fake_analyze(path,config):
        captured.update(config)
        return {'provenance':{}}
    monkeypatch.setattr(api,'analyze',fake_analyze)
    response=client.post('/api/analyze',data={'sample_id':'calibrated','settings':json.dumps({'roi':None,'pivot':None,'meters_per_pixel':None})})
    assert response.status_code==200
    assert captured['roi'] is None and captured['pivot'] is None and captured['meters_per_pixel'] is None
    assert captured['mode']=='pendulum'


def test_unspecified_sample_settings_keep_defaults(monkeypatch,tmp_path):
    from realityforge import api
    captured={};defaults={'mode':'sliding','roi':[10,20,30,40],'meters_per_pixel':.002}
    monkeypatch.setattr(api,'RUNTIME',tmp_path)
    monkeypatch.setattr(api,'samples',lambda:[{'id':'calibrated','video_url':'/samples/example.mp4','settings':defaults}])
    def fake_analyze(path,config):
        captured.update(config)
        return {'provenance':{}}
    monkeypatch.setattr(api,'analyze',fake_analyze)
    response=client.post('/api/analyze',data={'sample_id':'calibrated','settings':'{}'})
    assert response.status_code==200
    assert captured==defaults


@pytest.mark.parametrize('geometry',[
    {'origin':[0.],'direction':[1.,0.]},
    {'origin':[1.,2.],'direction':[1.,0.,0.]},
    {'origin':[1.,2.],'direction':[0.,0.]},
    {'origin':[1.,2.],'direction':[2.,0.]},
])
def test_invalid_sliding_geometry_is_rejected(geometry):
    sim=valid_simulation();sim['geometry']=geometry
    assert client.post('/api/simulate',json={'simulation':sim}).status_code==422


@pytest.mark.parametrize('geometry',[
    {'pivot':[320.,40.],'radius':-20.},
    {'pivot':[320.],'radius':200.},
])
def test_invalid_pendulum_geometry_is_rejected(geometry):
    sim={'model':'pendulum','parameters':{'k':19.62,'beta':.1,'theta0':.5,'omega0':0.},'geometry':geometry,'time':[0.,.5,1.]}
    assert client.post('/api/simulate',json={'simulation':sim}).status_code==422


def test_actual_uploaded_video_analysis_and_export():
    from realityforge.resources import ROOT
    path=ROOT/'samples/generated/sliding.mp4'
    settings={'mode':'sliding','roi':[111,285,48,50],'meters_per_pixel':.005,'fit_fraction':.65}
    with path.open('rb') as video:
        response=client.post('/api/analyze',files={'file':('sliding.mp4',video,'video/mp4')},data={'settings':json.dumps(settings)})
    assert response.status_code==200,response.text
    result=response.json()
    assert result['model']=='sliding'
    assert result['metrics']['forecast_rmse_px']<3
    assert result['video_url'].startswith('/runtime/')
    assert client.get(result['video_url']).content==path.read_bytes()
    exported=client.get('/api/export/'+result['id'])
    assert exported.status_code==200
    assert exported.json()['simulation']==result['simulation']
    edited=client.post('/api/simulate',json={'simulation':result['simulation'],'changes':{'acceleration':0.}})
    assert edited.status_code==200
    assert np.linalg.norm(np.array(edited.json()['xy'][-1])-result['predicted_xy'][-1])>50
