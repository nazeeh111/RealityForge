"""Shipped model artifacts and automatic selection on well-excited trajectories."""
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from realityforge.estimator import ParameterNet, estimate
from realityforge.inference import infer_track
from realityforge.tracking import Track

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('family,outputs',[('pendulum',4),('sliding',2)])
def test_exported_model_matches_shipped_checkpoint(family,outputs):
    checkpoint=torch.load(ROOT/'models'/f'{family}.pt',map_location='cpu',weights_only=True)
    network=ParameterNet(outputs).eval();network.load_state_dict(checkpoint['state_dict'])
    exported=torch.jit.load(str(ROOT/'models'/f'{family}.torchscript'),map_location='cpu').eval()
    signal=torch.randn(3,100,generator=torch.Generator().manual_seed(921))
    with torch.inference_mode():
        expected=network(signal);actual=exported(signal)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    assert sum(p.numel() for p in network.parameters())<17000


def test_shipped_pendulum_initializer_on_independent_solver():
    times=np.linspace(0,4,121)
    truth=[21.,.2,.55,.3]
    angle=solve_ivp(lambda _,z:[z[1],-truth[0]*np.sin(z[0])-truth[1]*z[1]],(0,4),truth[2:],t_eval=times,rtol=1e-10,atol=1e-11).y[0]
    initial=estimate(times,angle,'pendulum')
    assert abs(initial[0]-truth[0])/truth[0]<.08
    assert abs(initial[1]-truth[1])<.1
    assert np.isfinite(initial).all()


def test_auto_selects_well_excited_pendulum():
    times=np.linspace(0,5,151)
    angle=solve_ivp(lambda _,z:[z[1],-18*np.sin(z[0])-.15*z[1]],(0,5),[.6,.2],t_eval=times,rtol=1e-10,atol=1e-11).y[0]
    xy=np.array([320.,50.])+220*np.c_[np.sin(angle),np.cos(angle)]
    track=Track(times,xy,np.ones(len(times)),640,480,30,len(times),[])
    result=infer_track(track,{'mode':'auto'},method='analytical')
    assert result['model']=='pendulum'


def test_auto_selects_constant_acceleration_on_a_line():
    times=np.linspace(0,3,91)
    scalar=20*times+4*times**2
    xy=np.array([50.,80.])+scalar[:,None]*np.array([1.,.2])
    track=Track(times,xy,np.ones(len(times)),640,480,30,len(times),[])
    result=infer_track(track,{'mode':'auto'},method='analytical')
    assert result['model']=='sliding'
    assert result['metrics']['forecast_rmse_px']<1e-6


@pytest.mark.parametrize('amplitude',[.05,.1])
def test_auto_recognizes_small_angle_periodicity(amplitude):
    times=np.linspace(0,6,181)
    angle=solve_ivp(lambda _,z:[z[1],-19.62*np.sin(z[0])-.1*z[1]],(0,6),[amplitude,0],t_eval=times,rtol=1e-10,atol=1e-11).y[0]
    xy=np.array([320.,40.])+250*np.c_[np.sin(angle),np.cos(angle)]
    track=Track(times,xy,np.ones(len(times)),640,480,30,len(times),[])
    result=infer_track(track,{'mode':'auto'},method='analytical')
    assert result['model']=='pendulum'
    assert result['selection']['motion_reversals']>=2
    assert any('Automatic model selection' in w for w in result['warnings'])


@pytest.mark.parametrize('family,outputs',[('pendulum',4),('sliding',2)])
def test_modern_export_matches_checkpoint_and_provenance(family,outputs):
    import hashlib
    import json
    report=json.loads((ROOT/'models'/'exports.json').read_text())[family]
    checkpoint=ROOT/'models'/f'{family}.pt';artifact=ROOT/'models'/f'{family}.pt2'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256']
    assert hashlib.sha256(artifact.read_bytes()).hexdigest()==report['export_sha256']
    network=ParameterNet(outputs).eval()
    network.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    exported=torch.export.load(artifact).module()
    signal=torch.randn(1,100,generator=torch.Generator().manual_seed(922))
    with torch.inference_mode():
        torch.testing.assert_close(exported(signal),network(signal),atol=1e-6,rtol=1e-6)
