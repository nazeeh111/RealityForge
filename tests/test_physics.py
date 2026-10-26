"""Independent checks against conservation laws and SciPy's adaptive integrator."""
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp
from realityforge.physics import pendulum_torch, pendulum_numpy, sliding_torch, simulate
from realityforge.inference import refine, analytical, infer_track
from realityforge.tracking import Track


def test_rk4_matches_independent_adaptive_solver():
    p=np.array([19.62,.12,.7,-.3]);t=np.linspace(0,6,361)
    expected=solve_ivp(lambda _,z:[z[1],-p[0]*np.sin(z[0])-p[1]*z[1]],(0,6),p[2:],t_eval=t,rtol=1e-11,atol=1e-12).y[0]
    actual=pendulum_numpy(p,t)
    assert np.max(np.abs(actual-expected))<1e-5
    assert np.max(np.abs(pendulum_torch(torch.tensor(p),torch.tensor(t)).detach().numpy()-expected))<1e-5


def test_torch_gradient_matches_central_difference():
    p=torch.tensor([15.,.2,.4,.1],dtype=torch.float64,requires_grad=True)
    t=torch.linspace(0,2,81,dtype=torch.float64)
    terminal=pendulum_torch(p,t)[-1]
    gradient=torch.autograd.grad(terminal,p)[0].numpy()
    reference=[]
    for j in range(4):
        delta=np.zeros(4);delta[j]=1e-5
        reference.append((pendulum_numpy(p.detach().numpy()+delta,t.numpy())[-1]-pendulum_numpy(p.detach().numpy()-delta,t.numpy())[-1])/2e-5)
    np.testing.assert_allclose(gradient,reference,rtol=2e-5,atol=1e-7)


def test_undamped_pendulum_energy_is_conserved():
    t=np.linspace(0,4,801);k=12.
    angle=pendulum_numpy([k,0,.8,0],t)
    omega=np.gradient(angle,t)
    energy=.5*omega**2+k*(1-np.cos(angle))
    assert np.ptp(energy[2:-2])/energy[0]<.0002


def test_sliding_has_exact_constant_acceleration():
    p=torch.tensor([-8.,22.,3.],dtype=torch.float64,requires_grad=True)
    t=torch.linspace(0,2,41,dtype=torch.float64)
    actual=sliding_torch(p,t)
    np.testing.assert_allclose(actual.detach(),3+22*t.numpy()-4*t.numpy()**2)
    assert torch.autograd.grad(actual[-1],p)[0].tolist()==pytest.approx([2.,2.,1.])


def test_clean_and_noisy_pendulum_parameter_recovery():
    rng=np.random.default_rng(2026)
    t=np.linspace(0,4.5,136);truth=np.array([19.62,.18,.6,.3])
    clean=pendulum_numpy(truth,t)
    for noise in [0,.003]:
        y=clean+rng.normal(0,noise,len(t))
        recovered,ci,diagnostics=refine(t,y,'pendulum',np.array([18.,.2,y[0],.2]))
        assert abs(recovered[0]-truth[0])/truth[0]<.006
        assert abs(recovered[1]-truth[1])<.015
        assert np.isfinite(ci).all() and (ci>=0).all()
        assert diagnostics['jacobian_condition']>0


def synthetic_slide(noise=0):
    t=np.linspace(0,3,91);s=30*t+4*t*t
    xy=np.stack([80+s,200+.2*s],axis=1)
    xy+=np.random.default_rng(33).normal(0,noise,xy.shape)
    return Track(t,xy,np.ones(len(t)),640,360,30.,len(t),[])


def test_fit_ignores_forecast_positions_and_geometry_uses_fit_only():
    track=synthetic_slide(.1)
    result=infer_track(track,{'mode':'sliding'},method='analytical_refine')
    altered=synthetic_slide(.1)
    altered.xy[result['fit_count']:]+=np.array([2000,-700])
    repeat=infer_track(altered,{'mode':'sliding'},method='analytical_refine')
    assert result['simulation']==repeat['simulation']
    assert result['metrics']['forecast_rmse_px']<1
    assert repeat['metrics']['forecast_rmse_px']>1000


def test_metric_scale_changes_only_derived_units():
    track=synthetic_slide(.1)
    base=infer_track(track,{'mode':'sliding'},method='analytical')
    calibrated=infer_track(track,{'mode':'sliding','meters_per_pixel':.002},method='analytical')
    assert base['simulation']==calibrated['simulation']
    rows={p['name']:p for p in calibrated['parameters']}
    assert rows['metric_acceleration']['value']==pytest.approx(rows['acceleration']['value']*.002)
    assert not rows['metric_acceleration']['editable']
    assert any('not separately identifiable' in w for w in calibrated['warnings'])


def test_counterfactual_changes_trajectory():
    times=np.linspace(0,3,91)
    geometry={'pivot':[320,40],'radius':180}
    p={'k':19.62,'beta':.1,'theta0':.5,'omega0':0}
    a=simulate('pendulum',p,geometry,times)
    b=simulate('pendulum',{**p,'k':12.},geometry,times)
    assert np.linalg.norm(a[-1]-b[-1])>30
    np.testing.assert_allclose(a[0],b[0])


@pytest.mark.parametrize('fraction',[0,.2,.9,1,float('nan')])
def test_invalid_fit_fraction_is_rejected(fraction):
    with pytest.raises(ValueError): infer_track(synthetic_slide(),{'fit_fraction':fraction},method='analytical')


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError): infer_track(synthetic_slide(),{'mode':'3d'},method='analytical')


def test_sparse_pendulum_observations_use_stable_internal_steps():
    p=np.array([28.,.12,.7,-.3]);t=np.linspace(0,20,41)
    expected=solve_ivp(lambda _,z:[z[1],-p[0]*np.sin(z[0])-p[1]*z[1]],(0,20),p[2:],t_eval=t,rtol=1e-11,atol=1e-12).y[0]
    actual=pendulum_torch(torch.tensor(p),torch.tensor(t)).detach().numpy()
    assert np.max(np.abs(actual-expected))<.0002
