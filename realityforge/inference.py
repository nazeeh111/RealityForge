"""Geometry, learned initialization, differentiable shooting and conditional uncertainty."""
import time
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from .physics import pendulum_torch, sliding_torch, simulate
from .estimator import estimate
from .tracking import track_video

torch.set_num_threads(2)


def circle_geometry(xy,pivot=None):
    if pivot is None:
        center=xy.mean(0)
        z=xy-center
        a=np.c_[2*z,np.ones(len(z))]
        solution=np.linalg.lstsq(a,(z*z).sum(1),rcond=None)[0]
        guess=center+solution[:2]
        fit=least_squares(lambda p:np.linalg.norm(xy-p[:2],axis=1)-p[2],np.r_[guess,np.linalg.norm(xy-guess,axis=1).mean()],loss='soft_l1')
        pivot=fit.x[:2]
    pivot=np.asarray(pivot)
    radii=np.linalg.norm(xy-pivot,axis=1);radius=np.median(radii)
    if radius<5 or not np.isfinite(radius): raise ValueError('Pendulum radius is too small or degenerate.')
    angle=np.unwrap(np.arctan2(xy[:,0]-pivot[0],xy[:,1]-pivot[1]))
    return {'pivot':pivot.tolist(),'radius':float(radius),'radial_error':float(np.std(radii)/radius)},angle


def line_geometry(xy):
    origin=xy[0]
    _,_,vh=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
    direction=vh[0]
    if np.dot(xy[-1]-xy[0],direction)<0: direction=-direction
    s=(xy-origin)@direction
    error=np.sqrt(np.mean(np.sum((xy-origin-s[:,None]*direction)**2,axis=1)))
    return {'origin':origin.tolist(),'direction':direction.tolist(),'line_error':float(error)},s


def analytical(t,y,mode):
    if mode=='sliding':
        c=np.polyfit(t,y,2)
        return np.array([2*c[0],c[1],c[2]])
    smooth=savgol_filter(y,min(15,len(y)//2*2-1),3)
    vel=np.gradient(smooth,t);acc=np.gradient(vel,t)
    p=np.linalg.lstsq(np.c_[-np.sin(smooth[3:-3]),-vel[3:-3]],acc[3:-3],rcond=None)[0]
    return np.array([np.clip(p[0],.2,100),np.clip(p[1],0,3),smooth[0],vel[0]])


def refine(t,y,mode,initial,weights=None,max_iter=55):
    # Fits one initial-value problem, rather than differentiating noisy image positions twice.
    if mode=='pendulum':
        # More than 80 samples is rarely useful for this 4-parameter fit.
        index=np.unique(np.linspace(0,len(t)-1,min(80,len(t))).astype(int))
    else: index=np.arange(len(t))
    tt=torch.tensor(t[index],dtype=torch.float64)
    yy=torch.tensor(y[index],dtype=torch.float64)
    ww=torch.tensor(np.ones(len(index)) if weights is None else np.maximum(weights[index],.05),dtype=torch.float64)
    if mode=='pendulum':
        raw=torch.tensor([np.log(initial[0]),np.log(np.clip(initial[1]/3,.01,.9999)/(1-np.clip(initial[1]/3,.01,.9999))),initial[2],initial[3]],dtype=torch.float64,requires_grad=True)
        def unpack(): return torch.stack([raw[0].clamp(-2,4.6).exp(),3*raw[1].sigmoid(),raw[2],raw[3]])
        fn=pendulum_torch
    else:
        raw=torch.tensor(initial,dtype=torch.float64,requires_grad=True)
        def unpack(): return raw
        fn=sliding_torch
    opt=torch.optim.LBFGS([raw],lr=1.,max_iter=max_iter,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    scale=max(float(np.std(y)),.02)
    evaluations=0
    def closure():
        nonlocal evaluations
        opt.zero_grad();residual=(fn(unpack(),tt)-yy)/scale
        loss=(torch.nn.functional.huber_loss(residual,torch.zeros_like(residual),reduction='none',delta=.15)*ww).mean()
        loss.backward();evaluations+=1
        return loss
    opt.step(closure)
    result=unpack().detach()
    if not torch.isfinite(result).all(): raise ValueError('Physics optimization diverged. Check tracking and model choice.')
    jac=torch.autograd.functional.jacobian(lambda p:fn(p,tt),result,vectorize=True).detach().numpy()
    residual=(fn(result,tt)-yy).detach().numpy()
    dof=max(len(index)-len(result),1)
    sigma2=float(np.sum(residual**2)/dof)
    gram=jac.T@jac
    covariance=sigma2*np.linalg.pinv(gram,rcond=1e-10)
    ci=1.96*np.sqrt(np.maximum(np.diag(covariance),0))
    return result.numpy(),ci,{'jacobian_condition':float(np.linalg.cond(jac)),'residual_sigma':sigma2**.5,'function_evaluations':evaluations}


def infer_track(track,settings=None,method='hybrid'):
    settings=settings or {}
    if method not in ('hybrid','neural','analytical','analytical_refine','iris_recurrence'): raise ValueError('Unknown estimation method.')
    t=track.time-track.time[0];xy=track.xy
    fraction=float(settings.get('fit_fraction',.65))
    if not .3<=fraction<=.85: raise ValueError('Fit fraction must be between 0.30 and 0.85.')
    n=max(12,int(len(t)*fraction))
    if n>=len(t)-3: raise ValueError('Clip too short for a separate forecast segment.')
    line,line_y=line_geometry(xy[:n])
    mode=settings.get('mode','auto')
    if mode not in ('auto','pendulum','sliding'): raise ValueError('Choose auto, pendulum or sliding.')
    circle=None
    try: circle,angle=circle_geometry(xy[:n],settings.get('pivot'))
    except (ValueError,np.linalg.LinAlgError): pass
    selection={'requested':mode,'method':'user-selected supported family'}
    if mode=='auto':
        extent=max(np.ptp(xy[:n],axis=0).max(),1)
        line_ratio=line['line_error']/extent
        smooth=savgol_filter(line_y,min(15,len(line_y)//2*2-1),3)
        velocity=np.gradient(smooth,t[:n])
        active=velocity[np.abs(velocity)>max(np.max(np.abs(velocity))*.15,1e-6)]
        reversals=int(np.sum(np.diff(np.sign(active))!=0))
        circular=circle is not None and circle['radial_error']<.055 and circle['pivot'][1]<np.median(xy[:n,1])
        if circular and (line_ratio>=.018 or reversals>=2): mode='pendulum'
        elif line_ratio<.018: mode='sliding'
        else: raise ValueError('Motion is outside supported planar families. Select a target region or choose a model after inspecting the trajectory.')
        selection={'requested':'auto','method':'fit-prefix geometry and motion-reversal screening','line_relative_error':float(line_ratio),'motion_reversals':reversals,'circle_admissible':bool(circular)}
    warnings=list(track.warnings)
    if selection['requested']=='auto': warnings.append('Automatic model selection screens two supported families using geometry and motion reversals. A short monotonic pendulum arc may be indistinguishable from sliding; inspect the result or choose a family explicitly.')
    if mode=='pendulum':
        if circle is None: raise ValueError('Unable to fit a pendulum circle. Supply a pivot point.')
        geometry=circle
        y=np.unwrap(np.arctan2(xy[:,0]-circle['pivot'][0],xy[:,1]-circle['pivot'][1]))
        names=['k','beta','theta0','omega0'];units=['s⁻²','s⁻¹','rad','rad/s']
        if circle['radial_error']>.04: warnings.append('Motion deviates from a circle. Perspective, camera motion or tracking error may invalidate this pendulum model.')
        if np.ptp(y[:n])<.15: warnings.append('Small visible arc makes the pivot and damping weakly identifiable. Supply a measured pivot and a longer clip.')
        warnings.append('Pendulum assumes an upright fixed camera, planar point-mass motion, and viscous damping. Only k = g/L is recovered dynamically; g and L are not separately identifiable.')
    else:
        geometry=line;y=(xy-np.array(line['origin']))@np.array(line['direction'])
        names=['acceleration','v0','x0'];units=['px/s²','px/s','px']
        extent=max(np.ptp(y[:n]),1)
        if line['line_error']/extent>.03: warnings.append('Motion is not straight. Constant acceleration may be an inadequate model.')
        warnings.append('Sliding fits signed constant image-plane acceleration, with no contact or stopping law. Counterfactuals can pass through a physical stop. Friction, mass and applied force are not separately identifiable.')
    if method=='iris_recurrence':
        if mode!='pendulum': raise ValueError('The IRIS recurrence adapter supports pendulums only.')
        from .baselines import iris_recurrence
        parameters,angle=iris_recurrence(t[:n],y[:n],t)
        p=np.array([parameters[name] for name in names]);ci=np.full(len(p),np.nan);diagnostics={}
        predicted=np.array(geometry['pivot'])+geometry['radius']*np.c_[np.sin(angle),np.cos(angle)]
    else:
        initial=estimate(t[:n],y[:n],mode) if method in ('hybrid','neural') else analytical(t[:n],y[:n],mode)
        if method in ('neural','analytical'):
            p=initial;ci=np.full(len(p),np.nan);diagnostics={}
        else:
            p,ci,diagnostics=refine(t[:n],y[:n],mode,initial,track.confidence[:n])
        parameters=dict(zip(names,map(float,p)))
        predicted=simulate(mode,parameters,geometry,t)
    fit_error=float(np.sqrt(np.mean(np.sum((predicted[:n]-xy[:n])**2,axis=1))))
    forecast_error=float(np.sqrt(np.mean(np.sum((predicted[n:]-xy[n:])**2,axis=1))))
    metrics={'fit_rmse_px':fit_error,'forecast_rmse_px':forecast_error,**diagnostics}
    rows=[]
    for i,name in enumerate(names):
        low,high=float(p[i]-ci[i]),float(p[i]+ci[i])
        if name in ('k','beta'): low=max(0.,low)
        row={'name':name,'value':float(p[i]),'unit':units[i],'editable':True}
        if np.isfinite([low,high]).all(): row['ci95']=[low,high]
        rows.append(row)
    meters=settings.get('meters_per_pixel')
    if meters is not None:
        meters=float(meters)
        if not 0<meters<10: raise ValueError('Meters per pixel must be positive and less than 10.')
        if mode=='sliding': rows.append({'name':'metric_acceleration','value':parameters['acceleration']*meters,'unit':'m/s²','editable':False})
        else:
            rows.extend([{'name':'image_plane_length','value':geometry['radius']*meters,'unit':'m','editable':False},{'name':'gravity_given_length','value':parameters['k']*geometry['radius']*meters,'unit':'m/s²','editable':False}])
        warnings.append('Metric units assume a uniform scale at the motion plane. A single scale reference does not correct perspective or lens distortion.')
    else: warnings.append('No scale calibration supplied. Pixel distances are not meters. Pendulum angles assume a front-facing circular motion plane.')
    if diagnostics.get('jacobian_condition',0)>1000: warnings.append('Sensitivity matrix is poorly conditioned: some fitted parameters are weakly identifiable.')
    if forecast_error>max(5,fit_error*3): warnings.append('Forecast mismatch exceeds the fit error substantially. Review model assumptions before trusting counterfactuals.')
    warnings.append('Nominal 95% intervals are uncalibrated local linear approximations conditional on the chosen model, fitted geometry and independent residual noise. They exclude scale, camera, tracking bias and model uncertainty.')
    return {'model':mode,'width':track.width,'height':track.height,'fps':track.fps,'fit_end':float(t[n-1]),'fit_count':n,'parameters':rows,'warnings':warnings,'metrics':metrics,'time':t.tolist(),'observed_xy':xy.tolist(),'predicted_xy':predicted.tolist(),'fit_xy':predicted[:n].tolist(),'confidence':track.confidence.tolist(),'simulation':{'model':mode,'parameters':parameters,'geometry':geometry,'time':t.tolist()},'selection':selection,'provenance':{'training':'seeded synthetic trajectories only','method':method,'tracking':'adaptive ROI color centroid with template NCC fallback; fit-prefix foreground otherwise','evaluation':'initial segment fit, remaining frames forecast; no physical parameters fit on forecast frames','geometry_fit_frames':n}}


def analyze(path,settings=None,method='hybrid'):
    start=time.perf_counter();settings=settings or {}
    track=track_video(path,settings.get('roi'),settings.get('fit_fraction',.65))
    result=infer_track(track,settings,method)
    result['metrics']['runtime_s']=time.perf_counter()-start
    return result
