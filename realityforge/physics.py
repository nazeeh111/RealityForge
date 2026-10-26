"""Original small planar models. Coordinates: x right, y down; angles from down."""
import numpy as np
import torch


def pendulum_torch(params, times):
    """Differentiable RK4. params = [g/L, viscous angular damping, theta0, omega0]."""
    k, beta, theta, omega = params.unbind(-1)
    values = [theta]
    # Bound RK4 step size even when fit observations are sparse or the clip is long.
    for interval in (times[1:] - times[:-1]).unbind():
        steps=max(1,int(np.ceil(float(interval.detach())*30)))
        dt=interval/steps
        for _ in range(steps):
            def rhs(a, w): return w, -k * torch.sin(a) - beta * w
            a1, w1 = rhs(theta, omega)
            a2, w2 = rhs(theta + dt * a1 / 2, omega + dt * w1 / 2)
            a3, w3 = rhs(theta + dt * a2 / 2, omega + dt * w2 / 2)
            a4, w4 = rhs(theta + dt * a3, omega + dt * w3)
            theta = theta + dt * (a1 + 2*a2 + 2*a3 + a4) / 6
            omega = omega + dt * (w1 + 2*w2 + 2*w3 + w4) / 6
        values.append(theta)
    return torch.stack(values, -1)


def pendulum_numpy(params, times):
    p = np.asarray(params)
    k, b, a, w = [p[..., i].copy() for i in range(4)]
    out = [a.copy()]
    for dt in np.diff(times, axis=-1).T:
        def rhs(a, w): return w, -k * np.sin(a) - b * w
        a1, w1 = rhs(a, w)
        a2, w2 = rhs(a + dt*a1/2, w + dt*w1/2)
        a3, w3 = rhs(a + dt*a2/2, w + dt*w2/2)
        a4, w4 = rhs(a + dt*a3, w + dt*w3)
        a = a + dt*(a1+2*a2+2*a3+a4)/6
        w = w + dt*(w1+2*w2+2*w3+w4)/6
        out.append(a.copy())
    return np.stack(out, -1)


def sliding_torch(params, times):
    a, v, x = params.unbind(-1)
    return x[..., None] + v[..., None]*times + 0.5*a[..., None]*times**2


def simulate(model, parameters, geometry, times):
    t = np.asarray(times, dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or t[0]<0 or np.any(np.diff(t)<=0) or t[-1]>60:
        raise ValueError('Simulation times must increase within 0..60 seconds.')
    if model not in ('pendulum','sliding'): raise ValueError('Unsupported physical model.')
    if not np.isfinite(list(parameters.values())).all(): raise ValueError('Parameters must be finite.')
    if model == 'pendulum':
        p = [parameters[n] for n in ('k','beta','theta0','omega0')]
        # Integrate on a fine grid even if the caller requests sparse output.
        dense = np.linspace(0, max(t[-1], 0.001), max(2, int(t[-1]*120)+1))
        angle = np.interp(t, dense, pendulum_numpy(p, dense))
        pivot, r = np.asarray(geometry['pivot'],dtype=float), float(geometry['radius'])
        if pivot.shape!=(2,) or not np.isfinite(pivot).all() or not 0<r<=1e6:
            raise ValueError('Pendulum geometry requires a finite 2D pivot and positive radius.')
        return pivot + r*np.stack([np.sin(angle),np.cos(angle)], axis=-1)
    origin=np.asarray(geometry['origin'],dtype=float);direction=np.asarray(geometry['direction'],dtype=float)
    if origin.shape!=(2,) or direction.shape!=(2,) or not np.isfinite(np.r_[origin,direction]).all() or not np.isclose(np.linalg.norm(direction),1,atol=.001):
        raise ValueError('Sliding geometry requires a finite 2D origin and unit direction.')
    s = parameters['x0'] + parameters['v0']*t + .5*parameters['acceleration']*t*t
    return origin + s[:,None]*direction
