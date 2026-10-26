"""Original observed-angle adapter for the published IRIS pendulum recurrence.

Equation source (MIT): KurbanIntelligenceLab/iris-bench,
33106189830e766bb286aa625f160d599f14042d,
src/models/PhysModels.py, class Pendulum. No upstream code is copied.

This comparator bypasses EndPhys's learned image encoder and latent regularizer.
It compares the published discrete physics component on shared measured angles;
it is not a reproduction of the complete published video-to-parameter baseline.
"""
import numpy as np
from scipy.optimize import lsq_linear


def iris_recurrence(tfit, yfit, all_times):
    """Fit a constrained one-step recurrence, then roll out from two initial angles.

    Returns ``(parameters, angles)``. No suffix observations enter this function.
    ``k`` is g/L directly, avoiding ambiguity in upstream saved ``alpha`` maps.
    Inputs must begin at zero and share a uniform positive sampling interval.
    The physical adapter bounds k to [0.1,100] and damping to [0,3].
    """
    t = np.asarray(tfit, dtype=float)
    y = np.asarray(yfit, dtype=float)
    target = np.asarray(all_times, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 16:
        raise ValueError('IRIS recurrence requires at least 16 paired fit samples.')
    if target.ndim != 1 or len(target) < len(t):
        raise ValueError('IRIS recurrence output times must include the fit prefix.')
    if not np.isfinite(t).all() or not np.isfinite(y).all() or not np.isfinite(target).all():
        raise ValueError('IRIS recurrence requires finite angles and timestamps.')
    dt = float(t[1]-t[0])
    if dt <= 0 or not np.isclose(t[0], 0) or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-8):
        raise ValueError('IRIS recurrence requires uniform fit timestamps starting at zero.')
    if not np.allclose(target[:len(t)], t, rtol=1e-6, atol=1e-8) or not np.allclose(np.diff(target), dt, rtol=1e-6, atol=1e-8):
        raise ValueError('IRIS recurrence requires a shared uniform time axis.')
    # q_(i+1)=2*q_i-q_(i-1)-dt*beta*(q_i-q_(i-1))-dt²*k*sin(q_i).
    design = np.column_stack([dt*dt*np.sin(y[1:-1]), dt*(y[1:-1]-y[:-2])])
    rhs = 2*y[1:-1]-y[:-2]-y[2:]
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError('IRIS recurrence parameters are not identifiable from this fit prefix.')
    fit = lsq_linear(design, rhs, bounds=([.1, 0.], [100., 3.]), tol=1e-12)
    if not fit.success:
        raise ValueError('IRIS recurrence parameter fitting did not converge.')
    k, beta = map(float, fit.x)
    angle = np.empty(len(target), dtype=float)
    angle[:2] = y[:2]
    for i in range(1, len(angle)-1):
        angle[i+1] = (2*angle[i]-angle[i-1] - dt*beta*(angle[i]-angle[i-1])
                      - dt*dt*k*np.sin(angle[i]))
    if not np.isfinite(angle).all():
        raise ValueError('IRIS recurrence rollout diverged.')
    parameters = {'k': k, 'beta': beta, 'theta0': float(y[0]),
                  'omega0': float((y[1]-y[0])/dt)}
    return parameters, angle
