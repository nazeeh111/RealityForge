# From pixels to a conditional physical model

RealityForge separates measurement, model assumptions, parameter fitting and prediction. Its output is a supported planar model with diagnostics, not an unconstrained reconstruction of the filmed world.

## 1. Measure one object's image trajectory

OpenCV decodes the recording and preserves timestamps derived from the source frame rate. With a first-frame object rectangle, tracking uses the selected region's dominant saturated color when suitable, then falls back to normalized template correlation. Without a region, foreground components are compared with a median background constructed from the **fit prefix only**. Tracking confidence is a heuristic image-match score, not a calibrated probability.

The user can inspect the observed path and choose a more reliable rectangle. A fixed camera, visible object edges, little occlusion and a clearly bounded motion interval are important. Release hands, impacts, external pushes and stopped motion should be trimmed or treated as model violations.

## 2. Choose a supported geometry and dynamics family

Only the initial observation window estimates geometry and physical parameters. The default split fits the first 65% and reserves the remaining 35% for a forecast check. Manual family selection is available; automatic selection is limited to the supported line/circle motion hypotheses and must be checked against the displayed trajectory.

Image coordinates have `x` right and `y` down. A pendulum angle is measured from the downward image direction. A supplied pivot is used directly; otherwise a robust circle fit estimates the pivot and pixel radius from the fit prefix. Sliding uses the leading image-plane direction from a singular-value decomposition and measures signed displacement along that line.

These geometric assumptions require an upright, stationary, approximately front-facing camera. A scalar meters-per-pixel reference converts image lengths at the motion plane; it does not undo perspective, camera roll or lens distortion. An arbitrary oblique pendulum can project to an ellipse rather than the assumed circle. The application does not estimate a full camera calibration or homography.

## 3. Fit an explicit initial-value problem

The supported pendulum model is

\[
\ddot{\theta}+\beta\dot{\theta}+k\sin\theta=0,\qquad k=g/L.
\]

Its fitted parameters are restoring coefficient `k` in s⁻², viscous damping `beta` in s⁻¹, initial angle `theta0` and initial angular velocity `omega0`. A point mass and a fixed pivot are assumed. The image prediction is

\[
(x,y)=p+R(\sin\theta,\cos\theta),
\]

where `p` and `R` are the fixed fitted pixel pivot and radius.

Sliding uses

\[
s(t)=s_0+v_0t+\tfrac12at^2,\qquad (x,y)=o+s(t)d.
\]

Here `o` and unit direction `d` come from the fit-prefix track. The fitted acceleration is signed pixels/s². This is a kinematic constant-acceleration model. It has no collision, contact, friction transition or stopping law; edited trajectories may continue through a real-world stop.

A compact [ParameterNet](model-card.md) provides the initial estimate. PyTorch differentiates the predicted trajectory with respect to its physical parameters. L-BFGS then minimizes a confidence-weighted Huber residual on the fit segment. The pendulum uses fourth-order Runge–Kutta integration with a bounded internal step during refinement; sliding has an exact polynomial solution. Geometry remains fixed during physical refinement.

Pendulum refinement parameterizes a positive `k` and nonnegative damping through transformed variables. Robust residuals reduce the influence of isolated tracking errors but cannot correct a systematically incorrect object track or physical equation.

## 4. Make a forecast and a counterfactual

The fitted initial state and parameters generate a trajectory for the full recording interval. The withheld suffix is compared with the tracked object using Euclidean image-position RMSE:

\[
\operatorname{RMSE}=\sqrt{\frac1N\sum_i\|\hat{q}_i-q_i\|_2^2}.
\]

Forecast errors against real tracks combine tracking error and model error. They do not establish accuracy against independent position annotations. The synthetic benchmark can compare with clean known trajectories separately.

Changing a displayed parameter runs the same equations with the changed value. Initial state, geometry and time axis remain as recorded unless the corresponding initial-state parameter is edited. A counterfactual is conditional on the model continuing to apply; it is not experimental confirmation that a real object would behave that way.

## 5. State what is identifiable

| Observable or supplied evidence | Supported quantity | Not determined by that evidence alone |
|---|---|---|
| Pendulum angles and timing | `k = g/L`, conditional on the pendulum equation | Gravity and length separately; mass |
| Pendulum timing plus assumed gravity | Effective dynamic length `g/k` | Independently measured geometric length |
| Known image-plane metric length plus `k` | Gravity conditional on length and camera assumptions | Full camera calibration or 3D geometry |
| Sliding pixels and time | Image-plane acceleration | Metric acceleration without scale |
| Metric sliding acceleration | Net acceleration along the observed direction | Friction, applied force and mass separately |

Small arcs, short duration, little excitation and strong parameter correlations weaken identifiability. A physically plausible fit is not enough to distinguish alternative causal explanations.

## 6. Attach conditional uncertainty

Let `J` be the fitted trajectory's Jacobian with respect to the physical parameters and let residual variance be `sigma²`. The implementation approximates

\[
\operatorname{Cov}(p)\approx\sigma^2(J^TJ)^+,
\]

using a pseudoinverse, and displays `p ± 1.96 × standard_error`. The Jacobian condition number flags weak sensitivity. This is a local linear approximation, conditional on fixed fitted geometry and the assumed independent residual-noise model. It does not propagate uncertainty in scale, camera geometry, tracking bias or the governing equation. Robust fitting does not make this covariance a robust or fully calibrated uncertainty estimator.

On the small held-out synthetic snapshot, the hybrid primary-parameter interval contains truth in four of six cases per family. Thus the nominal **95% label is not validated as 95% empirical coverage**. Treat the interval as a sensitivity diagnostic, especially near parameter constraints or poor geometry.
