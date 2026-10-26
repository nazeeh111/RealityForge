# ParameterNet model card

**Version:** 0.1.0 · **Task:** initialize a small, explicit dynamics fit from a measured planar trajectory · **License:** MIT for the original code and newly trained weights.

ParameterNet is RealityForge's original compact estimator, coupled to differentiable trajectory optimization. It is a practical implemented contribution, not a claim of a new scientific state of the art. The measured analytical-initialization ablation reaches essentially the same refined accuracy. The strongest evidence supports the physics refinement, not superiority of the neural initializer.

## Input and architecture

One separate multilayer perceptron is trained for each supported family:

| Model | Layers | Trainable weights | Outputs before postprocessing |
|---|---|---:|---|
| Pendulum | 100 → 96 → 64 → 4, SiLU hidden activations | 16,164 | `k/20`, `beta`, `theta0`, `omega0/5` |
| Sliding | 100 → 96 → 64 → 2, SiLU hidden activations | 16,034 | displacement-normalized acceleration and initial velocity |

Inputs are **96 uniformly resampled trajectory values plus four engineered summaries**. This is not a video encoder or raw-image-only model.

- Pendulum values are angles in radians obtained from a supplied or fitted circular pivot. Summaries are duration divided by six, a clipped finite-difference least-squares restoring coefficient divided by 30, a clipped damping estimate, and initial smoothed angular velocity divided by five.
- Sliding values are displacement from the first observation divided by the observed displacement range. Summaries contain duration divided by six, the two leading quadratic-fit coefficients, and zero padding.
- Finite differences use an 11-sample, degree-three Savitzky–Golay smoothed sequence. Training and inference use the same feature function.

At inference the pendulum's learned `theta0` output is replaced by the first observed angle before refinement. Sliding's initial position is likewise taken from the first observation. Pendulum initial estimates are clipped to `k ∈ [1,60] s⁻²`, `beta ∈ [0,3] s⁻¹`, and `omega0 ∈ [−12,12] rad/s`; these are initializer bounds, not evidence that all values within them have been validated. Physics refinement can adjust the initial state.

## Training data and procedure

No real recording, IRIS parameter, external pretrained weight, cloud model, or paid API enters training.

| Setting | Value |
|---|---|
| Seed | 47 for NumPy generation and PyTorch initialization/shuffling |
| Per-family split | 6,000 training trajectories and 800 synthetic validation trajectories |
| Duration | Uniform 1.5–6 seconds; 121 simulation samples per trajectory |
| Pendulum prior | `k` uniform 3–38 s⁻²; `beta` 0–0.9 s⁻¹; `theta0` −1–1 rad; `omega0` −2.5–2.5 rad/s |
| Pendulum observation noise | Independent Gaussian angle noise; standard deviation sampled from 0.0005–0.015 rad |
| Sliding prior | Signed acceleration −1.5–1.5 and velocity −2–2 in synthetic coordinates; initial position zero |
| Sliding observation noise | Gaussian noise with standard deviation 0.003 × trajectory displacement range |
| Optimizer | AdamW, learning rate 0.002, weight decay 0.0001, batch size 256 |
| Objective | Mean squared error on scaled parameter targets |
| Selection | Lowest synthetic validation loss across 100 epochs |
| Execution | CPU, two PyTorch threads |

The held-out synthetic benchmark uses seed 2026 and a separate SciPy adaptive solver for pendulum truth. That checks a different trajectory realization and integrator, but its systems are still close to the training model family. It is not evidence of generalization to arbitrary dynamics.

The delivered run's normalized validation MSE is 0.00093766 for pendulums and 0.0000065085 for sliding. These losses average differently scaled parameter components and are **not physical-unit accuracy**. See [`models/training.json`](../models/training.json) and the parameter/forecast metrics in the [evaluation report](evaluation.md).

## Artifacts and reproducibility

- `models/pendulum.pt`, `models/sliding.pt`: tensor state dictionaries plus small training metadata, loaded on CPU with `weights_only=True`.
- `models/*.pt2`: modern `torch.export` inference networks, verified against the checkpoints in `models/exports.json`.
- `models/*.torchscript`: legacy exported inference networks. Export covers the neural network only; tracking, feature extraction, physical refinement and uncertainty stay in the Python pipeline.
- `scripts/train.py`: complete seeded generator and training procedure. See [exact commands](reproducibility.md).

Do not load arbitrary model files from untrusted sources. The application's bundled weights are the tested artifacts; upload inputs are videos, not checkpoints.

## Intended use and limits

Use as an inspectable starting estimate for a fixed-camera, front-facing planar pendulum or a straight-line constant-acceleration track. Refinement is expected to improve the initial guess. The real-footage ablation demonstrates that neural-only trajectories can drift badly despite plausible initial coefficients.

The network does not identify mass, materials, friction, contact laws, arbitrary equations, camera pose, metric scale, deformable structure or 3D geometry. Unknown perspective, camera motion, occlusion, an inaccurate pivot and dynamics outside the synthetic prior can bias its predictions. Local fit intervals are computed by the physics stage; they are not neural predictive uncertainty and have not achieved calibrated 95% coverage in the small test set.
