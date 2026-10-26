# Evaluation and evidence

RealityForge measures parameter recovery and future trajectory error separately. The current evidence is a small, reproducible supported-system study, not the full IRIS benchmark or a claim of superiority over published video reconstruction systems.

**Read the [measured result tables](benchmark-results.md), [synthetic records](../benchmarks/synthetic.json) and [real-recording records](../benchmarks/real.json) together.** The JSON artifacts retain per-case results, failures, runtime and source hashes. Numbers only support the artifact versions identified there.

## What is tested

| Evaluation layer | Inputs | Evidence it supports | Evidence it does not supply |
|---|---|---|---|
| Synthetic inverse dynamics | Independently generated planar tracks with known parameters | Parameter error, clean-trajectory forecast error, noise/outlier robustness and optimizer ablations | Real-video tracking quality or arbitrary-physics transfer |
| Rendered fixture pipeline | Original generated pixels with known object motion | Decode → tracking → model → forecast integration | Accuracy on real experiments |
| IRIS recording subset | Three pendulum recordings and one sliding recording | Transfer to real measurements, tracked-trajectory forecasts and nominal-length consistency | Official complete IRIS score, friction truth or independently annotated positions |
| Frozen IRIS recordings | Six unseen pendulum takes, preselected after core/model freeze | Recording transfer without tracker or estimator tuning on those clips | The complete official IRIS benchmark or broad physical coverage |
| Independent household footage | A separately sourced physical pendulum | Transfer beyond the IRIS recording setup | Precision length metrology or broad generalization |
| Automated tests | Physical laws, independent solver, gradients, tracking and API cases | The particular contracts and numerical tolerances asserted | A substitute for the real-recording evaluation |

## Split and leakage control

Training uses synthetic seed 47. Synthetic evaluation uses seed 2026 and a SciPy adaptive solver for pendulum ground truth rather than the training generator's Runge–Kutta solver. Default evaluation has 24 cases per family, totaling 48 trajectories.

Within each recording or synthetic trajectory, the initial 65% estimates physical parameters and geometry; the final 35% tests extrapolation. Background estimation uses only the fit prefix. Region selection is manual on the first frame for the bundled experimental samples. The withheld suffix is never used to choose physical parameters, pivot or sliding direction.

All real recordings are excluded from neural training. The five development recordings were inspected while developing the tracker and interface. They are **evaluation-only relative to training, not a blind untouched test set**. The six additional recordings in `benchmarks/blind-iris.json` were preselected after freezing source/model hashes and were not used for subsequent tracker or estimator tuning. The small selected subset is appropriate for a transparent transfer check; it cannot establish population-level performance.

## Comparators and ablations

| Method key | Computation | Question answered |
|---|---|---|
| `hybrid` | ParameterNet initialization, then robust differentiable trajectory refinement | Does the full implemented estimator predict the withheld trajectory? |
| `neural` | ParameterNet initialization alone | What does refinement add? |
| `analytical` | Smoothed finite-difference angular regression for pendulum; quadratic least squares for sliding | How does a simple classical estimate compare? |
| `analytical_refine` | Same physical refinement starting from the classical estimate | Does the learned initialization improve the final solution? |
| `constant_velocity` | Line fitted to observed image positions, extrapolated in time | Does the assumed acceleration/oscillation model add useful predictive structure? |
| `iris_recurrence` | Published IRIS pendulum recurrence, fitted to shared observed angles and rolled out discretely | How does the published discrete physical component compare on matched measurements? |

The observed-angle IRIS comparator uses the [pinned published pendulum recurrence](https://github.com/KurbanIntelligenceLab/iris-bench/blob/33106189830e766bb286aa625f160d599f14042d/src/models/PhysModels.py):

\[
\theta_{i+1}=2\theta_i-\theta_{i-1}-\Delta t\,\beta(\theta_i-\theta_{i-1})
-\Delta t^2\,k\sin\theta_i.
\]

A constrained one-step least-squares fit estimates `k` and `beta` from the fit prefix. The full autoregressive rollout starts with its first two observed angles; future observations are not supplied. This adapter uses shared tracking, geometry and split so image measurement quality is held constant. It constrains `k` to 0.1–100 s⁻² and damping to 0–3 s⁻¹. It is a **published-component comparison with an observed-angle adapter**, not a reproduction of EndPhys's learned image encoder, latent regularization or reported complete pipeline scores. Sliding has no corresponding pendulum recurrence and must not be included in an aggregate for that method.

The original estimator's raw trajectory inputs are accompanied by analytical summary features. These experiments ablate inference/refinement methods; they do not isolate the contribution of each input feature or establish that a neural network is necessary.

## Metrics and uncertainty

- **Primary parameter relative error:** `abs(estimate − truth) / abs(truth)`, for synthetic pendulum `k` and sliding acceleration. Damping error is reported separately in s⁻¹.
- **Forecast RMSE:** Euclidean image-position root mean squared error over the withheld suffix, in pixels. Synthetic evaluation compares against clean known positions. Real evaluation compares against tracked positions, so tracking and model errors are mixed.
- **Normalized forecast RMSE:** synthetic RMSE divided by the maximum clean image displacement range. This helps compare cases with different motion extents, without inventing a metric scale.
- **Runtime:** synthetic inference time excludes decoding/tracking. Real reports retain tracking, inference and total times separately. Single-run wall-clock values include host load and cold-start overhead; they are not a controlled hardware speed ranking.
- **Interval coverage:** fraction of known synthetic primary parameters inside the displayed local interval. A nominal 95% interval is not evidence of 95% empirical coverage. Inspect this value and the small sample count before interpreting the uncertainty display.

The final 48-trajectory benchmark's hybrid intervals cover the primary truth in 62.5% of pendulum cases and 70.8% of sliding cases. These values show substantial undercoverage, not a 95% coverage guarantee. The JSON/results tables preserve the counts and per-case outcomes. Scale uncertainty, perspective, geometry uncertainty, correlated tracking noise and physical model mismatch are not propagated by the local Jacobian covariance.

## Interpreting the real clips

IRIS supplies nominal pendulum rope length 0.50 m. RealityForge recovers `k = g/L` and can compute an **effective dynamic length assuming gravity 9.80665 m/s²**. A difference from 0.50 m is reported as nominal-length consistency, not precise parameter error: string length may exclude the bob radius, and metadata does not establish exact pivot-to-center metrology.

The sliding metadata supplies incline geometry but no independently measured acceleration or friction. Real sliding acceleration error is therefore unavailable. Reporting a low image-trajectory error does not manufacture that missing truth. The household pendulum's approximate length is author-reported and has no independently verified uncertainty.

See [`samples/provenance.json`](../samples/provenance.json) for exact source recording paths, clip windows, source revision, frame rates, transformations, licenses and hashes.

## Published results and the remaining comparison limit

The pinned IRIS source and released CSVs use inconsistent apparent interpretations of the saved pendulum `alpha` parameter. `PhysModels.Pendulum` treats `abs(alpha)` as length, whereas the comparison runner maps a positive `alpha` through `g/alpha`. The CSV header also has fewer fields than some rows. Published raw values therefore must not be silently converted into a matched physical-accuracy leaderboard.

The complete published encoder pipeline is not reproduced here. Neither PhysTwin's RGB-D deformable reconstruction nor SimFoundry's scene-generation stack solves the same small planar estimation problem with the same sensing and compute. Their reported metrics are not presented as head-to-head results. The concrete comparison here is the implemented published recurrence component plus the matched classical and learned/refined ablations. See [source inspection and rationale](research.md).

## Known failure cases

1. **Large-amplitude and perspective-affected pendulum footage:** the circular planar model can show a modest fit error yet accumulate a larger forecast error. Inspect the track and radial residual before interpreting damping or gravity.
2. **Release hands and occlusion:** an initial hand near the object can contaminate tracking or the fitted initial state. Select the object carefully and trim to a free-motion interval.
3. **Sliding during changing contact or near impact:** constant acceleration can fit the prefix while failing on the suffix. It does not recover an actual friction/contact law.
4. **Small pendulum arcs:** circle geometry is weakly constrained, and a nearly straight visible path can be ambiguous with sliding. Supply a measured pivot or select a model after checking the temporal motion.
5. **Neural-only forecasts:** plausible parameter guesses can still cause substantial phase drift. The refinement ablation is essential.
6. **Uncalibrated cameras:** a metric scale alone cannot correct a tilted motion plane or moving camera. A precise-looking interval can remain biased.
7. **Out-of-family motion:** collisions, deformables, multiple interacting objects, moving pivots and arbitrary 3D motion remain outside the validated scope.

A repeatable failure with a visible diagnostic is more useful than an unsupported successful reconstruction claim. Contributions should add matched evidence before expanding the supported model family.
