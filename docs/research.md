# Research foundations and scope

Inspected 25 October 2026. The source repositories are research references, not the product's implementation base. RealityForge's pipeline and interface are original code. No PhysTwin, SimFoundry or IRIS code or weights are vendored into the application.

| Foundation | Inspected revision | License and concrete inspection |
|---|---|---|
| [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) | `81c718790a37e5e0102eb77af2c6edd34a9db25f` | MIT. README, license and `qqtt/engine/trainer_warp.py`. |
| [SimFoundry](https://github.com/NVlabs/simfoundry) | `9e34ebefcd020583fbb755a8b57268dce78eca26` | Apache 2.0. README, license, pipeline entry point and `simfoundry/pipeline/articulation_physics.py`. |
| [IRIS code](https://github.com/KurbanIntelligenceLab/iris-bench) | `33106189830e766bb286aa625f160d599f14042d` | MIT. `src/models/PhysModels.py`, `src/models/model.py`, `src/losses/loss.py`, multi-clip runner and released result CSVs. |
| [IRIS dataset](https://huggingface.co/datasets/rasulkhanbayov/IRIS) | `c253822f55431ca80ef2084de4bc5e79d1a488f1` | **CC BY-NC 4.0**. Dataset card, file tree, parameters.json and selected recordings. Separate notices accompany the clips. |
| [IRIS study](https://arxiv.org/abs/2603.16432) | arXiv 2603.16432v3, 1 July 2026 | Same IRIS project, not a fourth independent algorithm. Primary paper, experimental protocol, identifiability and failure discussions. |

## What was useful

**PhysTwin:** reconstruction from RGB-D observations, spring-mass dynamics, tracked-geometry losses and first-order parameter optimization show the value of explicit physical fitting. Its native 3D deformable model relies on stronger sensing and GPU dependencies. RealityForge uses the fitting principle with an intentionally smaller 2D dynamics model; it does not claim a replacement for PhysTwin or its 3D reconstruction.

**SimFoundry:** its interchangeable scene-generation stages and explicit physics-property overrides inform the separation between video tracking, system identification and counterfactual simulation. Its complete path requires Linux/NVIDIA, large installation storage, gated models and cloud VLM access. Those dependencies are unnecessary for the supported pendulum and sliding systems. RealityForge implements its own CPU path and does not invoke a paid cloud service or infer detailed geometry/materials from appearance.

**IRIS paper:** the benchmark motivates separate checks of equation selection, parameter recovery, extrapolation and identifiability. Its latent-state pipelines and multi-step objectives provide useful comparisons, but published results include unstable regimes and calibration caveats. RealityForge uses observed 2D tracks, fixed dynamics families and explicit units. A small recording subset is never presented as the official complete benchmark.

## Data and baselines require care

IRIS `parameters.json` gives pendulum rope length 0.50 m, release angle and camera distance. Its sliding entries give incline geometry, **not friction or acceleration ground truth**. Zero standard deviations encode fixed settings; they do not establish perfect measurement accuracy. Rope length may differ from pivot-to-bob-center effective length. Report this distinction when comparing a dynamic estimate with nominal 0.50 m.

The pinned IRIS `PhysModels.Pendulum` computes its restoring coefficient as `9.80665 / abs(alpha)`, treating alpha as a length. The pinned comparison/multi-clip mapping instead converts alpha to length as `g / alpha`. These conventions differ. Published CSVs should therefore not be silently interpreted as a direct matched benchmark for this product. An observed-angle finite-difference baseline based on the same physical equation is a **reimplementation of a discrete residual idea**, not a reproduced IRIS encoder or paper baseline. A fair future paper comparison needs matched preprocessing, validated parameter mappings, official splits, repeated seeds and comparable compute.

Our selected take 10 recordings are never used to train the compact estimator. Their prefix can be fitted at inference time; future frames are withheld from physical fitting. This is distinct from generalization to unseen parameter distributions. Data selection and manual ROI initialization are documented in `samples/provenance.json`; the sample is small and selected for supported physics.

## Independent footage

The independent recording is [Schwingung (Definition).webm](https://commons.wikimedia.org/wiki/File:Schwingung_(Definition).webm), by Rhetos / Mathe-AC Lernwerkstatt Aachen, 28 May 2024, under **CC BY-SA 4.0**. The supplied clip is the normal-speed 15..25 second interval, resized with audio removed. The author reports an approximate 0.5 m pendulum; this is not independently verified metrology. The pivot is outside the frame. The excerpt and its adaptations remain CC BY-SA 4.0, separately from the application's MIT code.

## Scope boundary

Scale is unobservable from arbitrary pixels alone. Pendulum timing constrains `g/L`; reporting a length requires a stated gravity assumption, and reporting gravity requires a known length. Sliding motion constrains image acceleration; SI acceleration requires a reference scale in the motion plane. Inferring friction further requires the incline angle and a valid force model. Perspective, camera motion, contact, external forcing and out-of-plane motion can invalidate these equations. Pixel tracking precision and parameter fit uncertainty cannot establish correctness of the camera or physical model.

Collisions and deformables are out of the validated scope. They are intentionally deferred until tracking, contact modeling, identifiability and out-of-sample evaluation support them.

## Frozen recording check

After tracker development on the take10 clips, six additional pendulum recordings were preselected: take03 and take06 at 20°, 45° and 90°, always source seconds 2..10. The current core and weights were hashed before acquisition. Region initialization uses one fixed first-frame green-component rule; neither future frames nor later error measurements change it. All six attempts are retained in `benchmarks/blind-iris.json`. This is a genuinely unseen-recording transfer check within the IRIS pendulum setup, not the full official benchmark or evidence for arbitrary physics. The separate provenance file preserves the frozen hashes and exact transformations.
