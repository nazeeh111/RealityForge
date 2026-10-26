# Release evidence

RealityForge 0.1.0 is a working local application for two supported planar dynamics families. This page maps its concrete claims to checkable artifacts. It does not certify arbitrary-video reconstruction.

| Requirement | Current evidence |
|---|---|
| Upload → track → infer → predict | Positive uploaded-video API test; actual pendulum/sliding/IRIS browser runs; `benchmarks/verification/` logs |
| Original learned estimator | Seeded training generator, two 6,000-trajectory training runs, included checkpoints and `models/training.json`; no upstream weights/code vendored |
| Differentiable physical fitting | PyTorch pendulum/sliding equations and robust fitting; independent SciPy, gradient and sparse-time tests |
| Held-out forecast isolation | Geometry/parameters/background restricted to observation prefix; tests modify only future pixels/positions |
| Parameter observability and scale | Explicit k=g/L and image acceleration units, optional uniform calibration, conditional intervals and scope warnings |
| Interactivity and sharing | Actual numeric parameter edits, synchronized playback/scrub, downloaded JSON and actual comparison PNG |
| Measured synthetic evaluation | 48 independent trajectories, noise/outliers, parameter and forecast error, runtime and method ablations in `benchmarks/synthetic.json` |
| Real development transfer | Three IRIS pendulums, one IRIS slide, one independently sourced household pendulum; per-case artifacts in `benchmarks/real.json` |
| Frozen unseen-recording transfer | Six preselected IRIS recordings acquired after source/model freeze; all attempts and hashes retained in `benchmarks/blind-iris.json` |
| Published comparison | Original observed-angle adapter for the pinned IRIS pendulum recurrence, independently tested against that equation |
| Independent review | Recorded findings, fixes and regression checks in `validation-review.md` |
| Exportable models | Modern `.pt2` and legacy TorchScript artifacts; checkpoint equivalence and file hashes tested |
| Installable application | Wheel includes models/web/samples; isolated installed-wheel analysis, counterfactual and export smoke check |
| Authentic presentation | Actual 28.08-second app recording, generated fixture, no mocked responses or speedup; screenshots, mobile QA and video provenance |
| Open-source distribution | MIT original code/weights/fixtures; separate CC BY-NC/CC BY-SA recording notices, acquisition provenance and reproducibility commands |

The final local suite passes **65 tests**, with three dependency deprecation warnings. Logs and exact source/model/web hashes are in [`benchmarks/verification/checks.json`](../benchmarks/verification/checks.json). [Installed wheel log](../benchmarks/verification/wheel-smoke.txt). The complete installed-wheel check is recorded separately in that verification folder. CI runs on Linux and macOS and repeats a built-wheel smoke test; consult the live workflow for its current result.

The frozen test's median forecast is 12.59 pixels. Analytical initialization plus the same refinement matches the learned initializer; the observed-angle IRIS recurrence adapter reaches 54.01 pixels. These results compare the documented components, not complete published latent-video pipelines. They do not establish a neural advantage or a new state of the art.

The nominal 95% intervals are uncalibrated: primary-parameter coverage is 62.5% for the 24 synthetic pendulum cases and 70.8% for sliding. This substantial undercoverage, the lack of real sliding parameter truth, the approximate nominal pendulum rope length, and the restricted camera/model scope remain explicit limitations.

Full latent-image IRIS/EndPhys reproduction, general perspective/lens calibration, calibrated predictive uncertainty, collisions, deformables and arbitrary 3D reconstruction are not delivered or implied. The published recurrence adapter and supported planar pipeline are the independently verified result.
