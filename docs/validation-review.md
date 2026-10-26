# Independent release review

Reviewed source, numerical models, API, tracking, saved estimators, tests and raw benchmark artifacts on 25 October 2026. This review distinguishes implementation evidence from scientific claims. It is not a certification of arbitrary-video reconstruction.

## Checks observed

The existing 39 tests passed in 28.33 seconds in the task's Python 3.12 environment. The only warning concerned Starlette's deprecated use of httpx in its test client. The coverage includes an independent SciPy integrator reference, finite-difference gradient agreement, undamped energy conservation, exact sliding acceleration, clean/noisy parameter recovery, counterfactual changes, fit/forecast separation, units, decoding, pixel-centroid tracking, malformed uploads, simulation validation and export path boundaries.

The independent review pass observed **62 passing tests** in 108.78 seconds. In addition to the original coverage, it verifies shipped checkpoints against both TorchScript exports, neural initialization against an independent solver, small-angle periodic automatic selection, explicit clearing of sample settings, invalid geometry, sparse-observation integration stability, a published-recurrence component adapter and bounded long-video tracking. Three warnings concern deprecated dependency interfaces, not failed checks. A subsequent nine-test model subset also passed in 16.17 seconds, including two new checks that modern `.pt2` exports match the checkpoints and that both export hashes match their provenance records.

Reproduce from the project root with `python -m pytest -q`. The separate final release verification records **65 passing tests**, including a positive uploaded-video roundtrip. Its complete portable log and exact artifact hashes are in `../benchmarks/verification/`. Browser, wheel-installation and hosted-CI evidence are separate from this Python suite and are recorded by their owning release checks.

## Scientific interpretation

- The trained initializer consumes resampled trajectories plus analytical derivative features. Initial angle or position is taken from the observation. The neural-only comparison is therefore an initializer ablation, not a raw-video neural model.
- The hybrid and analytical-initialization-plus-refinement paths essentially tie in the recorded benchmark. The evidence supports the value of explicit physics refinement; it does not establish neural superiority or a new state of the art.
- The expanded final synthetic test has 24 trajectories per family. Its hybrid local nominal 95% primary-parameter intervals cover 62.5% of pendulum cases and 70.8% of sliding cases. These intervals are not calibrated 95% coverage. They exclude geometry, camera, tracking bias and model uncertainty.
- Real trajectory errors are measured against extracted image tracks. They are useful forecast consistency checks, not independent proof of tracking precision. The selected IRIS recordings informed tracker development and are not a blind test. No real recording entered neural training.
- IRIS's nominal string length is not necessarily effective pivot-to-bob-center length. Its sliding metadata supplies no friction or acceleration truth. The independent recording's length is author-reported. Parameter accuracy claims must preserve these limits.
- No complete published GPU/video baseline was reproduced. The new `iris_recurrence` comparator implements the published IRIS discrete pendulum physics component on shared observed angles. I compared it with the pinned source recurrence in `src/models/PhysModels.py`, lines 214-224, and found the equation consistent after reparameterizing `k = g/L`. Its constrained least-squares fit and observed-angle input differ from the full latent-video system. Its component-level scope must remain explicit. The separate analytical baseline uses smoothed derivatives.

## Findings and inspected resolutions

1. **Resolved:** automatic classification originally labeled clean 0.05 and 0.10 rad periodic pendulums as sliding. Fit-prefix motion-reversal screening now recognizes both; dedicated tests pass. Automatic selection remains a two-family heuristic and warns that a short monotonic arc can be ambiguous. Manual selection is available.
2. **Resolved:** optional nulls now survive server validation and override sample defaults. Two API tests distinguish explicitly cleared ROI/pivot/scale from unspecified defaults. The frontend sends null for blank optional inputs.
3. **Resolved:** decoding now streams frames in bounded passes. It retains at most 31 grayscale frames for foreground background construction. A 70-second, 320×200 decoded clip test passes with peak traced Python/NumPy allocation below 25 MiB. This excludes OpenCV's native allocations; it is not a total-process RSS claim. The output is capped at 60 seconds and includes a truncation warning, matching the counterfactual time limit.
4. **Resolved:** the Makefile now uses the accepted `--cases` synthetic-evaluation argument.
5. **Implementation inspected; installation evidence separate:** the build hook bundles web/model/sample assets in wheels; `resources.py` selects those bundled assets, and runtime writes go to a working-directory output folder. The separate installed-wheel smoke check passed video inference, counterfactual simulation, export and a positive upload. Its result is preserved in `../benchmarks/verification/wheel-smoke.txt`.
6. **Resolved:** pendulum RK4 now subdivides sparse observation intervals to at most 1/30 second. An independent SciPy reference checks a 20-second trajectory observed only every half-second, closing the long-clip integration instability.

## Remaining scope limits

The evidence establishes the supported planar pendulum and constant-acceleration pipeline, not arbitrary-video physics recovery. Full camera calibration, 3D reconstruction, collisions, deformables, general governing-equation discovery, calibrated uncertainty and matched complete IRIS video baselines remain outside the validated implementation. The README, model card and evaluation report must preserve these boundaries. GitHub publication, a polished interface or a green test suite does not remove them.
