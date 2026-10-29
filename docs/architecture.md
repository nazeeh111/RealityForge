# Recover, forecast, then explore

The schematic figure follows the default hybrid workflow. Its mechanics sketch, record marks and network glyph illustrate the method; they are not measured trajectories, neuron counts or accuracy evidence. Tracking produces image-plane positions and timestamps. Geometry, learned initialization and physical refinement then use the initial track window; the remaining track checks the original model's forecast. Parameter edits form a separate, conditional simulation.

| Figure stage | Implementation |
|---|---|
| Track one object | [`track_video`](../realityforge/tracking.py): ROI color centroids, template correlation or foreground components |
| Geometry and family | [`infer_track`, `circle_geometry`, `line_geometry`](../realityforge/inference.py): an initial-window circle or straight path; automatic selection screens these two families |
| Learned initial estimate | [`features`, `estimate`](../realityforge/estimator.py): 96 resampled signal values and four features; synthetic-only trained weights |
| Physical refinement | [`refine`](../realityforge/inference.py): confidence-weighted Huber residuals and L-BFGS; fitted geometry stays fixed |
| Forecast check | [`infer_track`](../realityforge/inference.py): compare the original model's predicted suffix with tracked positions, reporting pixel RMSE and warnings |
| Changed conditions | [`counterfactual`](../realityforge/api.py) calls [`simulate`](../realityforge/physics.py) with edited parameters and the recorded geometry/time axis |

Tracking precedes the inference split. The figure's fit-window labels describe geometry, initialization and physical fitting; they do not describe an independent tracking run for each segment.

The learned initializer is not a demonstrated accuracy gain over analytical initialization plus refinement in the current evidence. Local intervals are uncalibrated sensitivity estimates. Counterfactuals retain the same camera, scale and model assumptions. See the [method](method.md), [model card](model-card.md) and [evaluation](evaluation.md).
