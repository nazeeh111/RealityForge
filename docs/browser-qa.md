# Browser verification

Verified against the running local application on `http://127.0.0.1:8001/`, 25 October 2026, using Chrome and the documented CUA browser controls. These are actual API-backed interactions and current-source screenshots, not fabricated response fixtures.

| Check | Observed result |
|---|---|
| Generated pendulum recovery | 180 observations, fit0.44px, held-out forecast0.42px, final run1.06s. Parameters exposed, model and interval assumptions visible. |
| Numeric counterfactual | Changed damping from approximately0.160 to0.500. Number input and slider synchronized; server returned a simulation; UI displayed orange counterfactual and `Counterfactual ready`. |
| Forecast scrub | Moved timeline to its end. Time5.97s, `FORECAST` indicator, source video frame and both physical trajectories rendered. |
| Generated sliding recovery | 180 observations, fit0.28px, forecast0.39px, run1.61s. Signed acceleration approximately−20.969px/s² and calibrated acceleration shown. |
| Real IRIS20 recovery | 240 observations, fit16.67px, forecast9.43px, run16.51s during concurrent local work. Explicit no-scale warning visible. This demonstrates transfer, not precision ground truth or official benchmark equivalence. |
| Analysis download | Downloaded JSON read back successfully: pendulum,180timestamps and metrics matching the generated-pendulum run. |
| Comparison image download | Actual app-generated PNG exists and decodes at1440×900. CUA's blob-download event observation timed out, but the file was independently read back successfully. |
| Responsive layout | Temporary390×844 viewport; full-page screenshot visually inspected. Source controls, comparison, timeline, parameters, diagnostics and export controls remain available without horizontal overflow. Viewport override reset afterward. |
| JavaScript | `node --check web/app.js` passes. |

The current frontend explicitly sends null for cleared optional calibration fields, rewinds source video to frame0 before selecting an initial tracking region, scopes legend colors to avoid coloring whole edited parameter panels, and distinguishes original fit intervals from counterfactual values. Runtime-critical warnings remain expanded; model assumptions are in an expandable disclosure.

## Evidence

- `assets/realityforge-workbench.jpg`: full-page generated-fixture comparison, including counterfactual and held-out forecast.
- `assets/realityforge-mobile.jpg`: generated-fixture workbench at390pixels wide.
- `assets/realityforge-comparison.png`: actual in-app comparison-image export.
- `assets/iris-workbench-cc-by-nc.jpg`: actual IRIS20 result. Contains CC BY-NC4.0 footage; separate attribution/license applies.

Generated fixtures and their presentation assets are original MIT work. IRIS screenshot credit: Rasul Khanbayov, Mohamed Rayan Barhdadi, Erchin Serpedin and Hasan Kurban, CC BY-NC4.0. See `assets/NOTICE.md` and `../samples/iris/NOTICE.md`.

## Verification limits

This pass did not exercise an arbitrary personal upload through the native file picker. The generated and IRIS sample selections use the same analysis endpoint with server-side sample paths. The final backend suite also exercises a real uploaded generated video through inference, edited simulation and export. Browser UI checks do not establish calibrated uncertainty coverage or general camera/scene robustness.

The documented CDP `Page.startScreencast` observation call hung until interrupted. It produced no recorded frames and is not counted as demo evidence. Ordinary screenshots succeeded. The delivered demo was subsequently captured in a separate local test browser. Its actual response, timing and artifact hashes are in `assets/demo-provenance.json`.
