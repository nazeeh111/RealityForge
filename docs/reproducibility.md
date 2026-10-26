# Reproduce RealityForge

Run commands from the repository root. The application uses local CPU computation and needs no account, API key, cloud inference or separately billed service. Dependency installation and optional data acquisition need internet access. Bundled weights and samples allow the normal demo to run without retraining or redownloading data.

## Install and run

Python 3.11–3.14 is declared by the package. The recorded development environment uses Python 3.12.13 on Apple Silicon macOS, PyTorch 2.14.1 and NumPy 2.5.4. Use Python 3.12 for the closest reproduction; exact tested dependencies are recorded separately from the broad supported ranges in `pyproject.toml`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m realityforge.cli serve --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Choose a sample, inspect its first-frame region, and select **Recover dynamics**. Scrub into the held-out portion, edit a physical parameter, then download the analysis JSON or comparison image. The service binds to loopback only. Keep it local; it is not a hardened multi-user hosting service.

To analyze one original generated video without a browser:

```bash
mkdir -p local-results
python -m realityforge.cli analyze samples/generated/pendulum.mp4 \
  --settings '{"mode":"pendulum","roi":[620,310,46,46],"pivot":[480,105],"fit_fraction":0.65}' \
  --output local-results/pendulum.json
```

Coordinates refer to original video pixels. A rectangle is `[x,y,width,height]`; the pivot is `[x,y]`. Supply `meters_per_pixel` only if a reference length at the motion plane supports it. See [method assumptions](method.md).

## Run the checks

```bash
python -m pytest -q
```

Tests cover an independent SciPy solver, physical invariants, analytic derivatives versus automatic differentiation, known-parameter recovery, forecast isolation, decoded-video tracking, input rejection, simulation edits and export boundaries. The published-recurrence adapter has a separate known-discrete-system recovery check. Passing these tests supports those contracts; it is not a full scientific validation of arbitrary footage.

## Train both estimators from scratch

Preserve bundled weights while reproducing into a separate directory:

```bash
python scripts/train.py --count 6000 --epochs 100 --seed 47 \
  --output local-results/retrained-models
```

The command trains one network per family, chooses the best validation checkpoint, exports each network and writes training metadata. No real recordings enter this command. See the [model card](model-card.md) for the full data distribution, input normalization and optimizer settings.

The fixed seed makes the random data and initialization reproducible in a compatible environment. Different library versions, hardware and numerical kernels can change floating-point outputs or serialized bytes. Compare parameter predictions and evaluation tolerances, not checkpoint-file byte identity. To use replacement weights, copy them deliberately to the corresponding model directory after preserving the originals, or pass a model directory to the Python estimator API.

## Recreate original rendered sample inputs

FFmpeg must be available on `PATH` with an H.264 encoder. The installed application decodes video with OpenCV; FFmpeg is needed for regenerating media and downloading trimmed external samples.

```bash
ffmpeg -version
python scripts/generate_samples.py
```

The generator renders the original MIT synthetic pendulum and sliding fixtures and their truth JSON. It updates their sample-manifest entries while preserving external sample entries. This is a deterministic scene specification; encoded bytes can vary with FFmpeg/OpenCV versions. A successful synthetic render is not evidence of a real experiment.

## Re-run measured evaluations

```bash
python scripts/evaluate_synthetic.py --seed 2026 --cases 24 \
  --output local-results/synthetic.json
python scripts/evaluate_real.py --output local-results/real.json
python scripts/evaluate_blind_iris.py
python scripts/evaluate_selection.py
python scripts/report_benchmarks.py
```

The first command evaluates 48 independently generated trajectories and every available method; `--cases` means cases **per family**. The real command evaluates the bundled real-recording subset. It reuses tracking within a recording so methods receive the same image measurements and records the shared tracking cost separately. It also writes detailed hybrid analysis artifacts under `benchmarks/analyses/`; preserve those files if comparing to the shipped version.

The shipped JSON contains source/model hashes and the recording provenance has data hashes. When source or weights change, rerun affected measurements instead of carrying forward an old table. See [evaluation protocol and limitations](evaluation.md).

## Reacquire separately licensed experimental samples

The shipped clips are small trims of public sources. Their licenses remain separate from the application:

```bash
python scripts/fetch_evaluation_data.py \
  --output local-results/reacquired
```

This downloads the independent CC BY-SA recording into a separate destination. For the noncommercial IRIS recording subset, explicitly include:

```bash
python scripts/fetch_evaluation_data.py \
  --output local-results/reacquired --include-noncommercial-iris
```

Existing destination files are skipped. The script prints hashes and preserves a source manifest. It seeks to the documented source windows, resizes and transcodes without changing playback speed. Acquisition can fail if the public host, byte-range support or FFmpeg HTTPS support changes; retain the original source URL and exact clip transformations from [`samples/provenance.json`](../samples/provenance.json).

Do not treat the source MIT license as permission to relicense the IRIS recordings or independent footage. See the [top-level notices](../NOTICE.md) and recording-folder notices before redistribution.

## Reading and sharing an analysis

The exported JSON contains observed and predicted image trajectories, fit-window boundary, physical parameters, conditional intervals, fit/forecast errors, diagnostics, settings and provenance. This lets another person inspect how the result was produced. It does not embed a universal 3D scene or certify the model assumptions.

The comparison image captures the current viewing time and counterfactual. An image or demo that includes a third-party recording must retain that recording's attribution and license. Use the clearly labeled generated fixtures for an MIT-only presentation.

## Model export and installed-package check

The checked-in `.pt2` files use `torch.export`; legacy TorchScript exports are also included. Export checks compare network outputs with each checkpoint and record SHA-256 hashes:

```bash
python scripts/export_models.py --models models
python -m build --no-isolation --wheel
python -m pip install --force-reinstall --no-deps dist/*.whl
python scripts/smoke_installed.py
```

For a strict package test, run the smoke script from outside the source checkout, as the CI workflow does. The wheel includes the web interface, model files, generated fixtures and separately licensed recordings. Model-export compatibility is version-sensitive; use the recorded environment for the closest match.

## A simple scale reference

If a known 0.50 m object at the motion plane spans 300 pixels, enter `0.50 / 300 = 0.001667` meters per pixel. Measure at the motion plane rather than on a nearer wall. This establishes a uniform image-plane scale only; perspective and lens distortion still require a separate calibrated imaging setup.

The frozen IRIS evaluator uses the six bundled `samples/blind-iris/` recordings and `samples/blind-provenance.json`. Its optional `--fetch` reacquires the predefined source windows. Source/model hashes were frozen before acquisition; no parameter or tracker tuning followed. All six attempts, including any failure in a rerun, are retained. These clips retain the IRIS CC BY-NC 4.0 license.

## Record the actual app demo

The app itself needs no Node build. These optional developer tools capture a separate test browser and require Node.js, FFmpeg and FFprobe:

```bash
npm ci
npx playwright install chromium
# With RealityForge already running on localhost:8000:
REALITYFORGE_URL=http://127.0.0.1:8000 npm run demo:record
python scripts/encode_demo.py \
  --recording .realityforge/demo-recording/recording.json
```

`REALITYFORGE_CHROME` can select an existing Chrome executable instead of downloaded Chromium. The recording uses a fresh test context and the real application endpoints. No response is mocked. It runs a labeled generated fixture, performs parameter editing, scrubs to the forecast and reads back an analysis download. The event log retains the actual analysis response and errors.

The encoder adds explanatory captions without changing the recording's timing. The GIF is a short excerpt of actual counterfactual playback. `docs/assets/demo-provenance.json` records timestamps, metrics and video/GIF hashes. The delivered movie is 28.08 seconds; its 6.23-second recovery time is the measured run, not an asserted universal latency.
