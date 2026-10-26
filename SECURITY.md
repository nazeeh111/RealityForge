# Security

RealityForge is a local research tool. It binds to loopback and has no accounts or hosted upload service. Do not expose the server to the public internet without adding an appropriate deployment security design.

Uploaded videos are untrusted decoder input. Keep Python, PyTorch, OpenCV, FastAPI and their dependencies updated. Only the included trusted model checkpoints are loaded; the app does not provide a model-upload endpoint. Checkpoints are loaded with `weights_only=True`.

Results and uploaded videos persist in `.realityforge/` under the launch directory, or `REALITYFORGE_OUTPUT_DIR`. Keep that folder private and remove unwanted recordings after use. It is ignored by Git. The application has no telemetry or cloud inference calls.

Report reproducible security issues through GitHub's private vulnerability reporting when available. Avoid including private recordings, credentials or exploit-sensitive details in public issues. This project has no paid support commitment or promised response deadline.
