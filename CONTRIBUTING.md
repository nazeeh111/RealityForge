# Contributing

Reproduce the current behavior before changing it. Include a small example that exposes the problem and the assumptions under which a fix is valid.

Set up Python 3.12 and run `python -m pip install -e '.[dev]'`, then `python -m pytest -q`. For changes to physics, tracking or parameter inference, rerun the relevant evaluation scripts and compare per-case results rather than only an aggregate. Preserve the distinction between real footage, generated video and trajectory-only synthetic evaluation.

When adding a dynamics family, include its governing equation, observable quantities, scale/camera assumptions, independently generated recovery checks, a held-out forecast, failure examples, and data-license provenance. Avoid a new family that only works in a rendered showcase.

Interface changes should be checked in the running app at desktop and narrow-screen widths, including pending and error states. Keep inputs local and exports inspectable. Do not add cloud inference, network telemetry or paid-service requirements without an explicit product decision.

Do not commit uploaded user recordings, `.realityforge/`, `.runtime/`, caches, virtual environments or private information. New sample data needs a source, rights to redistribution and any required notice. Research references do not automatically permit copying their code or media.

Submit changes through a pull request with the problem, resulting behavior, validation and known limitations. Code and documentation contributions use the MIT license; separately licensed media must retain its own terms.
