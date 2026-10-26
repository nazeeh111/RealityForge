"""Locate checked-out assets or the equivalent assets bundled in a wheel."""
from pathlib import Path
import os

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE / 'assets' if (PACKAGE / 'assets').is_dir() else PACKAGE.parent
# Runtime output belongs to the working directory, never read-only site-packages.
RUNTIME = Path(os.environ.get('REALITYFORGE_OUTPUT_DIR', str(Path.cwd() / '.realityforge'))).resolve()
