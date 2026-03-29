"""Example modules for the FunctorFlow v0 package.

This package adjusts `sys.path` when run directly from a source checkout so
commands such as `python -m examples.tutorial_v0` can still resolve the
top-level `FunctorFlow` package without requiring installation first.
"""

from __future__ import annotations

import sys
from pathlib import Path


_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
_REPO_PARENT = _REPO_ROOT.parent

if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))
