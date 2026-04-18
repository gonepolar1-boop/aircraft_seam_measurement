"""Shared pytest configuration.

Adds ``src/`` to ``sys.path`` so tests can import ``pipeline.*`` and
``seam_training.*`` without installing the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
