"""Lightweight stage-timing helper for the gap/flush pipeline.

Produces a nested dict of elapsed wall-clock times per stage that is
attached to the pipeline result. Used both for quick eyeballing and to
feed the thesis figure ``ch6_fig11_pipeline_timing_breakdown.png``.

Usage:
    timer = StageTimer()
    with timer.stage("load_point_map"):
        point_map = load_point_map(...)
    with timer.stage("predict_mask"):
        ...
    summary = timer.summary()  # {"total_s": ..., "stages_s": {...}}
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


class StageTimer:
    """Minimal hierarchical stage timer using :func:`time.perf_counter`."""

    def __init__(self) -> None:
        self._stages: list[tuple[str, float]] = []
        self._start: float = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._stages.append((name, elapsed))

    def record(self, name: str, elapsed_seconds: float) -> None:
        """Explicitly record a pre-measured stage duration."""
        self._stages.append((str(name), float(elapsed_seconds)))

    def summary(self) -> dict[str, Any]:
        total = time.perf_counter() - self._start
        stages: dict[str, float] = {}
        for name, elapsed in self._stages:
            stages[name] = stages.get(name, 0.0) + float(elapsed)
        # Residual captures work not explicitly wrapped in a stage (e.g.
        # result-dict assembly, JSON export). Never goes negative; in the
        # unlikely event of clock noise we clamp to zero.
        attributed = sum(stages.values())
        stages["_residual"] = max(0.0, total - attributed)
        return {"total_s": float(total), "stages_s": stages}
