"""Structured dictionary contracts for the seam-measurement pipeline.

The pipeline historically passed data around as ``dict[str, Any]`` which
hid typos until runtime and blocked IDE/mypy autocomplete. These
``TypedDict`` aliases describe the shape of the most common payloads so
call sites can be annotated with concrete types. ``total=False`` is used
where a payload can legitimately be partial (e.g. an empty-section
placeholder that omits all numeric fields).

The runtime objects are still plain ``dict`` instances; ``TypedDict`` is
purely a static annotation that costs nothing at import time.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------

class PlotPoints(TypedDict):
    """Bundle of arrays describing a set of 2D/3D points used for plotting.

    ``u`` and ``z`` are the in-section reference coordinates; ``pixels_xy``
    is the integer pixel coordinate each point was sampled from; ``xyz``
    is the full 3D world coordinate.
    """

    u: np.ndarray      # (N,) float32
    z: np.ndarray      # (N,) float32
    pixels_xy: np.ndarray  # (N, 2) float32
    xyz: np.ndarray    # (N, 3) float32


class LineFit(TypedDict, total=False):
    """Output of ``np.polyfit`` applied to the top-surface points on one
    side of the seam. ``valid`` is ``False`` when the fit could not be
    performed reliably; ``z_ref`` is set on refined fits only."""

    slope: float
    intercept: float
    valid: bool
    z_ref: float


class ReferenceFrame(TypedDict):
    """Per-section local frame used to project gap/flush into a common
    coordinate system."""

    valid: bool
    origin_xy: np.ndarray  # (2,)
    tangent: np.ndarray    # (2,)
    normal: np.ndarray     # (2,)


# ---------------------------------------------------------------------------
# Top-surface detection
# ---------------------------------------------------------------------------

class TopSurfacePayload(TypedDict, total=False):
    """Output of :func:`pipeline.seam_measurement.top_surface.detect_top_surface_edges`.

    ``total=False`` because the empty / error payloads only populate a
    subset of the keys.
    """

    valid: bool
    reason: str
    z_top: float
    left_z_top: float
    right_z_top: float
    center_margin: float
    background_points: PlotPoints
    local_background: PlotPoints
    neighbor_filtered: PlotPoints
    left_fit: LineFit
    right_fit: LineFit
    left_model: LineFit
    right_model: LineFit
    top_band: PlotPoints
    left_candidates: PlotPoints
    right_candidates: PlotPoints
    left_segment: PlotPoints
    right_segment: PlotPoints
    left_edge: PlotPoints
    right_edge: PlotPoints


# ---------------------------------------------------------------------------
# Per-section measurement result
# ---------------------------------------------------------------------------

class SectionMeasurement(TypedDict, total=False):
    """Gap / flush measurement for a single section."""

    valid: bool
    reason: str
    gap: float
    flush: float
    left_point: dict[str, float]   # {"u": float, "z": float}
    right_point: dict[str, float]
    left_model: LineFit
    right_model: LineFit
    gap_reference_frame: dict[str, Any]
    reference_mode: str


class SectionResult(TypedDict, total=False):
    """The aggregate per-section record passed through the pipeline after
    bottom filtering, top-surface detection, and gap/flush computation."""

    valid: bool
    reason: str
    sample_index: int
    center_xy: np.ndarray
    local_mask_width: float
    gap: float
    flush: float
    filtered_points: PlotPoints
    isolated_filtered: PlotPoints
    top_surface: TopSurfacePayload
    measurement: SectionMeasurement


# ---------------------------------------------------------------------------
# Profile / summary (used by outputs.py and the C# host via summary.json)
# ---------------------------------------------------------------------------

class SectionProfileEntry(TypedDict):
    sample_index: int
    distance_mm: float
    valid: bool
    reason: str
    gap: float
    flush: float
    center_x: float
    center_y: float
    left_u: float
    left_z: float
    right_u: float
    right_z: float


class GapFlushSummary(TypedDict, total=False):
    unit: str
    num_sections: int
    num_measurement_sections: int
    gap_mean: float
    gap_mean_mm: float
    gap_std: float
    gap_std_mm: float
    gap_min: float
    gap_max: float
    flush_mean: float
    flush_mean_mm: float
    flush_std: float
    flush_std_mm: float
    flush_min: float
    flush_max: float
    bottom_count_mean: float
    bottom_count_std: float
