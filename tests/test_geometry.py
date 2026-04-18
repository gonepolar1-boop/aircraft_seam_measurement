"""Unit tests for pure-math helpers in :mod:`pipeline.seam_measurement.geometry`.

These tests intentionally avoid anything that requires torch / opencv /
open3d so they run in a minimal environment (numpy + pytest).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.seam_measurement import geometry as geo


def test_reference_basis_horizontal_line():
    tangent, normal = geo.reference_basis_from_fit({"slope": 0.0, "intercept": 5.0})
    assert tangent is not None and normal is not None
    np.testing.assert_allclose(tangent, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(normal, [0.0, 1.0], atol=1e-6)


def test_reference_basis_diagonal_line():
    tangent, normal = geo.reference_basis_from_fit({"slope": 1.0, "intercept": 0.0})
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    np.testing.assert_allclose(tangent, [inv_sqrt2, inv_sqrt2], atol=1e-6)
    # Normal is tangent rotated 90 degrees counter-clockwise.
    np.testing.assert_allclose(normal, [-inv_sqrt2, inv_sqrt2], atol=1e-6)


def test_reference_basis_rejects_nan_intercept():
    tangent, normal = geo.reference_basis_from_fit({"slope": 0.0, "intercept": float("nan")})
    assert tangent is None
    assert normal is None


def test_transform_xy_to_reference_projects_along_axes():
    fit = {"slope": 0.0, "intercept": 0.0}
    origin = np.asarray([0.0, 0.0], dtype=np.float32)
    points = np.asarray([[3.0, 4.0], [-1.0, 2.0]], dtype=np.float32)

    projected = geo.transform_xy_to_reference(points, origin, fit)

    # With slope=0 the reference frame is the identity: tangent=[1,0], normal=[0,1].
    np.testing.assert_allclose(projected, points, atol=1e-6)


def test_measure_point_line_gap_flush_horizontal_line():
    result = geo.measure_point_line_gap_flush(
        source_point=np.asarray([0.0, 5.0], dtype=np.float32),
        other_edge_point=np.asarray([3.0, 0.0], dtype=np.float32),
        fit={"slope": 0.0, "intercept": 0.0},
    )
    assert result["flush"] == pytest.approx(5.0, abs=1e-5)
    assert result["gap"] == pytest.approx(3.0, abs=1e-5)
    np.testing.assert_allclose(result["foot_point"], [0.0, 0.0], atol=1e-6)


def test_build_section_reference_frame_valid():
    left_fit = {"slope": 0.0, "intercept": 0.0}
    right_fit = {"slope": 0.0, "intercept": 0.0}
    frame = geo.build_section_reference_frame(
        left_fit=left_fit,
        right_fit=right_fit,
        left_point_xy=np.asarray([-1.0, 0.0], dtype=np.float32),
        right_point_xy=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert frame["valid"] is True
    np.testing.assert_allclose(frame["origin_xy"], [0.0, 0.0], atol=1e-6)
    # Both fits are horizontal so the averaged normal should be +z.
    np.testing.assert_allclose(frame["normal"], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(frame["tangent"], [1.0, 0.0], atol=1e-6)


def test_build_section_reference_frame_invalid_when_intercept_nan():
    frame = geo.build_section_reference_frame(
        left_fit={"slope": 0.0, "intercept": float("nan")},
        right_fit={"slope": 0.0, "intercept": 0.0},
        left_point_xy=np.asarray([-1.0, 0.0], dtype=np.float32),
        right_point_xy=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    assert frame["valid"] is False


def test_measure_gap_in_reference_frame_returns_distance_along_tangent():
    left_fit = {"slope": 0.0, "intercept": 0.0}
    right_fit = {"slope": 0.0, "intercept": 0.0}
    frame = geo.build_section_reference_frame(
        left_fit=left_fit,
        right_fit=right_fit,
        left_point_xy=np.asarray([-1.0, 0.0], dtype=np.float32),
        right_point_xy=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    result = geo.measure_gap_in_reference_frame(
        left_point_xy=np.asarray([-2.0, 0.3], dtype=np.float32),
        right_point_xy=np.asarray([3.0, -0.1], dtype=np.float32),
        frame=frame,
    )
    # Tangent direction is +x; gap = |3.0 - (-2.0)| = 5.0.
    assert result["gap"] == pytest.approx(5.0, abs=1e-5)


def test_measure_gap_in_reference_frame_invalid_frame_returns_nan():
    invalid_frame = {"valid": False}
    result = geo.measure_gap_in_reference_frame(
        left_point_xy=np.asarray([0.0, 0.0], dtype=np.float32),
        right_point_xy=np.asarray([1.0, 0.0], dtype=np.float32),
        frame=invalid_frame,
    )
    assert math.isnan(result["gap"])
