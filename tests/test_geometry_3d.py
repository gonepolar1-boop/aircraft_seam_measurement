"""Tests for scheme B 3D gap / flush helpers.

Covers the new ``fit_plane_3d`` and ``measure_gap_flush_3d`` in
:mod:`pipeline.seam_measurement.geometry`. Uses only numpy + pytest.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.seam_measurement.geometry import fit_plane_3d, measure_gap_flush_3d


# ---------------------------------------------------------------------------
# fit_plane_3d
# ---------------------------------------------------------------------------

def test_fit_plane_on_flat_xy_points_returns_plus_z_normal():
    rng = np.random.default_rng(0)
    pts = np.zeros((40, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-5, 5, size=40)
    pts[:, 1] = rng.uniform(-5, 5, size=40)
    pts[:, 2] = 0.0  # perfectly flat at z = 0
    result = fit_plane_3d(pts)
    assert result is not None
    np.testing.assert_allclose(result["normal"], [0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(result["centroid"], [0.0, 0.0, 0.0], atol=0.8)


def test_fit_plane_on_tilted_plane_recovers_correct_normal():
    # Plane equation z = 0.5 * x (tilt around the y axis).
    rng = np.random.default_rng(1)
    xs = rng.uniform(-5, 5, size=60)
    ys = rng.uniform(-5, 5, size=60)
    zs = 0.5 * xs
    pts = np.column_stack([xs, ys, zs]).astype(np.float32)
    result = fit_plane_3d(pts)
    assert result is not None
    normal = result["normal"]
    # Expected normal direction (from gradient): (-0.5, 0, 1), normalised.
    expected = np.asarray([-0.5, 0.0, 1.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    # Account for sign convention (normal oriented toward +z, which
    # matches our expected vector already because expected[2] > 0).
    np.testing.assert_allclose(normal, expected, atol=1e-4)


def test_fit_plane_rejects_collinear_points():
    # Points on a straight line in 3D - SVD sees only one non-zero SV.
    ts = np.linspace(-2.0, 2.0, 20, dtype=np.float32)
    pts = np.column_stack([ts, 2.0 * ts, 3.0 * ts]).astype(np.float32)
    assert fit_plane_3d(pts) is None


def test_fit_plane_rejects_insufficient_points():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    assert fit_plane_3d(pts, min_points=6) is None


def test_fit_plane_orients_normal_toward_plus_z():
    # Even when the raw SVD would return -z as the normal, we flip it.
    rng = np.random.default_rng(3)
    pts = np.zeros((50, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(-5, 5, size=50)
    pts[:, 1] = rng.uniform(-5, 5, size=50)
    pts[:, 2] = 10.0  # offset in z does not change the normal direction
    result = fit_plane_3d(pts)
    assert result is not None
    assert float(result["normal"][2]) > 0.0


# ---------------------------------------------------------------------------
# measure_gap_flush_3d
# ---------------------------------------------------------------------------

def _make_flat_plane(z: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "centroid": np.asarray([0.0, 0.0, z], dtype=np.float32),
        "normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "tangents": np.eye(3, dtype=np.float32)[:2],
        "singular_values": np.asarray([1.0, 1.0, 0.01], dtype=np.float32),
    }


def test_gap_flush_pure_in_plane_distance_on_flat_surface():
    left = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([10.0, 0.0, 0.0], dtype=np.float32)
    plane = _make_flat_plane()
    result = measure_gap_flush_3d(left, right, plane, plane)
    assert result["valid"] is True
    assert result["gap"] == pytest.approx(10.0, abs=1e-5)
    assert result["flush"] == pytest.approx(0.0, abs=1e-5)
    assert result["full_3d_mm"] == pytest.approx(10.0, abs=1e-5)


def test_gap_flush_pure_flush_step_on_flat_surface():
    left = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([0.0, 0.0, 2.0], dtype=np.float32)
    plane = _make_flat_plane()
    result = measure_gap_flush_3d(left, right, plane, plane)
    assert result["gap"] == pytest.approx(0.0, abs=1e-5)
    assert result["flush"] == pytest.approx(2.0, abs=1e-5)
    assert result["full_3d_mm"] == pytest.approx(2.0, abs=1e-5)


def test_gap_flush_pythagorean_decomposition():
    left = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([6.0, 0.0, 8.0], dtype=np.float32)
    plane = _make_flat_plane()
    result = measure_gap_flush_3d(left, right, plane, plane)
    # Expect gap = 6 (in xy plane), flush = 8 (along +z), full 3D = 10.
    assert result["gap"] == pytest.approx(6.0, abs=1e-5)
    assert result["flush"] == pytest.approx(8.0, abs=1e-5)
    assert result["full_3d_mm"] == pytest.approx(10.0, abs=1e-5)
    # And the invariant gap² + flush² == full_3d² must hold.
    assert math.sqrt(result["gap"] ** 2 + result["flush"] ** 2) == pytest.approx(
        result["full_3d_mm"], abs=1e-5
    )


def test_gap_flush_handles_tilted_plane():
    # Plane tilted 45° about y: normal = (-1, 0, 1)/√2. Edges placed so
    # the delta is entirely along the plane's in-plane tangent (1, 0, 1)/√2.
    tilt_normal = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32) / math.sqrt(2.0)
    tilted_plane = {
        "centroid": np.zeros(3, dtype=np.float32),
        "normal": tilt_normal,
        "tangents": np.asarray(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ) / np.asarray([[math.sqrt(2.0)], [1.0]], dtype=np.float32),
        "singular_values": np.asarray([1.0, 1.0, 0.01], dtype=np.float32),
    }
    along_plane = np.asarray([1.0, 0.0, 1.0], dtype=np.float32) / math.sqrt(2.0)
    delta = 10.0 * along_plane
    left = np.zeros(3, dtype=np.float32)
    right = (left + delta).astype(np.float32)
    result = measure_gap_flush_3d(left, right, tilted_plane, tilted_plane)
    assert result["gap"] == pytest.approx(10.0, abs=1e-4)
    assert result["flush"] == pytest.approx(0.0, abs=1e-4)


def test_gap_flush_average_normal_when_planes_differ_slightly():
    # Two mildly-different normals: the average should point between them.
    plane_left = _make_flat_plane()
    plane_right = {
        "centroid": np.zeros(3, dtype=np.float32),
        "normal": np.asarray([0.1, 0.0, 0.99498744], dtype=np.float32),  # 5.7° tilt
        "tangents": np.eye(3, dtype=np.float32)[:2],
        "singular_values": np.asarray([1.0, 1.0, 0.01], dtype=np.float32),
    }
    left = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([10.0, 0.0, 0.5], dtype=np.float32)
    result = measure_gap_flush_3d(left, right, plane_left, plane_right)
    # Gap should remain close to 10 (dominant xy distance); flush modest.
    assert result["gap"] == pytest.approx(10.0, abs=0.5)
    assert result["flush"] < 1.5
    assert math.sqrt(result["gap"] ** 2 + result["flush"] ** 2) == pytest.approx(
        result["full_3d_mm"], abs=1e-4
    )


def test_gap_flush_rejects_orthogonal_plane_pair_by_default():
    # Orthogonal plane normals indicate one side's fit caught a seam-edge
    # artefact (real skin patches should agree within a few degrees).
    # The default ``plane_agreement_min=0.7`` guard rejects this.
    plane_up = {
        "centroid": np.zeros(3, dtype=np.float32),
        "normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "tangents": np.eye(3, dtype=np.float32)[:2],
        "singular_values": np.asarray([1.0, 1.0, 0.01], dtype=np.float32),
    }
    plane_side = dict(plane_up)
    plane_side["normal"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    left = np.zeros(3, dtype=np.float32)
    right = np.asarray([5.0, 0.0, 0.0], dtype=np.float32)
    result = measure_gap_flush_3d(left, right, plane_up, plane_side)
    assert result["valid"] is False
    assert "plane_disagreement" in result["reason"]


def test_gap_flush_strips_along_seam_component_when_tangent_given():
    # Flat skin, delta has 10 mm cross-seam + 0.5 mm along-seam + 2 mm flush.
    # With seam_tangent_3d supplied we should recover gap = 10 mm exactly
    # (instead of sqrt(10^2 + 0.5^2) = 10.01 mm the 2-way decomposition
    # would return).
    plane = _make_flat_plane()
    seam_tangent = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)  # +y = along seam
    # Cross-seam direction is then +x; normal is +z.
    left = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    right = np.asarray([10.0, 0.5, 2.0], dtype=np.float32)
    result = measure_gap_flush_3d(left, right, plane, plane, seam_tangent_3d=seam_tangent)
    assert result["valid"] is True
    assert result["gap"] == pytest.approx(10.0, abs=1e-5)
    assert result["gap_along_residual"] == pytest.approx(0.5, abs=1e-5)
    assert result["flush"] == pytest.approx(2.0, abs=1e-5)
    # Pythagoras still holds on the FULL 3D distance:
    # gap_cross^2 + gap_along^2 + flush^2 = full_3d^2
    total = (result["gap"] ** 2
             + result["gap_along_residual"] ** 2
             + result["flush"] ** 2) ** 0.5
    assert total == pytest.approx(result["full_3d_mm"], abs=1e-5)


def test_gap_flush_guards_can_be_relaxed_for_edge_cases():
    # Callers who know their data has wider angular tolerance can relax
    # the guard. A near-orthogonal pair now goes through.
    plane_up = {
        "centroid": np.zeros(3, dtype=np.float32),
        "normal": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        "tangents": np.eye(3, dtype=np.float32)[:2],
        "singular_values": np.asarray([1.0, 1.0, 0.01], dtype=np.float32),
    }
    plane_tilted = dict(plane_up)
    plane_tilted["normal"] = np.asarray([0.5, 0.0, 0.866025], dtype=np.float32)  # 30° off
    left = np.zeros(3, dtype=np.float32)
    right = np.asarray([5.0, 0.0, 0.0], dtype=np.float32)
    result = measure_gap_flush_3d(left, right, plane_up, plane_tilted)
    # 30° agreement = cos(30°) ≈ 0.866, above default 0.7 threshold.
    assert result["valid"] is True
    assert np.isfinite(result["gap"])
    assert np.isfinite(result["flush"])
