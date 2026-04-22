"""Unit tests for :mod:`pipeline.seam_measurement.helpers`.

Targets the pure-numpy helpers whose correctness underpins section
filtering and seam-direction extraction. Tests avoid torch / open3d so
they run in the minimal CI environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.seam_measurement import helpers as h

# ``select_primary_mask_component`` lazily imports cv2. Tests that touch it
# are gated by an in-test importorskip so the rest of the suite runs in an
# environment that only has numpy + pytest installed.


# ---------------------------------------------------------------------------
# count_neighbors
# ---------------------------------------------------------------------------

def test_count_neighbors_empty_and_singleton():
    assert h.count_neighbors(np.array([]), np.array([]), 1.0, 0.1).shape == (0,)
    assert h.count_neighbors(np.array([0.0]), np.array([0.0]), 1.0, 0.1).tolist() == [0]


def test_count_neighbors_rejects_zero_radius_or_tol():
    u = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    z = np.zeros_like(u)
    assert h.count_neighbors(u, z, 0.0, 0.1).tolist() == [0, 0, 0]
    assert h.count_neighbors(u, z, 1.0, 0.0).tolist() == [0, 0, 0]


def test_count_neighbors_counts_within_window():
    # 5 points on a horizontal line spaced 1 apart.
    u = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    z = np.zeros_like(u)
    counts = h.count_neighbors(u, z, radius_u=1.5, tol_z=0.1)
    # Interior points see both neighbours (|du| <= 1.5); endpoints see one.
    assert counts.tolist() == [1, 2, 2, 2, 1]


def test_count_neighbors_rejects_points_outside_z_tolerance():
    u = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    z = np.array([0.0, 10.0, 0.0], dtype=np.float32)
    counts = h.count_neighbors(u, z, radius_u=1.5, tol_z=0.5)
    # Every point is inside the u-window of at least one neighbour, but no
    # pair agrees in z (z differs by 10). So nobody is counted for any row.
    assert counts.tolist() == [0, 0, 0]
    # Loosening tol_z to 10 re-includes all pairs.
    counts_wide = h.count_neighbors(u, z, radius_u=1.5, tol_z=11.0)
    assert counts_wide.tolist() == [1, 2, 1]


def test_count_neighbors_handles_unsorted_input():
    u = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    z = np.zeros_like(u)
    counts = h.count_neighbors(u, z, radius_u=1.5, tol_z=0.1)
    # After sorting -> counts [1, 2, 1]; remap to original order (2,0,1) -> (1,1,2).
    assert counts.tolist() == [1, 1, 2]


def test_count_neighbors_handles_duplicate_u():
    u = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    z = np.array([0.0, 0.05, 0.2], dtype=np.float32)
    counts = h.count_neighbors(u, z, radius_u=0.01, tol_z=0.1)
    # Indices 0 and 1 are within 0.1 of each other in z; 2 is far.
    assert counts.tolist() == [1, 1, 0]


# ---------------------------------------------------------------------------
# split_segments_by_u
# ---------------------------------------------------------------------------

def _points(u, z=None):
    u_arr = np.asarray(u, dtype=np.float32)
    z_arr = np.asarray(z if z is not None else [0.0] * len(u_arr), dtype=np.float32)
    return {
        "u": u_arr,
        "z": z_arr,
        "pixels_xy": np.column_stack([u_arr, z_arr]).astype(np.float32),
        "xyz": np.column_stack([u_arr, z_arr, z_arr]).astype(np.float32),
    }


def test_split_segments_empty_input():
    assert h.split_segments_by_u(h.empty_plot_points(), 1.0) == []


def test_split_segments_single_segment():
    pts = _points([0.0, 0.5, 1.0, 1.5])
    segments = h.split_segments_by_u(pts, continuity_gap_u=1.0)
    assert len(segments) == 1
    assert segments[0]["u"].tolist() == [0.0, 0.5, 1.0, 1.5]


def test_split_segments_splits_on_gap():
    pts = _points([0.0, 0.5, 5.0, 5.5])
    segments = h.split_segments_by_u(pts, continuity_gap_u=1.0)
    assert len(segments) == 2
    assert segments[0]["u"].tolist() == [0.0, 0.5]
    assert segments[1]["u"].tolist() == [5.0, 5.5]


# ---------------------------------------------------------------------------
# principal_axes
# ---------------------------------------------------------------------------

def test_principal_axes_recovers_known_orientation():
    rng = np.random.default_rng(0)
    # Cloud elongated along the x axis with slight y spread.
    x = rng.uniform(-5.0, 5.0, size=500)
    y = rng.normal(0.0, 0.05, size=500)
    cloud = np.column_stack([x, y]).astype(np.float32)
    center, tangent, normal = h.principal_axes(cloud)

    # Mean of a uniform[-5, 5] sample has std ~= 10/sqrt(12*n); 1.0 is a
    # very generous tolerance.
    np.testing.assert_allclose(center, [0.0, 0.0], atol=1.0)
    # Tangent should be along +/- x.
    assert abs(float(tangent[0])) > 0.99
    assert abs(float(tangent[1])) < 0.1
    # Normal is orthogonal to tangent.
    np.testing.assert_allclose(np.dot(tangent, normal), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# select_primary_mask_component
# ---------------------------------------------------------------------------

def test_select_primary_mask_component_empty_mask_returns_empty():
    pytest.importorskip("cv2")
    mask = np.zeros((16, 16), dtype=np.uint8)
    assert h.select_primary_mask_component(mask).shape == (0, 2)


def test_select_primary_mask_component_prefers_larger_component():
    pytest.importorskip("cv2")
    mask = np.zeros((64, 64), dtype=np.uint8)
    # Small component in the corner.
    mask[2:4, 2:4] = 1
    # Much larger component near the centre.
    mask[28:44, 28:44] = 1
    component = h.select_primary_mask_component(mask)
    # The large central block has 16x16 = 256 pixels.
    assert component.shape == (256, 2)
    # All returned pixels should fall inside the large block's bounding box.
    xs, ys = component[:, 0], component[:, 1]
    assert float(xs.min()) >= 28 and float(xs.max()) <= 43
    assert float(ys.min()) >= 28 and float(ys.max()) <= 43


def test_select_primary_mask_component_ties_broken_by_proximity_to_centre():
    pytest.importorskip("cv2")
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[1:4, 1:4] = 1           # 9 pixels, far corner
    mask[30:33, 30:33] = 1       # 9 pixels, near centre
    component = h.select_primary_mask_component(mask)
    # Same area -> centre-distance term breaks the tie for the one near centre.
    assert component.shape == (9, 2)
    assert float(np.min(component[:, 0])) >= 30


# ---------------------------------------------------------------------------
# difference_plot_points
# ---------------------------------------------------------------------------

def test_difference_plot_points_empty_cases():
    empty = h.empty_plot_points()
    pts = _points([0.0, 1.0, 2.0])
    assert h.difference_plot_points(empty, pts)["u"].size == 0
    # Removing an empty set returns the input (sorted).
    kept = h.difference_plot_points(pts, empty)
    assert kept["u"].tolist() == [0.0, 1.0, 2.0]


def test_difference_plot_points_removes_matching_pixels():
    pts = _points([0.0, 1.0, 2.0, 3.0])
    # Use the same pixel coords as in _points (u, z) -> (u, 0).
    to_remove = _points([1.0, 3.0])
    kept = h.difference_plot_points(pts, to_remove)
    assert kept["u"].tolist() == [0.0, 2.0]


def test_difference_plot_points_ignores_tiny_float_noise_in_pixels():
    pts = _points([0.0, 1.0, 2.0])
    # Perturb the pixel coordinate by a sub-pixel amount - rounding to int
    # should still treat them as the same point.
    noisy = _points([0.0, 1.0, 2.0])
    noisy["pixels_xy"] = noisy["pixels_xy"].copy()
    noisy["pixels_xy"][1, 0] = 1.0 + 1e-4
    kept = h.difference_plot_points(pts, noisy)
    assert kept["u"].size == 0
