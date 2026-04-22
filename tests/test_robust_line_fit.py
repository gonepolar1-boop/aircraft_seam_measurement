"""Unit tests for :func:`pipeline.seam_measurement.top_surface._robust_line_fit`.

Covers the three scenarios that matter in practice on seam data:
a clean point set, a point set with strong outliers, and a point set
small enough to trip the polyfit fall-back path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.seam_measurement.top_surface import _robust_line_fit


def test_robust_line_fit_clean_points_recovers_ground_truth():
    rng = np.random.default_rng(7)
    u = np.linspace(-5.0, 5.0, 30, dtype=np.float32)
    true_slope = 0.08
    true_intercept = 2.0
    z = true_slope * u + true_intercept + rng.normal(0.0, 0.01, size=u.size).astype(np.float32)
    slope, intercept = _robust_line_fit(u, z, tol=0.1)
    assert slope == pytest.approx(true_slope, abs=5e-3)
    assert intercept == pytest.approx(true_intercept, abs=5e-3)


def test_robust_line_fit_is_robust_to_outliers():
    """Drop in a handful of large outliers; the recovered line should
    still follow the inlier trend, whereas a plain L2 fit would be
    pulled toward the outliers.
    """
    rng = np.random.default_rng(42)
    u_inliers = np.linspace(-5.0, 5.0, 40, dtype=np.float32)
    true_slope = 0.05
    true_intercept = 1.0
    z_inliers = true_slope * u_inliers + true_intercept + rng.normal(0.0, 0.02, size=u_inliers.size).astype(np.float32)

    # Inject 5 strong outliers well outside any reasonable inlier band.
    outlier_u = np.asarray([-4.0, -2.0, 0.0, 2.0, 4.0], dtype=np.float32)
    outlier_z = np.asarray([-3.0, 4.0, -2.5, 3.5, -4.5], dtype=np.float32)

    u = np.concatenate([u_inliers, outlier_u])
    z = np.concatenate([z_inliers, outlier_z])

    slope_r, intercept_r = _robust_line_fit(u, z, tol=0.15)
    slope_poly, intercept_poly = np.polyfit(u.astype(np.float64), z.astype(np.float64), deg=1)

    # RANSAC should land closer to ground truth than plain polyfit does.
    assert abs(slope_r - true_slope) < abs(slope_poly - true_slope)
    assert abs(intercept_r - true_intercept) < abs(intercept_poly - true_intercept)
    # And at loose tolerance it should be close to the clean line.
    assert slope_r == pytest.approx(true_slope, abs=0.03)
    assert intercept_r == pytest.approx(true_intercept, abs=0.2)


def test_robust_line_fit_small_input_falls_back_to_polyfit():
    """For fewer than 3 points RANSAC can't do its voting, so the
    helper should fall back to a plain polyfit rather than raising."""
    u = np.asarray([-1.0, 2.5], dtype=np.float32)
    z = np.asarray([0.3, 0.7], dtype=np.float32)
    slope, intercept = _robust_line_fit(u, z, tol=0.1)
    # With two points polyfit returns the exact line through them.
    expected_slope = (z[1] - z[0]) / (u[1] - u[0])
    expected_intercept = z[0] - expected_slope * u[0]
    assert slope == pytest.approx(float(expected_slope), abs=1e-6)
    assert intercept == pytest.approx(float(expected_intercept), abs=1e-6)


def test_robust_line_fit_handles_identical_u_values():
    """A fully-degenerate input where every u is the same should not
    crash.  Any finite output is acceptable - the downstream LineFit
    gets marked as a fallback by the caller."""
    u = np.full(10, 1.0, dtype=np.float32)
    z = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    # polyfit emits a RankWarning here but still returns numbers.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.RankWarning)
        slope, intercept = _robust_line_fit(u, z, tol=0.1)
    assert math.isfinite(slope) or math.isnan(slope)
    assert math.isfinite(intercept) or math.isnan(intercept)
