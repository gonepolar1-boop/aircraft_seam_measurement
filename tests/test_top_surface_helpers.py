"""Tests for vectorised helpers in :mod:`pipeline.seam_measurement.top_surface`.

Currently scoped to ``_rolling_median`` which was previously a slow
Python for-loop and is now a ``sliding_window_view`` + ``nanmedian``
vectorisation. These tests lock in its behaviour so the rewrite is
guaranteed equivalent.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.seam_measurement.top_surface import _rolling_median


def _reference_rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """The original naive implementation, kept here as an oracle."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return values
    radius = max(1, int(window) // 2)
    output = np.full_like(values, np.nan)
    for index in range(len(values)):
        chunk = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        finite_chunk = chunk[np.isfinite(chunk)]
        if len(finite_chunk):
            output[index] = float(np.median(finite_chunk))
    return output


def test_rolling_median_empty_input():
    result = _rolling_median(np.array([], dtype=np.float32), window=3)
    assert result.size == 0


def test_rolling_median_no_nans_matches_reference():
    rng = np.random.default_rng(0)
    values = rng.normal(size=200).astype(np.float32)
    for window in (3, 5, 7, 11):
        fast = _rolling_median(values, window)
        slow = _reference_rolling_median(values, window)
        np.testing.assert_allclose(fast, slow, atol=1e-5)


def test_rolling_median_handles_scattered_nans():
    rng = np.random.default_rng(1)
    values = rng.normal(size=120).astype(np.float32)
    # Poke holes through the array.
    values[10:15] = np.nan
    values[40] = np.nan
    values[-5:] = np.nan
    for window in (3, 5, 9):
        fast = _rolling_median(values, window)
        slow = _reference_rolling_median(values, window)
        # Compare positions where both are finite OR both are NaN.
        np.testing.assert_array_equal(np.isnan(fast), np.isnan(slow))
        finite = ~np.isnan(fast)
        np.testing.assert_allclose(fast[finite], slow[finite], atol=1e-5)


def test_rolling_median_all_nan_row_returns_nan():
    values = np.full((5,), np.nan, dtype=np.float32)
    result = _rolling_median(values, window=3)
    assert np.all(np.isnan(result))


@pytest.mark.parametrize("window", [1, 2])
def test_rolling_median_degenerate_windows_still_run(window):
    values = np.arange(10.0, dtype=np.float32)
    result = _rolling_median(values, window=window)
    # Window=1/2 both resolve to radius=1 via max(1, window//2).
    slow = _reference_rolling_median(values, window=window)
    np.testing.assert_allclose(result, slow, atol=1e-5)
