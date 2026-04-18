"""Tests for the ASCII-PCD reader in :mod:`pipeline.seam_mapping.io`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline.seam_mapping import io as pcd_io


def _write_pcd(path: Path, points: np.ndarray) -> None:
    """Write ``points`` (shape [H, W, 3]) as an ASCII PCD file."""
    h, w, _ = points.shape
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {w}\n"
        f"HEIGHT {h}\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {w * h}\n"
        "DATA ascii\n"
    )
    body_rows = [
        " ".join(f"{v:.6f}" for v in row)
        for row in points.reshape(-1, 3)
    ]
    path.write_text(header + "\n".join(body_rows) + "\n", encoding="utf-8")


def test_load_point_map_from_ascii_pcd_roundtrip(tmp_path: Path):
    expected = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    pcd_path = tmp_path / "cloud.pcd"
    _write_pcd(pcd_path, expected)

    loaded = pcd_io.load_point_map(pcd_path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, expected, atol=1e-5)


def test_load_point_map_uses_cache_on_second_read(tmp_path: Path):
    expected = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    pcd_path = tmp_path / "cloud.pcd"
    _write_pcd(pcd_path, expected)

    first = pcd_io.load_point_map(pcd_path)
    cache_path = pcd_path.with_suffix(".pointmap.npy")
    assert cache_path.exists(), "PCD loader should cache to .pointmap.npy"

    second = pcd_io.load_point_map(pcd_path)
    np.testing.assert_array_equal(first, second)


def test_load_point_map_rejects_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "cloud.bin"
    bad.write_bytes(b"not a point map")
    with pytest.raises(ValueError):
        pcd_io.load_point_map(bad)


def test_load_point_map_rejects_binary_pcd(tmp_path: Path):
    pcd_path = tmp_path / "binary.pcd"
    pcd_path.write_text(
        "FIELDS x y z\n"
        "WIDTH 1\n"
        "HEIGHT 1\n"
        "POINTS 1\n"
        "DATA binary\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ASCII PCD"):
        pcd_io.load_point_map(pcd_path)
