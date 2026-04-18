from __future__ import annotations

from pathlib import Path

import numpy as np


def load_point_map(point_map_path: str | Path) -> np.ndarray:
    point_map_path = Path(point_map_path)
    extension = point_map_path.suffix.lower()
    if extension == ".npy":
        point_map = np.load(point_map_path)
    elif extension == ".npz":
        data = np.load(point_map_path)
        if not data.files:
            raise ValueError(f"No arrays found in point map file: {point_map_path}")
        point_map = data[data.files[0]]
    elif extension == ".pcd":
        point_map = load_point_map_from_pcd(point_map_path)
    else:
        raise ValueError("Point map must be a .npy, .npz, or ASCII .pcd file.")

    point_map = np.asarray(point_map, dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError(f"Point map must have shape [H, W, 3], got {point_map.shape}.")
    return point_map


def _build_point_map_cache_path(point_map_path: Path) -> Path:
    return point_map_path.with_suffix(".pointmap.npy")


def load_point_map_from_pcd(point_map_path: str | Path) -> np.ndarray:
    point_map_path = Path(point_map_path)
    cache_path = _build_point_map_cache_path(point_map_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= point_map_path.stat().st_mtime:
        point_map = np.load(cache_path, allow_pickle=False)
        return np.asarray(point_map, dtype=np.float32)

    width = None
    height = None
    points = None
    field_names: list[str] = []
    data_type = None

    with point_map_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith("FIELDS "):
                field_names = stripped.split()[1:]
            elif upper.startswith("WIDTH "):
                width = int(stripped.split()[1])
            elif upper.startswith("HEIGHT "):
                height = int(stripped.split()[1])
            elif upper.startswith("POINTS "):
                points = int(stripped.split()[1])
            elif upper.startswith("DATA "):
                data_type = stripped.split()[1].lower()
                break
        body = file.read()

    if data_type != "ascii":
        raise ValueError(f"Only ASCII PCD files are supported, got DATA {data_type!r} in {point_map_path}.")
    if width is None or height is None or points is None:
        raise ValueError(f"PCD header missing WIDTH/HEIGHT/POINTS in {point_map_path}.")
    if width * height != points:
        raise ValueError(f"PCD shape mismatch in {point_map_path}: width*height={width * height}, points={points}.")
    if not field_names:
        raise ValueError(f"PCD header missing FIELDS in {point_map_path}.")

    num_fields = len(field_names)
    xyz_indices = []
    for axis in ("x", "y", "z"):
        if axis not in field_names:
            raise ValueError(f"PCD file {point_map_path} is missing '{axis}' in FIELDS.")
        xyz_indices.append(field_names.index(axis))

    # ``np.fromstring`` is deprecated; ``np.fromiter`` over ``body.split()``
    # gives the same parse of whitespace-separated ASCII floats.
    flat_values = np.fromiter(
        (float(token) for token in body.split()),
        dtype=np.float32,
    )
    expected_values = points * num_fields
    if flat_values.size != expected_values:
        raise ValueError(
            f"PCD data size mismatch in {point_map_path}: expected {expected_values} values, got {flat_values.size}."
        )

    point_rows = flat_values.reshape(points, num_fields)
    xyz_points = point_rows[:, xyz_indices]
    point_map = xyz_points.reshape(height, width, 3).astype(np.float32, copy=False)
    np.save(cache_path, point_map, allow_pickle=False)
    return point_map
