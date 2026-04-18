from __future__ import annotations

from typing import Any

import numpy as np


def empty_plot_points() -> dict[str, np.ndarray]:
    empty = np.empty((0,), dtype=np.float32)
    return {
        "u": empty,
        "z": empty,
        "pixels_xy": np.empty((0, 2), dtype=np.float32),
        "xyz": np.empty((0, 3), dtype=np.float32),
    }


def make_plot_points(
    u: np.ndarray,
    z: np.ndarray,
    pixels_xy: np.ndarray,
    xyz: np.ndarray,
    *,
    sort: bool = False,
) -> dict[str, np.ndarray]:
    points = {
        "u": np.asarray(u, dtype=np.float32).reshape(-1),
        "z": np.asarray(z, dtype=np.float32).reshape(-1),
        "pixels_xy": np.asarray(pixels_xy, dtype=np.float32).reshape(-1, 2),
        "xyz": np.asarray(xyz, dtype=np.float32).reshape(-1, 3),
    }
    return sort_points(points) if sort else points


def sort_points(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(points["u"]) == 0:
        return points
    if len(points["u"]) == 1 or np.all(points["u"][1:] >= points["u"][:-1]):
        return points
    order = np.argsort(points["u"])
    return {key: value[order] for key, value in points.items()}


def subset_points(points: dict[str, np.ndarray], keep_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[keep_mask] for key, value in points.items()}


def empty_points_like(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value[:0] for key, value in points.items()}


def merge_plot_points(*point_sets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    empty = empty_plot_points()
    merged: dict[str, np.ndarray] = {}
    for key in ("u", "z", "pixels_xy", "xyz"):
        chunks = [points[key] for points in point_sets if points is not None and len(points.get(key, []))]
        merged[key] = np.concatenate(chunks, axis=0).astype(np.float32) if chunks else empty[key]
    return merged


def difference_plot_points(points: dict[str, np.ndarray], to_remove: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if len(points.get("u", [])) == 0:
        return empty_plot_points()
    if len(to_remove.get("u", [])) == 0:
        return sort_points(points)

    remove_keys = {
        (
            round(float(u), 6),
            round(float(z), 6),
            round(float(px), 3),
            round(float(py), 3),
        )
        for u, z, (px, py) in zip(to_remove["u"], to_remove["z"], to_remove["pixels_xy"])
    }
    keep_mask = np.asarray(
        [
            (
                round(float(u), 6),
                round(float(z), 6),
                round(float(px), 3),
                round(float(py), 3),
            ) not in remove_keys
            for u, z, (px, py) in zip(points["u"], points["z"], points["pixels_xy"])
        ],
        dtype=bool,
    )
    return sort_points(subset_points(points, keep_mask))


def collect_pixels_from_sections(result: dict[str, Any], key: str) -> np.ndarray:
    chunks = [
        points["pixels_xy"]
        for item in result.get("section_results", [])
        if (points := item.get(key)) is not None and len(points.get("pixels_xy", []))
    ]
    if not chunks:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def collect_xyz_from_sections(result: dict[str, Any], key: str) -> np.ndarray:
    chunks = [
        points["xyz"]
        for item in result.get("section_results", [])
        if (points := item.get(key)) is not None and len(points.get("xyz", []))
    ]
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0).astype(np.float32)


def collect_valid_pointcloud_xyz(result: dict[str, Any]) -> np.ndarray:
    point_map = np.asarray(result.get("point_map", []), dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        return np.empty((0, 3), dtype=np.float32)
    finite_mask = np.all(np.isfinite(point_map), axis=2)
    if not np.any(finite_mask):
        return np.empty((0, 3), dtype=np.float32)
    return point_map[finite_mask].reshape(-1, 3).astype(np.float32)


def subsample_xyz(xyz: np.ndarray, max_points: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if len(xyz) <= max_points or max_points <= 0:
        return xyz
    indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int32)
    return xyz[indices]


def set_equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def count_neighbors(u: np.ndarray, z: np.ndarray, radius_u: float, tol_z: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if len(u) == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(u) == 1:
        return np.zeros((1,), dtype=np.int32)
    radius_u = float(radius_u)
    tol_z = float(tol_z)
    if radius_u <= 0.0 or tol_z <= 0.0:
        return np.zeros((len(u),), dtype=np.int32)
    if np.all(u[1:] >= u[:-1]):
        order = None
        u_sorted = u
        z_sorted = z
    else:
        order = np.argsort(u)
        u_sorted = u[order]
        z_sorted = z[order]

    counts_sorted = np.zeros((len(u_sorted),), dtype=np.int32)
    left = 0
    right = 0
    for index, center_u in enumerate(u_sorted):
        while center_u - u_sorted[left] > radius_u:
            left += 1
        while right + 1 < len(u_sorted) and u_sorted[right + 1] - center_u <= radius_u:
            right += 1
        counts_sorted[index] = int(np.count_nonzero(np.abs(z_sorted[left : right + 1] - z_sorted[index]) <= tol_z) - 1)

    if order is None:
        return counts_sorted
    counts = np.zeros((len(u_sorted),), dtype=np.int32)
    counts[order] = counts_sorted
    return counts


def split_segments_by_u(points: dict[str, np.ndarray], continuity_gap_u: float) -> list[dict[str, np.ndarray]]:
    if len(points["u"]) == 0:
        return []
    points = sort_points(points)
    du = np.abs(np.diff(points["u"]))
    breaks = np.zeros(len(points["u"]), dtype=bool)
    breaks[0] = True
    breaks[1:] = du > continuity_gap_u
    segment_ids = np.cumsum(breaks) - 1
    return [subset_points(points, segment_ids == segment_id) for segment_id in np.unique(segment_ids)]


def select_primary_mask_component(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    coords = np.column_stack([xs, ys]).astype(np.int32)
    coord_set = {tuple(point) for point in coords.tolist()}
    visited: set[tuple[int, int]] = set()
    image_center = np.asarray([(mask.shape[1] - 1) * 0.5, (mask.shape[0] - 1) * 0.5], dtype=np.float32)

    best_component = np.empty((0, 2), dtype=np.float32)
    best_score = -np.inf
    for point in coord_set:
        if point in visited:
            continue
        stack = [point]
        visited.add(point)
        component: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (x + dx, y + dy)
                    if neighbor in coord_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

        component_xy = np.asarray(component, dtype=np.float32)
        centroid = np.mean(component_xy, axis=0)
        distances = np.linalg.norm(component_xy - image_center[None, :], axis=1)
        score = float(len(component_xy)) - 0.15 * float(np.min(distances)) - 0.05 * float(np.linalg.norm(centroid - image_center))
        if score > best_score:
            best_score = score
            best_component = component_xy
    return best_component


def principal_axes(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(points_xy, axis=0).astype(np.float32)
    centered = points_xy - center[None, :]
    covariance = centered.T @ centered / max(1, len(points_xy) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float32))
    order = np.argsort(eigenvalues)[::-1]
    tangent = eigenvectors[:, order[0]].astype(np.float32)
    tangent /= max(1e-6, float(np.linalg.norm(tangent)))
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    return center, tangent, normal


def snap_center_to_valid_pixel(center_xy: np.ndarray, pixels_xy: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if len(pixels_xy) == 0 or not np.any(valid_mask):
        return np.asarray(center_xy, dtype=np.float32)
    valid_pixels = pixels_xy[valid_mask].astype(np.float32)
    distances = np.sum((valid_pixels - np.asarray(center_xy, dtype=np.float32)[None, :]) ** 2, axis=1)
    return valid_pixels[int(np.argmin(distances))].astype(np.float32)


def validate_inputs(mask: np.ndarray, point_map: np.ndarray) -> None:
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}.")
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError(f"point_map must have shape (H, W, 3), got {point_map.shape}.")
    if mask.shape != point_map.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match point_map shape {point_map.shape[:2]}.")
