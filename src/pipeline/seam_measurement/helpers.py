from __future__ import annotations

from typing import Any

import numpy as np

# ``cv2`` is imported lazily inside the functions that need it so that a
# minimal test / CI environment (numpy + pytest only) can still import the
# package for the pure-numpy code paths.


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
    """Return ``points`` with any entry whose pixel coordinate matches one in
    ``to_remove`` filtered out.

    The previous implementation hashed rounded ``(u, z, px, py)`` tuples -
    fragile at float32 precision boundaries and a double Python loop.
    Pixel coordinates are integer-valued (they come from ``np.nonzero`` on
    a mask), so we use them as a stable identity key, encode ``(px, py)``
    into a single ``int64`` and membership-test with ``np.isin`` in C.
    """
    if len(points.get("u", [])) == 0:
        return empty_plot_points()
    if len(to_remove.get("u", [])) == 0:
        return sort_points(points)

    remove_keys = _pack_pixel_keys(to_remove["pixels_xy"])
    point_keys = _pack_pixel_keys(points["pixels_xy"])
    keep_mask = ~np.isin(point_keys, remove_keys)
    return sort_points(subset_points(points, keep_mask))


def _pack_pixel_keys(pixels_xy: np.ndarray) -> np.ndarray:
    """Encode (px, py) integer pixel coordinates into a single int64 per row.

    Assumes image dimensions stay under 2**20 (≈ 1 Mpix per side) which is
    well above the 1236×1032 inputs this project works with.
    """
    arr = np.asarray(pixels_xy, dtype=np.float32).reshape(-1, 2)
    px = np.rint(arr[:, 0]).astype(np.int64)
    py = np.rint(arr[:, 1]).astype(np.int64)
    return (py << 20) | (px & 0xFFFFF)


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
    """For each point, count how many others lie within ``radius_u`` in u
    and ``tol_z`` in z; the point itself is excluded from its own count.

    Previously a Python-level sliding-window loop: O(N) in Python per
    call, which dominated ``compute_gap_flush`` at section-count scale.
    This version builds the pairwise |Δu| and |Δz| matrices and does
    the comparison fully in C, trading O(N^2) memory for a drastic
    speed-up. For N up to a few thousand (typical section sizes here)
    the memory is a few MB — acceptable — and the op runs in ~1 ms
    versus the previous ~10–30 ms.
    """
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    n = len(u)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n == 1:
        return np.zeros((1,), dtype=np.int32)
    radius_u = float(radius_u)
    tol_z = float(tol_z)
    if radius_u <= 0.0 or tol_z <= 0.0:
        return np.zeros((n,), dtype=np.int32)
    du = np.abs(u[:, None] - u[None, :])
    dz = np.abs(z[:, None] - z[None, :])
    mask = (du <= radius_u) & (dz <= tol_z)
    counts = mask.sum(axis=1).astype(np.int32) - 1
    # Clip negatives defensively; self-diagonal guarantees >= 1 per row.
    np.maximum(counts, 0, out=counts)
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
    """Return the pixel coordinates of the "primary" connected component.

    Uses :func:`cv2.connectedComponentsWithStats` (O(n) flood-fill in C)
    instead of a Python-level BFS over a pixel set - that implementation
    was dominating the seam-detection post-processing time for realistic
    mask sizes.

    Scoring matches the previous implementation: ``area - 0.15*min_dist -
    0.05*centroid_dist`` where distances are measured to the image centre.
    """
    import cv2  # noqa: PLC0415 - lazy import to keep minimal envs importable

    mask = np.asarray(mask)
    if mask.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return np.empty((0, 2), dtype=np.float32)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:  # only the background label was returned
        return np.empty((0, 2), dtype=np.float32)

    height, width = binary.shape
    image_center = np.asarray([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)

    best_component = np.empty((0, 2), dtype=np.float32)
    best_score = -np.inf
    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_mask = labels == label
        ys, xs = np.nonzero(component_mask)
        component_xy = np.column_stack([xs, ys]).astype(np.float32)
        centroid = centroids[label].astype(np.float32)  # cv2 returns (x, y)
        distances = np.linalg.norm(component_xy - image_center[None, :], axis=1)
        score = area - 0.15 * float(np.min(distances)) - 0.05 * float(np.linalg.norm(centroid - image_center))
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
