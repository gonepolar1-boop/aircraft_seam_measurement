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


_VECTORISED_COUNT_NEIGHBORS_THRESHOLD = 1500


def count_neighbors(u: np.ndarray, z: np.ndarray, radius_u: float, tol_z: float) -> np.ndarray:
    """For each point, count how many others lie within ``radius_u`` in u
    and ``tol_z`` in z; the point itself is excluded from its own count.

    Hybrid implementation:

    * For modestly-sized inputs (``N <= ~1500``) the pairwise |Δu| / |Δz|
      matrices fit comfortably in memory and the fully-vectorised
      ``(du <= r) & (dz <= t)`` runs in ~1 ms. This beats the Python
      for-loop on the small section-filter inputs where the old loop
      dominated ``compute_gap_flush``.
    * For larger inputs (e.g. the ``local_background`` point set in
      top-surface detection, which can reach ~10⁴ points) the O(N²)
      pairwise matrices blow out memory bandwidth and the naive
      broadcast becomes *slower* than the sorted sliding-window loop.
      We fall back to the original sorted-pointer scan in that regime,
      which is O(N·K) with K the average neighbourhood size.
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

    if n <= _VECTORISED_COUNT_NEIGHBORS_THRESHOLD:
        du = np.abs(u[:, None] - u[None, :])
        dz = np.abs(z[:, None] - z[None, :])
        mask = (du <= radius_u) & (dz <= tol_z)
        counts = mask.sum(axis=1).astype(np.int32) - 1
        np.maximum(counts, 0, out=counts)
        return counts

    if np.all(u[1:] >= u[:-1]):
        order = None
        u_sorted = u
        z_sorted = z
    else:
        order = np.argsort(u)
        u_sorted = u[order]
        z_sorted = z[order]

    counts_sorted = np.zeros((n,), dtype=np.int32)
    left = 0
    right = 0
    for index in range(n):
        center_u = u_sorted[index]
        while center_u - u_sorted[left] > radius_u:
            left += 1
        while right + 1 < n and u_sorted[right + 1] - center_u <= radius_u:
            right += 1
        counts_sorted[index] = int(
            np.count_nonzero(np.abs(z_sorted[left : right + 1] - z_sorted[index]) <= tol_z) - 1
        )

    if order is None:
        return counts_sorted
    counts = np.zeros((n,), dtype=np.int32)
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
    """Return the pixel coordinates of the "primary" connected component.

    Uses :func:`cv2.connectedComponentsWithStats` and scores each component
    using only the stats table + bbox geometry, avoiding the per-component
    full-image ``np.nonzero`` scan that the naive implementation used to do.
    Only the winning component's pixel coordinates are actually materialised.

    Score formula (unchanged):
        ``area - 0.15 * min_dist_to_centre - 0.05 * centroid_dist_to_centre``

    ``min_dist`` is approximated by "closest point on the component bbox to
    the image centre" — exact for bboxes that contain the centre, a tight
    lower bound otherwise.  For the seam-mask use case (one large primary
    component dwarfing the rest) the approximation never flips the winner.
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
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    # Vectorised scoring over all non-background labels. For label i:
    #   bbox clamp cx/cy into [x1, x1+w) x [y1, y1+h) -> nearest-on-bbox
    #   min_dist = distance from (cx, cy) to that clamped point.
    labels_range = np.arange(1, num_labels)
    areas = stats[labels_range, cv2.CC_STAT_AREA].astype(np.float64)
    left = stats[labels_range, cv2.CC_STAT_LEFT].astype(np.float64)
    top = stats[labels_range, cv2.CC_STAT_TOP].astype(np.float64)
    w_box = stats[labels_range, cv2.CC_STAT_WIDTH].astype(np.float64)
    h_box = stats[labels_range, cv2.CC_STAT_HEIGHT].astype(np.float64)
    right = left + w_box - 1.0
    bottom = top + h_box - 1.0
    clamp_x = np.clip(cx, left, right)
    clamp_y = np.clip(cy, top, bottom)
    min_dist = np.sqrt((clamp_x - cx) ** 2 + (clamp_y - cy) ** 2)
    cent_xy = centroids[labels_range].astype(np.float64)
    cent_dist = np.sqrt((cent_xy[:, 0] - cx) ** 2 + (cent_xy[:, 1] - cy) ** 2)
    scores = areas - 0.15 * min_dist - 0.05 * cent_dist
    # Guard against all-zero-area components (shouldn't happen but keep safe).
    valid = areas > 0
    if not valid.any():
        return np.empty((0, 2), dtype=np.float32)
    scores = np.where(valid, scores, -np.inf)
    best_label = int(labels_range[int(np.argmax(scores))])

    ys, xs = np.nonzero(labels == best_label)
    return np.column_stack([xs, ys]).astype(np.float32)


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
