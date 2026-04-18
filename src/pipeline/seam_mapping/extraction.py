from __future__ import annotations

from typing import Any

import numpy as np


def extract_mask_point_cloud(mask: np.ndarray, point_map: np.ndarray) -> dict[str, Any]:
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}.")
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError(f"point_map must have shape (H, W, 3), got {point_map.shape}.")
    if mask.shape != point_map.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match point_map shape {point_map.shape[:2]}.")

    ys, xs = np.where(mask > 0)
    pixels_xy = np.column_stack([xs, ys]).astype(np.int32)
    points_xyz = point_map[ys, xs].astype(np.float32)
    valid_flags = np.all(np.isfinite(points_xyz), axis=1)

    valid_mask_2d = np.zeros(mask.shape, dtype=np.uint8)
    invalid_mask_2d = np.zeros(mask.shape, dtype=np.uint8)
    valid_mask_2d[ys[valid_flags], xs[valid_flags]] = 255
    invalid_mask_2d[ys[~valid_flags], xs[~valid_flags]] = 255

    return {
        "mask_pixels_xy": pixels_xy,
        "seam_points_xyz": points_xyz,
        "seam_points_valid_mask": valid_flags,
        "valid_mask_2d": valid_mask_2d,
        "invalid_mask_2d": invalid_mask_2d,
        "valid_pixels_xy": pixels_xy[valid_flags],
        "valid_points_xyz": points_xyz[valid_flags],
        "invalid_pixels_xy": pixels_xy[~valid_flags],
    }
