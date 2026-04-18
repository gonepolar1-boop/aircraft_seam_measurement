from __future__ import annotations

from typing import Any

import numpy as np


def build_section_reference_frame(
    left_fit: dict[str, float],
    right_fit: dict[str, float],
    left_point_xy: np.ndarray,
    right_point_xy: np.ndarray,
) -> dict[str, Any]:
    _, left_normal = reference_basis_from_fit(left_fit)
    _, right_normal = reference_basis_from_fit(right_fit)
    if left_normal is None or right_normal is None:
        return _empty_reference_frame()

    left_normal = np.asarray(left_normal, dtype=np.float32)
    right_normal = np.asarray(right_normal, dtype=np.float32)
    if float(np.dot(left_normal, right_normal)) < 0.0:
        right_normal = -right_normal

    normal = left_normal + right_normal
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-6:
        return _empty_reference_frame()
    normal = normal / normal_norm
    tangent = np.asarray([normal[1], -normal[0]], dtype=np.float32)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        return _empty_reference_frame()
    tangent = tangent / tangent_norm

    origin_xy = 0.5 * (np.asarray(left_point_xy, dtype=np.float32) + np.asarray(right_point_xy, dtype=np.float32))
    return {
        "valid": True,
        "origin_xy": origin_xy.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def reference_basis_from_fit(fit: dict[str, float]) -> tuple[np.ndarray | None, np.ndarray | None]:
    slope = float(fit.get("slope", 0.0))
    intercept = float(fit.get("intercept", np.nan))
    if not np.isfinite(intercept):
        return None, None
    tangent = np.asarray([1.0, slope], dtype=np.float32)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        return None, None
    tangent = tangent / tangent_norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    return tangent, normal


def transform_xy_to_reference(points_xy: np.ndarray, origin_xy: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    tangent, normal = reference_basis_from_fit(fit)
    if tangent is None or normal is None:
        return np.full_like(points_xy, np.nan, dtype=np.float32)
    deltas = np.asarray(points_xy, dtype=np.float32) - np.asarray(origin_xy, dtype=np.float32)
    tangent_coords = deltas @ tangent
    normal_coords = deltas @ normal
    return np.stack([tangent_coords, normal_coords], axis=1).astype(np.float32)


def transform_plot_points_to_reference(points: dict[str, np.ndarray], origin_xy: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    if points is None or len(points.get("u", [])) == 0:
        return np.empty((0, 2), dtype=np.float32)
    xy = np.stack(
        [
            np.asarray(points["u"], dtype=np.float32).reshape(-1),
            np.asarray(points["z"], dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    finite_mask = np.all(np.isfinite(xy), axis=1)
    if not np.any(finite_mask):
        return np.empty((0, 2), dtype=np.float32)
    return transform_xy_to_reference(xy[finite_mask], origin_xy, fit)


def transform_xy_to_frame(points_xy: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    if not bool(frame.get("valid", False)):
        return np.full_like(points_xy, np.nan, dtype=np.float32)
    deltas = np.asarray(points_xy, dtype=np.float32) - np.asarray(frame["origin_xy"], dtype=np.float32)
    tangent_coords = deltas @ np.asarray(frame["tangent"], dtype=np.float32)
    normal_coords = deltas @ np.asarray(frame["normal"], dtype=np.float32)
    return np.stack([tangent_coords, normal_coords], axis=1).astype(np.float32)


def transform_plot_points_to_frame(points: dict[str, np.ndarray], frame: dict[str, Any]) -> np.ndarray:
    if points is None or len(points.get("u", [])) == 0 or not bool(frame.get("valid", False)):
        return np.empty((0, 2), dtype=np.float32)
    xy = np.stack(
        [
            np.asarray(points["u"], dtype=np.float32).reshape(-1),
            np.asarray(points["z"], dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    finite_mask = np.all(np.isfinite(xy), axis=1)
    if not np.any(finite_mask):
        return np.empty((0, 2), dtype=np.float32)
    return transform_xy_to_frame(xy[finite_mask], frame)


def measure_point_line_gap_flush(
    source_point: np.ndarray,
    other_edge_point: np.ndarray,
    fit: dict[str, float],
) -> dict[str, Any]:
    tangent, _ = reference_basis_from_fit(fit)
    if tangent is None:
        return {"gap": np.nan, "flush": np.nan, "foot_point": np.asarray([np.nan, np.nan], dtype=np.float32)}

    intercept = float(fit.get("intercept", np.nan))
    line_anchor = np.asarray([0.0, intercept], dtype=np.float32)
    source_point = np.asarray(source_point, dtype=np.float32)
    other_edge_point = np.asarray(other_edge_point, dtype=np.float32)
    source_delta = source_point - line_anchor
    foot_point = line_anchor + float(np.dot(source_delta, tangent)) * tangent
    flush = float(np.linalg.norm(source_point - foot_point))
    gap = float(np.linalg.norm(other_edge_point - foot_point))
    return {"gap": gap, "flush": flush, "foot_point": foot_point}


def measure_gap_in_reference_frame(
    left_point_xy: np.ndarray,
    right_point_xy: np.ndarray,
    frame: dict[str, Any],
) -> dict[str, Any]:
    if not bool(frame.get("valid", False)):
        return {
            "gap": np.nan,
            "left_point_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
            "right_point_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
            "left_foot_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
            "right_foot_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
        }
    left_point_ref = transform_xy_to_frame(np.asarray([left_point_xy], dtype=np.float32), frame)[0]
    right_point_ref = transform_xy_to_frame(np.asarray([right_point_xy], dtype=np.float32), frame)[0]
    left_foot_ref = np.asarray([left_point_ref[0], 0.0], dtype=np.float32)
    right_foot_ref = np.asarray([right_point_ref[0], 0.0], dtype=np.float32)
    return {
        "gap": float(abs(right_foot_ref[0] - left_foot_ref[0])),
        "left_point_ref": left_point_ref,
        "right_point_ref": right_point_ref,
        "left_foot_ref": left_foot_ref,
        "right_foot_ref": right_foot_ref,
    }


def _empty_reference_frame() -> dict[str, Any]:
    return {
        "valid": False,
        "origin_xy": np.asarray([np.nan, np.nan], dtype=np.float32),
        "tangent": np.asarray([np.nan, np.nan], dtype=np.float32),
        "normal": np.asarray([np.nan, np.nan], dtype=np.float32),
    }
