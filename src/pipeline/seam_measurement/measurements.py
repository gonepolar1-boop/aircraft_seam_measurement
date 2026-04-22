from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import (
    build_section_reference_frame,
    fit_plane_3d,
    measure_gap_flush_3d,
    measure_gap_in_reference_frame,
    measure_point_line_gap_flush,
)
from .helpers import empty_plot_points
from .types import SectionMeasurement


def compute_section_gap_flush(section_result: dict[str, Any]) -> SectionMeasurement:
    top_surface = section_result.get("top_surface", {})
    left_edge = top_surface.get("left_edge", empty_plot_points())
    right_edge = top_surface.get("right_edge", empty_plot_points())
    left_fit = top_surface.get("left_model", {"slope": 0.0, "intercept": np.nan, "valid": False})
    right_fit = top_surface.get("right_model", {"slope": 0.0, "intercept": np.nan, "valid": False})

    if len(left_edge.get("u", [])) != 1 or len(right_edge.get("u", [])) != 1:
        return _empty_measurement_payload("missing_edge_point")

    left_point_xy = _extract_edge_xy(left_edge)
    right_point_xy = _extract_edge_xy(right_edge)

    # Legacy 2D measurements kept as diagnostic fields so downstream
    # consumers (and the thesis figures) can compare against the prior
    # mixed-unit formulation. These are never exposed as the primary
    # ``gap`` / ``flush`` values.
    left_reference = measure_point_line_gap_flush(
        source_point=right_point_xy,
        other_edge_point=left_point_xy,
        fit=left_fit,
    )
    right_reference = measure_point_line_gap_flush(
        source_point=left_point_xy,
        other_edge_point=right_point_xy,
        fit=right_fit,
    )
    reference_frame = build_section_reference_frame(
        left_fit=left_fit,
        right_fit=right_fit,
        left_point_xy=left_point_xy,
        right_point_xy=right_point_xy,
    )
    reference_gap = measure_gap_in_reference_frame(
        left_point_xy=left_point_xy,
        right_point_xy=right_point_xy,
        frame=reference_frame,
    )
    gap_legacy_mixed = float(reference_gap["gap"])
    flush_legacy_mm = float(left_reference["flush"])

    # ----- authoritative 3D scheme B measurement -----
    left_surface_xyz = _collect_side_surface_xyz(top_surface, "left")
    right_surface_xyz = _collect_side_surface_xyz(top_surface, "right")
    if left_surface_xyz is None or right_surface_xyz is None:
        return _empty_measurement_payload("insufficient_surface_points_for_plane_fit")
    left_plane = fit_plane_3d(left_surface_xyz)
    right_plane = fit_plane_3d(right_surface_xyz)
    if left_plane is None or right_plane is None:
        return _empty_measurement_payload("degenerate_plane_fit")

    left_edge_xyz = _extract_edge_xyz(left_edge)
    right_edge_xyz = _extract_edge_xyz(right_edge)
    if left_edge_xyz is None or right_edge_xyz is None:
        return _empty_measurement_payload("missing_edge_xyz")

    # Estimate the section's along-seam direction in 3D: the second
    # principal axis of the combined top-surface points on both sides.
    # In a section slab the cross-seam extent (~15 mm) strongly dominates
    # the along-seam extent (~1 mm), so the PCA ordering is reliably
    # (cross, along, normal) and ``vt[1]`` is the along-seam direction.
    seam_tangent_3d = _estimate_seam_tangent_3d(left_surface_xyz, right_surface_xyz)

    measurement_3d = measure_gap_flush_3d(
        left_edge_xyz=left_edge_xyz,
        right_edge_xyz=right_edge_xyz,
        left_plane=left_plane,
        right_plane=right_plane,
        seam_tangent_3d=seam_tangent_3d,
    )
    if not bool(measurement_3d["valid"]):
        return _empty_measurement_payload(str(measurement_3d.get("reason", "degenerate_3d_frame")))

    gap = float(measurement_3d["gap"])
    flush = float(measurement_3d["flush"])
    full_3d_mm = float(measurement_3d["full_3d_mm"])
    if not np.isfinite(gap) or not np.isfinite(flush):
        return _empty_measurement_payload("invalid_3d_gap_flush")

    return {
        "valid": True,
        "reason": "ok",
        "gap": gap,
        "gap_along_residual_mm": float(measurement_3d.get("gap_along_residual", float("nan"))),
        "gap_perp_full_mm": float(measurement_3d.get("gap_perp_full", gap)),
        "flush": flush,
        "full_3d_mm": full_3d_mm,
        "gap_legacy_mixed": gap_legacy_mixed,
        "flush_legacy_mm": flush_legacy_mm,
        "left_point": _xy_payload(left_point_xy),
        "right_point": _xy_payload(right_point_xy),
        "left_xyz": _xyz_payload(left_edge_xyz),
        "right_xyz": _xyz_payload(right_edge_xyz),
        "left_model": dict(left_fit),
        "right_model": dict(right_fit),
        "left_plane": _plane_payload(left_plane),
        "right_plane": _plane_payload(right_plane),
        "reference_normal": [float(x) for x in measurement_3d["n_avg"].tolist()],
        "seam_tangent_along_3d": [float(x) for x in measurement_3d["t_along_plane"].tolist()],
        "seam_tangent_cross_3d": [float(x) for x in measurement_3d["t_cross_plane"].tolist()],
        "parallel_component_signed": float(measurement_3d["parallel_component_signed"]),
        "gap_reference_frame": _frame_payload(reference_frame),
        "gap_reference_left_point": _tn_payload(reference_gap, "left_point_ref"),
        "gap_reference_right_point": _tn_payload(reference_gap, "right_point_ref"),
        "gap_reference_left_foot": _tn_payload(reference_gap, "left_foot_ref"),
        "gap_reference_right_foot": _tn_payload(reference_gap, "right_foot_ref"),
        "left_reference_gap": float(left_reference["gap"]),
        "left_reference_flush": float(left_reference["flush"]),
        "right_reference_gap": float(right_reference["gap"]),
        "right_reference_flush": float(right_reference["flush"]),
        "left_reference_foot": _xy_payload(left_reference["foot_point"]),
        "right_reference_foot": _xy_payload(right_reference["foot_point"]),
        "reference_mode": "two_plane_3d_mm",
    }


def summarize_gap_flush(section_results: list[dict[str, Any]]) -> dict[str, Any]:
    valid_measurements = [item.get("measurement", {}) for item in section_results if item.get("measurement", {}).get("valid")]
    gaps = np.asarray([item["gap"] for item in valid_measurements], dtype=np.float32)
    flushes = np.asarray([item["flush"] for item in valid_measurements], dtype=np.float32)
    return {
        "num_sections": int(len(section_results)),
        "num_measurement_sections": int(len(valid_measurements)),
        "gap_mean": float(np.mean(gaps)) if len(gaps) else np.nan,
        "gap_std": float(np.std(gaps)) if len(gaps) else np.nan,
        "flush_mean": float(np.mean(flushes)) if len(flushes) else np.nan,
        "flush_std": float(np.std(flushes)) if len(flushes) else np.nan,
    }


def _empty_measurement_payload(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "reason": reason,
        "gap": np.nan,
        "gap_along_residual_mm": np.nan,
        "gap_perp_full_mm": np.nan,
        "flush": np.nan,
        "full_3d_mm": np.nan,
        "gap_legacy_mixed": np.nan,
        "flush_legacy_mm": np.nan,
        "left_point": _xy_payload(np.asarray([np.nan, np.nan], dtype=np.float32)),
        "right_point": _xy_payload(np.asarray([np.nan, np.nan], dtype=np.float32)),
        "left_xyz": _xyz_payload(None),
        "right_xyz": _xyz_payload(None),
        "left_model": {"slope": 0.0, "intercept": np.nan, "valid": False},
        "right_model": {"slope": 0.0, "intercept": np.nan, "valid": False},
        "left_plane": _plane_payload(None),
        "right_plane": _plane_payload(None),
        "reference_normal": [np.nan, np.nan, np.nan],
        "seam_tangent_along_3d": [np.nan, np.nan, np.nan],
        "seam_tangent_cross_3d": [np.nan, np.nan, np.nan],
        "parallel_component_signed": np.nan,
        "gap_reference_frame": _frame_payload({"valid": False, "origin_xy": [np.nan, np.nan], "tangent": [np.nan, np.nan], "normal": [np.nan, np.nan]}),
        "gap_reference_left_point": _tn_payload({}, "missing"),
        "gap_reference_right_point": _tn_payload({}, "missing"),
        "gap_reference_left_foot": _tn_payload({}, "missing"),
        "gap_reference_right_foot": _tn_payload({}, "missing"),
        "left_reference_gap": np.nan,
        "left_reference_flush": np.nan,
        "right_reference_gap": np.nan,
        "right_reference_flush": np.nan,
        "left_reference_foot": _xy_payload(np.asarray([np.nan, np.nan], dtype=np.float32)),
        "right_reference_foot": _xy_payload(np.asarray([np.nan, np.nan], dtype=np.float32)),
        "reference_mode": "two_plane_3d_mm",
    }


def _extract_edge_xy(edge: dict[str, Any]) -> np.ndarray:
    return np.asarray([edge["u"][0], edge["z"][0]], dtype=np.float32)


def _extract_edge_xyz(edge: dict[str, Any]) -> np.ndarray | None:
    xyz = edge.get("xyz") if edge else None
    if xyz is None or len(xyz) == 0:
        return None
    point = np.asarray(xyz[0], dtype=np.float32).reshape(-1)
    if len(point) < 3 or not np.all(np.isfinite(point[:3])):
        return None
    return point[:3]


def _estimate_seam_tangent_3d(
    left_surface_xyz: np.ndarray | None,
    right_surface_xyz: np.ndarray | None,
) -> np.ndarray | None:
    """Return a unit 3D vector pointing along the seam, estimated from the
    combined left + right top-surface point sets.

    The slab geometry gives (cross_seam, along_seam, normal) as the
    PCA axis order because the cross-seam extent (~15 mm) is much larger
    than the along-seam slab width (~1 mm) which in turn is larger than
    the surface-normal noise (~0.2 mm). We therefore take ``vt[1]`` as
    the along-seam direction. Returns ``None`` if either side's point
    set is missing or the SVD is degenerate.
    """
    chunks = [arr for arr in (left_surface_xyz, right_surface_xyz)
              if arr is not None and len(arr) > 0]
    if not chunks:
        return None
    combined = np.concatenate(chunks, axis=0).astype(np.float64)
    finite = np.all(np.isfinite(combined), axis=1)
    combined = combined[finite]
    if len(combined) < 4:
        return None
    centered = combined - combined.mean(axis=0)
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size < 2 or float(singular_values[1]) < 1e-6:
        return None
    return vt[1].astype(np.float32)


def _collect_side_surface_xyz(top_surface: dict[str, Any], side: str) -> np.ndarray | None:
    """Gather 3D surface points for one side's plane fit.

    Preference order: segment (cleanest) -> candidates (more data) ->
    neighbor_filtered (noisiest). Requires at least 6 finite points
    before a plane fit is attempted.
    """
    keys = (f"{side}_segment", f"{side}_candidates", f"{side}_neighbor_filtered")
    for key in keys:
        points = top_surface.get(key) if top_surface else None
        if not points:
            continue
        xyz = points.get("xyz")
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
        if arr.shape[0] == 0:
            continue
        finite = np.all(np.isfinite(arr), axis=1)
        arr = arr[finite]
        if arr.shape[0] >= 6:
            return arr
    return None


def _xy_payload(xy: np.ndarray) -> dict[str, float]:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1)
    if len(xy) < 2 or not np.all(np.isfinite(xy[:2])):
        return {"u": np.nan, "z": np.nan}
    return {"u": float(xy[0]), "z": float(xy[1])}


def _xyz_payload(xyz: np.ndarray | None) -> dict[str, float]:
    if xyz is None:
        return {"x": np.nan, "y": np.nan, "z": np.nan}
    arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
    if len(arr) < 3 or not np.all(np.isfinite(arr[:3])):
        return {"x": np.nan, "y": np.nan, "z": np.nan}
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}


def _plane_payload(plane: dict[str, np.ndarray] | None) -> dict[str, Any]:
    if plane is None:
        return {
            "valid": False,
            "centroid": [np.nan, np.nan, np.nan],
            "normal": [np.nan, np.nan, np.nan],
            "singular_values": [np.nan, np.nan, np.nan],
        }
    centroid = np.asarray(plane.get("centroid", [np.nan] * 3), dtype=np.float32).reshape(-1)
    normal = np.asarray(plane.get("normal", [np.nan] * 3), dtype=np.float32).reshape(-1)
    singular = np.asarray(plane.get("singular_values", [np.nan] * 3), dtype=np.float32).reshape(-1)
    return {
        "valid": True,
        "centroid": [float(x) for x in centroid.tolist()[:3]],
        "normal": [float(x) for x in normal.tolist()[:3]],
        "singular_values": [float(x) for x in singular.tolist()[:3]],
    }


def _tn_payload(container: dict[str, Any], key: str) -> dict[str, float]:
    values = np.asarray(container.get(key, [np.nan, np.nan]), dtype=np.float32).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values[:2])):
        return {"t": np.nan, "n": np.nan}
    return {"t": float(values[0]), "n": float(values[1])}


def _frame_payload(frame: dict[str, Any]) -> dict[str, Any]:
    origin_xy = np.asarray(frame.get("origin_xy", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
    tangent = np.asarray(frame.get("tangent", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
    normal = np.asarray(frame.get("normal", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
    return {
        "valid": bool(frame.get("valid", False)),
        "origin_xy": [float(origin_xy[0]) if len(origin_xy) > 0 else np.nan, float(origin_xy[1]) if len(origin_xy) > 1 else np.nan],
        "tangent": [float(tangent[0]) if len(tangent) > 0 else np.nan, float(tangent[1]) if len(tangent) > 1 else np.nan],
        "normal": [float(normal[0]) if len(normal) > 0 else np.nan, float(normal[1]) if len(normal) > 1 else np.nan],
    }
