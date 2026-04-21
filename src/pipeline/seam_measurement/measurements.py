from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import build_section_reference_frame, measure_gap_in_reference_frame, measure_point_line_gap_flush
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
    u_l, z_l = float(left_point_xy[0]), float(left_point_xy[1])
    u_r, z_r = float(right_point_xy[0]), float(right_point_xy[1])

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
    # Flush keeps using the left fitted surface as the only reference line.
    gap = float(reference_gap["gap"])
    flush = float(left_reference["flush"])
    if not np.isfinite(gap):
        return _empty_measurement_payload("invalid_gap")

    return {
        "valid": bool(np.isfinite(gap) and np.isfinite(flush)),
        "reason": "ok" if (np.isfinite(gap) and np.isfinite(flush)) else "invalid_flush",
        "gap": float(gap),
        "flush": float(flush),
        "left_point": _xy_payload(left_point_xy),
        "right_point": _xy_payload(right_point_xy),
        "left_model": dict(left_fit),
        "right_model": dict(right_fit),
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
        "reference_mode": "flush_left_surface_gap_section_middle",
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
        "flush": np.nan,
        "left_point": _xy_payload(np.asarray([np.nan, np.nan], dtype=np.float32)),
        "right_point": _xy_payload(np.asarray([np.nan, np.nan], dtype=np.float32)),
        "left_model": {"slope": 0.0, "intercept": np.nan, "valid": False},
        "right_model": {"slope": 0.0, "intercept": np.nan, "valid": False},
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
        "reference_mode": "flush_left_surface_gap_section_middle",
    }


def _extract_edge_xy(edge: dict[str, Any]) -> np.ndarray:
    return np.asarray([edge["u"][0], edge["z"][0]], dtype=np.float32)


def _xy_payload(xy: np.ndarray) -> dict[str, float]:
    xy = np.asarray(xy, dtype=np.float32).reshape(-1)
    if len(xy) < 2 or not np.all(np.isfinite(xy[:2])):
        return {"u": np.nan, "z": np.nan}
    return {"u": float(xy[0]), "z": float(xy[1])}


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
