from __future__ import annotations

from typing import Any

import numpy as np

from .helpers import count_neighbors, empty_points_like, make_plot_points, sort_points, subset_points
from .params import GapFlushParams


def filter_section_points(section: dict[str, Any], params: GapFlushParams) -> dict[str, np.ndarray]:
    return _filter_isolated_points(_valid_section_points(section), params)


def collect_section_filter_debug(section: dict[str, Any], params: GapFlushParams) -> dict[str, Any]:
    raw_points = _raw_section_points(section)
    valid_points = _valid_section_points(section)
    isolated_filtered = _filter_isolated_points(valid_points, params)
    empty_bottom = empty_points_like(isolated_filtered)
    return {
        "raw_points": raw_points,
        "valid_points": valid_points,
        "isolated_filtered": isolated_filtered,
        "surface_filtered": isolated_filtered,
        "rough_bottom": empty_bottom,
        "rough_bottom_debug": {"bottom_candidates": empty_bottom, "bottom_segments": [], "bottom_selected": empty_bottom},
        "bottom_debug": {
            "bottom_candidates": empty_bottom,
            "bottom_segments": [],
            "bottom_selected": isolated_filtered,
            "bottom_selected_raw": isolated_filtered,
            "recovered_bottom": isolated_filtered,
            "recovery_candidates": isolated_filtered,
        },
    }


def collect_bottom_segment_debug(points: dict[str, np.ndarray], params: GapFlushParams) -> dict[str, Any]:
    del params
    points = sort_points(points)
    return {"bottom_candidates": points, "bottom_segments": [], "bottom_selected": points}


def select_bottom_segment(points: dict[str, np.ndarray], params: GapFlushParams) -> dict[str, np.ndarray]:
    del params
    return sort_points(points)


def compute_section_bottom(section: dict[str, Any], params: GapFlushParams) -> dict[str, Any]:
    filter_debug = collect_section_filter_debug(section, params)
    filtered = filter_debug["isolated_filtered"]
    filter_debug["section_points"] = filtered
    if len(filtered["u"]) < params.min_section_points:
        return _section_payload(section, False, "too_few_points", filtered, filter_debug)
    return _section_payload(section, True, "ok", filtered, filter_debug)


def summarize_bottom(bottom_counts: np.ndarray, num_sections: int) -> dict[str, Any]:
    return {
        "num_sections": int(num_sections),
        "num_valid_sections": int(len(bottom_counts)),
        "bottom_count_mean": float(np.mean(bottom_counts)) if len(bottom_counts) else np.nan,
        "bottom_count_std": float(np.std(bottom_counts)) if len(bottom_counts) else np.nan,
    }


def _valid_section_points(section: dict[str, Any]) -> dict[str, np.ndarray]:
    valid_mask = np.asarray(section["valid"], dtype=bool)
    return make_plot_points(
        section["u"][valid_mask],
        section["z"][valid_mask],
        section["pixels_xy"][valid_mask],
        section["xyz"][valid_mask],
        sort=True,
    )


def _filter_isolated_points(points: dict[str, np.ndarray], params: GapFlushParams) -> dict[str, np.ndarray]:
    filtered = points
    while len(filtered["u"]):
        counts = count_neighbors(filtered["u"], filtered["z"], params.neighbor_radius_u, params.neighbor_height_tol)
        next_filtered = subset_points(filtered, counts >= params.min_neighbors)
        if len(next_filtered["u"]) == len(filtered["u"]):
            return next_filtered
        filtered = next_filtered
    return filtered


def _raw_section_points(section: dict[str, Any]) -> dict[str, np.ndarray]:
    return make_plot_points(
        section["raw_u"],
        section["raw_z"],
        section["raw_pixels_xy"],
        section["raw_xyz"],
    )


def _section_payload(
    section: dict[str, Any],
    valid: bool,
    reason: str,
    filtered: dict[str, np.ndarray],
    filter_debug: dict[str, Any],
) -> dict[str, Any]:
    return {
        "valid": valid,
        "reason": reason,
        "sample_index": section["sample_index"],
        "center_xy": section["center_xy"],
        "local_mask_width": section.get("local_mask_width", np.nan),
        "gap": np.nan,
        "flush": np.nan,
        "filter_debug": filter_debug,
        "filtered_points": filtered,
        "isolated_filtered": filter_debug["isolated_filtered"],
        "bottom_debug": filter_debug["bottom_debug"],
        "bottom_selected": filter_debug["bottom_debug"]["bottom_selected"],
    }
