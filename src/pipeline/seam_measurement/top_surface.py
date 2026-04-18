from __future__ import annotations

from typing import Any

import numpy as np

from .helpers import (
    count_neighbors,
    empty_plot_points,
    make_plot_points,
    merge_plot_points,
    sort_points,
    split_segments_by_u,
    subset_points,
)
from .params import GapFlushParams


def detect_top_surface_edges(section: dict[str, Any], params: GapFlushParams) -> dict[str, Any]:
    background = _finite_background_points(section)
    if len(background["u"]) == 0:
        return _empty_top_surface_payload("no_background_points")

    local_width = max(float(section.get("local_mask_width", 0.0)), 1.0)
    search_radius = max(8.0, float(params.top_surface_search_radius_ratio) * local_width)
    local_background = sort_points(subset_points(background, np.abs(background["u"]) <= search_radius))
    if len(local_background["u"]) == 0:
        return _empty_top_surface_payload("no_local_background", background_points=background)

    counts = count_neighbors(
        local_background["u"],
        local_background["z"],
        float(params.top_surface_neighbor_radius_u),
        float(params.top_surface_neighbor_height_tol),
    )
    neighbor_filtered = subset_points(local_background, counts >= int(params.top_surface_min_neighbors))
    if len(neighbor_filtered["u"]) == 0:
        return _empty_top_surface_payload("no_neighbor_points", background_points=background)

    center_margin = max(0.5, float(params.top_surface_center_margin_ratio) * local_width)
    left_neighbor = sort_points(subset_points(neighbor_filtered, neighbor_filtered["u"] < -center_margin))
    right_neighbor = sort_points(subset_points(neighbor_filtered, neighbor_filtered["u"] > center_margin))

    left_z_top = _estimate_top_z(left_neighbor, float(params.top_surface_quantile))
    right_z_top = _estimate_top_z(right_neighbor, float(params.top_surface_quantile))
    left_fit = _fit_top_surface_line(left_neighbor, left_z_top, params)
    right_fit = _fit_top_surface_line(right_neighbor, right_z_top, params)
    if not np.isfinite(left_z_top) and not np.isfinite(right_z_top):
        return _empty_top_surface_payload(
            "invalid_top_z",
            background_points=background,
            neighbor_filtered=neighbor_filtered,
            center_margin=center_margin,
        )

    left_candidates = _select_points_near_top_fit(left_neighbor, left_fit, left_z_top, float(params.top_surface_band_height))
    right_candidates = _select_points_near_top_fit(right_neighbor, right_fit, right_z_top, float(params.top_surface_band_height))
    top_band = merge_plot_points(left_candidates, right_candidates)
    if len(top_band["u"]) == 0:
        return _empty_top_surface_payload(
            "no_top_band_points",
            background_points=background,
            neighbor_filtered=neighbor_filtered,
            left_z_top=left_z_top,
            right_z_top=right_z_top,
            center_margin=center_margin,
        )

    left_segment = _select_center_nearest_segment(left_candidates, params, side="left")
    right_segment = _select_center_nearest_segment(right_candidates, params, side="right")
    left_model = _fit_segment_surface_line(left_segment, left_fit, left_z_top)
    right_model = _fit_segment_surface_line(right_segment, right_fit, right_z_top)
    left_edge = _build_edge_point_from_model(left_segment, left_model, side="left")
    right_edge = _build_edge_point_from_model(right_segment, right_model, side="right")
    valid = len(left_edge["u"]) == 1 and len(right_edge["u"]) == 1

    return {
        "valid": bool(valid),
        "reason": "ok" if valid else "missing_side_edge",
        "z_top": float(np.nanmax(np.asarray([left_z_top, right_z_top], dtype=np.float32))),
        "left_z_top": float(left_z_top),
        "right_z_top": float(right_z_top),
        "center_margin": float(center_margin),
        "background_points": background,
        "local_background": local_background,
        "neighbor_filtered": neighbor_filtered,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "left_model": left_model,
        "right_model": right_model,
        "top_band": top_band,
        "left_candidates": left_candidates,
        "right_candidates": right_candidates,
        "left_segment": left_segment,
        "right_segment": right_segment,
        "left_edge": left_edge,
        "right_edge": right_edge,
    }


def refine_top_surface_edge_sequence(
    section_results: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    params: GapFlushParams,
) -> list[dict[str, Any]]:
    if not section_results:
        return section_results
    refined_results = list(section_results)
    for _ in range(max(1, int(params.top_surface_refine_passes))):
        for side in ("left", "right"):
            refined_results = _refine_side_edges(refined_results, sections, params, side)
    return refined_results


def _finite_background_points(section: dict[str, Any]) -> dict[str, np.ndarray]:
    points = section.get("background_points", empty_plot_points())
    u = np.asarray(points.get("u", []), dtype=np.float32).reshape(-1)
    z = np.asarray(points.get("z", []), dtype=np.float32).reshape(-1)
    pixels_xy = np.asarray(points.get("pixels_xy", []), dtype=np.float32).reshape(-1, 2)
    xyz = np.asarray(points.get("xyz", []), dtype=np.float32).reshape(-1, 3)
    if len(u) == 0:
        return empty_plot_points()
    finite_mask = np.isfinite(u) & np.isfinite(z) & np.all(np.isfinite(xyz), axis=1)
    return make_plot_points(u[finite_mask], z[finite_mask], pixels_xy[finite_mask], xyz[finite_mask], sort=True)


def _estimate_top_z(points: dict[str, np.ndarray], quantile: float) -> float:
    z = np.asarray(points.get("z", []), dtype=np.float32).reshape(-1)
    finite_z = z[np.isfinite(z)]
    if len(finite_z) == 0:
        return np.nan
    return float(np.quantile(finite_z, quantile))


def _fit_top_surface_line(points: dict[str, np.ndarray], z_ref: float, params: GapFlushParams) -> dict[str, float]:
    if len(points.get("u", [])) < int(params.top_surface_fit_min_points) or not np.isfinite(z_ref):
        return {"slope": 0.0, "intercept": float(z_ref), "valid": False}
    u = np.asarray(points["u"], dtype=np.float32).reshape(-1)
    z = np.asarray(points["z"], dtype=np.float32).reshape(-1)
    pre_mask = np.abs(z - float(z_ref)) <= max(float(params.top_surface_band_height) * 1.5, 0.25)
    if int(np.count_nonzero(pre_mask)) >= int(params.top_surface_fit_min_points):
        u_fit = u[pre_mask]
        z_fit = z[pre_mask]
    else:
        u_fit = u
        z_fit = z
    if len(u_fit) < 2:
        return {"slope": 0.0, "intercept": float(z_ref), "valid": False}
    slope, intercept = np.polyfit(u_fit.astype(np.float64), z_fit.astype(np.float64), deg=1)
    return {"slope": float(slope), "intercept": float(intercept), "valid": True}


def _fit_segment_surface_line(segment: dict[str, np.ndarray], fallback_fit: dict[str, float], z_ref: float) -> dict[str, float]:
    if len(segment.get("u", [])) < 2:
        return dict(fallback_fit)
    u = np.asarray(segment["u"], dtype=np.float32).reshape(-1)
    z = np.asarray(segment["z"], dtype=np.float32).reshape(-1)
    if len(u) < 2:
        return dict(fallback_fit)
    slope, intercept = np.polyfit(u.astype(np.float64), z.astype(np.float64), deg=1)
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return dict(fallback_fit)
    return {"slope": float(slope), "intercept": float(intercept), "valid": True, "z_ref": float(z_ref)}


def _evaluate_top_surface_fit(u: np.ndarray, fit: dict[str, float], z_ref: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    if bool(fit.get("valid", False)):
        return (float(fit["slope"]) * u + float(fit["intercept"])).astype(np.float32)
    return np.full_like(u, float(z_ref), dtype=np.float32)


def _select_points_near_top_fit(points: dict[str, np.ndarray], fit: dict[str, float], z_ref: float, tol: float) -> dict[str, np.ndarray]:
    if len(points.get("z", [])) == 0 or not np.isfinite(z_ref):
        return empty_plot_points()
    fit_z = _evaluate_top_surface_fit(points["u"], fit, z_ref)
    keep_mask = np.abs(np.asarray(points["z"], dtype=np.float32) - fit_z) <= float(tol)
    return sort_points(subset_points(points, keep_mask))


def _select_center_nearest_segment(points: dict[str, np.ndarray], params: GapFlushParams, side: str) -> dict[str, np.ndarray]:
    if len(points["u"]) == 0:
        return empty_plot_points()
    segments = split_segments_by_u(points, float(params.top_surface_continuity_gap_u))
    segments = [segment for segment in segments if len(segment["u"]) >= int(params.top_surface_min_segment_points)]
    if not segments:
        return empty_plot_points()
    if side == "left":
        return max(segments, key=lambda segment: float(np.max(segment["u"])))
    return min(segments, key=lambda segment: float(np.min(segment["u"])))


def _select_edge_point(points: dict[str, np.ndarray], side: str) -> dict[str, np.ndarray]:
    if len(points["u"]) == 0:
        return empty_plot_points()
    index = int(np.argmax(points["u"])) if side == "left" else int(np.argmin(points["u"]))
    return {key: value[index : index + 1] for key, value in points.items()}


def _build_edge_point_from_model(points: dict[str, np.ndarray], fit: dict[str, float], side: str) -> dict[str, np.ndarray]:
    if len(points.get("u", [])) == 0:
        return empty_plot_points()
    u_values = np.asarray(points["u"], dtype=np.float32).reshape(-1)
    edge_u = float(np.max(u_values)) if side == "left" else float(np.min(u_values))
    fit_z = _evaluate_top_surface_fit(np.asarray([edge_u], dtype=np.float32), fit, np.nan)
    nearest_index = int(np.argmin(np.abs(u_values - edge_u)))
    edge_point = {key: value[nearest_index : nearest_index + 1].copy() for key, value in points.items()}
    edge_point["u"][0] = edge_u
    if np.isfinite(fit_z[0]):
        edge_point["z"][0] = float(fit_z[0])
        if edge_point["xyz"].shape[1] >= 3:
            edge_point["xyz"][0, 2] = float(fit_z[0])
    return edge_point


def _refine_side_edges(
    section_results: list[dict[str, Any]],
    sections: list[dict[str, Any]],
    params: GapFlushParams,
    side: str,
) -> list[dict[str, Any]]:
    edge_key = f"{side}_edge"
    candidates_key = f"{side}_candidates"
    segment_key = f"{side}_segment"
    current_u = np.full((len(section_results),), np.nan, dtype=np.float32)
    local_widths = np.full((len(section_results),), np.nan, dtype=np.float32)

    for index, item in enumerate(section_results):
        top_surface = item.get("top_surface", {})
        edge = top_surface.get(edge_key, empty_plot_points())
        if len(edge.get("u", [])):
            current_u[index] = float(edge["u"][0])
        sample_index = int(item.get("sample_index", index))
        if 0 <= sample_index < len(sections):
            local_widths[index] = float(sections[sample_index].get("local_mask_width", np.nan))

    target_u = _rolling_median(current_u, max(3, int(params.top_surface_smooth_window)))
    threshold_u = np.maximum(
        float(params.top_surface_outlier_u_min),
        np.nan_to_num(local_widths, nan=float(params.top_surface_outlier_u_min)) * float(params.top_surface_outlier_u_ratio),
    )

    for index, item in enumerate(section_results):
        if not np.isfinite(current_u[index]) or not np.isfinite(target_u[index]):
            continue
        if abs(float(current_u[index] - target_u[index])) <= float(threshold_u[index]):
            continue

        top_surface = dict(item.get("top_surface", {}))
        candidates = top_surface.get(candidates_key, empty_plot_points())
        if len(candidates.get("u", [])) == 0:
            continue

        updated_segment = _select_segment_near_target(candidates, params, target_u[index])
        if len(updated_segment["u"]) == 0:
            continue
        fallback_fit_key = f"{side}_fit"
        model_key = f"{side}_model"
        z_top_key = f"{side}_z_top"
        updated_model = _fit_segment_surface_line(
            updated_segment,
            top_surface.get(fallback_fit_key, {"slope": 0.0, "intercept": np.nan, "valid": False}),
            float(top_surface.get(z_top_key, np.nan)),
        )
        updated_edge = _build_edge_point_from_model(updated_segment, updated_model, side=side)
        top_surface[segment_key] = updated_segment
        top_surface[model_key] = updated_model
        top_surface[edge_key] = updated_edge
        has_left = len(top_surface.get("left_edge", empty_plot_points()).get("u", [])) == 1
        has_right = len(top_surface.get("right_edge", empty_plot_points()).get("u", [])) == 1
        top_surface["valid"] = bool(has_left and has_right)
        top_surface["reason"] = "ok_refined" if top_surface["valid"] else "missing_side_edge"

        updated_item = dict(item)
        updated_item["top_surface"] = top_surface
        section_results[index] = updated_item
    return section_results


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return values
    radius = max(1, int(window) // 2)
    output = np.full_like(values, np.nan)
    for index in range(len(values)):
        chunk = values[max(0, index - radius) : min(len(values), index + radius + 1)]
        finite_chunk = chunk[np.isfinite(chunk)]
        if len(finite_chunk):
            output[index] = float(np.median(finite_chunk))
    return output


def _select_segment_near_target(points: dict[str, np.ndarray], params: GapFlushParams, target_u: float) -> dict[str, np.ndarray]:
    if len(points.get("u", [])) == 0 or not np.isfinite(target_u):
        return empty_plot_points()
    segments = split_segments_by_u(points, float(params.top_surface_continuity_gap_u))
    segments = [segment for segment in segments if len(segment["u"]) >= int(params.top_surface_min_segment_points)]
    if not segments:
        return empty_plot_points()
    return min(segments, key=lambda segment: float(np.min(np.abs(segment["u"] - target_u))))




def _empty_top_surface_payload(
    reason: str,
    *,
    background_points: dict[str, np.ndarray] | None = None,
    neighbor_filtered: dict[str, np.ndarray] | None = None,
    left_z_top: float = np.nan,
    right_z_top: float = np.nan,
    z_top: float = np.nan,
    center_margin: float = np.nan,
) -> dict[str, Any]:
    empty = empty_plot_points()
    return {
        "valid": False,
        "reason": reason,
        "z_top": float(z_top),
        "left_z_top": float(left_z_top),
        "right_z_top": float(right_z_top),
        "center_margin": float(center_margin),
        "background_points": empty if background_points is None else background_points,
        "local_background": empty,
        "neighbor_filtered": empty if neighbor_filtered is None else neighbor_filtered,
        "left_fit": {"slope": 0.0, "intercept": float(left_z_top), "valid": False},
        "right_fit": {"slope": 0.0, "intercept": float(right_z_top), "valid": False},
        "left_model": {"slope": 0.0, "intercept": float(left_z_top), "valid": False},
        "right_model": {"slope": 0.0, "intercept": float(right_z_top), "valid": False},
        "top_band": empty,
        "left_candidates": empty,
        "right_candidates": empty,
        "left_segment": empty,
        "right_segment": empty,
        "left_edge": empty,
        "right_edge": empty,
    }
