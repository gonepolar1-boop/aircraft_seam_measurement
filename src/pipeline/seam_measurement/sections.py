from __future__ import annotations

from typing import Any

import numpy as np

from .helpers import principal_axes, select_primary_mask_component, snap_center_to_valid_pixel, validate_inputs
from .params import GapFlushParams


def extract_seam_direction(mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}.")
    component = select_primary_mask_component(mask > 0)
    if len(component) < 2:
        raise ValueError("Unable to extract seam direction: too few valid mask pixels.")

    principal_center, principal_tangent, principal_normal = principal_axes(component)
    if float(principal_normal[0]) < 0.0 or (
        abs(float(principal_normal[0])) <= 1e-6 and float(principal_normal[1]) < 0.0
    ):
        principal_normal = -principal_normal
    centered_component = component - principal_center[None, :]
    return {
        "component_pixels": component.astype(np.float32),
        "principal_center_xy": principal_center.astype(np.float32),
        "principal_tangent_xy": principal_tangent.astype(np.float32),
        "principal_normal_xy": principal_normal.astype(np.float32),
        "component_t": (centered_component @ principal_tangent).astype(np.float32),
        "component_n": (centered_component @ principal_normal).astype(np.float32),
    }


def extract_sections(
    mask: np.ndarray,
    point_map: np.ndarray,
    seam_direction: dict[str, np.ndarray] | None = None,
    params: GapFlushParams | None = None,
) -> list[dict[str, Any]]:
    params = GapFlushParams() if params is None else params
    validate_inputs(mask, point_map)
    seam_direction = extract_seam_direction(mask) if seam_direction is None else seam_direction

    mask_pixels = np.asarray(seam_direction["component_pixels"], dtype=np.float32).reshape(-1, 2)
    component_t = np.asarray(seam_direction["component_t"], dtype=np.float32).reshape(-1)
    component_n = np.asarray(seam_direction["component_n"], dtype=np.float32).reshape(-1)
    tangent = np.asarray(seam_direction["principal_tangent_xy"], dtype=np.float32).reshape(2)
    normal = np.asarray(seam_direction["principal_normal_xy"], dtype=np.float32).reshape(2)
    if len(mask_pixels) == 0 or len(component_t) == 0:
        return []

    height, width = point_map.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    all_pixels_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)
    flat_xyz = point_map.reshape(-1, 3).astype(np.float32, copy=False)
    flat_valid = np.all(np.isfinite(flat_xyz), axis=1)
    base_u = (all_pixels_xy @ normal).astype(np.float32)
    base_v = (all_pixels_xy @ tangent).astype(np.float32)

    sections: list[dict[str, Any]] = []
    step = max(1.0, float(params.seam_step))
    sample_positions = np.arange(float(np.min(component_t)), float(np.max(component_t)) + step, step, dtype=np.float32)
    for sample_index, sample_t in enumerate(sample_positions):
        slab_mask = np.abs(component_t - sample_t) <= float(params.section_half_length_px)
        if not np.any(slab_mask):
            continue

        seam_pixels_xy = mask_pixels[slab_mask].astype(np.float32)
        if len(seam_pixels_xy) == 0:
            continue

        raw_center_xy = np.mean(seam_pixels_xy, axis=0).astype(np.float32)
        local_mask_width = float(np.max(component_n[slab_mask]) - np.min(component_n[slab_mask])) if np.sum(slab_mask) > 1 else 0.0
        pix_x = seam_pixels_xy[:, 0].astype(np.int32)
        pix_y = seam_pixels_xy[:, 1].astype(np.int32)
        xyz = point_map[pix_y, pix_x].astype(np.float32)
        valid = np.all(np.isfinite(xyz), axis=1)
        center_xy = snap_center_to_valid_pixel(raw_center_xy, seam_pixels_xy, valid)
        valid_xyz = xyz[valid].astype(np.float32, copy=False)
        center_xyz = np.mean(valid_xyz, axis=0).astype(np.float32) if len(valid_xyz) else np.full((3,), np.nan, dtype=np.float32)
        rel_xy = seam_pixels_xy - center_xy[None, :]
        center_u = float(center_xy @ normal)
        center_v = float(center_xy @ tangent)
        slab_point_mask = np.abs(base_v - center_v) <= float(params.section_half_length_px)
        slab_valid = slab_point_mask & flat_valid
        background_points = {
            "u": (base_u[slab_valid] - center_u).astype(np.float32),
            "z": flat_xyz[slab_valid, 2].astype(np.float32),
            "pixels_xy": all_pixels_xy[slab_valid].astype(np.float32),
            "xyz": flat_xyz[slab_valid].astype(np.float32),
        }

        sections.append(
            {
                "sample_index": int(sample_index),
                "section_half_length_px": float(params.section_half_length_px),
                "center_xy": center_xy.astype(np.float32),
                "center_xyz": center_xyz.astype(np.float32),
                "raw_center_xy": raw_center_xy.astype(np.float32),
                "tangent_xy": tangent.astype(np.float32),
                "normal_xy": normal.astype(np.float32),
                "pixels_xy": seam_pixels_xy.astype(np.float32),
                "u": (rel_xy @ normal).astype(np.float32),
                "v": (rel_xy @ tangent).astype(np.float32),
                "xyz": xyz.astype(np.float32),
                "z": xyz[:, 2].astype(np.float32),
                "valid": valid,
                "local_half_width": 0.5 * float(local_mask_width),
                "local_mask_width": float(local_mask_width),
                "raw_pixels_xy": seam_pixels_xy.astype(np.float32),
                "raw_u": (rel_xy @ normal).astype(np.float32),
                "raw_v": (rel_xy @ tangent).astype(np.float32),
                "raw_xyz": xyz.astype(np.float32),
                "raw_z": xyz[:, 2].astype(np.float32),
                "background_points": background_points,
            }
        )
    return sections


def extract_sections_fast(
    mask: np.ndarray,
    point_map: np.ndarray,
    seam_direction: dict[str, np.ndarray] | None = None,
    params: GapFlushParams | None = None,
) -> list[dict[str, Any]]:
    params = GapFlushParams() if params is None else params
    validate_inputs(mask, point_map)
    seam_direction = extract_seam_direction(mask) if seam_direction is None else seam_direction

    mask_pixels = np.asarray(seam_direction["component_pixels"], dtype=np.float32).reshape(-1, 2)
    component_t = np.asarray(seam_direction["component_t"], dtype=np.float32).reshape(-1)
    component_n = np.asarray(seam_direction["component_n"], dtype=np.float32).reshape(-1)
    tangent = np.asarray(seam_direction["principal_tangent_xy"], dtype=np.float32).reshape(2)
    normal = np.asarray(seam_direction["principal_normal_xy"], dtype=np.float32).reshape(2)
    if len(mask_pixels) == 0 or len(component_t) == 0:
        return []

    component_order = np.argsort(component_t)
    sorted_component_t = component_t[component_order]
    sorted_component_n = component_n[component_order]
    sorted_mask_pixels = mask_pixels[component_order]

    height, width = point_map.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    all_pixels_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32, copy=False)
    flat_xyz = point_map.reshape(-1, 3).astype(np.float32, copy=False)
    flat_valid = np.all(np.isfinite(flat_xyz), axis=1)
    valid_pixels_xy = all_pixels_xy[flat_valid]
    valid_xyz = flat_xyz[flat_valid]
    valid_base_u = (valid_pixels_xy @ normal).astype(np.float32)
    valid_base_v = (valid_pixels_xy @ tangent).astype(np.float32)
    valid_order = np.argsort(valid_base_v)
    sorted_valid_base_v = valid_base_v[valid_order]
    sorted_valid_base_u = valid_base_u[valid_order]
    sorted_valid_pixels_xy = valid_pixels_xy[valid_order]
    sorted_valid_xyz = valid_xyz[valid_order]

    sections: list[dict[str, Any]] = []
    step = max(1.0, float(params.seam_step))
    sample_positions = np.arange(float(np.min(component_t)), float(np.max(component_t)) + step, step, dtype=np.float32)
    half_length = float(params.section_half_length_px)
    boundary_eps = 1e-5
    for sample_index, sample_t in enumerate(sample_positions):
        slab_start = int(np.searchsorted(sorted_component_t, sample_t - half_length - boundary_eps, side="left"))
        slab_end = int(np.searchsorted(sorted_component_t, sample_t + half_length + boundary_eps, side="right"))
        if slab_end <= slab_start:
            continue

        seam_pixels_xy = sorted_mask_pixels[slab_start:slab_end]
        if len(seam_pixels_xy) == 0:
            continue

        raw_center_xy = np.mean(seam_pixels_xy, axis=0).astype(np.float32)
        slab_component_n = sorted_component_n[slab_start:slab_end]
        local_mask_width = float(np.max(slab_component_n) - np.min(slab_component_n)) if len(slab_component_n) > 1 else 0.0
        pix_x = seam_pixels_xy[:, 0].astype(np.int32)
        pix_y = seam_pixels_xy[:, 1].astype(np.int32)
        xyz = point_map[pix_y, pix_x].astype(np.float32)
        valid = np.all(np.isfinite(xyz), axis=1)
        center_xy = snap_center_to_valid_pixel(raw_center_xy, seam_pixels_xy, valid)
        valid_xyz = xyz[valid].astype(np.float32, copy=False)
        center_xyz = np.mean(valid_xyz, axis=0).astype(np.float32) if len(valid_xyz) else np.full((3,), np.nan, dtype=np.float32)
        rel_xy = seam_pixels_xy - center_xy[None, :]
        center_u = float(center_xy @ normal)
        center_v = float(center_xy @ tangent)
        valid_start = int(np.searchsorted(sorted_valid_base_v, center_v - half_length - boundary_eps, side="left"))
        valid_end = int(np.searchsorted(sorted_valid_base_v, center_v + half_length + boundary_eps, side="right"))
        background_points = {
            "u": (sorted_valid_base_u[valid_start:valid_end] - center_u).astype(np.float32, copy=False),
            "z": sorted_valid_xyz[valid_start:valid_end, 2].astype(np.float32, copy=False),
            "pixels_xy": sorted_valid_pixels_xy[valid_start:valid_end].astype(np.float32, copy=False),
            "xyz": sorted_valid_xyz[valid_start:valid_end].astype(np.float32, copy=False),
        }

        sections.append(
            {
                "sample_index": int(sample_index),
                "section_half_length_px": half_length,
                "center_xy": center_xy.astype(np.float32),
                "center_xyz": center_xyz.astype(np.float32),
                "raw_center_xy": raw_center_xy.astype(np.float32),
                "tangent_xy": tangent.astype(np.float32),
                "normal_xy": normal.astype(np.float32),
                "pixels_xy": seam_pixels_xy.astype(np.float32),
                "u": (rel_xy @ normal).astype(np.float32),
                "v": (rel_xy @ tangent).astype(np.float32),
                "xyz": xyz.astype(np.float32),
                "z": xyz[:, 2].astype(np.float32),
                "valid": valid,
                "local_half_width": 0.5 * float(local_mask_width),
                "local_mask_width": float(local_mask_width),
                "raw_pixels_xy": seam_pixels_xy.astype(np.float32),
                "raw_u": (rel_xy @ normal).astype(np.float32),
                "raw_v": (rel_xy @ tangent).astype(np.float32),
                "raw_xyz": xyz.astype(np.float32),
                "raw_z": xyz[:, 2].astype(np.float32),
                "background_points": background_points,
            }
        )
    return sections
