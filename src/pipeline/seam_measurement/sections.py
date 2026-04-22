from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .helpers import principal_axes, select_primary_mask_component, snap_center_to_valid_pixel, validate_inputs
from .params import GapFlushParams

_SECTION_EXTRACT_THREADS = int(os.environ.get("GAP_FLUSH_MAX_WORKERS", "0"))
if _SECTION_EXTRACT_THREADS <= 0:
    _SECTION_EXTRACT_THREADS = min(8, max(2, (os.cpu_count() or 1)))


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
    """Reference implementation. Uses boolean masks for slab selection."""
    ctx = _prepare_section_context(mask, point_map, seam_direction, params)
    if ctx is None:
        return []

    sample_positions = ctx["sample_positions"]
    half_length = float(ctx["params"].section_half_length_px)

    def _build_one(sample_index: int) -> dict[str, Any] | None:
        sample_t = float(sample_positions[sample_index])
        slab_mask = np.abs(ctx["component_t"] - sample_t) <= half_length
        if not np.any(slab_mask):
            return None
        seam_pixels_xy = ctx["mask_pixels"][slab_mask].astype(np.float32)
        slab_component_n = ctx["component_n"][slab_mask]
        return _build_section_from_slab(
            ctx=ctx,
            sample_index=sample_index,
            seam_pixels_xy=seam_pixels_xy,
            slab_component_n=slab_component_n,
            background_selector=_slab_mask_selector,
        )

    # ctx is read-only after _prepare_section_context; the per-sample
    # boolean mask selection is independent for each sample, so parallel
    # execution is safe. Mirrors the P6 change already applied to
    # extract_sections_fast.
    payloads: list[dict[str, Any] | None] = [None] * len(sample_positions)
    if _SECTION_EXTRACT_THREADS > 1 and len(sample_positions) > 0:
        with ThreadPoolExecutor(max_workers=_SECTION_EXTRACT_THREADS) as pool:
            futures = {pool.submit(_build_one, i): i for i in range(len(sample_positions))}
            for future, i in futures.items():
                payloads[i] = future.result()
    else:
        for i in range(len(sample_positions)):
            payloads[i] = _build_one(i)
    sections = [p for p in payloads if p is not None]
    return sections


def extract_sections_fast(
    mask: np.ndarray,
    point_map: np.ndarray,
    seam_direction: dict[str, np.ndarray] | None = None,
    params: GapFlushParams | None = None,
) -> list[dict[str, Any]]:
    """Faster variant: binary-search on a pre-sorted valid-point array."""
    ctx = _prepare_section_context(mask, point_map, seam_direction, params)
    if ctx is None:
        return []

    # Sort component arrays by component_t for O(log n) slab lookup.
    component_order = np.argsort(ctx["component_t"])
    ctx["sorted_component_t"] = ctx["component_t"][component_order]
    ctx["sorted_component_n"] = ctx["component_n"][component_order]
    ctx["sorted_mask_pixels"] = ctx["mask_pixels"][component_order]

    # Pre-sort valid-only projection arrays by v.
    valid_pixels_xy = ctx["all_pixels_xy"][ctx["flat_valid"]]
    valid_xyz = ctx["flat_xyz"][ctx["flat_valid"]]
    valid_base_u = (valid_pixels_xy @ ctx["normal"]).astype(np.float32)
    valid_base_v = (valid_pixels_xy @ ctx["tangent"]).astype(np.float32)
    valid_order = np.argsort(valid_base_v)
    ctx["sorted_valid_base_v"] = valid_base_v[valid_order]
    ctx["sorted_valid_base_u"] = valid_base_u[valid_order]
    ctx["sorted_valid_pixels_xy"] = valid_pixels_xy[valid_order]
    ctx["sorted_valid_xyz"] = valid_xyz[valid_order]

    half_length = float(ctx["params"].section_half_length_px)
    boundary_eps = 1e-5
    selector = _searchsorted_selector(boundary_eps)
    sample_positions = ctx["sample_positions"]

    def _build_one(sample_index: int) -> dict[str, Any] | None:
        sample_t = float(sample_positions[sample_index])
        slab_start = int(np.searchsorted(ctx["sorted_component_t"], sample_t - half_length - boundary_eps, side="left"))
        slab_end = int(np.searchsorted(ctx["sorted_component_t"], sample_t + half_length + boundary_eps, side="right"))
        if slab_end <= slab_start:
            return None
        seam_pixels_xy = ctx["sorted_mask_pixels"][slab_start:slab_end]
        slab_component_n = ctx["sorted_component_n"][slab_start:slab_end]
        return _build_section_from_slab(
            ctx=ctx,
            sample_index=sample_index,
            seam_pixels_xy=seam_pixels_xy.astype(np.float32),
            slab_component_n=slab_component_n,
            background_selector=selector,
        )

    # ctx arrays are read-only after _prepare_section_context + the pre-sorts
    # above, so threads can all index into them safely.  selector is pure-
    # functional (returns a dict built from numpy slices).
    payloads: list[dict[str, Any] | None] = [None] * len(sample_positions)
    if _SECTION_EXTRACT_THREADS > 1 and len(sample_positions) > 0:
        with ThreadPoolExecutor(max_workers=_SECTION_EXTRACT_THREADS) as pool:
            futures = {pool.submit(_build_one, i): i for i in range(len(sample_positions))}
            for future, i in futures.items():
                payloads[i] = future.result()
    else:
        for i in range(len(sample_positions)):
            payloads[i] = _build_one(i)
    return [p for p in payloads if p is not None]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_section_context(
    mask: np.ndarray,
    point_map: np.ndarray,
    seam_direction: dict[str, np.ndarray] | None,
    params: GapFlushParams | None,
) -> dict[str, Any] | None:
    params = GapFlushParams() if params is None else params
    validate_inputs(mask, point_map)
    seam_direction = extract_seam_direction(mask) if seam_direction is None else seam_direction

    mask_pixels = np.asarray(seam_direction["component_pixels"], dtype=np.float32).reshape(-1, 2)
    component_t = np.asarray(seam_direction["component_t"], dtype=np.float32).reshape(-1)
    component_n = np.asarray(seam_direction["component_n"], dtype=np.float32).reshape(-1)
    tangent = np.asarray(seam_direction["principal_tangent_xy"], dtype=np.float32).reshape(2)
    normal = np.asarray(seam_direction["principal_normal_xy"], dtype=np.float32).reshape(2)
    if len(mask_pixels) == 0 or len(component_t) == 0:
        return None

    height, width = point_map.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    all_pixels_xy = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32, copy=False)
    flat_xyz = point_map.reshape(-1, 3).astype(np.float32, copy=False)
    flat_valid = np.all(np.isfinite(flat_xyz), axis=1)
    base_u = (all_pixels_xy @ normal).astype(np.float32)
    base_v = (all_pixels_xy @ tangent).astype(np.float32)

    step = max(1.0, float(params.seam_step))
    sample_positions = np.arange(float(np.min(component_t)), float(np.max(component_t)) + step, step, dtype=np.float32)

    return {
        "params": params,
        "mask_pixels": mask_pixels,
        "component_t": component_t,
        "component_n": component_n,
        "tangent": tangent,
        "normal": normal,
        "all_pixels_xy": all_pixels_xy,
        "flat_xyz": flat_xyz,
        "flat_valid": flat_valid,
        "base_u": base_u,
        "base_v": base_v,
        "sample_positions": sample_positions,
        "point_map": point_map,
    }


def _slab_mask_selector(ctx: dict[str, Any], center_u: float, center_v: float) -> dict[str, np.ndarray]:
    slab_point_mask = np.abs(ctx["base_v"] - center_v) <= float(ctx["params"].section_half_length_px)
    slab_valid = slab_point_mask & ctx["flat_valid"]
    return {
        "u": (ctx["base_u"][slab_valid] - center_u).astype(np.float32),
        "z": ctx["flat_xyz"][slab_valid, 2].astype(np.float32),
        "pixels_xy": ctx["all_pixels_xy"][slab_valid].astype(np.float32),
        "xyz": ctx["flat_xyz"][slab_valid].astype(np.float32),
    }


def _searchsorted_selector(boundary_eps: float):
    def _select(ctx: dict[str, Any], center_u: float, center_v: float) -> dict[str, np.ndarray]:
        half_length = float(ctx["params"].section_half_length_px)
        valid_start = int(np.searchsorted(ctx["sorted_valid_base_v"], center_v - half_length - boundary_eps, side="left"))
        valid_end = int(np.searchsorted(ctx["sorted_valid_base_v"], center_v + half_length + boundary_eps, side="right"))
        return {
            "u": (ctx["sorted_valid_base_u"][valid_start:valid_end] - center_u).astype(np.float32, copy=False),
            "z": ctx["sorted_valid_xyz"][valid_start:valid_end, 2].astype(np.float32, copy=False),
            "pixels_xy": ctx["sorted_valid_pixels_xy"][valid_start:valid_end].astype(np.float32, copy=False),
            "xyz": ctx["sorted_valid_xyz"][valid_start:valid_end].astype(np.float32, copy=False),
        }

    return _select


def _build_section_from_slab(
    *,
    ctx: dict[str, Any],
    sample_index: int,
    seam_pixels_xy: np.ndarray,
    slab_component_n: np.ndarray,
    background_selector,
) -> dict[str, Any] | None:
    """Factor out the per-slab section-dict construction shared by both
    reference and fast extractors. ``background_selector`` abstracts over the
    slab selection strategy (boolean-mask vs searchsorted)."""
    if len(seam_pixels_xy) == 0:
        return None

    raw_center_xy = np.mean(seam_pixels_xy, axis=0).astype(np.float32)
    local_mask_width = float(np.max(slab_component_n) - np.min(slab_component_n)) if len(slab_component_n) > 1 else 0.0
    pix_x = seam_pixels_xy[:, 0].astype(np.int32)
    pix_y = seam_pixels_xy[:, 1].astype(np.int32)
    xyz = ctx["point_map"][pix_y, pix_x].astype(np.float32)
    valid = np.all(np.isfinite(xyz), axis=1)
    center_xy = snap_center_to_valid_pixel(raw_center_xy, seam_pixels_xy, valid)
    valid_xyz = xyz[valid].astype(np.float32, copy=False)
    center_xyz = np.mean(valid_xyz, axis=0).astype(np.float32) if len(valid_xyz) else np.full((3,), np.nan, dtype=np.float32)
    rel_xy = seam_pixels_xy - center_xy[None, :]
    center_u = float(center_xy @ ctx["normal"])
    center_v = float(center_xy @ ctx["tangent"])

    background_points = background_selector(ctx, center_u, center_v)

    return {
        "sample_index": int(sample_index),
        "section_half_length_px": float(ctx["params"].section_half_length_px),
        "center_xy": center_xy.astype(np.float32),
        "center_xyz": center_xyz.astype(np.float32),
        "raw_center_xy": raw_center_xy.astype(np.float32),
        "tangent_xy": ctx["tangent"].astype(np.float32),
        "normal_xy": ctx["normal"].astype(np.float32),
        "pixels_xy": seam_pixels_xy.astype(np.float32),
        "u": (rel_xy @ ctx["normal"]).astype(np.float32),
        "v": (rel_xy @ ctx["tangent"]).astype(np.float32),
        "xyz": xyz.astype(np.float32),
        "z": xyz[:, 2].astype(np.float32),
        "valid": valid,
        "local_half_width": 0.5 * float(local_mask_width),
        "local_mask_width": float(local_mask_width),
        "raw_pixels_xy": seam_pixels_xy.astype(np.float32),
        "raw_u": (rel_xy @ ctx["normal"]).astype(np.float32),
        "raw_v": (rel_xy @ ctx["tangent"]).astype(np.float32),
        "raw_xyz": xyz.astype(np.float32),
        "raw_z": xyz[:, 2].astype(np.float32),
        "background_points": background_points,
    }
