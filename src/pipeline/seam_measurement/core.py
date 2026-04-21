from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .bottom import compute_section_bottom
from .measurements import compute_section_gap_flush, summarize_gap_flush
from .params import GapFlushParams
from .sections import extract_seam_direction, extract_sections, extract_sections_fast
from .top_surface import detect_top_surface_edges, refine_top_surface_edge_sequence

# Thread workers for the per-section loops. numpy releases the GIL for
# most of its compute kernels so threading can help here even on CPython.
# Capped so a huge CPU does not thrash the L3 cache.
_MAX_SECTION_THREADS = int(os.environ.get("GAP_FLUSH_MAX_WORKERS", "0"))
if _MAX_SECTION_THREADS <= 0:
    _MAX_SECTION_THREADS = min(8, max(2, (os.cpu_count() or 1)))


def _process_section_initial(section: dict[str, Any], params: GapFlushParams) -> dict[str, Any]:
    item = compute_section_bottom(section, params)
    item["top_surface"] = detect_top_surface_edges(section, params)
    return item


def _process_section_measurement(item: dict[str, Any]) -> dict[str, Any]:
    item["measurement"] = compute_section_gap_flush(item)
    return item


def compute_gap_flush(
    mask: np.ndarray,
    point_map: np.ndarray,
    params: GapFlushParams | None = None,
    *,
    fast_mode: bool = False,
) -> dict[str, Any]:
    params = GapFlushParams() if params is None else params
    seam_direction = extract_seam_direction(mask)
    sections_fn = extract_sections_fast if fast_mode else extract_sections
    sections = sections_fn(mask, point_map, seam_direction=seam_direction, params=params)
    section_results: list[dict[str, Any]] = [None] * len(sections)  # type: ignore[list-item]
    if len(sections):
        with ThreadPoolExecutor(max_workers=_MAX_SECTION_THREADS) as pool:
            futures = {pool.submit(_process_section_initial, section, params): idx
                       for idx, section in enumerate(sections)}
            for future, idx in futures.items():
                section_results[idx] = future.result()
    section_results = refine_top_surface_edge_sequence(section_results, sections, params)
    if len(section_results):
        with ThreadPoolExecutor(max_workers=_MAX_SECTION_THREADS) as pool:
            futures = {pool.submit(_process_section_measurement, item): idx
                       for idx, item in enumerate(section_results)}
            for future, idx in futures.items():
                section_results[idx] = future.result()
    section_index = {int(section["sample_index"]): section for section in sections}
    bottom_counts = np.asarray(
        [len(item["bottom_selected"]["u"]) for item in section_results if item.get("valid")],
        dtype=np.float32,
    )
    gap_list = np.asarray(
        [item["measurement"]["gap"] for item in section_results if item.get("measurement", {}).get("valid")],
        dtype=np.float32,
    )
    flush_list = np.asarray(
        [item["measurement"]["flush"] for item in section_results if item.get("measurement", {}).get("valid")],
        dtype=np.float32,
    )
    summary = summarize_gap_flush(section_results)
    summary["bottom_count_mean"] = float(np.mean(bottom_counts)) if len(bottom_counts) else np.nan
    summary["bottom_count_std"] = float(np.std(bottom_counts)) if len(bottom_counts) else np.nan
    return {
        "seam_direction": seam_direction,
        "sections": sections,
        "section_index": section_index,
        "section_results": section_results,
        "point_map": point_map.astype(np.float32, copy=False),
        "params": params,
        "gap_list": gap_list,
        "flush_list": flush_list,
        "summary": summary,
    }


def compute_gap_flush_from_mapping(
    mapping_result: dict[str, Any],
    params: GapFlushParams | None = None,
    *,
    fast_mode: bool = False,
) -> dict[str, Any]:
    if "pred_mask" not in mapping_result or "point_map" not in mapping_result:
        raise ValueError("mapping_result must contain 'pred_mask' and 'point_map'.")
    result = compute_gap_flush(
        mask=np.asarray(mapping_result["pred_mask"], dtype=np.uint8),
        point_map=np.asarray(mapping_result["point_map"], dtype=np.float32),
        params=params,
        fast_mode=fast_mode,
    )
    result["mapping_inputs"] = mapping_result.get("inputs", {})
    return result
