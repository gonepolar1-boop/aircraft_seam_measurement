"""Gap/flush pipeline parameters.

Defaults are loaded at import time from ``configs/gap_flush.yaml`` (next to
the project root). Every field on :class:`GapFlushParams` has a built-in
fallback in :data:`_BUILTIN_DEFAULTS` so the pipeline keeps running even
if the YAML file is missing or unparsable.

See ``configs/gap_flush.yaml`` for per-parameter documentation (meaning,
unit, typical range).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "configs" / "gap_flush.yaml"

# Hard-coded safety-net values. Kept in sync with configs/gap_flush.yaml;
# the YAML file is the documented source of truth, this dict only takes
# effect when the YAML cannot be loaded.
_BUILTIN_DEFAULTS: dict[str, float | int] = {
    "seam_step": 1,
    "section_half_length_px": 5.0,
    "min_section_points": 32,
    "neighbor_radius_u": 2.16,
    "neighbor_height_tol": 0.096,
    "min_neighbors": 2,
    "height_window_u": 2.5,
    "height_outlier_tol": 0.12,
    "continuity_gap_u": 2.5,
    "continuity_gap_z": 0.12,
    "min_segment_points": 6,
    "bottom_band_height": 1.0,
    "bottom_candidate_height_ratio": 0.18,
    "bottom_span_weight": 2.0,
    "bottom_count_weight": 0.12,
    "bottom_flatness_weight": 4.0,
    "bottom_low_weight": 1.5,
    "bottom_min_span_ratio": 0.2,
    "bottom_refine_height_tol": 0.08,
    "bottom_refine_height_ratio": 0.18,
    "bottom_refine_num_bins": 64,
    "bottom_refine_min_bin_points": 6,
    "bottom_refine_low_quantile": 0.2,
    "bottom_refine_smooth_window": 5,
    "bottom_refine_sigma_alpha": 2.5,
    "bottom_refine_residual_alpha": 2.5,
    "display_max_pointcloud_points": 40000,
    "display_max_bottom_points": 12000,
    "display_z_scale": 1.0,
    "top_surface_quantile": 0.90,
    "top_surface_band_height": 0.8,
    "top_surface_center_margin_ratio": 0.2,
    "top_surface_neighbor_radius_u": 2.16,
    "top_surface_neighbor_height_tol": 0.096,
    "top_surface_min_neighbors": 2,
    "top_surface_continuity_gap_u": 2.5,
    "top_surface_min_segment_points": 10,
    "top_surface_search_radius_ratio": 3.0,
    "top_surface_smooth_window": 9,
    "top_surface_outlier_u_ratio": 0.35,
    "top_surface_outlier_u_min": 8.0,
    "top_surface_refine_passes": 5,
    "top_surface_fit_min_points": 16,
    "outlier_sigma": 2.0,
}


def _load_yaml_defaults(path: Path = _CONFIG_PATH) -> dict[str, float | int]:
    """Return the YAML defaults merged on top of :data:`_BUILTIN_DEFAULTS`.

    Missing file, missing dependency (PyYAML), or parse errors all degrade
    gracefully to the built-in defaults with a log warning.
    """
    merged = dict(_BUILTIN_DEFAULTS)
    try:
        import yaml  # noqa: PLC0415 - intentional lazy import
    except ImportError:
        logger.warning(
            "PyYAML not installed; using built-in GapFlushParams defaults. "
            "Install pyyaml to edit configs/gap_flush.yaml at runtime."
        )
        return merged

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        logger.warning("Config file %s not found; using built-in defaults.", path)
        return merged
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse %s (%s); using built-in defaults.", path, exc)
        return merged

    if not isinstance(loaded, dict):
        logger.warning("Config %s did not contain a mapping; using built-in defaults.", path)
        return merged

    unknown = set(loaded) - set(_BUILTIN_DEFAULTS)
    if unknown:
        logger.warning("Ignoring unknown keys in %s: %s", path, sorted(unknown))

    for key in _BUILTIN_DEFAULTS:
        if key in loaded and loaded[key] is not None:
            merged[key] = loaded[key]
    return merged


_CFG = _load_yaml_defaults()


@dataclass
class GapFlushParams:
    seam_step: int = int(_CFG["seam_step"])
    section_half_length_px: float = float(_CFG["section_half_length_px"])
    min_section_points: int = int(_CFG["min_section_points"])
    neighbor_radius_u: float = float(_CFG["neighbor_radius_u"])
    neighbor_height_tol: float = float(_CFG["neighbor_height_tol"])
    min_neighbors: int = int(_CFG["min_neighbors"])
    height_window_u: float = float(_CFG["height_window_u"])
    height_outlier_tol: float = float(_CFG["height_outlier_tol"])
    continuity_gap_u: float = float(_CFG["continuity_gap_u"])
    continuity_gap_z: float = float(_CFG["continuity_gap_z"])
    min_segment_points: int = int(_CFG["min_segment_points"])
    bottom_band_height: float = float(_CFG["bottom_band_height"])
    bottom_candidate_height_ratio: float = float(_CFG["bottom_candidate_height_ratio"])
    bottom_span_weight: float = float(_CFG["bottom_span_weight"])
    bottom_count_weight: float = float(_CFG["bottom_count_weight"])
    bottom_flatness_weight: float = float(_CFG["bottom_flatness_weight"])
    bottom_low_weight: float = float(_CFG["bottom_low_weight"])
    bottom_min_span_ratio: float = float(_CFG["bottom_min_span_ratio"])
    bottom_refine_height_tol: float = float(_CFG["bottom_refine_height_tol"])
    bottom_refine_height_ratio: float = float(_CFG["bottom_refine_height_ratio"])
    bottom_refine_num_bins: int = int(_CFG["bottom_refine_num_bins"])
    bottom_refine_min_bin_points: int = int(_CFG["bottom_refine_min_bin_points"])
    bottom_refine_low_quantile: float = float(_CFG["bottom_refine_low_quantile"])
    bottom_refine_smooth_window: int = int(_CFG["bottom_refine_smooth_window"])
    bottom_refine_sigma_alpha: float = float(_CFG["bottom_refine_sigma_alpha"])
    bottom_refine_residual_alpha: float = float(_CFG["bottom_refine_residual_alpha"])
    display_max_pointcloud_points: int = int(_CFG["display_max_pointcloud_points"])
    display_max_bottom_points: int = int(_CFG["display_max_bottom_points"])
    display_z_scale: float = float(_CFG["display_z_scale"])
    top_surface_quantile: float = float(_CFG["top_surface_quantile"])
    top_surface_band_height: float = float(_CFG["top_surface_band_height"])
    top_surface_center_margin_ratio: float = float(_CFG["top_surface_center_margin_ratio"])
    top_surface_neighbor_radius_u: float = float(_CFG["top_surface_neighbor_radius_u"])
    top_surface_neighbor_height_tol: float = float(_CFG["top_surface_neighbor_height_tol"])
    top_surface_min_neighbors: int = int(_CFG["top_surface_min_neighbors"])
    top_surface_continuity_gap_u: float = float(_CFG["top_surface_continuity_gap_u"])
    top_surface_min_segment_points: int = int(_CFG["top_surface_min_segment_points"])
    top_surface_search_radius_ratio: float = float(_CFG["top_surface_search_radius_ratio"])
    top_surface_smooth_window: int = int(_CFG["top_surface_smooth_window"])
    top_surface_outlier_u_ratio: float = float(_CFG["top_surface_outlier_u_ratio"])
    top_surface_outlier_u_min: float = float(_CFG["top_surface_outlier_u_min"])
    top_surface_refine_passes: int = int(_CFG["top_surface_refine_passes"])
    top_surface_fit_min_points: int = int(_CFG["top_surface_fit_min_points"])
    outlier_sigma: float = float(_CFG["outlier_sigma"])
