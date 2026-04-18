from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GapFlushParams:
    seam_step: int = 4
    section_half_length_px: float = 4.0
    min_section_points: int = 24
    neighbor_radius_u: float = 2.16
    neighbor_height_tol: float = 0.096
    min_neighbors: int = 2
    height_window_u: float = 2.5
    height_outlier_tol: float = 0.12
    continuity_gap_u: float = 2.5
    continuity_gap_z: float = 0.12
    min_segment_points: int = 6
    bottom_band_height: float = 1.0
    bottom_candidate_height_ratio: float = 0.18
    bottom_span_weight: float = 2.0
    bottom_count_weight: float = 0.12
    bottom_flatness_weight: float = 4.0
    bottom_low_weight: float = 1.5
    bottom_min_span_ratio: float = 0.2
    bottom_refine_height_tol: float = 0.08
    bottom_refine_height_ratio: float = 0.18
    bottom_refine_num_bins: int = 64
    bottom_refine_min_bin_points: int = 6
    bottom_refine_low_quantile: float = 0.2
    bottom_refine_smooth_window: int = 5
    bottom_refine_sigma_alpha: float = 2.5
    bottom_refine_residual_alpha: float = 2.5
    display_max_pointcloud_points: int = 40000
    display_max_bottom_points: int = 12000
    display_z_scale: float = 1.0
    top_surface_quantile: float = 0.85
    top_surface_band_height: float = 1.0
    top_surface_center_margin_ratio: float = 0.2
    top_surface_neighbor_radius_u: float = 2.16
    top_surface_neighbor_height_tol: float = 0.096
    top_surface_min_neighbors: int = 2
    top_surface_continuity_gap_u: float = 2.5
    top_surface_min_segment_points: int = 6
    top_surface_search_radius_ratio: float = 3.0
    top_surface_smooth_window: int = 5
    top_surface_outlier_u_ratio: float = 0.5
    top_surface_outlier_u_min: float = 12.0
    top_surface_refine_passes: int = 3
    top_surface_fit_min_points: int = 12
