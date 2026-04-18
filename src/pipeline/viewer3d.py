from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_gap_flush_viewer_payload(
    *,
    measurement_result: dict[str, Any],
    section_profile: list[dict[str, Any]],
    max_background_points: int = 30000,
    max_seam_points: int = 8000,
    max_surface_points: int = 12000,
) -> dict[str, np.ndarray]:
    point_map = np.asarray(measurement_result.get("point_map", np.empty((0, 0, 3), dtype=np.float32)), dtype=np.float32)
    seam_direction = measurement_result.get("seam_direction", {})
    sections = measurement_result.get("sections", [])
    section_results = measurement_result.get("section_results", [])

    background_xyz = _subsample_xyz(_finite_xyz(point_map.reshape(-1, 3)), int(max_background_points))
    seam_xyz = _subsample_xyz(_collect_seam_xyz(seam_direction, point_map), int(max_seam_points))
    left_candidates_xyz = _subsample_xyz(_collect_nested_xyz(section_results, "top_surface", "left_candidates"), int(max_surface_points))
    right_candidates_xyz = _subsample_xyz(_collect_nested_xyz(section_results, "top_surface", "right_candidates"), int(max_surface_points))
    left_edge_xyz = _collect_nested_edge_xyz(section_results, "top_surface", "left_edge")
    right_edge_xyz = _collect_nested_edge_xyz(section_results, "top_surface", "right_edge")

    anomaly_indices = np.asarray(_collect_anomaly_sample_indices(section_profile), dtype=np.int32)
    normal_centers_xyz, anomaly_centers_xyz = _split_section_centers(sections, set(anomaly_indices.tolist()))
    return {
        "background_xyz": np.asarray(background_xyz, dtype=np.float32),
        "seam_xyz": np.asarray(seam_xyz, dtype=np.float32),
        "left_candidates_xyz": np.asarray(left_candidates_xyz, dtype=np.float32),
        "right_candidates_xyz": np.asarray(right_candidates_xyz, dtype=np.float32),
        "left_edge_xyz": np.asarray(left_edge_xyz, dtype=np.float32),
        "right_edge_xyz": np.asarray(right_edge_xyz, dtype=np.float32),
        "normal_centers_xyz": np.asarray(normal_centers_xyz, dtype=np.float32),
        "anomaly_centers_xyz": np.asarray(anomaly_centers_xyz, dtype=np.float32),
        "anomaly_sample_indices": anomaly_indices,
    }


def save_gap_flush_viewer_bundle(
    *,
    save_path: str | Path,
    measurement_result: dict[str, Any],
    section_profile: list[dict[str, Any]],
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_gap_flush_viewer_payload(
        measurement_result=measurement_result,
        section_profile=section_profile,
    )
    np.savez_compressed(save_path, **payload)
    return save_path


def load_gap_flush_viewer_bundle(bundle_path: str | Path) -> dict[str, np.ndarray]:
    bundle_path = Path(bundle_path)
    with np.load(bundle_path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key], dtype=np.float32 if key != "anomaly_sample_indices" else np.int32) for key in archive.files}


def show_gap_flush_open3d_viewer(
    *,
    measurement_result: dict[str, Any],
    section_profile: list[dict[str, Any]],
    window_title: str,
    max_background_points: int = 30000,
    max_seam_points: int = 8000,
    max_surface_points: int = 12000,
) -> None:
    payload = build_gap_flush_viewer_payload(
        measurement_result=measurement_result,
        section_profile=section_profile,
        max_background_points=max_background_points,
        max_seam_points=max_seam_points,
        max_surface_points=max_surface_points,
    )
    show_gap_flush_open3d_viewer_from_payload(payload=payload, window_title=window_title)


def show_gap_flush_open3d_viewer_from_bundle(
    *,
    bundle_path: str | Path,
    window_title: str,
) -> None:
    payload = load_gap_flush_viewer_bundle(bundle_path)
    show_gap_flush_open3d_viewer_from_payload(payload=payload, window_title=window_title)


def show_gap_flush_open3d_viewer_from_payload(
    *,
    payload: dict[str, np.ndarray],
    window_title: str,
) -> None:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("open3d is not installed in the current environment.") from exc

    background_xyz = _finite_xyz(payload.get("background_xyz", np.empty((0, 3), dtype=np.float32)))
    seam_xyz = _finite_xyz(payload.get("seam_xyz", np.empty((0, 3), dtype=np.float32)))
    left_candidates_xyz = _finite_xyz(payload.get("left_candidates_xyz", np.empty((0, 3), dtype=np.float32)))
    right_candidates_xyz = _finite_xyz(payload.get("right_candidates_xyz", np.empty((0, 3), dtype=np.float32)))
    left_edge_xyz = _finite_xyz(payload.get("left_edge_xyz", np.empty((0, 3), dtype=np.float32)))
    right_edge_xyz = _finite_xyz(payload.get("right_edge_xyz", np.empty((0, 3), dtype=np.float32)))
    normal_centers_xyz = _finite_xyz(payload.get("normal_centers_xyz", np.empty((0, 3), dtype=np.float32)))
    anomaly_centers_xyz = _finite_xyz(payload.get("anomaly_centers_xyz", np.empty((0, 3), dtype=np.float32)))

    geometries = [
        _make_point_cloud(o3d, background_xyz, color=(0.66, 0.70, 0.78)),
        _make_point_cloud(o3d, seam_xyz, color=(0.92, 0.25, 0.25)),
        _make_point_cloud(o3d, left_candidates_xyz, color=(0.18, 0.82, 0.38)),
        _make_point_cloud(o3d, right_candidates_xyz, color=(0.95, 0.46, 0.26)),
        _make_point_cloud(o3d, normal_centers_xyz, color=(0.22, 0.82, 0.90)),
        _make_point_cloud(o3d, anomaly_centers_xyz, color=(0.95, 0.84, 0.22)),
    ]
    geometries.extend(_make_edge_markers(o3d, left_edge_xyz, color=(1.0, 1.0, 1.0), radius=0.30))
    geometries.extend(_make_edge_markers(o3d, right_edge_xyz, color=(1.0, 0.92, 0.25), radius=0.30))
    geometries.extend(_make_edge_markers(o3d, anomaly_centers_xyz, color=(0.98, 0.84, 0.18), radius=0.42))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title, width=1180, height=860, left=120, top=80)
    for geometry in geometries:
        vis.add_geometry(geometry)
    _style_viewer(vis)
    _apply_shared_view(vis, background_xyz if len(background_xyz) else seam_xyz)
    while True:
        alive = vis.poll_events()
        vis.update_renderer()
        if not alive:
            break
    vis.destroy_window()


def _style_viewer(vis: Any) -> None:
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.04, 0.06, 0.10], dtype=np.float64)
    render_option.point_size = 3.0
    render_option.show_coordinate_frame = True


def _apply_shared_view(vis: Any, xyz: np.ndarray) -> None:
    xyz = _finite_xyz(xyz)
    if len(xyz) == 0:
        return
    control = vis.get_view_control()
    center = xyz.mean(axis=0).astype(np.float64)
    extent = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    radius = float(np.max(extent)) if np.any(np.isfinite(extent)) else 1.0
    control.set_front(np.asarray([0.45, -0.35, -0.82], dtype=np.float64))
    control.set_lookat(center)
    control.set_up(np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    control.set_zoom(max(0.08, min(0.65, 120.0 / max(radius, 1.0))))


def _make_point_cloud(o3d: Any, xyz: np.ndarray, color: tuple[float, float, float]):
    xyz = _finite_xyz(xyz)
    point_cloud = o3d.geometry.PointCloud()
    if len(xyz):
        point_cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        colors = np.tile(np.asarray(color, dtype=np.float64).reshape(1, 3), (len(xyz), 1))
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def _make_edge_markers(o3d: Any, xyz_points: np.ndarray, color: tuple[float, float, float], radius: float) -> list[Any]:
    markers: list[Any] = []
    for xyz in _finite_xyz(np.asarray(xyz_points, dtype=np.float32).reshape(-1, 3)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
        sphere.translate(np.asarray(xyz, dtype=np.float64))
        markers.append(sphere)
    return markers


def _collect_seam_xyz(seam_direction: dict[str, Any], point_map: np.ndarray) -> np.ndarray:
    pixels_xy = np.asarray(seam_direction.get("component_pixels", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    if len(pixels_xy) == 0 or point_map.ndim != 3 or point_map.shape[2] != 3:
        return np.empty((0, 3), dtype=np.float32)
    xs = np.clip(np.round(pixels_xy[:, 0]).astype(np.int32), 0, point_map.shape[1] - 1)
    ys = np.clip(np.round(pixels_xy[:, 1]).astype(np.int32), 0, point_map.shape[0] - 1)
    return _finite_xyz(point_map[ys, xs].reshape(-1, 3))


def _collect_nested_xyz(section_results: list[dict[str, Any]], parent_key: str, child_key: str) -> np.ndarray:
    chunks = []
    for item in section_results:
        points = item.get(parent_key, {}).get(child_key, {})
        xyz = _finite_xyz(points.get("xyz", np.empty((0, 3), dtype=np.float32)))
        if len(xyz):
            chunks.append(xyz)
    return np.concatenate(chunks, axis=0).astype(np.float32) if chunks else np.empty((0, 3), dtype=np.float32)


def _collect_nested_edge_xyz(section_results: list[dict[str, Any]], parent_key: str, child_key: str) -> np.ndarray:
    points = []
    for item in section_results:
        xyz = _finite_xyz(item.get(parent_key, {}).get(child_key, {}).get("xyz", np.empty((0, 3), dtype=np.float32)))
        if len(xyz):
            points.append(xyz[0])
    return np.asarray(points, dtype=np.float32).reshape(-1, 3) if points else np.empty((0, 3), dtype=np.float32)


def _split_section_centers(sections: list[dict[str, Any]], anomaly_indices: set[int]) -> tuple[np.ndarray, np.ndarray]:
    normal_points = []
    anomaly_points = []
    for section in sections:
        sample_index = int(section.get("sample_index", -1))
        center_xyz = np.asarray(section.get("center_xyz", [np.nan, np.nan, np.nan]), dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(center_xyz)):
            continue
        if sample_index in anomaly_indices:
            anomaly_points.append(center_xyz)
        else:
            normal_points.append(center_xyz)
    normal_xyz = np.asarray(normal_points, dtype=np.float32).reshape(-1, 3) if normal_points else np.empty((0, 3), dtype=np.float32)
    anomaly_xyz = np.asarray(anomaly_points, dtype=np.float32).reshape(-1, 3) if anomaly_points else np.empty((0, 3), dtype=np.float32)
    return normal_xyz, anomaly_xyz


def _collect_anomaly_sample_indices(section_profile: list[dict[str, Any]]) -> list[int]:
    anomaly = []
    gap_values = np.asarray(
        [float(item.get("gap", np.nan)) for item in section_profile if bool(item.get("valid", False)) and np.isfinite(item.get("gap", np.nan))],
        dtype=np.float32,
    )
    flush_values = np.asarray(
        [float(item.get("flush", np.nan)) for item in section_profile if bool(item.get("valid", False)) and np.isfinite(item.get("flush", np.nan))],
        dtype=np.float32,
    )
    gap_mean = float(np.mean(gap_values)) if len(gap_values) else np.nan
    gap_std = float(np.std(gap_values)) if len(gap_values) else np.nan
    flush_mean = float(np.mean(flush_values)) if len(flush_values) else np.nan
    flush_std = float(np.std(flush_values)) if len(flush_values) else np.nan

    for item in section_profile:
        sample_index = int(item.get("sample_index", -1))
        if not bool(item.get("valid", False)):
            anomaly.append(sample_index)
            continue
        gap = float(item.get("gap", np.nan))
        flush = float(item.get("flush", np.nan))
        gap_outlier = np.isfinite(gap_std) and gap_std > 1e-6 and np.isfinite(gap) and abs(gap - gap_mean) > 2.0 * gap_std
        flush_outlier = np.isfinite(flush_std) and flush_std > 1e-6 and np.isfinite(flush) and abs(flush - flush_mean) > 2.0 * flush_std
        if gap_outlier or flush_outlier:
            anomaly.append(sample_index)
    return sorted(set(anomaly))


def _finite_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if len(xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return xyz[np.all(np.isfinite(xyz), axis=1)]


def _subsample_xyz(xyz: np.ndarray, max_points: int) -> np.ndarray:
    xyz = _finite_xyz(xyz)
    if len(xyz) <= max_points or max_points <= 0:
        return xyz
    indices = np.linspace(0, len(xyz) - 1, num=max_points, dtype=np.int32)
    return xyz[indices]
