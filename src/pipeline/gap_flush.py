from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np

from .outputs import save_pipeline_outputs
from .seam_mapping.inference import predict_mask_from_point_map, preload_model
from .seam_mapping.io import load_point_map
from .seam_measurement import GapFlushParams, compute_gap_flush_from_mapping
from .seam_training.utils import REAL_ONLY_DEFAULTS
from .timing import StageTimer
from .viewer3d import show_gap_flush_open3d_viewer


def run_gap_flush_pipeline(
    pcd_path: str | Path,
    checkpoint_path: str | Path,
    *,
    threshold: float = REAL_ONLY_DEFAULTS["preview_threshold"],
    output_root: str | Path | None = None,
    params: GapFlushParams | None = None,
    fast_mode: bool = False,
    save_profile_plots: bool = True,
    save_viewer_bundle: bool = False,
    show_3d_viewer: bool = False,
) -> dict[str, Any]:
    params = GapFlushParams() if params is None else params
    resolved_pcd_path = Path(pcd_path).resolve()
    resolved_checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = _build_output_dir(resolved_pcd_path, output_root)
    timer = StageTimer()

    measurement_result, counts = _run_measurement(
        pcd_path=resolved_pcd_path,
        checkpoint_path=resolved_checkpoint_path,
        threshold=float(threshold),
        params=params,
        fast_mode=fast_mode,
        timer=timer,
    )
    with timer.stage("build_profile"):
        section_profile = _build_section_profile(measurement_result["section_results"], measurement_result.get("sections", []))
        summary = _build_summary(measurement_result["summary"], section_profile)
    result = {
        "inputs": {
            "pcd_path": str(resolved_pcd_path),
            "checkpoint_path": str(resolved_checkpoint_path),
            "threshold": float(threshold),
        },
        "sample_name": output_dir.name,
        "output_dir": str(output_dir),
        "summary": summary,
        "section_profile": section_profile,
        "counts": counts,
        "params": _params_payload(params),
    }
    with timer.stage("save_outputs"):
        result["exports"] = save_pipeline_outputs(
            output_dir=output_dir,
            result=result,
            measurement_result=measurement_result,
            save_profile_plots=save_profile_plots,
            save_viewer_bundle=save_viewer_bundle,
            outlier_sigma=float(params.outlier_sigma),
        )
    result["timing"] = timer.summary()
    if show_3d_viewer:
        show_gap_flush_open3d_viewer(
            measurement_result=measurement_result,
            section_profile=section_profile,
            window_title=f"Gap/Flush 3D Viewer - {output_dir.name}",
        )
    return result


def preload_pipeline_model(checkpoint_path: str | Path) -> None:
    preload_model(checkpoint_path)


def _build_output_dir(pcd_path: Path, output_root: str | Path | None) -> Path:
    root = Path(output_root) if output_root is not None else Path(__file__).resolve().parents[2] / "outputs" / "pipeline"
    return root / _build_sample_name(pcd_path)


def _build_sample_name(pcd_path: Path) -> str:
    parent_name = pcd_path.parent.name.strip()
    stem = pcd_path.stem.strip()
    return f"{parent_name}_{stem}" if parent_name else stem


def _run_measurement(
    *,
    pcd_path: Path,
    checkpoint_path: Path,
    threshold: float,
    params: GapFlushParams,
    fast_mode: bool,
    timer: StageTimer | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    timer = StageTimer() if timer is None else timer
    with timer.stage("load_point_map"):
        point_map = load_point_map(pcd_path)
    with timer.stage("predict_mask"):
        prediction = predict_mask_from_point_map(
            point_map=point_map,
            checkpoint_path=checkpoint_path,
            threshold=threshold,
        )
    pred_mask = np.asarray(prediction["pred_mask"], dtype=np.uint8)
    with timer.stage("compute_gap_flush"):
        measurement_result = compute_gap_flush_from_mapping(
            {
                "inputs": {
                    "pcd_path": str(pcd_path),
                    "checkpoint_path": str(checkpoint_path),
                    "threshold": float(threshold),
                },
                "point_map": point_map,
                "pred_mask": pred_mask,
            },
            params=params,
            fast_mode=fast_mode,
        )
    return measurement_result, _build_counts(point_map=point_map, pred_mask=pred_mask)


def _build_counts(*, point_map: np.ndarray, pred_mask: np.ndarray) -> dict[str, int]:
    mask = np.asarray(pred_mask, dtype=np.uint8) > 0
    valid_points = np.all(np.isfinite(np.asarray(point_map, dtype=np.float32)), axis=2)
    mask_pixel_count = int(np.count_nonzero(mask))
    valid_point_count = int(np.count_nonzero(mask & valid_points))
    return {
        "mask_pixel_count": mask_pixel_count,
        "valid_point_count": valid_point_count,
        "invalid_point_count": int(mask_pixel_count - valid_point_count),
    }


def _build_section_profile(section_results: list[dict[str, Any]], sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    distance_by_index = _build_distance_lookup(sections)
    section_profile: list[dict[str, Any]] = []
    for item in section_results:
        measurement = item.get("measurement", {})
        center_xy = np.asarray(item.get("center_xy", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
        sample_index = int(item.get("sample_index", -1))
        section_profile.append(
            {
                "sample_index": sample_index,
                "distance_mm": float(distance_by_index.get(sample_index, np.nan)),
                "valid": bool(measurement.get("valid", False)),
                "reason": str(measurement.get("reason", "unknown")),
                "gap": float(measurement.get("gap", np.nan)),
                "flush": float(measurement.get("flush", np.nan)),
                "center_x": float(center_xy[0]) if len(center_xy) > 0 else np.nan,
                "center_y": float(center_xy[1]) if len(center_xy) > 1 else np.nan,
                "left_u": float(measurement.get("left_point", {}).get("u", np.nan)),
                "left_z": float(measurement.get("left_point", {}).get("z", np.nan)),
                "right_u": float(measurement.get("right_point", {}).get("u", np.nan)),
                "right_z": float(measurement.get("right_point", {}).get("z", np.nan)),
            }
        )
    return section_profile


def _build_distance_lookup(sections: list[dict[str, Any]]) -> dict[int, float]:
    if not sections:
        return {}

    sample_indices = np.asarray([int(section.get("sample_index", -1)) for section in sections], dtype=np.int32)
    centers_xyz = np.asarray(
        [np.asarray(section.get("center_xyz", [np.nan, np.nan, np.nan]), dtype=np.float32).reshape(3) for section in sections],
        dtype=np.float32,
    )
    valid = np.all(np.isfinite(centers_xyz), axis=1)
    cumulative = np.full((len(sections),), np.nan, dtype=np.float32)

    last_valid_index: int | None = None
    running_distance = 0.0
    for index in range(len(sections)):
        if not bool(valid[index]):
            continue
        if last_valid_index is not None:
            delta = float(np.linalg.norm(centers_xyz[index] - centers_xyz[last_valid_index]))
            if np.isfinite(delta):
                running_distance += delta
        cumulative[index] = float(running_distance)
        last_valid_index = index

    valid_indices = np.flatnonzero(np.isfinite(cumulative))
    if len(valid_indices) == 1:
        cumulative[:] = cumulative[valid_indices[0]]
    elif len(valid_indices) >= 2:
        missing = np.flatnonzero(~np.isfinite(cumulative))
        if len(missing):
            cumulative[missing] = np.interp(missing.astype(np.float32), valid_indices.astype(np.float32), cumulative[valid_indices]).astype(np.float32)

    return {int(sample_index): float(distance_mm) for sample_index, distance_mm in zip(sample_indices, cumulative)}


def _build_summary(raw_summary: dict[str, Any], section_profile: list[dict[str, Any]]) -> dict[str, Any]:
    gap_values = np.asarray(
        [item["gap"] for item in section_profile if item["valid"] and np.isfinite(item["gap"])],
        dtype=np.float32,
    )
    flush_values = np.asarray(
        [item["flush"] for item in section_profile if item["valid"] and np.isfinite(item["flush"])],
        dtype=np.float32,
    )
    return {
        "unit": "mm",
        "num_sections": int(raw_summary.get("num_sections", len(section_profile))),
        "num_measurement_sections": int(raw_summary.get("num_measurement_sections", len(gap_values))),
        "gap_mean": float(raw_summary.get("gap_mean", np.nan)),
        "gap_mean_mm": float(raw_summary.get("gap_mean", np.nan)),
        "gap_std": float(raw_summary.get("gap_std", np.nan)),
        "gap_std_mm": float(raw_summary.get("gap_std", np.nan)),
        "gap_min": float(np.min(gap_values)) if len(gap_values) else np.nan,
        "gap_max": float(np.max(gap_values)) if len(gap_values) else np.nan,
        "flush_mean": float(raw_summary.get("flush_mean", np.nan)),
        "flush_mean_mm": float(raw_summary.get("flush_mean", np.nan)),
        "flush_std": float(raw_summary.get("flush_std", np.nan)),
        "flush_std_mm": float(raw_summary.get("flush_std", np.nan)),
        "flush_min": float(np.min(flush_values)) if len(flush_values) else np.nan,
        "flush_max": float(np.max(flush_values)) if len(flush_values) else np.nan,
    }


def _params_payload(params: GapFlushParams) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(params).items()
    }
