from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

from .viewer3d import save_gap_flush_viewer_bundle

PROFILE_FIELDNAMES = [
    "sample_index",
    "distance_mm",
    "valid",
    "reason",
    "gap",
    "gap_unit",
    "flush",
    "flush_unit",
    "center_x",
    "center_y",
    "left_u",
    "left_z",
    "right_u",
    "right_z",
]


def save_pipeline_outputs(
    *,
    output_dir: str | Path,
    result: dict[str, Any],
    measurement_result: dict[str, Any] | None = None,
    save_profile_plots: bool = True,
    save_viewer_bundle: bool = False,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    section_profile_csv = output_dir / "section_profile.csv"
    gap_profile_png = output_dir / "gap_profile.png"
    flush_profile_png = output_dir / "flush_profile.png"
    summary_json = output_dir / "summary.json"
    depth_overlay_png = output_dir / "depth_section_overlay.png"
    section_debug_dir = output_dir / "section_debug"
    viewer_bundle_npz = output_dir / "viewer_bundle.npz"
    write_section_profile_csv(section_profile_csv, result["section_profile"])
    if save_profile_plots:
        save_metric_profile_plot(
            save_path=gap_profile_png,
            section_profile=result["section_profile"],
            metric_key="gap",
            title="Gap Profile",
            ylabel="gap (mm)",
            color="#f59e0b",
            mean_value=float(result["summary"].get("gap_mean", np.nan)),
        )
        save_metric_profile_plot(
            save_path=flush_profile_png,
            section_profile=result["section_profile"],
            metric_key="flush",
            title="Flush Profile",
            ylabel="flush (mm)",
            color="#22c55e",
            mean_value=float(result["summary"].get("flush_mean", np.nan)),
        )
    if save_profile_plots and measurement_result is not None:
        save_depth_overlay_plot(
            save_path=depth_overlay_png,
            point_map=np.asarray(measurement_result.get("point_map", np.empty((0, 0, 3), dtype=np.float32)), dtype=np.float32),
            seam_direction=measurement_result.get("seam_direction", {}),
            sections=measurement_result.get("sections", []),
            section_results=measurement_result.get("section_results", []),
            section_profile=result["section_profile"],
        )
        save_section_debug_detail_plots(
            save_dir=section_debug_dir,
            section_results=measurement_result.get("section_results", []),
            section_profile=result["section_profile"],
        )
    if save_viewer_bundle and measurement_result is not None:
        save_gap_flush_viewer_bundle(
            save_path=viewer_bundle_npz,
            measurement_result=measurement_result,
            section_profile=result["section_profile"],
        )
    write_summary_json(summary_json, result)

    exports = {
        "section_profile_csv": str(section_profile_csv),
        "summary_json": str(summary_json),
    }
    if save_profile_plots:
        exports["gap_profile_png"] = str(gap_profile_png)
        exports["flush_profile_png"] = str(flush_profile_png)
    if save_profile_plots and measurement_result is not None:
        exports["depth_overlay_png"] = str(depth_overlay_png)
        exports["section_debug_dir"] = str(section_debug_dir)
    if save_viewer_bundle and measurement_result is not None:
        exports["viewer_bundle_npz"] = str(viewer_bundle_npz)
    return exports


def write_section_profile_csv(csv_path: str | Path, section_profile: list[dict[str, Any]]) -> None:
    with Path(csv_path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PROFILE_FIELDNAMES)
        writer.writeheader()
        for item in section_profile:
            writer.writerow(
                {
                    "sample_index": int(item["sample_index"]),
                    "distance_mm": _format_metric(item.get("distance_mm", np.nan)),
                    "valid": int(bool(item["valid"])),
                    "reason": str(item["reason"]),
                    "gap": _format_metric(item["gap"]),
                    "gap_unit": "mm",
                    "flush": _format_metric(item["flush"]),
                    "flush_unit": "mm",
                    "center_x": _format_metric(item["center_x"]),
                    "center_y": _format_metric(item["center_y"]),
                    "left_u": _format_metric(item["left_u"]),
                    "left_z": _format_metric(item["left_z"]),
                    "right_u": _format_metric(item["right_u"]),
                    "right_z": _format_metric(item["right_z"]),
                }
            )


def save_metric_profile_plot(
    *,
    save_path: str | Path,
    section_profile: list[dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    color: str,
    mean_value: float,
) -> None:
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    distance_mm = np.asarray([float(item.get("distance_mm", np.nan)) for item in section_profile], dtype=np.float32)
    sample_indices = np.asarray([int(item["sample_index"]) for item in section_profile], dtype=np.int32)
    metric_values = np.asarray([float(item[metric_key]) for item in section_profile], dtype=np.float32)
    valid_mask = np.asarray(
        [bool(item["valid"]) and np.isfinite(item[metric_key]) for item in section_profile],
        dtype=bool,
    )
    x_values = distance_mm if np.any(np.isfinite(distance_mm)) else sample_indices.astype(np.float32)
    xlabel = "distance along seam (mm)" if np.any(np.isfinite(distance_mm)) else "section sample index"

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if len(x_values):
        if np.any(valid_mask):
            ax.plot(x_values[valid_mask], metric_values[valid_mask], color=color, linewidth=1.6, alpha=0.8)
            ax.scatter(x_values[valid_mask], metric_values[valid_mask], color=color, s=24, alpha=0.95, label="valid")
        invalid_x = x_values[~valid_mask]
        if len(invalid_x):
            # Use NaN (matplotlib drops them) rather than 0.0 - plotting a
            # row of zeros along the x axis misleads the reader into thinking
            # zero was measured at those sections.
            invalid_y = np.full((len(invalid_x),), mean_value if np.isfinite(mean_value) else np.nan, dtype=np.float32)
            ax.scatter(invalid_x, invalid_y, color="#94a3b8", marker="x", s=24, alpha=0.85, label="invalid")
        if np.isfinite(mean_value):
            ax.axhline(mean_value, color="#475569", linestyle="--", linewidth=1.2, alpha=0.9, label=f"mean={mean_value:.4f}")
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def write_summary_json(summary_json_path: str | Path, result: dict[str, Any]) -> None:
    with Path(summary_json_path).open("w", encoding="utf-8") as file:
        json.dump(_json_ready(result), file, ensure_ascii=False, indent=2)


def save_depth_overlay_plot(
    *,
    save_path: str | Path,
    point_map: np.ndarray,
    seam_direction: dict[str, Any],
    sections: list[dict[str, Any]],
    section_results: list[dict[str, Any]],
    section_profile: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    point_map = np.asarray(point_map, dtype=np.float32)
    if point_map.ndim != 3 or point_map.shape[2] < 3:
        return

    z_map = point_map[:, :, 2]
    finite_mask = np.isfinite(z_map)
    if not np.any(finite_mask):
        return

    display = np.zeros_like(z_map, dtype=np.float32)
    z_min = float(np.min(z_map[finite_mask]))
    z_max = float(np.max(z_map[finite_mask]))
    if z_max > z_min:
        display[finite_mask] = (z_map[finite_mask] - z_min) / (z_max - z_min)
    anomaly_indices = _collect_anomaly_sample_indices(section_results, section_profile)
    section_result_index = {int(item.get("sample_index", -1)): item for item in section_results}

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.imshow(display, cmap="gray", origin="upper")
    seam_pixels = np.asarray(seam_direction.get("component_pixels", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
    if len(seam_pixels):
        ax.scatter(seam_pixels[:, 0], seam_pixels[:, 1], s=1.5, c="#22d3ee", alpha=0.35, linewidths=0)

    for section in sections:
        center_xy = np.asarray(section.get("center_xy", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
        normal_xy = np.asarray(section.get("normal_xy", [np.nan, np.nan]), dtype=np.float32).reshape(-1)
        sample_index = int(section.get("sample_index", -1))
        if len(center_xy) < 2 or not np.all(np.isfinite(center_xy[:2])):
            continue
        color = "#ef4444" if sample_index in anomaly_indices else "#84cc16"
        ax.scatter([float(center_xy[0])], [float(center_xy[1])], s=18 if sample_index in anomaly_indices else 10, c=color, alpha=0.95, linewidths=0)
        if len(normal_xy) >= 2 and np.all(np.isfinite(normal_xy[:2])):
            half_len = 6.0
            dx = float(normal_xy[0]) * half_len
            dy = float(normal_xy[1]) * half_len
            ax.plot(
                [float(center_xy[0]) - dx, float(center_xy[0]) + dx],
                [float(center_xy[1]) - dy, float(center_xy[1]) + dy],
                color=color,
                linewidth=0.8,
                alpha=0.8,
            )
        if sample_index in anomaly_indices:
            item = section_result_index.get(sample_index, {})
            measurement = item.get("measurement", {})
            label = _format_section_tag(sample_index, _distance_for_index(section_profile, sample_index))
            if bool(measurement.get("valid", False)):
                label += f"\nG={float(measurement.get('gap', np.nan)):.3f} F={float(measurement.get('flush', np.nan)):.3f}"
            ax.text(
                float(center_xy[0]) + 4.0,
                float(center_xy[1]) + 4.0,
                label,
                fontsize=6,
                color="#fee2e2",
                bbox={"facecolor": "#111827", "edgecolor": "#ef4444", "alpha": 0.8, "pad": 2},
            )

    ax.set_title("Depth Overlay With Seam and Section Positions")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_section_debug_detail_plots(
    *,
    save_dir: str | Path,
    section_results: list[dict[str, Any]],
    section_profile: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    anomaly_indices = set(_collect_anomaly_sample_indices(section_results, section_profile))
    if not anomaly_indices:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    distance_lookup = {int(item.get("sample_index", -1)): float(item.get("distance_mm", np.nan)) for item in section_profile}
    for item in section_results:
        sample_index = int(item.get("sample_index", -1))
        if sample_index not in anomaly_indices:
            continue
        save_path = save_dir / f"section_{sample_index:04d}.png"
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
        _draw_section_debug_axes(axes, item, float(distance_lookup.get(sample_index, np.nan)))
        fig.tight_layout()
        fig.savefig(save_path, dpi=180)
        plt.close(fig)


def _draw_section_debug_axes(axes: Any, item: dict[str, Any], distance_mm: float) -> None:
    left_ax, right_ax = axes
    for ax in (left_ax, right_ax):
        ax.set_facecolor("#0f172a")
        ax.axvline(0.0, color="#e2e8f0", linewidth=1.0, alpha=0.45, linestyle="--")
        ax.grid(True, color="#475569", alpha=0.2)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#64748b")
            spine.set_alpha(0.45)

    filtered = item.get("filtered_points", {})
    top_surface = item.get("top_surface", {})
    measurement = item.get("measurement", {})

    _scatter_points(left_ax, filtered, "#64748b", 0.45, "filtered", 8)
    _scatter_points(left_ax, top_surface.get("left_candidates", {}), "#22c55e", 0.35, "left candidates", 10)
    _scatter_points(left_ax, top_surface.get("left_segment", {}), "#86efac", 0.95, "left segment", 16)
    _scatter_points(left_ax, top_surface.get("left_edge", {}), "#f8fafc", 1.0, "left edge", 44)

    _scatter_points(right_ax, filtered, "#64748b", 0.45, "filtered", 8)
    _scatter_points(right_ax, top_surface.get("right_candidates", {}), "#ef4444", 0.35, "right candidates", 10)
    _scatter_points(right_ax, top_surface.get("right_segment", {}), "#fca5a5", 0.95, "right segment", 16)
    _scatter_points(right_ax, top_surface.get("right_edge", {}), "#f8fafc", 1.0, "right edge", 44)

    left_ax.set_title("Left Surface", fontsize=9)
    right_ax.set_title("Right Surface", fontsize=9)
    left_ax.set_ylabel("world z", color="#e2e8f0")
    left_ax.set_xlabel("local u", color="#e2e8f0")
    right_ax.set_xlabel("local u", color="#e2e8f0")
    left_ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="#e2e8f0")
    right_ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="#e2e8f0")

    label = _format_section_tag(int(item.get("sample_index", -1)), distance_mm)
    if bool(measurement.get("valid", False)):
        label += "\n" + f"gap={float(measurement.get('gap', np.nan)):.3f} mm"
        label += "\n" + f"flush={float(measurement.get('flush', np.nan)):.3f} mm"
    else:
        label += "\n" + f"invalid: {str(measurement.get('reason', 'unknown'))}"
    left_ax.text(
        0.02,
        0.98,
        label,
        transform=left_ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="#e2e8f0",
        bbox={"facecolor": "#0f172a", "edgecolor": "#475569", "alpha": 0.85, "pad": 3},
    )


def _scatter_points(ax: Any, points: dict[str, Any], color: str, alpha: float, label: str, size: float) -> None:
    if not points or len(points.get("u", [])) == 0:
        return
    ax.scatter(points["u"], points["z"], s=size, c=color, alpha=alpha, label=label)


def _collect_anomaly_sample_indices(section_results: list[dict[str, Any]], section_profile: list[dict[str, Any]]) -> list[int]:
    anomaly = []
    gap_values = np.asarray([float(item.get("gap", np.nan)) for item in section_profile if bool(item.get("valid", False)) and np.isfinite(item.get("gap", np.nan))], dtype=np.float32)
    flush_values = np.asarray([float(item.get("flush", np.nan)) for item in section_profile if bool(item.get("valid", False)) and np.isfinite(item.get("flush", np.nan))], dtype=np.float32)
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

    if anomaly:
        return sorted(set(anomaly))

    scores: list[tuple[float, int]] = []
    for item in section_profile:
        if not bool(item.get("valid", False)):
            continue
        sample_index = int(item.get("sample_index", -1))
        score = 0.0
        gap = float(item.get("gap", np.nan))
        flush = float(item.get("flush", np.nan))
        if np.isfinite(gap_std) and gap_std > 1e-6 and np.isfinite(gap):
            score = max(score, abs(gap - gap_mean) / gap_std)
        if np.isfinite(flush_std) and flush_std > 1e-6 and np.isfinite(flush):
            score = max(score, abs(flush - flush_mean) / flush_std)
        scores.append((score, sample_index))
    scores.sort(reverse=True)
    return [sample_index for _score, sample_index in scores[: min(6, len(scores))]]


def _distance_for_index(section_profile: list[dict[str, Any]], sample_index: int) -> float:
    for item in section_profile:
        if int(item.get("sample_index", -1)) == int(sample_index):
            return float(item.get("distance_mm", np.nan))
    return np.nan


def _format_section_tag(sample_index: int, distance_mm: float) -> str:
    if np.isfinite(distance_mm):
        return f"section {sample_index} | {distance_mm:.3f} mm"
    return f"section {sample_index}"


def _format_metric(value: Any) -> str:
    return "" if not np.isfinite(value) else f"{float(value):.6f}"


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value
