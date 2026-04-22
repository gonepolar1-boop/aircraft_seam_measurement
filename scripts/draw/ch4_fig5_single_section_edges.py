"""图 4.5 单截面边缘定位.

单个截面的 u-z 散点 + 左右拟合线 + 边缘点 + gap/flush 辅助标注。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402

from pipeline.seam_mapping.inference import predict_mask_from_point_map  # noqa: E402
from pipeline.seam_mapping.io import load_point_map  # noqa: E402
from pipeline.seam_measurement import GapFlushParams  # noqa: E402
from pipeline.seam_measurement.bottom import compute_section_bottom  # noqa: E402
from pipeline.seam_measurement.measurements import compute_section_gap_flush  # noqa: E402
from pipeline.seam_measurement.sections import extract_seam_direction, extract_sections  # noqa: E402
from pipeline.seam_measurement.top_surface import detect_top_surface_edges  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 4.5 单截面边缘定位.")
    parser.add_argument(
        "--point-map",
        type=Path,
        default=PROJECT_ROOT / "data" / "process" / "manual_crop" / "1" / "crop.pcd",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model" / "03301416_loocv_attention_unet"
            / "holdout_1" / "checkpoints" / "best.pth",
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--section-index", type=int, default=-1)
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch4_fig5_single_section_edges.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_map = load_point_map(args.point_map)
    prediction = predict_mask_from_point_map(
        point_map=point_map, checkpoint_path=args.checkpoint, threshold=float(args.threshold),
    )
    pred_mask = prediction["pred_mask"]
    seam_direction = extract_seam_direction(pred_mask > 0)
    params = GapFlushParams()
    sections = extract_sections(pred_mask > 0, point_map, seam_direction=seam_direction, params=params)
    if not sections:
        raise RuntimeError("No sections were extracted.")

    index = len(sections) // 2 if args.section_index < 0 else min(int(args.section_index), len(sections) - 1)
    section = sections[index]
    item = compute_section_bottom(section, params)
    item["top_surface"] = detect_top_surface_edges(section, params)
    item["measurement"] = compute_section_gap_flush(item)

    top_surface = item["top_surface"]
    measurement = item["measurement"]
    filtered = item.get("filtered_points", {})

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    if len(filtered.get("u", [])):
        ax.scatter(filtered["u"], filtered["z"], s=6, c=PALETTE["muted"], alpha=0.5, label="有效点")

    for label, colour, seg_key, edge_key, model_key in (
        ("左顶面", PALETTE["accent"], "left_segment", "left_edge", "left_model"),
        ("右顶面", PALETTE["secondary"], "right_segment", "right_edge", "right_model"),
    ):
        segment = top_surface.get(seg_key, {})
        edge = top_surface.get(edge_key, {})
        fit = top_surface.get(model_key, {})
        if len(segment.get("u", [])):
            ax.scatter(segment["u"], segment["z"], s=14, c=colour, alpha=0.9, label=f"{label}选中段")
        if len(edge.get("u", [])):
            ax.scatter(edge["u"], edge["z"], s=110, c=colour, marker="*", edgecolors="white", linewidths=0.8,
                       label=f"{label}边缘点")
        if bool(fit.get("valid", False)):
            u_span = np.linspace(-20, 20, 80, dtype=np.float32)
            z_vals = float(fit["slope"]) * u_span + float(fit["intercept"])
            ax.plot(u_span, z_vals, color=colour, linestyle="--", linewidth=1.5, alpha=0.85)

    # Annotate gap / flush.
    left_pt = measurement.get("left_point", {})
    right_pt = measurement.get("right_point", {})
    gap = float(measurement.get("gap", np.nan))
    flush = float(measurement.get("flush", np.nan))
    if np.isfinite(gap) and np.all([np.isfinite(left_pt.get("u", np.nan)), np.isfinite(right_pt.get("u", np.nan))]):
        ax.annotate(
            "",
            xy=(right_pt["u"], max(left_pt["z"], right_pt["z"]) + 0.04),
            xytext=(left_pt["u"], max(left_pt["z"], right_pt["z"]) + 0.04),
            arrowprops=dict(arrowstyle="<->", color="#0f172a", lw=1.6),
        )
        ax.text(
            0.5 * (left_pt["u"] + right_pt["u"]),
            max(left_pt["z"], right_pt["z"]) + 0.08,
            f"gap = {gap:.3f} mm",
            ha="center",
            fontsize=10,
            color="#0f172a",
        )
    if np.isfinite(flush) and np.isfinite(left_pt.get("z", np.nan)) and np.isfinite(right_pt.get("z", np.nan)):
        ax.annotate(
            "",
            xy=(right_pt["u"], right_pt["z"]),
            xytext=(right_pt["u"], left_pt["z"]),
            arrowprops=dict(arrowstyle="<->", color=PALETTE["purple"], lw=1.4),
        )
        ax.text(
            right_pt["u"] + 0.2,
            0.5 * (left_pt["z"] + right_pt["z"]),
            f"flush = {flush:.3f} mm",
            color=PALETTE["purple"],
            fontsize=10,
        )

    ax.axvline(0.0, color="#94a3b8", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.set_xlabel("u / px (局部)")
    ax.set_ylabel("z / mm (世界)")
    ax.set_title(f"单截面边缘定位（section #{index}, reason={measurement.get('reason', 'ok')})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
