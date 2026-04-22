"""图 4.4 顶面检测多阶段可视化.

对单个截面运行 ``detect_top_surface_edges`` 并把 payload 中的中间结果
按 2×3 的顺序画在 u-z 坐标系里：
(a) 背景点 → (b) 邻域过滤 → (c) 顶面候选 → (d) 选中段 → (e) 边缘点
 (+ 左右拟合直线).
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
from pipeline.seam_measurement.sections import extract_seam_direction, extract_sections  # noqa: E402
from pipeline.seam_measurement.top_surface import detect_top_surface_edges  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 4.4 顶面检测多阶段可视化.")
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
    parser.add_argument("--section-index", type=int, default=-1,
                        help="Section to visualise; -1 picks the middle section.")
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch4_fig4_top_surface_stages.png",
    )
    return parser.parse_args()


def _scatter(ax, payload, key, colour, size, label, marker="o"):
    points = payload.get(key)
    if not points or len(points.get("u", [])) == 0:
        return
    ax.scatter(points["u"], points["z"], c=colour, s=size, marker=marker,
               alpha=0.85, linewidths=0, label=label)


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
        raise RuntimeError("No sections were extracted - cannot plot top-surface stages.")
    index = len(sections) // 2 if args.section_index < 0 else min(int(args.section_index), len(sections) - 1)
    section = sections[index]
    payload = detect_top_surface_edges(section, params)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes = axes.flatten()

    _scatter(axes[0], payload, "background_points", PALETTE["muted"], 6, "背景点")
    axes[0].set_title("(a) 背景点 (完整切片)")

    _scatter(axes[1], payload, "background_points", PALETTE["muted"], 4, "背景点")
    _scatter(axes[1], payload, "local_background", "#38bdf8", 6, "局部窗口")
    axes[1].set_title("(b) 局部背景窗口")

    _scatter(axes[2], payload, "local_background", PALETTE["muted"], 4, "局部背景")
    _scatter(axes[2], payload, "neighbor_filtered", PALETTE["primary"], 6, "邻域保留")
    axes[2].set_title("(c) 邻域过滤后")

    _scatter(axes[3], payload, "neighbor_filtered", PALETTE["muted"], 4, "邻域保留")
    _scatter(axes[3], payload, "left_candidates", PALETTE["accent"], 8, "左顶面候选")
    _scatter(axes[3], payload, "right_candidates", PALETTE["secondary"], 8, "右顶面候选")
    axes[3].set_title("(d) 顶面候选")

    _scatter(axes[4], payload, "left_candidates", PALETTE["muted"], 4, "左候选")
    _scatter(axes[4], payload, "right_candidates", PALETTE["muted"], 4, "右候选")
    _scatter(axes[4], payload, "left_segment", PALETTE["accent"], 10, "左选中段")
    _scatter(axes[4], payload, "right_segment", PALETTE["secondary"], 10, "右选中段")
    axes[4].set_title("(e) 中心最近段")

    _scatter(axes[5], payload, "left_segment", PALETTE["muted"], 4, "左段")
    _scatter(axes[5], payload, "right_segment", PALETTE["muted"], 4, "右段")
    _scatter(axes[5], payload, "left_edge", PALETTE["accent"], 72, "左边缘点", marker="*")
    _scatter(axes[5], payload, "right_edge", PALETTE["secondary"], 72, "右边缘点", marker="*")

    for label, colour, fit_key in (("左拟合线", PALETTE["accent"], "left_model"),
                                   ("右拟合线", PALETTE["secondary"], "right_model")):
        fit = payload.get(fit_key, {})
        if bool(fit.get("valid", False)):
            u_vals = np.linspace(-10, 10, 50, dtype=np.float32)
            z_vals = float(fit["slope"]) * u_vals + float(fit["intercept"])
            axes[5].plot(u_vals, z_vals, color=colour, linestyle="--", linewidth=1.4, alpha=0.8, label=label)

    axes[5].set_title("(f) 边缘点 + 拟合线")

    for ax in axes:
        ax.set_xlabel("u / px (局部)")
        ax.set_ylabel("z / mm (世界)")
        ax.axvline(0.0, color="#94a3b8", linewidth=0.9, linestyle=":", alpha=0.8)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(f"顶面检测多阶段（section #{index}, reason={payload.get('reason', 'ok')})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
