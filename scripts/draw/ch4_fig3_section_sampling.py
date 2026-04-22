"""图 4.3 截面采样示意.

跑一次 ``extract_sections`` 得到截面列表，把每个截面的中心 + 法向
短线叠加到灰度深度图上。用不同颜色区分有效/无效截面。
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

from pipeline.seam_mapping.inference import build_depth_image_from_point_map, predict_mask_from_point_map  # noqa: E402
from pipeline.seam_mapping.io import load_point_map  # noqa: E402
from pipeline.seam_measurement import GapFlushParams  # noqa: E402
from pipeline.seam_measurement.sections import extract_seam_direction, extract_sections  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 4.3 截面采样示意.")
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
    parser.add_argument("--seam-step", type=int, default=8,
                        help="Sub-sampling step - larger value = fewer sections drawn.")
    parser.add_argument("--half-len-px", type=float, default=6.0,
                        help="Drawn half-length of each section normal line (px).")
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch4_fig3_section_sampling.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_map = load_point_map(args.point_map)
    depth_image, _ = build_depth_image_from_point_map(point_map)
    prediction = predict_mask_from_point_map(
        point_map=point_map,
        checkpoint_path=args.checkpoint,
        threshold=float(args.threshold),
    )
    pred_mask = prediction["pred_mask"]
    seam_direction = extract_seam_direction(pred_mask > 0)
    params = GapFlushParams(seam_step=int(args.seam_step))
    sections = extract_sections(pred_mask > 0, point_map, seam_direction=seam_direction, params=params)

    fig, ax = plt.subplots(figsize=(10.0, 7.5))
    ax.imshow(depth_image, cmap="gray")

    seam_pixels = seam_direction["component_pixels"]
    ax.scatter(seam_pixels[:, 0], seam_pixels[:, 1], s=0.5, c="#38bdf8", alpha=0.35, linewidths=0)

    half_len = float(args.half_len_px)
    valid_count = 0
    invalid_count = 0
    for section in sections:
        center_xy = np.asarray(section["center_xy"], dtype=np.float32)
        normal_xy = np.asarray(section["normal_xy"], dtype=np.float32)
        if not np.all(np.isfinite(center_xy)):
            continue
        is_valid = bool(np.any(section["valid"]))
        colour = PALETTE["accent"] if is_valid else PALETTE["secondary"]
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
        ax.plot([center_xy[0]], [center_xy[1]], marker="o", color=colour,
                markersize=3.2, alpha=0.95, markeredgewidth=0)
        ax.plot(
            [center_xy[0] - normal_xy[0] * half_len, center_xy[0] + normal_xy[0] * half_len],
            [center_xy[1] - normal_xy[1] * half_len, center_xy[1] + normal_xy[1] * half_len],
            color=colour,
            linewidth=0.6,
            alpha=0.85,
        )

    ax.scatter([], [], s=24, c=PALETTE["accent"], label=f"有效截面 ({valid_count})")
    ax.scatter([], [], s=24, c=PALETTE["secondary"], label=f"无效截面 ({invalid_count})")
    ax.set_title(f"截面采样示意（seam_step={args.seam_step} px, 共 {len(sections)} 截面）")
    ax.set_xlabel("列 / 像素")
    ax.set_ylabel("行 / 像素")
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
