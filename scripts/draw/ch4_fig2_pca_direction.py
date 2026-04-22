"""图 4.2 PCA 主方向可视化.

在预测 mask 上运行 ``extract_seam_direction``，把主轴箭头（切向）
与法向箭头叠加到灰度深度图上。
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
from pipeline.seam_measurement.sections import extract_seam_direction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 4.2 PCA 主方向可视化.")
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
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch4_fig2_pca_direction.png",
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

    component = seam_direction["component_pixels"]
    centre = seam_direction["principal_center_xy"]
    tangent = seam_direction["principal_tangent_xy"]
    normal = seam_direction["principal_normal_xy"]
    extent_t = float(np.max(np.abs(seam_direction["component_t"])))
    extent_n = float(np.max(np.abs(seam_direction["component_n"])))

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.imshow(depth_image, cmap="gray")
    ax.scatter(component[:, 0], component[:, 1], s=0.4, c=PALETTE["primary"], alpha=0.35, linewidths=0, label="缝隙 mask")

    # Tangent (long axis).
    ax.annotate(
        "",
        xy=(centre[0] + tangent[0] * extent_t, centre[1] + tangent[1] * extent_t),
        xytext=(centre[0] - tangent[0] * extent_t, centre[1] - tangent[1] * extent_t),
        arrowprops=dict(arrowstyle="<|-|>", color=PALETTE["secondary"], lw=2.2),
    )
    ax.plot(centre[0], centre[1], "o", color=PALETTE["secondary"], markersize=9, label="主成分中心")
    ax.text(
        centre[0] + tangent[0] * extent_t * 1.05,
        centre[1] + tangent[1] * extent_t * 1.05,
        "主轴 (切向)",
        color=PALETTE["secondary"],
        fontsize=10,
    )

    # Normal arrow.
    ax.annotate(
        "",
        xy=(centre[0] + normal[0] * extent_n, centre[1] + normal[1] * extent_n),
        xytext=(centre[0], centre[1]),
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["accent"], lw=2.0),
    )
    ax.text(
        centre[0] + normal[0] * extent_n * 1.05,
        centre[1] + normal[1] * extent_n * 1.05,
        "法向",
        color=PALETTE["accent"],
        fontsize=10,
    )

    ax.set_title("PCA 主方向可视化：主轴与法向叠加到深度图")
    ax.set_xlabel("列 / 像素")
    ax.set_ylabel("行 / 像素")
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
