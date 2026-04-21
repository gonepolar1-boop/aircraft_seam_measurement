"""图 3.1 深度图构造示例.

从 point_map 读取 (H, W, 3) → 取 z 通道 → 归一化到 [0, 255] uint8。
以 3 联图展示：(a) z 原始值, (b) 有效掩膜, (c) 归一化深度图。
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

from pipeline.seam_mapping.io import load_point_map  # noqa: E402

from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402,F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 3.1 深度图构造示例.")
    parser.add_argument(
        "--point-map",
        type=Path,
        default=PROJECT_ROOT / "data" / "process" / "manual_crop" / "1" / "crop.pcd",
        help="Path to a .pcd / .npy / .npz point map.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch3_fig1_depth_image_demo.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point_map = load_point_map(args.point_map)
    z = point_map[:, :, 2]
    finite_mask = np.isfinite(z)

    normalised = np.zeros(z.shape, dtype=np.uint8)
    if finite_mask.any():
        z_valid = z[finite_mask]
        z_min, z_max = float(z_valid.min()), float(z_valid.max())
        if z_max > z_min:
            normalised[finite_mask] = np.clip(
                (z[finite_mask] - z_min) / (z_max - z_min) * 255.0, 0, 255
            ).astype(np.uint8)
        else:
            normalised[finite_mask] = 255

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2))

    axes[0].imshow(np.where(finite_mask, z, np.nan), cmap="viridis")
    axes[0].set_title("(a) 原始深度 z (mm)")

    axes[1].imshow(finite_mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("(b) 有效像素掩膜")

    axes[2].imshow(normalised, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title(f"(c) 归一化深度图 (uint8)  z∈[{z_valid.min():.2f}, {z_valid.max():.2f}] mm"
                      if finite_mask.any() else "(c) 归一化深度图 (uint8)")

    for ax in axes:
        ax.set_xlabel("列 / 像素")
        ax.set_ylabel("行 / 像素")

    fig.suptitle(f"来源: {args.point_map.name}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
