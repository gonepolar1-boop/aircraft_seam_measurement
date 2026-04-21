"""图 3.6 Patch 采样可视化.

在整张训练图上用红色框画出正样本中心 patch、再采若干随机 patch，
配合正样本概率热图便于读者理解 PatchSampler 的 positive-center 策略。
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seam_training.data import PatchSampler  # noqa: E402
from seam_training.utils import PATCH_SAMPLING_DEFAULTS  # noqa: E402

from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 3.6 Patch 采样可视化.")
    parser.add_argument(
        "--image",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "images" / "1.png",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "masks" / "1.png",
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "valids",
    )
    parser.add_argument("--num-patches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch3_fig6_patch_sampling.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError("Missing real_train sample image / mask.")

    cfg = dict(PATCH_SAMPLING_DEFAULTS)
    cfg["positive_only"] = True
    sampler = PatchSampler(cfg)
    sample_record = sampler.build_sample_record(args.image, args.mask, args.valid_dir)
    valid = cv2.imread(str(sample_record["valid_path"]), cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    # --- Left panel: overlay the drawn patches onto the grayscale image ---
    axes[0].imshow(image, cmap="gray")
    seam_ys, seam_xs = np.nonzero(mask > 127)
    axes[0].scatter(seam_xs, seam_ys, s=0.6, c=PALETTE["secondary"], alpha=0.35, linewidths=0, label="缝隙 mask")

    patch_colours = [PALETTE["primary"], PALETTE["accent"], PALETTE["highlight"], PALETTE["purple"], PALETTE["brown"]]
    for idx in range(args.num_patches):
        patch = sampler.sample_patch(image, mask, valid, sample=sample_record)
        box = patch["crop_box"]
        colour = patch_colours[idx % len(patch_colours)]
        rect = mpatches.Rectangle(
            (box["left"], box["top"]),
            box["right"] - box["left"],
            box["bottom"] - box["top"],
            linewidth=1.8,
            edgecolor=colour,
            facecolor="none",
            label=f"patch #{idx + 1}",
        )
        axes[0].add_patch(rect)
        cy, cx = patch["sample_center"]
        axes[0].plot([cx], [cy], marker="x", color=colour, markersize=9, markeredgewidth=1.8)

    axes[0].set_title("(a) Patch 采样窗口叠加到整图")
    axes[0].set_xlabel("列 / 像素")
    axes[0].set_ylabel("行 / 像素")
    axes[0].legend(loc="upper right", framealpha=0.85)

    # --- Right panel: heatmap of positive pixel density (gaussian-blurred mask) ---
    density = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0) / 255.0
    axes[1].imshow(image, cmap="gray", alpha=0.5)
    heat = axes[1].imshow(density, cmap="hot", alpha=0.55)
    axes[1].set_title("(b) 正样本密度热图")
    axes[1].set_xlabel("列 / 像素")
    axes[1].set_ylabel("行 / 像素")
    cbar = fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("缝隙像素密度")

    fig.suptitle(f"Patch 采样示例（来源: {args.image.name}）")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
