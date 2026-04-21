"""图 6.3 训练集样本示例.

拼接 N 张训练样本的 "深度图 / 有效掩膜 / GT mask 叠加" 三列网格图。
默认使用 real_train 下的 5 张样本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _style import THESIS_FIGURES_DIR, savefig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 6.3 训练样本网格.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "images",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "masks",
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "valids",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Sample names; default discovers all .png stems.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch6_fig3_train_samples_grid.png",
    )
    return parser.parse_args()


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    mask_bin = mask > 127
    base[mask_bin] = 0.35 * base[mask_bin] + 0.65 * np.array([239, 68, 68], dtype=np.float32)
    return base.clip(0, 255).astype(np.uint8)


def main() -> None:
    args = parse_args()
    sample_names = args.samples
    if sample_names is None:
        sample_names = sorted(p.stem for p in args.image_dir.glob("*.png"))
    if not sample_names:
        raise FileNotFoundError(f"No samples found under {args.image_dir}")

    rows = len(sample_names)
    fig, axes = plt.subplots(rows, 3, figsize=(11.0, 3.0 * rows))
    if rows == 1:
        axes = axes.reshape(1, 3)

    for row_idx, name in enumerate(sample_names):
        image = cv2.imread(str(args.image_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(args.mask_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(str(args.valid_dir / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or valid is None:
            raise FileNotFoundError(f"Sample {name} missing one of image/mask/valid.")

        axes[row_idx, 0].imshow(image, cmap="gray")
        axes[row_idx, 0].set_ylabel(f"样本 {name}", fontsize=11)
        axes[row_idx, 0].set_title("深度图" if row_idx == 0 else "")

        axes[row_idx, 1].imshow(valid, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 1].set_title("有效像素掩膜" if row_idx == 0 else "")

        axes[row_idx, 2].imshow(overlay_mask(image, mask))
        axes[row_idx, 2].set_title("GT 叠加" if row_idx == 0 else "")

        for ax in axes[row_idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"训练集全部 {rows} 个真实样本")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
