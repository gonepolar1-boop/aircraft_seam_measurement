"""图 3.5 数据增强可视化.

对一张训练样本依次施加 rotate / flip / affine / brightness / contrast /
noise / blur / illumination 八种增强，拼成 3×3 网格（左上原图）。
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seam_training.data import SeamAugmenter  # noqa: E402
from seam_training.utils import AUGMENTATION_DEFAULTS  # noqa: E402

from _style import THESIS_FIGURES_DIR, savefig  # noqa: E402


AUGMENTATIONS = (
    ("原始", None),
    ("旋转+缩放", "_rotate"),
    ("翻转", "_flip"),
    ("仿射", "_affine"),
    ("亮度偏移", "_brightness"),
    ("对比度缩放", "_contrast"),
    ("高斯噪声", "_noise"),
    ("高斯模糊", "_blur"),
    ("光照梯度", "_illumination"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 3.5 数据增强可视化.")
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
        "--valid-mask",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "valids" / "1.png",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch3_fig5_augmentation_demo.png",
    )
    return parser.parse_args()


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask_bin = mask > 127
    base[mask_bin] = (
        0.4 * base[mask_bin].astype(np.float32) + 0.6 * np.array([239, 68, 68], dtype=np.float32)
    ).astype(np.uint8)
    return base


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    valid = cv2.imread(str(args.valid_mask), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None or valid is None:
        raise FileNotFoundError("Missing real_train sample image / mask / valid.")

    augmenter = SeamAugmenter(cfg=copy.deepcopy(AUGMENTATION_DEFAULTS))

    fig, axes = plt.subplots(3, 3, figsize=(11.0, 11.0))
    for (title, method_name), ax in zip(AUGMENTATIONS, axes.flat, strict=True):
        if method_name is None:
            current_image, current_mask = image.copy(), mask.copy()
        else:
            method = getattr(augmenter, method_name)
            img_aug, mask_aug, _ = method(image.copy(), mask.copy(), valid.copy())
            current_image, current_mask = img_aug, mask_aug
        ax.imshow(overlay_mask(current_image, current_mask))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("单张样本上逐项应用的数据增强（红色叠加层为 GT mask）")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
