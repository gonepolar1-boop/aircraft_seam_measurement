"""图 3.8 分割结果三联图.

选定一个 holdout 样本，用训练好的 checkpoint 做推理，左到右展示
(a) 深度图, (b) 人工标注 GT, (c) 预测 mask + 叠加错误类型色码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _style import THESIS_FIGURES_DIR, savefig  # noqa: E402

from seam_training.model import build_model  # noqa: E402
from seam_training.utils import MODEL_INPUT_SIZE, REAL_ONLY_DEFAULTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 3.8 分割结果三联图.")
    parser.add_argument("--sample", type=str, default="1",
                        help="Sample name (stem of real_train/images/<name>.png).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model" / "03301416_loocv_attention_unet"
            / "holdout_1" / "checkpoints" / "best.pth",
    )
    parser.add_argument("--threshold", type=float, default=float(REAL_ONLY_DEFAULTS["preview_threshold"]))
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch3_fig8_seg_triplet.png",
    )
    return parser.parse_args()


def _load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_name = str(checkpoint.get("model_name", REAL_ONLY_DEFAULTS["model_name"]))
    base_channels = int(checkpoint.get("model_base_channels", REAL_ONLY_DEFAULTS["model_base_channels"]))
    model = build_model(model_name=model_name, base_channels=base_channels).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    images_dir = PROJECT_ROOT / "data" / "real_train" / "images"
    masks_dir = PROJECT_ROOT / "data" / "real_train" / "masks"
    image_path = images_dir / f"{args.sample}.png"
    mask_path = masks_dir / f"{args.sample}.png"
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Missing image/mask for sample {args.sample}.")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, device)
    input_h, input_w = MODEL_INPUT_SIZE
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None, ...].to(device)
    with torch.inference_mode():
        probs = torch.sigmoid(model(tensor))[0, 0].detach().cpu().numpy()
    probs = cv2.resize(probs, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred_mask = (probs > args.threshold).astype(np.uint8) * 255

    gt_bin = mask > 127
    pr_bin = pred_mask > 127

    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
    tp = gt_bin & pr_bin
    fp = pr_bin & ~gt_bin
    fn = ~pr_bin & gt_bin
    overlay[tp] = 0.35 * overlay[tp] + 0.65 * np.array([34, 197, 94], dtype=np.float32)
    overlay[fp] = 0.35 * overlay[fp] + 0.65 * np.array([239, 68, 68], dtype=np.float32)
    overlay[fn] = 0.35 * overlay[fn] + 0.65 * np.array([234, 179, 8], dtype=np.float32)
    overlay = overlay.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("(a) 深度图输入")

    axes[1].imshow(image, cmap="gray", alpha=0.6)
    axes[1].imshow(np.where(gt_bin, 1.0, np.nan), cmap="autumn", alpha=0.6)
    axes[1].set_title("(b) 人工标注 GT")

    axes[2].imshow(overlay)
    handles = [
        mpatches.Patch(color="#22c55e", label="TP (预测 ∩ GT)"),
        mpatches.Patch(color="#ef4444", label="FP (过分割)"),
        mpatches.Patch(color="#eab308", label="FN (漏分割)"),
    ]
    axes[2].legend(handles=handles, loc="lower right", framealpha=0.85, fontsize=9)
    axes[2].set_title("(c) 预测 mask 叠加到原图")

    for ax in axes:
        ax.set_xlabel("列 / 像素")
        ax.set_ylabel("行 / 像素")

    fig.suptitle(f"分割结果三联图（sample={args.sample}）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
