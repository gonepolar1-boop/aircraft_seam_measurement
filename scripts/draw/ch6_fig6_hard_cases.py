"""图 6.6 复杂工况分割结果.

从 LOOCV 结果里挑出 Dice 最低的 N 个 fold，把对应的
``best_holdout.png`` preview（train.py / run_loocv 生成的 5 联图）
拼在一张大图里。preview 本身已经包含 image/valid/GT/pred/overlay，
这里只做读取 + 标注。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _style import THESIS_FIGURES_DIR, savefig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 6.6 复杂工况分割结果.")
    parser.add_argument(
        "--loocv-run",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model" / "03301416_loocv_attention_unet",
    )
    parser.add_argument("--num-cases", type=int, default=3,
                        help="How many hardest folds (lowest Dice) to include.")
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch6_fig6_hard_cases.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = sorted(p for p in args.loocv_run.glob("holdout_*") if p.is_dir())
    records = []
    for fold in folds:
        summary_path = fold / "metrics" / "best_summary.json"
        preview_path = fold / "previews" / "best_holdout.png"
        if not summary_path.exists() or not preview_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
        records.append({
            "holdout": str(summary.get("holdout_sample", fold.name)),
            "dice": float(summary.get("dice", 0.0)),
            "iou": float(summary.get("iou", 0.0)),
            "preview": preview_path,
        })
    if not records:
        raise FileNotFoundError(f"No holdout previews found under {args.loocv_run}")

    records.sort(key=lambda item: item["dice"])
    selected = records[: max(1, int(args.num_cases))]

    fig, axes = plt.subplots(len(selected), 1, figsize=(14.0, 2.4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]
    for ax, record in zip(axes, selected, strict=True):
        preview = cv2.imread(str(record["preview"]), cv2.IMREAD_COLOR)
        if preview is None:
            ax.set_axis_off()
            continue
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        ax.imshow(preview_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"样本 {record['holdout']}  Dice={record['dice']:.3f}  IoU={record['iou']:.3f}",
            fontsize=10,
            loc="left",
        )

    fig.suptitle("困难样本上的分割结果（LOOCV Dice 最低的 "
                 f"{len(selected)} 个 fold；左→右：原图 | valid | GT | 预测 | 误差叠加）")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
