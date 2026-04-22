"""图 3.7 / 6.4 训练曲线.

从单模型 run 或 LOOCV run 聚合读取 history.json，画 train/val loss 与
dice/IoU/precision/recall 随 epoch 变化的曲线。LOOCV 模式下以折叠平均
为主线、折叠包络为阴影。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402

METRIC_KEYS = ("train_losses", "val_losses", "val_dice", "val_iou", "val_precision", "val_recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 3.7 / 6.4 训练曲线.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model" / "03301416_loocv_attention_unet",
        help=(
            "Either a single-sample training run (contains metrics/history.json) "
            "or a LOOCV run (contains holdout_*/metrics/history.json)."
        ),
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch3_fig7_train_curves.png",
    )
    return parser.parse_args()


def _load_history(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {key: [] for key in METRIC_KEYS}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {key: [float(value) for value in data.get(key, [])] for key in METRIC_KEYS}


def _discover_histories(run_dir: Path) -> list[tuple[str, dict[str, list[float]]]]:
    # Single-sample training run.
    single = run_dir / "metrics" / "history.json"
    if single.exists():
        return [(run_dir.name, _load_history(single))]
    # LOOCV run: aggregate over holdout_* folders.
    folds = sorted(p for p in run_dir.glob("holdout_*") if p.is_dir())
    histories: list[tuple[str, dict[str, list[float]]]] = []
    for fold in folds:
        history = _load_history(fold / "metrics" / "history.json")
        if any(history[key] for key in METRIC_KEYS):
            histories.append((fold.name, history))
    return histories


def _stack_metric(histories: list[tuple[str, dict[str, list[float]]]], key: str) -> np.ndarray:
    series = [np.asarray(history[key], dtype=np.float32) for _, history in histories if history.get(key)]
    if not series:
        return np.empty((0, 0), dtype=np.float32)
    min_len = min(len(s) for s in series)
    return np.stack([s[:min_len] for s in series], axis=0)


def main() -> None:
    args = parse_args()
    histories = _discover_histories(args.run_dir)
    if not histories:
        raise FileNotFoundError(f"No history.json found under {args.run_dir}")

    title = args.title or f"训练曲线 ({args.run_dir.name})"

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    loss_ax, metric_ax = axes

    for metric, colour, style, label in (
        ("train_losses", PALETTE["primary"], "-", "训练 loss"),
        ("val_losses", PALETTE["secondary"], "--", "验证 loss"),
    ):
        stack = _stack_metric(histories, metric)
        if stack.size == 0:
            continue
        x = np.arange(1, stack.shape[1] + 1)
        mean = np.nanmean(stack, axis=0)
        loss_ax.plot(x, mean, color=colour, linestyle=style, label=label, linewidth=1.8)
        if stack.shape[0] > 1:
            lo = np.nanmin(stack, axis=0)
            hi = np.nanmax(stack, axis=0)
            loss_ax.fill_between(x, lo, hi, color=colour, alpha=0.12)
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("(a) loss 曲线")
    loss_ax.legend(loc="upper right")

    for metric, colour, style, marker, label in (
        ("val_dice", PALETTE["accent"], "-", "o", "Dice"),
        ("val_iou", PALETTE["primary"], "--", "s", "IoU"),
        ("val_precision", PALETTE["highlight"], "-.", "^", "Precision"),
        ("val_recall", PALETTE["purple"], ":", "D", "Recall"),
    ):
        stack = _stack_metric(histories, metric)
        if stack.size == 0:
            continue
        x = np.arange(1, stack.shape[1] + 1)
        mean = np.nanmean(stack, axis=0)
        metric_ax.plot(x, mean, color=colour, linestyle=style, marker=marker, markersize=3,
                        markevery=max(1, len(x) // 20), label=label, linewidth=1.6)
        if stack.shape[0] > 1:
            lo = np.nanmin(stack, axis=0)
            hi = np.nanmax(stack, axis=0)
            metric_ax.fill_between(x, lo, hi, color=colour, alpha=0.10)
    metric_ax.set_xlabel("epoch")
    metric_ax.set_ylabel("指标")
    metric_ax.set_ylim(0.0, 1.02)
    metric_ax.set_title(f"(b) 分割指标 (n_folds={len(histories)})")
    metric_ax.legend(loc="lower right", ncol=2)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
