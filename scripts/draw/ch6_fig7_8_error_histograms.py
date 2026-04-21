"""图 6.7 / 6.8 算法误差分布直方图.

需要一个成对的 "算法 vs 人工" CSV，列如下（第一列为必需）：

    sample_id,algo_gap,manual_gap,algo_flush,manual_flush

输出 2 联图：左 gap 误差直方图、右 flush 误差直方图，标注均值 / std / p95。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 6.7 / 6.8 误差直方图.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV with columns: algo_gap, manual_gap, algo_flush, manual_flush.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch6_fig7_8_error_histograms.png",
    )
    parser.add_argument("--bin-count", type=int, default=30)
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    gaps = []
    flushes = []
    with path.open("r", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                algo_gap = float(row["algo_gap"])
                manual_gap = float(row["manual_gap"])
                gaps.append(algo_gap - manual_gap)
            except (KeyError, ValueError):
                pass
            try:
                algo_flush = float(row["algo_flush"])
                manual_flush = float(row["manual_flush"])
                flushes.append(algo_flush - manual_flush)
            except (KeyError, ValueError):
                pass
    return np.asarray(gaps, dtype=np.float32), np.asarray(flushes, dtype=np.float32)


def _annotate(ax, errors: np.ndarray, unit: str, colour: str):
    if errors.size == 0:
        ax.text(0.5, 0.5, "缺少数据", ha="center", va="center", transform=ax.transAxes)
        return
    mean = float(np.mean(errors))
    std = float(np.std(errors, ddof=0))
    p95 = float(np.percentile(np.abs(errors), 95))
    ax.axvline(0.0, color="#475569", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.axvline(mean, color=colour, linewidth=1.4, linestyle="-", alpha=0.95, label=f"均值 = {mean:+.4f} {unit}")
    ax.axvline(mean + std, color=colour, linewidth=1.0, linestyle=":", alpha=0.8, label=f"±1σ = {std:.4f} {unit}")
    ax.axvline(mean - std, color=colour, linewidth=1.0, linestyle=":", alpha=0.8)
    ax.text(0.02, 0.95, f"N = {errors.size}\n|err| P95 = {p95:.4f} {unit}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#e2e8f0", alpha=0.92, pad=4))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)


def main() -> None:
    args = parse_args()
    gap_err, flush_err = _read_csv(args.csv)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    axes[0].hist(gap_err, bins=int(args.bin_count), color=PALETTE["highlight"], alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[0].set_title("gap 误差分布 (算法 − 人工)")
    axes[0].set_xlabel("gap 误差 / mm")
    axes[0].set_ylabel("频数")
    _annotate(axes[0], gap_err, "mm", PALETTE["secondary"])

    axes[1].hist(flush_err, bins=int(args.bin_count), color=PALETTE["accent"], alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[1].set_title("flush 误差分布 (算法 − 人工)")
    axes[1].set_xlabel("flush 误差 / mm")
    axes[1].set_ylabel("频数")
    _annotate(axes[1], flush_err, "mm", PALETTE["secondary"])

    fig.suptitle(f"算法与人工测量的误差分布（来源: {args.csv.name}）")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
