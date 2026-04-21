"""图 6.9 算法 vs 人工测量散点图 + Bland-Altman.

接受同一份 "algo_gap / manual_gap / algo_flush / manual_flush" CSV，
左列画散点 + y=x 对角线，右列画 Bland-Altman（差分 vs 均值）。
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
    parser = argparse.ArgumentParser(description="图 6.9 算法 vs 人工散点 + Bland-Altman.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch6_fig9_scatter_bland_altman.png",
    )
    return parser.parse_args()


def _extract_pairs(rows: list[dict[str, str]], algo_key: str, manual_key: str) -> tuple[np.ndarray, np.ndarray]:
    algo, manual = [], []
    for row in rows:
        try:
            a = float(row[algo_key])
            m = float(row[manual_key])
        except (KeyError, ValueError):
            continue
        if np.isfinite(a) and np.isfinite(m):
            algo.append(a)
            manual.append(m)
    return np.asarray(algo, dtype=np.float32), np.asarray(manual, dtype=np.float32)


def _plot_scatter(ax, algo: np.ndarray, manual: np.ndarray, title: str, colour: str):
    if algo.size == 0:
        ax.text(0.5, 0.5, "缺少数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.scatter(manual, algo, c=colour, s=28, alpha=0.8, edgecolors="black", linewidths=0.4)
    lo = float(min(np.min(manual), np.min(algo)))
    hi = float(max(np.max(manual), np.max(algo)))
    pad = 0.05 * (hi - lo) if hi > lo else 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#334155", linewidth=1.2, label="y = x")
    slope, intercept = np.polyfit(manual, algo, deg=1)
    x_fit = np.linspace(lo - pad, hi + pad, 50)
    ax.plot(x_fit, slope * x_fit + intercept, "-", color=colour, linewidth=1.4, alpha=0.95,
            label=f"拟合 y = {slope:.3f}x + {intercept:+.3f}")
    ax.set_xlabel("人工测量 / mm")
    ax.set_ylabel("算法测量 / mm")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


def _plot_bland_altman(ax, algo: np.ndarray, manual: np.ndarray, title: str, colour: str):
    if algo.size == 0:
        ax.text(0.5, 0.5, "缺少数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    mean = 0.5 * (algo + manual)
    diff = algo - manual
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=0))
    ax.scatter(mean, diff, c=colour, s=28, alpha=0.8, edgecolors="black", linewidths=0.4)
    ax.axhline(mean_diff, color=colour, linestyle="-", linewidth=1.4, label=f"偏差 = {mean_diff:+.4f} mm")
    ax.axhline(mean_diff + 1.96 * std_diff, color="#334155", linestyle="--", linewidth=1.0,
               label=f"+1.96σ = {mean_diff + 1.96 * std_diff:+.4f}")
    ax.axhline(mean_diff - 1.96 * std_diff, color="#334155", linestyle="--", linewidth=1.0,
               label=f"−1.96σ = {mean_diff - 1.96 * std_diff:+.4f}")
    ax.set_xlabel("(算法 + 人工) / 2, mm")
    ax.set_ylabel("算法 − 人工, mm")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def main() -> None:
    args = parse_args()
    with args.csv.open("r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    gap_algo, gap_manual = _extract_pairs(rows, "algo_gap", "manual_gap")
    flush_algo, flush_manual = _extract_pairs(rows, "algo_flush", "manual_flush")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.5))
    _plot_scatter(axes[0, 0], gap_algo, gap_manual, "(a) gap 散点对比", PALETTE["highlight"])
    _plot_bland_altman(axes[0, 1], gap_algo, gap_manual, "(b) gap Bland-Altman", PALETTE["highlight"])
    _plot_scatter(axes[1, 0], flush_algo, flush_manual, "(c) flush 散点对比", PALETTE["accent"])
    _plot_bland_altman(axes[1, 1], flush_algo, flush_manual, "(d) flush Bland-Altman", PALETTE["accent"])

    fig.suptitle(f"算法与人工测量的 1-对-1 对照（来源: {args.csv.name}）")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
