"""图 4.8 gap / flush 沿距离 profile.

从一次 ``section_profile.csv`` 读取每个截面的 gap / flush，画出双轴
曲线：x 轴是沿缝隙的累计距离，y 轴左侧 gap，右侧 flush。
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
    parser = argparse.ArgumentParser(description="图 4.8 gap / flush profile.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to section_profile.csv produced by a pipeline run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch4_fig8_gap_flush_profile.png",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _to_float(raw: str | None) -> float:
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def main() -> None:
    args = parse_args()
    rows = _read_csv(args.csv)
    if not rows:
        raise FileNotFoundError(f"Empty section profile: {args.csv}")

    distance = np.asarray([_to_float(row.get("distance_mm")) for row in rows], dtype=np.float32)
    gap = np.asarray([_to_float(row.get("gap")) for row in rows], dtype=np.float32)
    flush = np.asarray([_to_float(row.get("flush")) for row in rows], dtype=np.float32)
    valid = np.asarray([int(row.get("valid") or 0) > 0 for row in rows], dtype=bool)

    x_axis = distance if np.any(np.isfinite(distance)) else np.arange(len(rows), dtype=np.float32)
    x_label = "沿缝隙累计距离 / mm" if np.any(np.isfinite(distance)) else "截面采样索引"

    fig, ax1 = plt.subplots(figsize=(12.0, 4.8))
    ax2 = ax1.twinx()

    gap_mask = valid & np.isfinite(gap)
    flush_mask = valid & np.isfinite(flush)

    ax1.plot(x_axis[gap_mask], gap[gap_mask], color=PALETTE["highlight"], linestyle="-", marker="o",
             markersize=4, markevery=max(1, int(gap_mask.sum() // 30)), label="gap (mm)", linewidth=1.6)
    ax2.plot(x_axis[flush_mask], flush[flush_mask], color=PALETTE["accent"], linestyle="--", marker="s",
             markersize=4, markevery=max(1, int(flush_mask.sum() // 30)), label="flush (mm)", linewidth=1.6)

    # Invalid scatter - x marks at both axes' mean levels for visibility.
    invalid = ~valid
    if invalid.any():
        ax1.scatter(x_axis[invalid], np.full(invalid.sum(), np.nanmean(gap[gap_mask]) if gap_mask.any() else np.nan),
                    marker="x", color="#94a3b8", s=24, alpha=0.75, label="invalid")

    gap_mean = float(np.nanmean(gap[gap_mask])) if gap_mask.any() else np.nan
    flush_mean = float(np.nanmean(flush[flush_mask])) if flush_mask.any() else np.nan
    if np.isfinite(gap_mean):
        ax1.axhline(gap_mean, color=PALETTE["highlight"], linestyle=":", alpha=0.6,
                    label=f"gap 均值 = {gap_mean:.3f}")
    if np.isfinite(flush_mean):
        ax2.axhline(flush_mean, color=PALETTE["accent"], linestyle=":", alpha=0.6,
                    label=f"flush 均值 = {flush_mean:.3f}")

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("gap / mm", color=PALETTE["highlight"])
    ax2.set_ylabel("flush / mm", color=PALETTE["accent"])
    ax1.tick_params(axis="y", colors=PALETTE["highlight"])
    ax2.tick_params(axis="y", colors=PALETTE["accent"])

    # Merge legends into one block.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9, fontsize=9)

    ax1.set_title(f"gap / flush 测量 profile (n_valid={int(valid.sum())} / {len(rows)})")
    fig.tight_layout()
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
