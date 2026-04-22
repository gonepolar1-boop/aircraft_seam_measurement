"""图 6.11 单帧耗时分解.

跑一次管线、拿 result["timing"]，把 stages_s 以水平条形图展示；每条
上标耗时和占比。
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

from pipeline.gap_flush import run_gap_flush_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 6.11 单帧耗时分解.")
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--warmups", type=int, default=1,
                        help="Warm-up runs (not recorded) to account for cold-start effects.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed runs (median reported).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments" / "timing",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch6_fig11_timing_breakdown.png",
    )
    return parser.parse_args()


def _run_once(args: argparse.Namespace, suffix: str) -> dict[str, float]:
    result = run_gap_flush_pipeline(
        pcd_path=args.pcd,
        checkpoint_path=args.checkpoint,
        threshold=float(args.threshold),
        output_root=args.output_root / suffix,
        fast_mode=True,
        save_profile_plots=False,
        save_viewer_bundle=False,
        show_3d_viewer=False,
    )
    return result["timing"]


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for idx in range(max(0, int(args.warmups))):
        _run_once(args, suffix=f"warmup_{idx}")

    runs: list[dict[str, float]] = []
    for idx in range(max(1, int(args.repeats))):
        timing = _run_once(args, suffix=f"run_{idx}")
        runs.append(timing)
        print(f"run {idx}: total_s={timing['total_s']:.3f}")

    with (args.output_root / "timing_runs.json").open("w", encoding="utf-8") as fh:
        json.dump(runs, fh, ensure_ascii=False, indent=2)

    stage_keys = sorted({key for run in runs for key in run["stages_s"]})
    # Take per-stage medians across the repeat runs.
    stage_medians: dict[str, float] = {}
    for key in stage_keys:
        values = [float(run["stages_s"].get(key, 0.0)) for run in runs]
        stage_medians[key] = float(np.median(values))
    total_median = float(np.median([float(run["total_s"]) for run in runs]))

    sorted_stages = sorted(stage_medians.items(), key=lambda item: item[1], reverse=True)
    labels = [name for name, _ in sorted_stages]
    values = np.asarray([time for _, time in sorted_stages], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10.5, 0.6 * len(labels) + 2.0))
    palette_cycle = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["highlight"],
                     PALETTE["purple"], PALETTE["brown"], PALETTE["muted"]]
    colours = [palette_cycle[i % len(palette_cycle)] for i in range(len(labels))]
    bars = ax.barh(labels, values, color=colours, edgecolor="black", linewidth=0.5)
    for bar, time in zip(bars, values):
        share = time / total_median if total_median > 0 else 0.0
        ax.text(bar.get_width() + 0.005 * total_median,
                bar.get_y() + bar.get_height() / 2.0,
                f"{time:.4f} s  ({share * 100.0:.1f}%)",
                va="center", fontsize=9)

    ax.set_xlabel("单帧耗时 / s (median over "
                  f"{len(runs)} runs)")
    ax.set_title(f"管线各阶段耗时分解  总 {total_median:.3f} s")
    ax.invert_yaxis()
    fig.tight_layout()
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
