"""图 6.5 分割指标对比 bar chart.

读取两个 LOOCV run 的 ``loocv_summary.json``（默认 attention_unet vs
unet），对比 Dice / IoU / Precision / Recall 的均值，误差棒为 std。
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

DEFAULT_RUNS = (
    ("U-Net", PROJECT_ROOT / "outputs" / "model" / "03312131_loocv_unet" / "loocv_summary.json"),
    ("Attention U-Net", PROJECT_ROOT / "outputs" / "model" / "03301416_loocv_attention_unet" / "loocv_summary.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="图 6.5 模型分割指标对比.")
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Format: 'LABEL=path/to/loocv_summary.json'. Repeat for each model.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_FIGURES_DIR / "ch6_fig5_model_compare_bars.png",
    )
    return parser.parse_args()


def _resolve_runs(raw: list[str] | None) -> list[tuple[str, Path]]:
    if not raw:
        return list(DEFAULT_RUNS)
    runs = []
    for item in raw:
        if "=" not in item:
            raise ValueError(f"--run expects LABEL=PATH, got {item!r}")
        label, path = item.split("=", 1)
        runs.append((label.strip(), Path(path.strip())))
    return runs


def _extract_metrics(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    aggregates = data.get("aggregates", {})
    return {
        key: {"mean": float(value.get("mean", 0.0)), "std": float(value.get("std", 0.0))}
        for key, value in aggregates.items()
    }


def main() -> None:
    args = parse_args()
    runs = _resolve_runs(args.run)
    metrics_order = ("dice", "iou", "precision", "recall")
    palette = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["highlight"], PALETTE["purple"]]

    labels = []
    per_metric: dict[str, list[tuple[float, float]]] = {key: [] for key in metrics_order}
    for label, path in runs:
        if not path.exists():
            raise FileNotFoundError(f"LOOCV summary not found: {path}")
        data = _extract_metrics(path)
        labels.append(label)
        for key in metrics_order:
            entry = data.get(key, {"mean": 0.0, "std": 0.0})
            per_metric[key].append((entry["mean"], entry["std"]))

    num_models = len(labels)
    x = np.arange(len(metrics_order), dtype=np.float32)
    bar_width = 0.8 / max(num_models, 1)

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for model_idx, label in enumerate(labels):
        offsets = x + (model_idx - (num_models - 1) / 2.0) * bar_width
        means = [per_metric[key][model_idx][0] for key in metrics_order]
        stds = [per_metric[key][model_idx][1] for key in metrics_order]
        colour = palette[model_idx % len(palette)]
        ax.bar(offsets, means, width=bar_width * 0.9, color=colour, edgecolor="black", linewidth=0.6,
               label=label, yerr=stds, capsize=3.0, ecolor="#334155")
        for offset, mean in zip(offsets, means):
            ax.text(offset, mean + 0.01, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([key.upper() for key in metrics_order])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("指标均值 (LOOCV)")
    ax.set_title("U-Net vs Attention U-Net 分割指标对比")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    saved = savefig(fig, args.out)
    print(f"saved: {saved}")


if __name__ == "__main__":
    main()
