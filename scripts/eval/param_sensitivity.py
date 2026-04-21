"""参数敏感性扫描（支撑 图 6.10）.

对 ``GapFlushParams`` 的选定字段做 1D 扫描，在同一个 PCD/checkpoint 上
跑管线、记录 gap_mean / flush_mean / num_measurement_sections。输出
CSV 到 outputs/experiments/param_sensitivity/，并可选直接画图。

用例：
    python scripts/eval/param_sensitivity.py \\
        --pcd data/process/manual_crop/1/crop.pcd \\
        --checkpoint outputs/model/03301416_loocv_attention_unet/holdout_1/checkpoints/best.pth \\
        --param seam_step --values 1 2 4 8 16 \\
        --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.gap_flush import run_gap_flush_pipeline  # noqa: E402
from pipeline.seam_measurement import GapFlushParams  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="参数敏感性扫描.")
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--param", type=str, required=True,
                        help="GapFlushParams field to sweep, e.g. seam_step, top_surface_quantile.")
    parser.add_argument("--values", nargs="+", required=True,
                        help="Values to try; will be cast to match the field type.")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "experiments" / "param_sensitivity")
    parser.add_argument("--plot", action="store_true", help="Also render a PNG chart.")
    return parser.parse_args()


def _cast(raw: str, target_type: type) -> object:
    if target_type is int:
        return int(float(raw))
    if target_type is float:
        return float(raw)
    return raw


def _infer_field_type(name: str) -> type:
    for field in fields(GapFlushParams):
        if field.name == name:
            return field.type if isinstance(field.type, type) else type(getattr(GapFlushParams(), name))
    raise KeyError(f"GapFlushParams has no field {name!r}")


def main() -> None:
    args = parse_args()
    field_type = _infer_field_type(args.param)
    typed_values = [_cast(value, field_type) for value in args.values]

    sweep_dir = args.output_dir / args.param
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in typed_values:
        params = replace(GapFlushParams(), **{args.param: value})
        run_output = sweep_dir / f"{args.param}={value}"
        result = run_gap_flush_pipeline(
            pcd_path=args.pcd,
            checkpoint_path=args.checkpoint,
            threshold=float(args.threshold),
            output_root=run_output,
            params=params,
            fast_mode=True,
            save_profile_plots=False,
            save_viewer_bundle=False,
            show_3d_viewer=False,
        )
        summary = result["summary"]
        timing = result.get("timing", {})
        rows.append({
            "param": args.param,
            "value": value,
            "gap_mean": summary.get("gap_mean"),
            "gap_std": summary.get("gap_std"),
            "flush_mean": summary.get("flush_mean"),
            "flush_std": summary.get("flush_std"),
            "num_sections": summary.get("num_sections"),
            "num_measurement_sections": summary.get("num_measurement_sections"),
            "total_s": timing.get("total_s"),
        })
        print(
            f"{args.param}={value}  gap_mean={summary.get('gap_mean'):.4f} "
            f"flush_mean={summary.get('flush_mean'):.4f} "
            f"n_valid={summary.get('num_measurement_sections')} "
            f"total_s={timing.get('total_s', 0.0):.3f}"
        )

    csv_path = sweep_dir / "sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (sweep_dir / "sensitivity.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2, default=float)
    print(f"saved: {csv_path}")

    if args.plot:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "draw"))
        from _style import PALETTE, THESIS_FIGURES_DIR, savefig  # noqa: E402,PLC0415

        values_axis = np.asarray([float(row["value"]) for row in rows], dtype=np.float32)
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        axes[0].errorbar(values_axis, [row["gap_mean"] for row in rows],
                         yerr=[row["gap_std"] for row in rows], fmt="-o",
                         color=PALETTE["highlight"], label="gap 均值 ± σ")
        axes[0].errorbar(values_axis, [row["flush_mean"] for row in rows],
                         yerr=[row["flush_std"] for row in rows], fmt="-s",
                         color=PALETTE["accent"], label="flush 均值 ± σ")
        axes[0].set_xlabel(args.param)
        axes[0].set_ylabel("测量值 / mm")
        axes[0].set_title("(a) 测量值随参数变化")
        axes[0].legend(loc="best", fontsize=9)

        axes[1].plot(values_axis, [row["num_measurement_sections"] for row in rows], "-^",
                     color=PALETTE["primary"], label="有效截面数")
        axes[1].set_xlabel(args.param)
        axes[1].set_ylabel("有效截面数")
        axes[1].set_title("(b) 有效截面数随参数变化")
        ax_right = axes[1].twinx()
        ax_right.plot(values_axis, [row["total_s"] for row in rows], "--D",
                      color=PALETTE["secondary"], alpha=0.85, label="总耗时 / s")
        ax_right.set_ylabel("耗时 / s", color=PALETTE["secondary"])
        ax_right.tick_params(axis="y", colors=PALETTE["secondary"])
        axes[1].legend(loc="upper left", fontsize=9)

        fig.suptitle(f"参数敏感性: {args.param}")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        saved = savefig(fig, THESIS_FIGURES_DIR / f"ch6_fig10_param_sensitivity_{args.param}.png")
        print(f"saved: {saved}")


if __name__ == "__main__":
    main()
