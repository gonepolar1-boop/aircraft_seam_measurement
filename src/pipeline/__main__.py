from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.gap_flush import run_gap_flush_pipeline
from pipeline.seam_measurement.params import GapFlushParams
from pipeline.seam_training.utils import REAL_ONLY_DEFAULTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v1 gap/flush pipeline on a PCD file and checkpoint.")
    parser.add_argument("--pcd-path", type=Path, required=True, help="Path to input PCD file.")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Path to segmentation checkpoint (.pth).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(REAL_ONLY_DEFAULTS["preview_threshold"]),
        help="Segmentation threshold.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output root. Defaults to outputs/pipeline.",
    )
    parser.add_argument(
        "--seam-step",
        type=int,
        default=None,
        help="Section sampling step size in pixels. If omitted, uses the "
             "value from configs/gap_flush.yaml (GapFlushParams.seam_step).",
    )
    parser.add_argument(
        "--show-3d-viewer",
        action="store_true",
        help="Open a single Open3D viewer for the current gap/flush pipeline result.",
    )
    parser.add_argument(
        "--save-3d-viewer-bundle",
        action="store_true",
        help="Export viewer_bundle.npz for launching the 3D viewer later from the app.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # If --seam-step was not supplied, fall through to GapFlushParams'
    # yaml-sourced default instead of clobbering it with a hardcoded 4.
    params = GapFlushParams() if args.seam_step is None else GapFlushParams(seam_step=int(args.seam_step))
    # The CLI surface runs in precision mode only: the reference
    # extraction path (``fast_mode=False``) and full profile plot
    # generation are always used.  Bench / internal users can still
    # pass ``fast_mode=True`` / ``save_profile_plots=False`` through
    # the Python API, but no command-line switch exposes them here.
    result = run_gap_flush_pipeline(
        pcd_path=args.pcd_path,
        checkpoint_path=args.checkpoint_path,
        threshold=float(args.threshold),
        output_root=args.output_root,
        params=params,
        fast_mode=False,
        save_profile_plots=True,
        save_viewer_bundle=args.save_3d_viewer_bundle or args.show_3d_viewer,
        show_3d_viewer=args.show_3d_viewer,
    )
    summary = result["summary"]
    print(f"gap_mean: {summary['gap_mean']:.6f}")
    print(f"gap_std: {summary['gap_std']:.6f}" if summary["gap_std"] == summary["gap_std"] else "gap_std: nan")
    print(f"gap_min: {summary['gap_min']:.6f}" if summary["gap_min"] == summary["gap_min"] else "gap_min: nan")
    print(f"gap_max: {summary['gap_max']:.6f}" if summary["gap_max"] == summary["gap_max"] else "gap_max: nan")
    print(f"flush_mean: {summary['flush_mean']:.6f}")
    print(f"flush_std: {summary['flush_std']:.6f}" if summary["flush_std"] == summary["flush_std"] else "flush_std: nan")
    print(f"flush_min: {summary['flush_min']:.6f}" if summary["flush_min"] == summary["flush_min"] else "flush_min: nan")
    print(f"flush_max: {summary['flush_max']:.6f}" if summary["flush_max"] == summary["flush_max"] else "flush_max: nan")
    print(f"num_sections: {summary['num_sections']}")
    print(f"num_measurement_sections: {summary['num_measurement_sections']}")
    if summary["num_sections"]:
        print(f"valid_ratio: {float(summary['num_measurement_sections']) / float(summary['num_sections']):.6f}")
    else:
        print("valid_ratio: nan")
    print(f"measurement_unit: {summary.get('unit', 'mm')}")
    print(f"output_dir: {result['output_dir']}")
    print(f"summary_json: {result['exports']['summary_json']}")
    print(f"section_profile_csv: {result['exports']['section_profile_csv']}")
    if "gap_profile_png" in result["exports"]:
        print(f"gap_profile_png: {result['exports']['gap_profile_png']}")
    if "flush_profile_png" in result["exports"]:
        print(f"flush_profile_png: {result['exports']['flush_profile_png']}")
    if "depth_overlay_png" in result["exports"]:
        print(f"depth_overlay_png: {result['exports']['depth_overlay_png']}")
    if "section_debug_dir" in result["exports"]:
        print(f"section_debug_dir: {result['exports']['section_debug_dir']}")
    if "viewer_bundle_npz" in result["exports"]:
        print(f"viewer_bundle_npz: {result['exports']['viewer_bundle_npz']}")


if __name__ == "__main__":
    main()
