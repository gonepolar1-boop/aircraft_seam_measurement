from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.viewer3d import show_gap_flush_open3d_viewer_from_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the standalone Gap/Flush Open3D viewer from an exported bundle.")
    parser.add_argument("--bundle-path", type=Path, required=True, help="Path to viewer_bundle.npz.")
    parser.add_argument("--window-title", type=str, default="Gap/Flush 3D Viewer", help="Window title for the viewer.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_gap_flush_open3d_viewer_from_bundle(
        bundle_path=args.bundle_path,
        window_title=args.window_title,
    )


if __name__ == "__main__":
    main()
