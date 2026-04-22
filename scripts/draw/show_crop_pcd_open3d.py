from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline.seam_mapping.io import load_point_map  # noqa: E402

DEFAULT_PCD_PATH = PROJECT_ROOT / "data" / "process" / "manual_crop" / "1" / "crop.pcd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show a crop.pcd file in a simple Open3D viewer.")
    parser.add_argument(
        "--pcd-path",
        type=Path,
        default=DEFAULT_PCD_PATH,
        help="Path to an ASCII .pcd point map file.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=60000,
        help="Maximum number of finite points to render.",
    )
    return parser.parse_args()


def build_colored_point_cloud(xyz: np.ndarray):
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise RuntimeError("open3d is not installed in the current environment.") from exc

    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    finite_mask = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[finite_mask]
    if len(xyz) == 0:
        raise ValueError("No finite 3D points were found in the input point cloud.")

    z = xyz[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    if z_max > z_min:
        z_norm = (z - z_min) / (z_max - z_min)
    else:
        z_norm = np.zeros_like(z, dtype=np.float32)

    colors = np.column_stack(
        [
            0.15 + 0.80 * z_norm,
            0.35 + 0.50 * (1.0 - np.abs(z_norm - 0.5) * 2.0),
            0.95 - 0.75 * z_norm,
        ]
    ).astype(np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return o3d, cloud


def maybe_subsample(xyz: np.ndarray, max_points: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    if max_points <= 0 or len(xyz) <= max_points:
        return xyz
    indices = np.linspace(0, len(xyz) - 1, num=max_points, dtype=np.int32)
    return xyz[indices]


def main() -> None:
    args = parse_args()
    pcd_path = args.pcd_path.resolve()
    if not pcd_path.exists():
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")

    point_map = load_point_map(pcd_path)
    xyz = maybe_subsample(point_map.reshape(-1, 3), int(args.max_points))
    o3d, cloud = build_colored_point_cloud(xyz)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    o3d.visualization.draw_geometries(
        [cloud, frame],
        window_name=f"crop.pcd viewer - {pcd_path.name}",
        width=1180,
        height=860,
        left=120,
        top=80,
    )


if __name__ == "__main__":
    main()
