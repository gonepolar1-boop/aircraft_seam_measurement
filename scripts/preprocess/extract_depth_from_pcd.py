from pathlib import Path
import shutil
import numpy as np
import cv2


def read_pcd_depth(pcd_path):
    with open(pcd_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    width = height = data_start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("WIDTH"):
            width = int(s.split()[1])
        elif s.startswith("HEIGHT"):
            height = int(s.split()[1])
        elif s.startswith("DATA"):
            data_start = i + 1
            break

    xyz = np.loadtxt(lines[data_start:], usecols=(0, 1, 2), dtype=np.float32)
    return xyz[:, 2].reshape(height, width)


def depth_to_gray(depth):
    valid = np.isfinite(depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)

    if valid.any():
        d_min = depth[valid].min()
        d_max = depth[valid].max()
        if d_max > d_min:
            gray[valid] = ((depth[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            gray[valid] = 255

    return gray


def find_crop_pcds(input_root):
    return sorted(input_root.rglob("crop.pcd"))


def main():
    input_root = Path("data/process/manual_crop")
    out_root = Path("data/process/depth")
    pcd_paths = find_crop_pcds(input_root)

    if not pcd_paths:
        raise ValueError(f"No crop.pcd found under: {input_root}")

    print(f"Found {len(pcd_paths)} sample(s) under {input_root}")

    for pcd_path in pcd_paths:
        sample_name = pcd_path.parent.name
        out_dir = out_root / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        depth = read_pcd_depth(pcd_path)
        gray = depth_to_gray(depth)

        out_png = out_dir / "depth.png"
        cv2.imwrite(str(out_png), gray)
        print(f"saved: {out_png}")

        meta_path = pcd_path.parent / "crop_meta.json"
        if meta_path.exists():
            out_meta = out_dir / "crop_meta.json"
            shutil.copyfile(meta_path, out_meta)
            print(f"saved: {out_meta}")


if __name__ == "__main__":
    main()
