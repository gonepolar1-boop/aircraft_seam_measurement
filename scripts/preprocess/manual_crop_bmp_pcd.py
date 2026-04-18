from pathlib import Path
import json
import numpy as np
import cv2


def load_pcd_xyz(pcd_path):
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
            if "ascii" not in s.lower():
                raise ValueError("Only ASCII PCD is supported")
            data_start = i + 1
            break

    xyz = np.loadtxt(lines[data_start:], usecols=(0, 1, 2), dtype=np.float32)
    return xyz.reshape(height, width, 3)


def save_pcd_xyz(xyz, out_path):
    h, w, _ = xyz.shape
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# .PCD v0.7\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {w}\n")
        f.write(f"HEIGHT {h}\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {w * h}\n")
        f.write("DATA ascii\n")
        np.savetxt(f, xyz.reshape(-1, 3), fmt="%.6f %.6f %.6f")


def save_json(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rotate_keep_all(image, angle, interp, border_value):
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_v = abs(m[0, 0])
    sin_v = abs(m[0, 1])
    new_w = int(np.ceil(h * sin_v + w * cos_v))
    new_h = int(np.ceil(h * cos_v + w * sin_v))
    m[0, 2] += new_w / 2.0 - center[0]
    m[1, 2] += new_h / 2.0 - center[1]
    return cv2.warpAffine(
        image,
        m,
        (new_w, new_h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def rotate_pcd(pcd, angle):
    xs = rotate_keep_all(pcd[:, :, 0], angle, cv2.INTER_NEAREST, float("nan"))
    ys = rotate_keep_all(pcd[:, :, 1], angle, cv2.INTER_NEAREST, float("nan"))
    zs = rotate_keep_all(pcd[:, :, 2], angle, cv2.INTER_NEAREST, float("nan"))
    return np.stack([xs, ys, zs], axis=2)


def choose_angle(image):
    window = "Angle"
    state = {"v": 1800}

    def on_change(v):
        state["v"] = v

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1000, 700)
    cv2.createTrackbar("angle x10 + 1800", window, 1800, 3600, on_change)

    while True:
        angle = (state["v"] - 1800) / 10.0
        show = rotate_keep_all(image, angle, cv2.INTER_LINEAR, (0, 0, 0))
        cv2.putText(show, f"angle={angle:.1f}  Left/Right:0.1  A/D:1  Enter:OK  Esc:Exit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(window, show)
        key = cv2.waitKeyEx(30)
        if key in (13, 32):
            cv2.destroyWindow(window)
            return angle
        if key == 27:
            cv2.destroyWindow(window)
            return None
        if key == 2424832:
            state["v"] = max(0, state["v"] - 1)
            cv2.setTrackbarPos("angle x10 + 1800", window, state["v"])
        if key == 2555904:
            state["v"] = min(3600, state["v"] + 1)
            cv2.setTrackbarPos("angle x10 + 1800", window, state["v"])
        if key in (ord("a"), ord("A")):
            state["v"] = max(0, state["v"] - 10)
            cv2.setTrackbarPos("angle x10 + 1800", window, state["v"])
        if key in (ord("d"), ord("D")):
            state["v"] = min(3600, state["v"] + 10)
            cv2.setTrackbarPos("angle x10 + 1800", window, state["v"])


def find_input_images(input_root):
    samples = []
    for bmp_path in sorted(input_root.glob("*/*.bmp")):
        pcd_path = bmp_path.with_suffix(".pcd")
        if not pcd_path.exists():
            raise FileNotFoundError(f"Missing pcd: {pcd_path}")
        samples.append((bmp_path, pcd_path))
    return samples


def process_one_image(image_path, pcd_path, out_dir):
    print(f"Processing: {image_path}")

    bmp = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bmp is None:
        raise ValueError(f"Failed to load image: {image_path}")

    pcd = load_pcd_xyz(pcd_path)

    h, w = bmp.shape[:2]
    hp, wp = pcd.shape[:2]
    if hp == h and wp == w:
        scale = 1
    elif hp == 2 * h and wp == 2 * w:
        scale = 2
    else:
        raise ValueError(f"Expected pcd to be 2x bmp, got bmp={(h, w)}, pcd={(hp, wp)}")

    angle = choose_angle(bmp)
    if angle is None:
        print(f"Canceled: {image_path}")
        return

    work_bmp = bmp if angle == 0 else rotate_keep_all(bmp, angle, cv2.INTER_LINEAR, (0, 0, 0))
    work_pcd = pcd if angle == 0 else rotate_pcd(pcd, angle)
    valid_src = np.full((h, w), 255, dtype=np.uint8)
    work_valid = valid_src if angle == 0 else rotate_keep_all(valid_src, angle, cv2.INTER_NEAREST, 0)

    while True:
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 1000, 700)
        roi = cv2.selectROI("Select ROI", work_bmp, False, False)
        cv2.destroyWindow("Select ROI")
        x, y, rw, rh = map(int, roi)

        if rw == 0 or rh == 0:
            print("ROI empty, redraw")
            continue
        break

    bmp_crop = work_bmp[y:y + rh, x:x + rw]
    pcd_crop = work_pcd[scale * y:scale * (y + rh), scale * x:scale * (x + rw), :]
    roi_mask = work_valid[y:y + rh, x:x + rw]
    roi_mask_depth = np.full((scale * rh, scale * rw), 255, dtype=np.uint8)

    sample_dir = out_dir / image_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    out_bmp = sample_dir / "crop.bmp"
    out_pcd = sample_dir / "crop.pcd"
    out_valid = sample_dir / "valid.png"
    out_roi = sample_dir / "roi_mask.png"
    out_valid_depth = sample_dir / "valid_depth.png"
    out_roi_depth = sample_dir / "roi_mask_depth.png"
    out_meta = sample_dir / "crop_meta.json"

    cv2.imwrite(str(out_bmp), bmp_crop)
    save_pcd_xyz(pcd_crop, out_pcd)
    cv2.imwrite(str(out_valid), roi_mask)
    cv2.imwrite(str(out_roi), roi_mask)
    cv2.imwrite(str(out_valid_depth), roi_mask_depth)
    cv2.imwrite(str(out_roi_depth), roi_mask_depth)
    save_json(
        {
            "sample_name": image_path.stem,
            "source_bmp": str(image_path),
            "source_pcd": str(pcd_path),
            "scale": scale,
            "angle_deg": float(angle),
            "rotated_bmp_shape_hw": [int(work_bmp.shape[0]), int(work_bmp.shape[1])],
            "rotated_pcd_shape_hw": [int(work_pcd.shape[0]), int(work_pcd.shape[1])],
            "bmp_roi_xywh": [int(x), int(y), int(rw), int(rh)],
            "pcd_roi_xywh": [int(scale * x), int(scale * y), int(scale * rw), int(scale * rh)],
            "crop_bmp_shape_hw": [int(bmp_crop.shape[0]), int(bmp_crop.shape[1])],
            "crop_pcd_shape_hw": [int(pcd_crop.shape[0]), int(pcd_crop.shape[1])],
            "files": {
                "crop_bmp": out_bmp.name,
                "crop_pcd": out_pcd.name,
                "valid_bmp": out_valid.name,
                "roi_bmp": out_roi.name,
                "valid_depth": out_valid_depth.name,
                "roi_depth": out_roi_depth.name,
            },
        },
        out_meta,
    )

    print("Saved:", out_bmp)
    print("Saved:", out_pcd)
    print("Saved:", out_valid)
    print("Saved:", out_roi)
    print("Saved:", out_valid_depth)
    print("Saved:", out_roi_depth)
    print("Saved:", out_meta)


def main():
    input_root = Path("data/input_data")
    out_dir = Path("data/process/manual_crop")

    samples = find_input_images(input_root)
    if not samples:
        raise ValueError(f"No bmp samples found under: {input_root}")

    print(f"Found {len(samples)} sample(s) under {input_root}")
    for image_path, pcd_path in samples:
        process_one_image(image_path, pcd_path, out_dir)


if __name__ == "__main__":
    main()
