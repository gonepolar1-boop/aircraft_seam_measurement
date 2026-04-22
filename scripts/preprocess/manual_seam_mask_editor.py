import json
from pathlib import Path

import cv2
import numpy as np


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def load_mask_or_empty(path: Path, shape_hw):
    if path.exists():
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {path}")
        mask = resize_mask_to_shape(mask, shape_hw)
        return (mask > 0).astype(np.uint8) * 255
    return np.zeros(shape_hw, dtype=np.uint8)


def ensure_gray(mask: np.ndarray):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resize_mask_to_shape(mask: np.ndarray, shape_hw):
    target_h, target_w = shape_hw
    src_h, src_w = mask.shape[:2]
    if (src_h, src_w) == (target_h, target_w):
        return mask

    scale_h = target_h / src_h
    scale_w = target_w / src_w
    if abs(scale_h - scale_w) > 1e-6:
        raise ValueError(f"Mask aspect mismatch: {(src_h, src_w)} vs {(target_h, target_w)}")

    resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    print(f"Resized mask from {(src_h, src_w)} to {(target_h, target_w)}")
    return resized


def find_roi_mask(sample_dir: Path):
    for name in ("roi_mask_depth.png", "valid_depth.png", "roi_mask.png", "valid.png"):
        path = sample_dir / name
        if path.exists():
            return path
    return None


def load_precise_roi_mask(sample_dir: Path, shape_hw):
    roi_path = find_roi_mask(sample_dir)
    if roi_path is None:
        return None, None

    roi_mask = ensure_gray(load_image(roi_path))
    if roi_mask.shape == shape_hw:
        return roi_mask, roi_path

    meta_path = sample_dir / "crop_meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        crop_pcd_shape_hw = tuple(meta.get("crop_pcd_shape_hw", []))
        if crop_pcd_shape_hw == shape_hw:
            files = meta.get("files", {})
            precise_name = files.get("roi_depth") or files.get("valid_depth")
            if precise_name:
                precise_path = sample_dir / precise_name
                if precise_path.exists():
                    precise_mask = ensure_gray(load_image(precise_path))
                    if precise_mask.shape == shape_hw:
                        return precise_mask, precise_path

    return resize_mask_to_shape(roi_mask, shape_hw), roi_path


def load_training_valid_mask(sample_dir: Path, shape_hw):
    precise_mask, precise_path = load_precise_roi_mask(sample_dir, shape_hw)
    if precise_mask is None or precise_path is None:
        raise ValueError(f"ROI mask not found under: {sample_dir}")
    aligned_mask = resize_mask_to_shape(precise_mask, shape_hw)
    return (aligned_mask > 0).astype(np.uint8) * 255, precise_path


class SeamMaskEditor:
    def __init__(self, image, roi_mask, init_mask):
        self.image = image
        self.h, self.w = image.shape[:2]

        self.roi_mask = ensure_gray(roi_mask)
        self.roi_mask = (self.roi_mask > 0).astype(np.uint8) * 255

        self.mask = ensure_gray(init_mask)
        self.mask = (self.mask > 0).astype(np.uint8) * 255
        self.mask = cv2.bitwise_and(self.mask, self.roi_mask)

        self.prev_mask = self.mask.copy()

        self.brush_size = 8
        self.drawing = False
        self.draw_value = 255
        self.last_point = None
        self.cursor_point = (self.w // 2, self.h // 2)
        self.line_mode = False
        self.line_start_point = None
        self.line_preview_point = None
        self.line_value = 255

        self.show_overlay = True
        self.view_scale = 1.0
        self.min_view_scale = 0.25
        self.max_view_scale = 12.0
        self.pan_x = 0
        self.pan_y = 0
        self.panning = False
        self.pan_start_mouse = None
        self.pan_start_offset = None
        self.window_name = "Manual Seam Mask Editor"

    def draw_hint_text(self, canvas, text, origin, color, font_scale=0.65):
        overlay = canvas.copy()
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        pad = max(1, int(round(6 * font_scale / 0.65)))
        x0 = max(0, x - pad)
        y0 = max(0, y - th - pad)
        x1 = min(canvas.shape[1], x + tw + pad)
        y1 = min(canvas.shape[0], y + baseline + pad)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.22, canvas, 0.78, 0, dst=canvas)
        cv2.putText(canvas, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_viewport_hint(self, canvas, text, row, color):
        font_scale = max(0.08, 0.65 / self.view_scale)
        pad_x = max(1, int(round(10 / self.view_scale)))
        step_y = max(1, int(round(28 / self.view_scale)))
        x = self.pan_x + pad_x
        y = self.pan_y + step_y * (row + 1)
        y = min(canvas.shape[0] - 2, max(2, y))
        self.draw_hint_text(canvas, text, (x, y), color, font_scale=font_scale)

    def backup(self):
        self.prev_mask = self.mask.copy()

    def undo(self):
        temp = self.mask.copy()
        self.mask = self.prev_mask.copy()
        self.prev_mask = temp

    def clamp_pan(self, view_w, view_h):
        max_pan_x = max(0, self.w - view_w)
        max_pan_y = max(0, self.h - view_h)
        self.pan_x = min(max(0, self.pan_x), max_pan_x)
        self.pan_y = min(max(0, self.pan_y), max_pan_y)

    def get_view_size(self):
        view_w = max(1, int(round(self.w / self.view_scale)))
        view_h = max(1, int(round(self.h / self.view_scale)))
        return min(self.w, view_w), min(self.h, view_h)

    def screen_to_image(self, x, y, window_w, window_h):
        window_w = max(1, window_w)
        window_h = max(1, window_h)
        view_w, view_h = self.get_view_size()
        self.clamp_pan(view_w, view_h)

        img_x = self.pan_x + x * view_w / window_w
        img_y = self.pan_y + y * view_h / window_h
        img_x = int(np.clip(round(img_x), 0, self.w - 1))
        img_y = int(np.clip(round(img_y), 0, self.h - 1))
        return img_x, img_y

    def zoom_at(self, factor, anchor_point):
        old_scale = self.view_scale
        new_scale = min(self.max_view_scale, max(self.min_view_scale, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-6:
            return

        ax, ay = anchor_point
        rel_x = (ax - self.pan_x) / max(1e-6, self.w / old_scale)
        rel_y = (ay - self.pan_y) / max(1e-6, self.h / old_scale)
        new_view_w, new_view_h = max(1, int(round(self.w / new_scale))), max(1, int(round(self.h / new_scale)))
        self.view_scale = new_scale
        self.pan_x = int(round(ax - rel_x * new_view_w))
        self.pan_y = int(round(ay - rel_y * new_view_h))
        self.clamp_pan(new_view_w, new_view_h)

    def draw_on_mask(self, p1, p2, value):
        temp = np.zeros_like(self.mask)
        thickness = max(1, self.brush_size * 2)

        if p2 is None:
            cv2.circle(temp, p1, self.brush_size, 255, -1)
        else:
            cv2.line(temp, p1, p2, 255, thickness=thickness)

        if value == 255:
            temp = cv2.bitwise_and(temp, self.roi_mask)
            self.mask[temp > 0] = 255
        else:
            self.mask[temp > 0] = 0

    def commit_line(self, end_point):
        if self.line_start_point is None:
            return
        self.draw_on_mask(self.line_start_point, end_point, self.line_value)
        self.mask = cv2.bitwise_and(self.mask, self.roi_mask)
        self.line_start_point = None
        self.line_preview_point = None

    def on_mouse(self, event, x, y, flags, param):
        window_w, window_h = param
        img_point = self.screen_to_image(x, y, window_w, window_h)
        self.cursor_point = img_point

        if event == cv2.EVENT_MBUTTONDOWN:
            self.panning = True
            self.pan_start_mouse = (x, y)
            self.pan_start_offset = (self.pan_x, self.pan_y)
            return

        elif event == cv2.EVENT_MBUTTONUP:
            self.panning = False
            self.pan_start_mouse = None
            self.pan_start_offset = None
            return

        elif event == cv2.EVENT_MOUSEMOVE and self.panning:
            view_w, view_h = self.get_view_size()
            dx = (x - self.pan_start_mouse[0]) * view_w / max(1, window_w)
            dy = (y - self.pan_start_mouse[1]) * view_h / max(1, window_h)
            self.pan_x = int(round(self.pan_start_offset[0] - dx))
            self.pan_y = int(round(self.pan_start_offset[1] - dy))
            self.clamp_pan(view_w, view_h)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.line_mode and self.line_start_point is not None:
            self.line_preview_point = img_point

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.line_mode:
                if self.line_start_point is None:
                    self.backup()
                    self.line_start_point = img_point
                    self.line_preview_point = img_point
                    self.line_value = 255
                else:
                    self.commit_line(img_point)
                return
            self.backup()
            self.drawing = True
            self.draw_value = 255
            self.last_point = img_point
            self.draw_on_mask(img_point, None, 255)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.line_mode:
                if self.line_start_point is None:
                    self.backup()
                    self.line_start_point = img_point
                    self.line_preview_point = img_point
                    self.line_value = 0
                else:
                    self.commit_line(img_point)
                return
            self.backup()
            self.drawing = True
            self.draw_value = 0
            self.last_point = img_point
            self.draw_on_mask(img_point, None, 0)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            current_point = img_point
            self.draw_on_mask(self.last_point, current_point, self.draw_value)
            self.last_point = current_point

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            self.last_point = None
            self.mask = cv2.bitwise_and(self.mask, self.roi_mask)

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.zoom_at(1.25, img_point)
            else:
                self.zoom_at(0.8, img_point)

    def render(self):
        if self.image.ndim == 2:
            base = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            base = self.image.copy()

        # roi 边界
        roi_edge = cv2.Canny(self.roi_mask, 50, 150)
        base[roi_edge > 0] = (0, 255, 0)

        if self.show_overlay:
            color_mask = np.zeros_like(base)
            color_mask[self.mask > 0] = (0, 0, 255)
            vis = cv2.addWeighted(base, 0.78, color_mask, 0.22, 0)
        else:
            vis = base

        if self.line_mode and self.line_start_point is not None and self.line_preview_point is not None:
            preview = vis.copy()
            thickness = max(1, self.brush_size * 2)
            color = (0, 255, 255) if self.line_value == 255 else (255, 255, 0)
            cv2.line(preview, self.line_start_point, self.line_preview_point, color, thickness=thickness)
            vis = cv2.addWeighted(vis, 0.75, preview, 0.25, 0)

        view_w, view_h = self.get_view_size()
        self.clamp_pan(view_w, view_h)
        x0 = self.pan_x
        y0 = self.pan_y
        x1 = x0 + view_w
        y1 = y0 + view_h
        mode_text = "line" if self.line_mode else "freehand"
        info1 = "L:add  R:erase  Wheel:zoom  Mid-drag:pan"
        info2 = "F:freehand  T:line  +/-:brush  0:fit  O:overlay"
        info3 = f"mode={mode_text}  brush={self.brush_size}  zoom={self.view_scale:.2f}x"
        info4 = "S:save  U:undo  R:reset  N:next  Q/Esc:quit"

        self.draw_viewport_hint(vis, info1, 0, (120, 220, 220))
        self.draw_viewport_hint(vis, info2, 1, (120, 220, 220))
        self.draw_viewport_hint(vis, info3, 2, (180, 220, 180))
        self.draw_viewport_hint(vis, info4, 3, (180, 220, 180))

        view = vis[y0:y1, x0:x1].copy()
        cursor_x = int(round((self.cursor_point[0] - x0) * view.shape[1] / max(1, view_w)))
        cursor_y = int(round((self.cursor_point[1] - y0) * view.shape[0] / max(1, view_h)))
        brush_r = max(1, int(round(self.brush_size * view.shape[1] / max(1, view_w))))
        cv2.circle(view, (cursor_x, cursor_y), brush_r, (255, 255, 0), 1, cv2.LINE_AA)

        return view


def edit_one_sample(image_path: Path, sample_dir: Path, out_path: Path):
    mask_path = out_path if out_path.exists() else None

    image = load_image(image_path)
    h, w = image.shape[:2]

    roi_mask, roi_path = load_precise_roi_mask(sample_dir, (h, w))
    if roi_mask is None or roi_path is None:
        raise ValueError(f"ROI mask not found under: {sample_dir}")

    if mask_path is not None:
        init_mask = load_mask_or_empty(mask_path, (h, w))
    else:
        init_mask = np.zeros((h, w), dtype=np.uint8)

    editor = SeamMaskEditor(image, roi_mask, init_mask)

    cv2.namedWindow(editor.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(editor.window_name, 1400, 900)
    cv2.setMouseCallback(editor.window_name, editor.on_mouse, (w, h))

    print(f"Editing: {image_path}")
    print(f"ROI: {roi_path}")
    print(f"Output: {out_path}")

    while True:
        vis = editor.render()
        cv2.setMouseCallback(editor.window_name, editor.on_mouse, (vis.shape[1], vis.shape[0]))
        cv2.imshow(editor.window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("s"):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), editor.mask)
            print(f"saved: {out_path}")

        elif key == ord("u"):
            editor.undo()
            print("undo")

        elif key == ord("r"):
            editor.backup()
            editor.mask = np.zeros((h, w), dtype=np.uint8)
            print("reset to empty mask")

        elif key == ord("o"):
            editor.show_overlay = not editor.show_overlay

        elif key in (ord("f"), ord("F")):
            editor.line_mode = False
            editor.line_start_point = None
            editor.line_preview_point = None
            print("mode: freehand")

        elif key in (ord("t"), ord("T")):
            editor.line_mode = True
            editor.line_start_point = None
            editor.line_preview_point = None
            print("mode: line")

        elif key in (ord("+"), ord("=")):
            editor.brush_size = min(100, editor.brush_size + 1)
            print(f"brush: {editor.brush_size}")

        elif key in (ord("-"), ord("_")):
            editor.brush_size = max(1, editor.brush_size - 1)
            print(f"brush: {editor.brush_size}")

        elif key == ord("0"):
            editor.view_scale = 1.0
            editor.pan_x = 0
            editor.pan_y = 0
            print("view reset")

        elif key == ord("n"):
            break

        elif key == ord("q") or key == 27:
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()
    return True


def main():
    depth_root = Path("data/process/depth")
    crop_root = Path("data/process/manual_crop")
    train_image_root = Path("data/real_train/images")
    train_mask_root = Path("data/real_train/masks")
    train_valid_root = Path("data/real_train/valids")

    image_paths = sorted(depth_root.glob("*/depth.png"))
    if not image_paths:
        raise ValueError(f"No depth images found under: {depth_root}")

    print(f"Found {len(image_paths)} sample(s) under {depth_root}")

    for image_path in image_paths:
        sample_name = image_path.parent.name
        sample_crop_dir = crop_root / sample_name
        if find_roi_mask(sample_crop_dir) is None:
            raise FileNotFoundError(f"Missing roi mask under: {sample_crop_dir}")

        train_image_path = train_image_root / f"{sample_name}.png"
        train_valid_path = train_valid_root / f"{sample_name}.png"
        out_path = train_mask_root / f"{sample_name}.png"
        train_image_path.parent.mkdir(parents=True, exist_ok=True)
        train_valid_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = load_image(image_path)
        cv2.imwrite(str(train_image_path), image)
        valid_image, valid_source_path = load_training_valid_mask(sample_crop_dir, image.shape[:2])
        cv2.imwrite(str(train_valid_path), valid_image)
        print(f"prepared train image: {train_image_path}")
        print(f"prepared train valid: {train_valid_path} (from {valid_source_path})")
        should_continue = edit_one_sample(image_path, sample_crop_dir, out_path)
        if not should_continue:
            break


if __name__ == "__main__":
    main()
