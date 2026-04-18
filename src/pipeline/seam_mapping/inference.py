from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ..seam_training.model import build_model
from ..seam_training.utils import MODEL_INPUT_SIZE, REAL_ONLY_DEFAULTS


_MODEL_CACHE: Final[dict[tuple[str, str, str], torch.nn.Module]] = {}


def build_depth_image_from_point_map(point_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if point_map.ndim != 3 or point_map.shape[2] != 3:
        raise ValueError(f"point_map must have shape (H, W, 3), got {point_map.shape}.")
    depth = point_map[:, :, 2]
    finite_mask = np.isfinite(depth)
    depth_image = np.zeros(depth.shape, dtype=np.uint8)
    if finite_mask.any():
        valid_depth = depth[finite_mask]
        depth_min = float(valid_depth.min())
        depth_max = float(valid_depth.max())
        if depth_max > depth_min:
            depth_image[finite_mask] = np.clip(
                (depth[finite_mask] - depth_min) / (depth_max - depth_min) * 255.0,
                0,
                255,
            ).astype(np.uint8)
        else:
            depth_image[finite_mask] = 255
    return depth_image, finite_mask.astype(np.uint8) * 255


def _load_model(checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    resolved_checkpoint = str(Path(checkpoint_path).resolve())
    checkpoint = torch.load(resolved_checkpoint, map_location=device, weights_only=True)
    model_name = str(checkpoint.get("model_name", REAL_ONLY_DEFAULTS["model_name"]))
    base_channels = int(checkpoint.get("model_base_channels", REAL_ONLY_DEFAULTS["model_base_channels"]))
    cache_key = (resolved_checkpoint, str(device), model_name)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    model = build_model(model_name=model_name, base_channels=base_channels).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def preload_model(checkpoint_path: str | Path, device: torch.device | None = None) -> torch.nn.Module:
    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    return _load_model(checkpoint_path, runtime_device)


def predict_mask_from_depth_image(
    depth_image: np.ndarray,
    checkpoint_path: str | Path,
    threshold: float = REAL_ONLY_DEFAULTS["preview_threshold"],
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)
    input_h, input_w = MODEL_INPUT_SIZE
    resized = cv2.resize(depth_image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None, ...].to(device)
    with torch.inference_mode():
        probs = torch.sigmoid(model(tensor))
        probs = F.interpolate(
            probs,
            size=(depth_image.shape[0], depth_image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = (probs[0, 0] >= float(threshold)).to(torch.uint8).mul_(255).cpu().numpy()
    return pred_mask


def predict_mask_from_point_map(
    point_map: np.ndarray,
    checkpoint_path: str | Path,
    threshold: float = REAL_ONLY_DEFAULTS["preview_threshold"],
) -> dict[str, np.ndarray]:
    depth_image, point_valid_mask = build_depth_image_from_point_map(point_map)
    pred_mask = predict_mask_from_depth_image(depth_image, checkpoint_path, threshold)
    valid = point_valid_mask > 0
    pred_mask = pred_mask.astype(np.uint8)
    pred_mask[~valid] = 0
    return {
        "depth_image": depth_image,
        "point_valid_mask_2d": point_valid_mask,
        "pred_mask": pred_mask,
    }
