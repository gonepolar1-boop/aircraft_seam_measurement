from pathlib import Path

import cv2
import numpy as np
import torch


def build_annotated_source_image(meta, target_h):
    source_image = cv2.imread(meta["image_path"], cv2.IMREAD_GRAYSCALE)
    if source_image is None:
        raise FileNotFoundError(f"Unable to build preview image for sample: {meta['sample_name']}")

    image_bgr = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
    crop_box = meta["crop_box"]
    top = max(0, int(crop_box["top"]))
    left = max(0, int(crop_box["left"]))
    bottom = min(source_image.shape[0], int(crop_box["bottom"]))
    right = min(source_image.shape[1], int(crop_box["right"]))
    cv2.rectangle(
        image_bgr,
        (left, top),
        (max(left, right - 1), max(top, bottom - 1)),
        (0, 0, 255),
        3,
    )
    center_y = int(meta.get("sample_center_y", (top + bottom) // 2))
    center_x = int(meta.get("sample_center_x", (left + right) // 2))
    center_y = min(max(center_y, 0), source_image.shape[0] - 1)
    center_x = min(max(center_x, 0), source_image.shape[1] - 1)
    cv2.drawMarker(
        image_bgr,
        (center_x, center_y),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(
        image_bgr,
        (center_x, center_y),
        8,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        meta["sample_mode"],
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if image_bgr.shape[0] == target_h:
        return image_bgr
    target_w = int(round(image_bgr.shape[1] * target_h / max(image_bgr.shape[0], 1)))
    return cv2.resize(image_bgr, (max(1, target_w), target_h), interpolation=cv2.INTER_LINEAR)


def save_preview_image(image, mask, pred, valid_mask, meta, save_path):
    image_u8 = image.astype(np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    pred_u8 = (pred > 0).astype(np.uint8) * 255
    valid_u8 = (valid_mask > 0).astype(np.uint8) * 255
    annotated_source_image = build_annotated_source_image(meta, image_u8.shape[0])
    overlay = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    overlay[(pred_u8 > 0) & (mask_u8 == 0)] = (0, 0, 255)
    overlay[(pred_u8 > 0) & (mask_u8 > 0)] = (0, 255, 0)
    overlay[(pred_u8 == 0) & (mask_u8 > 0)] = (255, 255, 0)
    canvas = np.concatenate(
        [
            annotated_source_image,
            cv2.cvtColor(valid_u8, cv2.COLOR_GRAY2BGR),
            overlay,
        ],
        axis=1,
    )
    cv2.imwrite(str(save_path), canvas)


def save_epoch_previews(model, dataset, device, preview_dir, epoch, threshold, preview_samples):
    epoch_dir = Path(preview_dir) / f"epoch_{epoch + 1:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for idx in range(min(preview_samples, len(dataset))):
            image_tensor, mask_tensor, valid_mask_tensor, meta = dataset[idx]
            logits = model(image_tensor.unsqueeze(0).to(device))
            pred_tensor = (torch.sigmoid(logits) > threshold).float()[0, 0].detach().cpu()
            image = (image_tensor[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mask = (mask_tensor[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            pred = (pred_tensor.numpy() > 0.5).astype(np.uint8) * 255
            valid_mask = (valid_mask_tensor[0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            save_preview_image(image, mask, pred, valid_mask, meta, epoch_dir / f"sample_{idx:02d}.png")
    if model_was_training:
        model.train()
