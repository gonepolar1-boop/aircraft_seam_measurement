from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seam_training.model import build_model
from seam_training.utils import MODEL_INPUT_SIZE, REAL_ONLY_DEFAULTS


def parse_args():
    parser = argparse.ArgumentParser(description="Compare all checkpoints under outputs with full-image evaluation.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model",
        help="Root directory to scan for .pth checkpoints.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "images",
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "masks",
        help="Directory containing ground-truth masks.",
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "valids",
        help="Directory containing valid masks.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=REAL_ONLY_DEFAULTS["preview_threshold"],
        help="Probability threshold used to binarize predictions.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        default=True,
        help="Search the best threshold for each checkpoint based on mean dice over all evaluation samples.",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.1,
        help="Start of threshold search range when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.9,
        help="End of threshold search range when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Step size of threshold search range when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=REAL_ONLY_DEFAULTS["model_base_channels"],
        help="Base channel width used to instantiate the model.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=REAL_ONLY_DEFAULTS["model_name"],
        choices=("unet", "attention_unet"),
        help="Model architecture used to instantiate checkpoints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of checkpoints to evaluate after sorting by path.",
    )
    parser.add_argument(
        "--save-previews",
        action="store_true",
        help="Save per-checkpoint overlays for each evaluation sample.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model" / "checkpoint_eval",
        help="Directory to store CSV/JSON reports and optional previews.",
    )
    return parser.parse_args()


def discover_checkpoints(outputs_root: Path) -> list[Path]:
    checkpoints = sorted(
        path
        for path in outputs_root.rglob("*.pth")
        if path.is_file() and "latest" in path.stem.lower()
    )
    if not checkpoints:
        raise FileNotFoundError(f"No latest checkpoint files found under: {outputs_root}")
    return checkpoints


def discover_samples(image_dir: Path, mask_dir: Path, valid_dir: Path) -> list[dict[str, Path]]:
    samples = []
    for image_path in sorted(image_dir.glob("*")):
        if not image_path.is_file():
            continue
        mask_path = mask_dir / image_path.name
        valid_path = valid_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing valid mask for {image_path.name}: {valid_path}")
        samples.append(
            {
                "name": image_path.stem,
                "image_path": image_path,
                "mask_path": mask_path,
                "valid_path": valid_path,
            }
        )
    if not samples:
        raise FileNotFoundError(f"No evaluation samples found under: {image_dir}")
    return samples


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def infer_model_name_from_path(checkpoint_path: Path, fallback: str) -> str:
    tokens = [part.lower() for part in checkpoint_path.parts]
    joined = "/".join(tokens)
    if "attention_unet" in joined:
        return "attention_unet"
    if "unet" in joined:
        return "unet"
    return fallback


def load_model(checkpoint_path: Path, device: torch.device, model_name: str, base_channels: int) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name is None:
        checkpoint_model_name = infer_model_name_from_path(checkpoint_path, model_name)
    checkpoint_model_name = str(checkpoint_model_name)
    checkpoint_base_channels = int(checkpoint.get("model_base_channels", base_channels))
    model = build_model(model_name=checkpoint_model_name, base_channels=checkpoint_base_channels).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path} with inferred model_name={checkpoint_model_name!r} "
            f"and base_channels={checkpoint_base_channels}. Original error: {exc}"
        ) from exc
    model.eval()
    return model


def predict_mask(model: torch.nn.Module, image: np.ndarray, threshold: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    input_h, input_w = MODEL_INPUT_SIZE
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None, ...].to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    probs = cv2.resize(probs, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    preds = (probs > threshold).astype(np.uint8)
    return probs, preds


def build_threshold_candidates(args) -> list[float]:
    if not args.auto_threshold:
        return [float(args.threshold)]
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if args.threshold_end < args.threshold_start:
        raise ValueError("--threshold-end must be greater than or equal to --threshold-start.")

    thresholds = []
    value = float(args.threshold_start)
    end = float(args.threshold_end)
    step = float(args.threshold_step)
    while value <= end + 1e-9:
        thresholds.append(round(value, 6))
        value += step
    if not thresholds:
        raise ValueError("No threshold candidates generated.")
    return thresholds


def compute_metrics(pred_mask: np.ndarray, true_mask: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    valid = valid_mask > 0
    pred = pred_mask[valid] > 0
    true = true_mask[valid] > 0

    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, ~true).sum())
    fn = float(np.logical_and(~pred, true).sum())

    dice = (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)
    iou = tp / max(tp + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def save_preview(image: np.ndarray, mask: np.ndarray, pred: np.ndarray, valid_mask: np.ndarray, save_path: Path, title: str):
    image_u8 = image.astype(np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    pred_u8 = (pred > 0).astype(np.uint8) * 255
    valid_u8 = (valid_mask > 0).astype(np.uint8) * 255
    overlay = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)
    overlay[(pred_u8 > 0) & (mask_u8 == 0)] = (0, 0, 255)
    overlay[(pred_u8 > 0) & (mask_u8 > 0)] = (0, 255, 0)
    overlay[(pred_u8 == 0) & (mask_u8 > 0)] = (255, 255, 0)
    canvas = np.concatenate(
        [
            cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(valid_u8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(pred_u8, cv2.COLOR_GRAY2BGR),
            overlay,
        ],
        axis=1,
    )
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), canvas)


def evaluate_checkpoint(
    checkpoint_path: Path,
    samples: list[dict[str, Path]],
    thresholds: list[float],
    device: torch.device,
    model_name: str,
    base_channels: int,
    preview_root: Path | None,
) -> dict[str, object]:
    model = load_model(checkpoint_path, device, model_name, base_channels)
    cached_samples = []
    for sample in samples:
        image = load_gray(sample["image_path"])
        mask = load_gray(sample["mask_path"])
        valid_mask = load_gray(sample["valid_path"])
        probs, _ = predict_mask(model, image, thresholds[0], device)
        cached_samples.append(
            {
                "sample": sample,
                "image": image,
                "mask": mask,
                "valid_mask": valid_mask,
                "probs": probs,
            }
        )

    best_threshold = thresholds[0]
    best_mean_dice = -1.0
    for threshold in thresholds:
        dice_scores = []
        for item in cached_samples:
            pred = (item["probs"] > threshold).astype(np.uint8)
            metrics = compute_metrics(pred, item["mask"], item["valid_mask"])
            dice_scores.append(metrics["dice"])
        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_threshold = threshold

    per_sample_rows = []
    preview_dir = None if preview_root is None else preview_root / checkpoint_path.parent.parent.name / checkpoint_path.stem
    for item in cached_samples:
        sample = item["sample"]
        pred = (item["probs"] > best_threshold).astype(np.uint8)
        metrics = compute_metrics(pred, item["mask"], item["valid_mask"])
        row = {
            "checkpoint": str(checkpoint_path),
            "sample_name": sample["name"],
            "threshold": best_threshold,
            **metrics,
        }
        per_sample_rows.append(row)
        if preview_dir is not None:
            save_preview(
                item["image"],
                item["mask"],
                pred,
                item["valid_mask"],
                preview_dir / f"{sample['name']}.png",
                title=f"{checkpoint_path.parent.parent.name}/{checkpoint_path.stem}:thr={best_threshold:.2f}:{sample['name']}",
            )

    metric_names = ("dice", "iou", "precision", "recall")
    summary = {
        metric: float(np.mean([row[metric] for row in per_sample_rows])) for metric in metric_names
    }
    summary["checkpoint"] = str(checkpoint_path)
    summary["checkpoint_name"] = checkpoint_path.stem
    summary["checkpoint_dir"] = str(checkpoint_path.parent)
    summary["num_samples"] = len(per_sample_rows)
    summary["threshold"] = float(best_threshold)
    summary["per_sample"] = per_sample_rows
    return summary


def write_csv(rows: list[dict[str, object]], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "checkpoint_name", "checkpoint", "threshold", "dice", "iou", "precision", "recall", "num_samples"]
    with save_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, object], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    checkpoints = discover_checkpoints(args.outputs_root)
    if args.limit is not None:
        checkpoints = checkpoints[: max(0, args.limit)]
    samples = discover_samples(args.image_dir, args.mask_dir, args.valid_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = build_threshold_candidates(args)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    preview_root = report_dir / "previews" if args.save_previews else None

    results = []
    skipped = []
    for checkpoint_path in checkpoints:
        try:
            result = evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                samples=samples,
                thresholds=thresholds,
                device=device,
                model_name=args.model_name,
                base_channels=args.base_channels,
                preview_root=preview_root,
            )
        except Exception as exc:
            skipped.append({"checkpoint": str(checkpoint_path), "error": str(exc)})
            print(f"SKIP {checkpoint_path} | {exc}")
            continue
        results.append(result)
        print(
            f"{checkpoint_path} | threshold={result['threshold']:.2f} "
            f"dice={result['dice']:.4f} iou={result['iou']:.4f} "
            f"precision={result['precision']:.4f} recall={result['recall']:.4f}"
        )

    if not results:
        raise RuntimeError("No compatible checkpoints were evaluated successfully.")

    ranked = sorted(results, key=lambda item: item["dice"], reverse=True)
    csv_rows = []
    for rank, result in enumerate(ranked, start=1):
        csv_rows.append(
            {
                "rank": rank,
                "checkpoint_name": result["checkpoint_name"],
                "checkpoint": result["checkpoint"],
                "threshold": f"{result['threshold']:.6f}",
                "dice": f"{result['dice']:.6f}",
                "iou": f"{result['iou']:.6f}",
                "precision": f"{result['precision']:.6f}",
                "recall": f"{result['recall']:.6f}",
                "num_samples": result["num_samples"],
            }
        )

    write_csv(csv_rows, report_dir / "checkpoint_ranking.csv")
    write_json(
        {
            "auto_threshold": args.auto_threshold,
            "threshold_candidates": thresholds,
            "device": str(device),
            "model_name": args.model_name,
            "base_channels": args.base_channels,
            "outputs_root": str(args.outputs_root),
            "samples": [sample["name"] for sample in samples],
            "skipped_checkpoints": skipped,
            "ranked_results": ranked,
        },
        report_dir / "checkpoint_ranking.json",
    )

    best = ranked[0]
    print("\nBest checkpoint")
    print(
        f"{best['checkpoint']} | threshold={best['threshold']:.2f} "
        f"dice={best['dice']:.4f} iou={best['iou']:.4f} "
        f"precision={best['precision']:.4f} recall={best['recall']:.4f}"
    )
    print(f"Saved CSV report to: {report_dir / 'checkpoint_ranking.csv'}")
    print(f"Saved JSON report to: {report_dir / 'checkpoint_ranking.json'}")


if __name__ == "__main__":
    main()
