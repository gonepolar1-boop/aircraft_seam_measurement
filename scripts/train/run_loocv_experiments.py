from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from seam_training.data import SeamDataset
from seam_training.model import SegmentationCriterion, build_model
from seam_training.train import load_checkpoint, save_checkpoint
from seam_training.utils import MODEL_INPUT_SIZE, REAL_ONLY_DEFAULTS, build_patch_sampling_cfg, set_seed


HISTORY_KEYS = (
    "train_losses",
    "val_losses",
    "val_dice",
    "val_iou",
    "val_precision",
    "val_recall",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Attention U-Net with leave-one-out validation on the five real seam samples.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "images",
        help="Directory containing real training images.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "masks",
        help="Directory containing real training masks.",
    )
    parser.add_argument(
        "--valid-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "real_train" / "valids",
        help="Directory containing real training valid masks.",
    )
    parser.add_argument(
        "--samples",
        nargs="*",
        default=None,
        help="Optional ordered sample names to use, for example: 1 2 3 4 5",
    )
    parser.add_argument(
        "--allow-any-count",
        action="store_true",
        help="Allow running LOOCV even when the discovered sample count is not exactly five.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=REAL_ONLY_DEFAULTS["epochs"],
        help="Training epochs for each fold.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=REAL_ONLY_DEFAULTS["train_steps"],
        help="Number of sampled training batches per epoch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=REAL_ONLY_DEFAULTS["batch_size"],
        help="Batch size for each training step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=REAL_ONLY_DEFAULTS["lr"],
        help="Learning rate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=REAL_ONLY_DEFAULTS["num_workers"],
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=REAL_ONLY_DEFAULTS["model_name"],
        choices=("unet", "attention_unet"),
        help="Segmentation model to train. Defaults to the current config.json value.",
    )
    parser.add_argument(
        "--model-base-channels",
        type=int,
        default=REAL_ONLY_DEFAULTS["model_base_channels"],
        help="Base channel count used to instantiate the model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=REAL_ONLY_DEFAULTS["preview_threshold"],
        help="Fixed probability threshold used for holdout metrics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=REAL_ONLY_DEFAULTS["seed"],
        help="Base random seed. Each fold uses seed + fold_index.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Output folder name under outputs/model. Auto-generated when omitted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "model",
        help="Root directory for fold artifacts.",
    )
    patch_group = parser.add_mutually_exclusive_group()
    patch_group.add_argument("--use-patch-sampling", dest="use_patch_sampling", action="store_true")
    patch_group.add_argument("--no-patch-sampling", dest="use_patch_sampling", action="store_false")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(
        use_patch_sampling=REAL_ONLY_DEFAULTS["use_patch_sampling"],
        resume=REAL_ONLY_DEFAULTS["resume"],
    )
    return parser.parse_args()


def build_run_name(model_name: str) -> str:
    timestamp = datetime.now().strftime("%m%d%H%M")
    return f"{timestamp}_loocv_{model_name}"


def configure_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def save_json(payload: dict[str, object], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def save_csv(rows: list[dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_history(history_path: Path) -> dict[str, list[float]]:
    if not history_path.exists():
        return {key: [] for key in HISTORY_KEYS}
    with history_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    history = {key: payload.get(key, []) for key in HISTORY_KEYS}
    for key in HISTORY_KEYS:
        history[key] = [float(value) for value in history[key]]
    return history


def save_history(history: dict[str, list[float]], history_path: Path):
    save_json(history, history_path)


def discover_samples(image_dir: Path, mask_dir: Path, valid_dir: Path, selected_names: list[str] | None) -> list[dict[str, Path]]:
    if selected_names is None:
        candidate_names = sorted(path.stem for path in image_dir.glob("*") if path.is_file())
    else:
        candidate_names = [str(name) for name in selected_names]
    if not candidate_names:
        raise FileNotFoundError(f"No samples found under: {image_dir}")

    samples = []
    for name in candidate_names:
        image_path = image_dir / f"{name}.png"
        mask_path = mask_dir / f"{name}.png"
        valid_path = valid_dir / f"{name}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image for sample {name}: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for sample {name}: {mask_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing valid mask for sample {name}: {valid_path}")
        samples.append(
            {
                "sample_name": name,
                "image_path": image_path,
                "mask_path": mask_path,
                "valid_path": valid_path,
            }
        )
    return samples


def build_fold_paths(run_dir: Path, holdout_name: str) -> dict[str, Path]:
    fold_dir = run_dir / f"holdout_{holdout_name}"
    checkpoints_dir = fold_dir / "checkpoints"
    previews_dir = fold_dir / "previews"
    metrics_dir = fold_dir / "metrics"
    logs_dir = fold_dir / "logs"
    return {
        "fold_dir": fold_dir,
        "checkpoints_dir": checkpoints_dir,
        "previews_dir": previews_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
        "latest_checkpoint": checkpoints_dir / "latest.pth",
        "best_checkpoint": checkpoints_dir / "best.pth",
        "history_path": metrics_dir / "history.json",
        "best_summary_path": metrics_dir / "best_summary.json",
        "split_path": metrics_dir / "split.json",
        "train_log": logs_dir / "train.log",
        "best_preview_path": previews_dir / "best_holdout.png",
    }


def ensure_fold_dirs(paths: dict[str, Path]):
    for key in ("checkpoints_dir", "previews_dir", "metrics_dir", "logs_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


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


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    model_name: str,
    base_channels: int,
) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_model_name = str(checkpoint.get("model_name", model_name))
    checkpoint_base_channels = int(checkpoint.get("model_base_channels", base_channels))
    model = build_model(model_name=checkpoint_model_name, base_channels=checkpoint_base_channels).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def predict_mask(model: torch.nn.Module, image: np.ndarray, threshold: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    input_h, input_w = MODEL_INPUT_SIZE
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None, ...].to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    probs = cv2.resize(probs, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = (probs > threshold).astype(np.uint8)
    return probs, pred


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


def make_dataset(
    cfg: dict[str, object],
    samples: list[dict[str, Path]],
    augment: bool,
    length: int,
    use_patch_sampling: bool,
) -> SeamDataset:
    return SeamDataset(
        image_dir=cfg["image_dir"],
        mask_dir=cfg["mask_dir"],
        valid_dir=cfg["valid_dir"],
        img_size=cfg["img_size"],
        augment=augment,
        samples=samples,
        length=length,
        use_patch_sampling=use_patch_sampling,
        patch_sampling_cfg=cfg["patch_sampling"],
    )


def prepare_samples_for_dataset(
    cfg: dict[str, object],
    samples: list[dict[str, Path]],
    use_patch_sampling: bool,
) -> list[dict[str, Path]]:
    if not use_patch_sampling:
        return list(samples)

    discovery_dataset = SeamDataset(
        image_dir=cfg["image_dir"],
        mask_dir=cfg["mask_dir"],
        valid_dir=cfg["valid_dir"],
        img_size=cfg["img_size"],
        augment=False,
        length=max(1, len(samples)),
        use_patch_sampling=True,
        patch_sampling_cfg=cfg["patch_sampling"],
    )
    sample_lookup = {item["sample_name"]: item for item in discovery_dataset.samples}
    prepared_samples = []
    for sample in samples:
        sample_name = sample["sample_name"]
        if sample_name not in sample_lookup:
            raise KeyError(f"Sample {sample_name!r} was not found while preparing patch-sampling metadata.")
        prepared_samples.append(sample_lookup[sample_name])
    return prepared_samples


def evaluate_holdout(
    model: torch.nn.Module,
    criterion: SegmentationCriterion,
    val_dataset: SeamDataset,
    holdout_sample: dict[str, Path],
    threshold: float,
    device: torch.device,
) -> dict[str, object]:
    image_tensor, mask_tensor, valid_tensor, _meta = val_dataset[0]
    image_batch = image_tensor.unsqueeze(0).to(device)
    mask_batch = mask_tensor.unsqueeze(0).to(device)
    valid_batch = valid_tensor.unsqueeze(0).to(device)
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        logits = model(image_batch)
        loss = criterion(logits, mask_batch, valid_batch).item()

    original_image = load_gray(holdout_sample["image_path"])
    original_mask = load_gray(holdout_sample["mask_path"])
    original_valid = load_gray(holdout_sample["valid_path"])
    _probs, pred = predict_mask(model, original_image, threshold, device)
    metrics = compute_metrics(pred, original_mask, original_valid)
    if model_was_training:
        model.train()
    return {
        "val_loss": float(loss),
        "metrics": metrics,
        "image": original_image,
        "mask": original_mask,
        "valid_mask": original_valid,
        "pred": pred,
    }


def evaluate_saved_checkpoint(
    checkpoint_path: Path,
    holdout_sample: dict[str, Path],
    threshold: float,
    device: torch.device,
    model_name: str,
    base_channels: int,
    expected_epoch_field: int | None = None,
) -> dict[str, object]:
    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
        base_channels=base_channels,
    )
    checkpoint_epoch_field = int(checkpoint.get("epoch", -1))
    if expected_epoch_field is not None and checkpoint_epoch_field != expected_epoch_field:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has epoch field {checkpoint_epoch_field}, "
            f"expected {expected_epoch_field}. This fold may not have finished training."
        )

    image = load_gray(holdout_sample["image_path"])
    mask = load_gray(holdout_sample["mask_path"])
    valid_mask = load_gray(holdout_sample["valid_path"])
    _probs, pred = predict_mask(model, image, threshold, device)
    metrics = compute_metrics(pred, mask, valid_mask)
    return {
        "holdout_sample": holdout_sample["sample_name"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch_field": checkpoint_epoch_field,
        "dice": float(metrics["dice"]),
        "iou": float(metrics["iou"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
    }


def train_one_fold(
    cfg: dict[str, object],
    fold_index: int,
    train_samples: list[dict[str, Path]],
    holdout_sample: dict[str, Path],
    run_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    holdout_name = holdout_sample["sample_name"]
    paths = build_fold_paths(run_dir, holdout_name)
    ensure_fold_dirs(paths)
    configure_logging(paths["train_log"])
    logging.info("Starting fold %d | holdout=%s", fold_index, holdout_name)
    logging.info("Fold artifacts will be saved under %s", paths["fold_dir"])
    logging.info("Training samples: %s", ", ".join(sample["sample_name"] for sample in train_samples))
    logging.info("Validation sample: %s", holdout_name)

    save_json(
        {
            "fold_index": fold_index,
            "train_samples": [sample["sample_name"] for sample in train_samples],
            "holdout_sample": holdout_name,
        },
        paths["split_path"],
    )

    set_seed(int(cfg["seed"]) + fold_index)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_dataset_samples = prepare_samples_for_dataset(
        cfg=cfg,
        samples=train_samples,
        use_patch_sampling=bool(cfg["use_patch_sampling"]),
    )
    train_dataset = make_dataset(
        cfg,
        train_dataset_samples,
        augment=True,
        length=int(cfg["batch_size"]) * int(cfg["train_steps"]),
        use_patch_sampling=bool(cfg["use_patch_sampling"]),
    )
    val_dataset = make_dataset(
        cfg,
        [holdout_sample],
        augment=False,
        length=1,
        use_patch_sampling=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(cfg["num_workers"]) > 0),
    )

    model = build_model(
        model_name=str(cfg["model_name"]),
        base_channels=int(cfg["model_base_channels"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    criterion = SegmentationCriterion(
        pos_weight=float(cfg["loss_pos_weight"]),
        bce_weight=float(cfg["loss_bce_weight"]),
        dice_weight=float(cfg["loss_dice_weight"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {key: [] for key in HISTORY_KEYS}
    start_epoch = 0
    best_dice = -1.0
    best_summary = None
    if bool(cfg["resume"]) and paths["latest_checkpoint"].exists():
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, paths["latest_checkpoint"], device, cfg)
        history = load_history(paths["history_path"])
        if history["val_dice"]:
            best_epoch_index = int(np.argmax(np.array(history["val_dice"], dtype=np.float32)))
            best_dice = float(history["val_dice"][best_epoch_index])
        if paths["best_summary_path"].exists():
            with paths["best_summary_path"].open("r", encoding="utf-8") as fh:
                best_summary = json.load(fh)
        logging.info("Resuming fold %d from epoch=%d", fold_index, start_epoch)

    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        progress = tqdm(train_loader, desc=f"Fold {fold_index} Epoch {epoch + 1}/{cfg['epochs']}")
        for images, masks, valid_masks, _meta in progress:
            images = images.to(device)
            masks = masks.to(device)
            valid_masks = valid_masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks, valid_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            batch_count += 1
            progress.set_postfix(loss=float(loss.item()))

        train_loss = epoch_loss / max(batch_count, 1)
        evaluation = evaluate_holdout(
            model=model,
            criterion=criterion,
            val_dataset=val_dataset,
            holdout_sample=holdout_sample,
            threshold=float(cfg["threshold"]),
            device=device,
        )
        metrics = evaluation["metrics"]

        history["train_losses"].append(float(train_loss))
        history["val_losses"].append(float(evaluation["val_loss"]))
        history["val_dice"].append(float(metrics["dice"]))
        history["val_iou"].append(float(metrics["iou"]))
        history["val_precision"].append(float(metrics["precision"]))
        history["val_recall"].append(float(metrics["recall"]))
        save_history(history, paths["history_path"])

        epoch_number = epoch + 1
        save_checkpoint(model, optimizer, epoch, paths["latest_checkpoint"], cfg)
        logging.info(
            "Fold %d Epoch %d/%d train_loss=%.4f val_loss=%.4f dice=%.4f iou=%.4f precision=%.4f recall=%.4f",
            fold_index,
            epoch_number,
            int(cfg["epochs"]),
            train_loss,
            float(evaluation["val_loss"]),
            float(metrics["dice"]),
            float(metrics["iou"]),
            float(metrics["precision"]),
            float(metrics["recall"]),
        )

        if float(metrics["dice"]) >= best_dice:
            best_dice = float(metrics["dice"])
            save_checkpoint(model, optimizer, epoch, paths["best_checkpoint"], cfg)
            save_preview(
                evaluation["image"],
                evaluation["mask"],
                evaluation["pred"],
                evaluation["valid_mask"],
                paths["best_preview_path"],
                title=f"fold={fold_index} holdout={holdout_name} epoch={epoch_number} dice={best_dice:.4f}",
            )
            best_summary = {
                "fold_index": fold_index,
                "holdout_sample": holdout_name,
                "best_epoch": epoch_number,
                "best_checkpoint": str(paths["best_checkpoint"]),
                "threshold": float(cfg["threshold"]),
                "train_samples": [sample["sample_name"] for sample in train_samples],
                "val_loss": float(evaluation["val_loss"]),
                "dice": float(metrics["dice"]),
                "iou": float(metrics["iou"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "history_path": str(paths["history_path"]),
                "preview_path": str(paths["best_preview_path"]),
            }
            save_json(best_summary, paths["best_summary_path"])

    if best_summary is None:
        raise RuntimeError(f"Fold {fold_index} finished without producing a best summary.")
    return {
        "fold_index": fold_index,
        "holdout_sample": holdout_name,
        "latest_checkpoint": str(paths["latest_checkpoint"]),
        "best_checkpoint": str(paths["best_checkpoint"]),
        "best_epoch": int(best_summary["best_epoch"]),
        "best_dice": float(best_summary["dice"]),
    }


def build_cfg(args) -> dict[str, object]:
    run_name = args.run_name if args.run_name else build_run_name(args.model_name)
    return {
        "image_dir": args.image_dir,
        "mask_dir": args.mask_dir,
        "valid_dir": args.valid_dir,
        "resume": args.resume,
        "use_patch_sampling": args.use_patch_sampling,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "img_size": MODEL_INPUT_SIZE,
        "num_workers": args.num_workers,
        "model_base_channels": args.model_base_channels,
        "model_name": args.model_name,
        "run_name": run_name,
        "loss_pos_weight": REAL_ONLY_DEFAULTS["loss_pos_weight"],
        "loss_bce_weight": REAL_ONLY_DEFAULTS["loss_bce_weight"],
        "loss_dice_weight": REAL_ONLY_DEFAULTS["loss_dice_weight"],
        "patch_sampling": build_patch_sampling_cfg(),
        "threshold": args.threshold,
        "output_root": args.output_root,
    }


def summarize_folds(
    fold_summaries: list[dict[str, object]],
    metric_names: tuple[str, ...] = ("dice", "iou", "precision", "recall", "val_loss"),
) -> dict[str, object]:
    aggregates = {}
    for metric_name in metric_names:
        values = np.array([float(item[metric_name]) for item in fold_summaries], dtype=np.float32)
        aggregates[metric_name] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std(ddof=0)) if len(values) else 0.0,
            "min": float(values.min()) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 0.0,
        }
    best_fold = max(fold_summaries, key=lambda item: float(item["dice"]))
    return {
        "num_folds": len(fold_summaries),
        "aggregates": aggregates,
        "best_fold": best_fold,
        "folds": fold_summaries,
    }


def evaluate_loocv_latest_checkpoints(
    cfg: dict[str, object],
    samples: list[dict[str, Path]],
    run_dir: Path,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    expected_epoch_field = int(cfg["epochs"]) - 1
    rows = []
    for fold_index, holdout_sample in enumerate(samples, start=1):
        holdout_name = holdout_sample["sample_name"]
        checkpoint_path = run_dir / f"holdout_{holdout_name}" / "checkpoints" / "latest.pth"
        row = evaluate_saved_checkpoint(
            checkpoint_path=checkpoint_path,
            holdout_sample=holdout_sample,
            threshold=float(cfg["threshold"]),
            device=device,
            model_name=str(cfg["model_name"]),
            base_channels=int(cfg["model_base_channels"]),
            expected_epoch_field=expected_epoch_field,
        )
        row["fold_index"] = fold_index
        rows.append(row)
        print(
            f"epoch{int(cfg['epochs'])} holdout={holdout_name} "
            f"dice={row['dice']:.6f} iou={row['iou']:.6f} "
            f"precision={row['precision']:.6f} recall={row['recall']:.6f}"
        )

    summary = summarize_folds(rows, metric_names=("dice", "iou", "precision", "recall"))
    summary.update(
        {
            "run_dir": str(run_dir),
            "device": str(device),
            "checkpoint_file": "latest.pth",
            "expected_epoch_field": expected_epoch_field,
            "threshold": float(cfg["threshold"]),
            "config": {
                "model_name": cfg["model_name"],
                "model_base_channels": cfg["model_base_channels"],
                "epochs": cfg["epochs"],
                "train_steps": cfg["train_steps"],
                "batch_size": cfg["batch_size"],
                "lr": cfg["lr"],
                "use_patch_sampling": cfg["use_patch_sampling"],
                "seed": cfg["seed"],
                "sample_names": [sample["sample_name"] for sample in samples],
            },
        }
    )
    return rows, summary


def main():
    args = parse_args()
    cfg = build_cfg(args)
    if int(cfg["epochs"]) < 5:
        print(
            "Warning: epochs is currently very small. "
            "For a meaningful leave-one-out run, consider passing --epochs 60 or --epochs 100."
        )

    samples = discover_samples(
        image_dir=Path(cfg["image_dir"]),
        mask_dir=Path(cfg["mask_dir"]),
        valid_dir=Path(cfg["valid_dir"]),
        selected_names=args.samples,
    )
    if not args.allow_any_count and len(samples) != 5:
        raise ValueError(
            f"Expected exactly five real samples for this experiment, but found {len(samples)}. "
            "Pass --allow-any-count if you intentionally want a different count."
        )

    run_dir = Path(cfg["output_root"]) / str(cfg["run_name"])
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running LOOCV with {len(samples)} samples on device={device}")
    print(f"Artifacts: {run_dir}")

    fold_summaries = []
    for fold_index, holdout_sample in enumerate(samples, start=1):
        train_samples = [sample for sample in samples if sample["sample_name"] != holdout_sample["sample_name"]]
        fold_summary = train_one_fold(
            cfg=cfg,
            fold_index=fold_index,
            train_samples=train_samples,
            holdout_sample=holdout_sample,
            run_dir=run_dir,
            device=device,
        )
        fold_summaries.append(fold_summary)
        print(
            f"Fold {fold_index} finished holdout={fold_summary['holdout_sample']} "
            f"best_epoch={int(fold_summary['best_epoch'])} best_dice={float(fold_summary['best_dice']):.4f}"
        )

    save_csv(fold_summaries, run_dir / "loocv_training_progress.csv")
    save_json({"folds": fold_summaries}, run_dir / "loocv_training_progress.json")

    eval_rows, eval_summary = evaluate_loocv_latest_checkpoints(
        cfg=cfg,
        samples=samples,
        run_dir=run_dir,
        device=device,
    )

    csv_rows = []
    for item in eval_rows:
        csv_rows.append(
            {
                "fold_index": item["fold_index"],
                "holdout_sample": item["holdout_sample"],
                "checkpoint_epoch_field": item["checkpoint_epoch_field"],
                "dice": f"{float(item['dice']):.6f}",
                "iou": f"{float(item['iou']):.6f}",
                "precision": f"{float(item['precision']):.6f}",
                "recall": f"{float(item['recall']):.6f}",
                "checkpoint": item["checkpoint"],
            }
        )

    save_csv(csv_rows, run_dir / "loocv_summary.csv")
    save_json(eval_summary, run_dir / "loocv_summary.json")

    mean_dice = float(eval_summary["aggregates"]["dice"]["mean"])
    std_dice = float(eval_summary["aggregates"]["dice"]["std"])
    print(
        f"Finished LOOCV epoch{int(cfg['epochs'])} evaluation | "
        f"mean_dice={mean_dice:.4f} std_dice={std_dice:.4f}"
    )
    print(f"Saved CSV summary to: {run_dir / 'loocv_summary.csv'}")
    print(f"Saved JSON summary to: {run_dir / 'loocv_summary.json'}")


if __name__ == "__main__":
    main()
