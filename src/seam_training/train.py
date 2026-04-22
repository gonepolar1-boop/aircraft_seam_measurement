import argparse
import json
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_datasets
from .model import SegmentationCriterion, build_model
from .preview import save_epoch_previews
from .utils import (
    REAL_ONLY_DEFAULTS,
    build_artifact_paths,
    build_cfg,
    reset_training_artifacts,
    set_seed,
    setup_logging,
)

METRIC_KEYS = ("train_losses",)
# Schema shared with scripts/train/run_loocv_experiments.py so downstream
# plotting tools can consume either source uniformly.
HISTORY_JSON_KEYS = (
    "train_losses",
    "val_losses",
    "val_dice",
    "val_iou",
    "val_precision",
    "val_recall",
)


def save_checkpoint(model, optimizer, epoch, save_path, cfg):
    torch.save({
        "epoch": epoch,
        "model_name": cfg["model_name"],
        "model_base_channels": cfg["model_base_channels"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)


def load_checkpoint(model, optimizer, load_path, device, cfg):
    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    checkpoint_model_name = checkpoint.get("model_name")
    checkpoint_base_channels = checkpoint.get("model_base_channels")
    if checkpoint_model_name is not None and checkpoint_model_name != cfg["model_name"]:
        raise ValueError(
            f"Checkpoint model_name={checkpoint_model_name!r} does not match current config "
            f"model_name={cfg['model_name']!r}."
        )
    if checkpoint_base_channels is not None and int(checkpoint_base_channels) != int(cfg["model_base_channels"]):
        raise ValueError(
            f"Checkpoint model_base_channels={checkpoint_base_channels} does not match current config "
            f"model_base_channels={cfg['model_base_channels']}."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"] + 1


def save_history(history, paths, train_loss):
    history["train_losses"].append(train_loss)
    for key in METRIC_KEYS:
        np.save(paths[key], np.array(history[key]))
    # Also emit a history.json side-car with the same schema as the LOOCV
    # script. Keeps a single source-of-truth for the plotting tools
    # regardless of whether you ran the single-sample trainer or LOOCV.
    history_path = paths["metrics_dir"] / "history.json"
    serialisable = {key: [float(value) for value in history.get(key, [])] for key in HISTORY_JSON_KEYS}
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, ensure_ascii=False, indent=2)


def load_history(paths):
    legacy = {key: np.load(paths[key]).tolist() if paths[key].exists() else [] for key in METRIC_KEYS}
    # Prefer the JSON side-car if present - carries the full metric list.
    history_path = paths["metrics_dir"] / "history.json"
    history = {key: [] for key in HISTORY_JSON_KEYS}
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            for key in HISTORY_JSON_KEYS:
                history[key] = [float(value) for value in loaded.get(key, [])]
        except (OSError, json.JSONDecodeError):
            history = {key: [] for key in HISTORY_JSON_KEYS}
    # Merge legacy train_losses if the json was empty but the npy wasn't.
    if not history["train_losses"] and legacy.get("train_losses"):
        history["train_losses"] = legacy["train_losses"]
    return history


def train_model(cfg):
    paths = build_artifact_paths(cfg)
    if not cfg["resume"]:
        reset_training_artifacts(paths)
    setup_logging(paths["train_log"], paths)
    logging.info("Training artifacts will be saved under %s", paths["artifacts_dir"])
    set_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, preview_dataset = build_datasets(cfg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg["num_workers"] > 0),
    )

    model = build_model(model_name=cfg["model_name"], base_channels=cfg["model_base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = SegmentationCriterion(
        pos_weight=cfg["loss_pos_weight"],
        bce_weight=cfg["loss_bce_weight"],
        dice_weight=cfg["loss_dice_weight"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    history = {key: [] for key in METRIC_KEYS}
    if cfg["resume"] and paths["latest_checkpoint"].exists():
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, paths["latest_checkpoint"], device, cfg)
        history = load_history(paths)
        logging.info("Resume training from epoch=%d using model=%s", start_epoch, cfg["model_name"])

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss, batch_count = 0.0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for images, masks, valid_masks, _meta in progress:
            images, masks, valid_masks = images.to(device), masks.to(device), valid_masks.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks, valid_masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batch_count += 1
            progress.set_postfix(loss=loss.item())

        train_loss = epoch_loss / max(batch_count, 1)
        save_history(history, paths, train_loss=train_loss)
        save_epoch_previews(
            model,
            preview_dataset,
            device,
            paths["previews_dir"],
            epoch,
            threshold=cfg["preview_threshold"],
            preview_samples=cfg["preview_samples"],
        )
        logging.info(
            "Epoch %d/%d train_loss=%.4f",
            epoch + 1,
            cfg["epochs"],
            history["train_losses"][-1],
        )
        save_checkpoint(model, optimizer, epoch, paths["latest_checkpoint"], cfg)
        if (epoch + 1) % 20 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                paths["checkpoints_dir"] / f"epoch_{epoch + 1}.pth",
                cfg,
            )

def parse_args():
    parser = argparse.ArgumentParser(description="Train seam segmentation model on real data only")
    parser.add_argument("--epochs", type=int, default=REAL_ONLY_DEFAULTS["epochs"])
    parser.add_argument("--batch-size", type=int, default=REAL_ONLY_DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--model-name", type=str, default=None, choices=("unet", "attention_unet"))
    parser.add_argument("--model-base-channels", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    patch_group = parser.add_mutually_exclusive_group()
    patch_group.add_argument("--use-patch-sampling", dest="use_patch_sampling", action="store_true")
    patch_group.add_argument("--no-patch-sampling", dest="use_patch_sampling", action="store_false")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None, use_patch_sampling=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(build_cfg(args))


if __name__ == "__main__":
    main()
