import json
import logging
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(__file__).with_name("config.json")
with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    RAW_CONFIG = json.load(fh)

MODEL_INPUT_SIZE = tuple(RAW_CONFIG["model_input_size"])
TRAINING_DEFAULTS = dict(RAW_CONFIG["training"])
TRAINING_DEFAULTS["img_size"] = MODEL_INPUT_SIZE
PREVIEW_DEFAULTS = dict(RAW_CONFIG["preview"])
MODEL_DEFAULTS = dict(RAW_CONFIG["model"])
LOSS_DEFAULTS = dict(RAW_CONFIG["loss"])
REAL_ONLY_DEFAULTS = {
    **TRAINING_DEFAULTS,
    **PREVIEW_DEFAULTS,
    **MODEL_DEFAULTS,
    **LOSS_DEFAULTS,
}
PATCH_SAMPLING_DEFAULTS = {
    "min_size_ratio_to_model_input": float(RAW_CONFIG["patch_sampling"].get("min_size_ratio_to_model_input", 0.5)),
    "positive_only": RAW_CONFIG["patch_sampling"].get("positive_only", False),
    "min_positive_pixels": int(RAW_CONFIG["patch_sampling"].get("min_positive_pixels", 1)),
    "sampling_attempts": int(RAW_CONFIG["patch_sampling"].get("sampling_attempts", 8)),
    "center_jitter": int(RAW_CONFIG["patch_sampling"].get("center_jitter", 48)),
}
AUGMENTATION_DEFAULTS = {
    "rotate_prob": RAW_CONFIG["augmentation"]["rotate_prob"],
    "rotate_deg_range": tuple(RAW_CONFIG["augmentation"]["rotate_deg_range"]),
    "rotate_scale_range": tuple(RAW_CONFIG["augmentation"]["rotate_scale_range"]),
    "flip_prob": RAW_CONFIG["augmentation"]["flip_prob"],
    "flip_codes": tuple(RAW_CONFIG["augmentation"]["flip_codes"]),
    "affine_prob": RAW_CONFIG["augmentation"]["affine_prob"],
    "affine_shear_range": tuple(RAW_CONFIG["augmentation"]["affine_shear_range"]),
    "affine_shift_range": tuple(RAW_CONFIG["augmentation"]["affine_shift_range"]),
    "brightness_prob": RAW_CONFIG["augmentation"]["brightness_prob"],
    "brightness_beta_range": tuple(RAW_CONFIG["augmentation"]["brightness_beta_range"]),
    "contrast_prob": RAW_CONFIG["augmentation"]["contrast_prob"],
    "contrast_alpha_range": tuple(RAW_CONFIG["augmentation"]["contrast_alpha_range"]),
    "noise_prob": RAW_CONFIG["augmentation"]["noise_prob"],
    "noise_std_range": tuple(RAW_CONFIG["augmentation"]["noise_std_range"]),
    "blur_prob": RAW_CONFIG["augmentation"]["blur_prob"],
    "blur_kernel_choices": tuple(tuple(item) for item in RAW_CONFIG["augmentation"]["blur_kernel_choices"]),
    "blur_sigma_range": tuple(RAW_CONFIG["augmentation"]["blur_sigma_range"]),
    "illumination_prob": RAW_CONFIG["augmentation"]["illumination_prob"],
    "illumination_strength_range": tuple(RAW_CONFIG["augmentation"]["illumination_strength_range"]),
    "illumination_modes": tuple(RAW_CONFIG["augmentation"]["illumination_modes"]),
}


def build_patch_sampling_cfg(cfg=None):
    cfg = {} if cfg is None else dict(cfg)
    built_cfg = {
        "min_size_ratio_to_model_input": float(PATCH_SAMPLING_DEFAULTS["min_size_ratio_to_model_input"]),
        "positive_only": bool(PATCH_SAMPLING_DEFAULTS["positive_only"]),
        "min_positive_pixels": int(PATCH_SAMPLING_DEFAULTS["min_positive_pixels"]),
        "sampling_attempts": int(PATCH_SAMPLING_DEFAULTS["sampling_attempts"]),
        "center_jitter": int(PATCH_SAMPLING_DEFAULTS["center_jitter"]),
    }
    if "patch_sampling" in cfg:
        patch_cfg = cfg["patch_sampling"]
        for key in ("min_size_ratio_to_model_input", "positive_only", "min_positive_pixels", "sampling_attempts", "center_jitter"):
            if key in patch_cfg:
                built_cfg[key] = patch_cfg[key]
    return built_cfg


def _build_run_name(model_name):
    timestamp = datetime.now().strftime("%m%d%H%M")
    normalized_model_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(model_name).strip().lower())
    return f"{timestamp}_{normalized_model_name}"


def build_artifact_paths(cfg):
    run_name = cfg["run_name"]
    artifacts_dir = PROJECT_ROOT / "outputs" / "model" / run_name
    checkpoints_dir = artifacts_dir / "checkpoints"
    metrics_dir = artifacts_dir / "metrics"
    logs_dir = artifacts_dir / "logs"
    previews_dir = artifacts_dir / "previews"
    return {
        "artifacts_dir": artifacts_dir,
        "checkpoints_dir": checkpoints_dir,
        "metrics_dir": metrics_dir,
        "logs_dir": logs_dir,
        "previews_dir": previews_dir,
        "latest_checkpoint": checkpoints_dir / "latest.pth",
        "train_log": logs_dir / "train.log",
        "train_losses": metrics_dir / "train_losses.npy",
    }


def ensure_training_dirs(paths):
    for path in (paths["checkpoints_dir"], paths["metrics_dir"], paths["logs_dir"], paths["previews_dir"]):
        path.mkdir(parents=True, exist_ok=True)


def reset_training_artifacts(paths):
    for path in (paths["checkpoints_dir"], paths["metrics_dir"], paths["logs_dir"], paths["previews_dir"]):
        if path.exists():
            shutil.rmtree(path)


def setup_logging(log_file, paths):
    ensure_training_dirs(paths)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_cfg(args):
    model_name = REAL_ONLY_DEFAULTS["model_name"] if getattr(args, "model_name", None) is None else args.model_name
    run_name = getattr(args, "run_name", None)
    if run_name is None:
        run_name = _build_run_name(model_name)
    return {
        "image_dir": PROJECT_ROOT / "data" / "real_train" / "images",
        "mask_dir": PROJECT_ROOT / "data" / "real_train" / "masks",
        "valid_dir": PROJECT_ROOT / "data" / "real_train" / "valids",
        "resume": REAL_ONLY_DEFAULTS["resume"] if args.resume is None else args.resume,
        "use_patch_sampling": REAL_ONLY_DEFAULTS["use_patch_sampling"] if args.use_patch_sampling is None else args.use_patch_sampling,
        "seed": REAL_ONLY_DEFAULTS["seed"],
        "train_steps": REAL_ONLY_DEFAULTS["train_steps"],
        "batch_size": args.batch_size,
        "lr": REAL_ONLY_DEFAULTS["lr"] if args.lr is None else args.lr,
        "epochs": args.epochs,
        "img_size": REAL_ONLY_DEFAULTS["img_size"],
        "num_workers": REAL_ONLY_DEFAULTS["num_workers"],
        "preview_samples": REAL_ONLY_DEFAULTS["preview_samples"],
        "preview_threshold": REAL_ONLY_DEFAULTS["preview_threshold"],
        "model_base_channels": REAL_ONLY_DEFAULTS["model_base_channels"] if getattr(args, "model_base_channels", None) is None else args.model_base_channels,
        "model_name": model_name,
        "run_name": run_name,
        "loss_pos_weight": REAL_ONLY_DEFAULTS["loss_pos_weight"],
        "loss_bce_weight": REAL_ONLY_DEFAULTS["loss_bce_weight"],
        "loss_dice_weight": REAL_ONLY_DEFAULTS["loss_dice_weight"],
        "patch_sampling": build_patch_sampling_cfg(),
    }
