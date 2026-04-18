from __future__ import annotations

MODEL_INPUT_SIZE = (1032, 1236)
REAL_ONLY_DEFAULTS = {
    "resume": True,
    "use_patch_sampling": True,
    "seed": 42,
    "train_steps": 128,
    "batch_size": 1,
    "lr": 0.0002,
    "epochs": 100,
    "num_workers": 2,
    "preview_samples": 5,
    "preview_threshold": 0.75,
    "model_name": "attention_unet",
    "backup": "unet",
    "model_base_channels": 32,
    "loss_pos_weight": 3.0,
    "loss_bce_weight": 1.0,
    "loss_dice_weight": 1.0,
}
