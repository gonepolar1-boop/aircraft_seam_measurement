from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import AUGMENTATION_DEFAULTS, REAL_ONLY_DEFAULTS


def _normalize_hw(size):
    if isinstance(size, int):
        value = int(size)
        return value, value
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError(f"Unsupported spatial size: {size!r}")


class SeamAugmenter:
    def __init__(self, cfg=None):
        self.cfg = dict(AUGMENTATION_DEFAULTS if cfg is None else cfg)

    def apply(self, image, mask, valid_mask):
        if random.random() < self.cfg["rotate_prob"]:
            image, mask, valid_mask = self._rotate(image, mask, valid_mask)
        if random.random() < self.cfg["flip_prob"]:
            image, mask, valid_mask = self._flip(image, mask, valid_mask)
        if random.random() < self.cfg["affine_prob"]:
            image, mask, valid_mask = self._affine(image, mask, valid_mask)
        if random.random() < self.cfg["brightness_prob"]:
            image, mask, valid_mask = self._brightness(image, mask, valid_mask)
        if random.random() < self.cfg["contrast_prob"]:
            image, mask, valid_mask = self._contrast(image, mask, valid_mask)
        if random.random() < self.cfg["noise_prob"]:
            image, mask, valid_mask = self._noise(image, mask, valid_mask)
        if random.random() < self.cfg["blur_prob"]:
            image, mask, valid_mask = self._blur(image, mask, valid_mask)
        if random.random() < self.cfg["illumination_prob"]:
            image, mask, valid_mask = self._illumination(image, mask, valid_mask)
        return image, mask, valid_mask

    def _rotate(self, image, mask, valid_mask):
        h, w = image.shape
        angle_deg = random.uniform(*self.cfg["rotate_deg_range"])
        scale = random.uniform(*self.cfg["rotate_scale_range"])
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, scale)
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = cv2.warpAffine(
            valid_mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image, mask, valid_mask

    def _flip(self, image, mask, valid_mask):
        flip_code = random.choice(self.cfg["flip_codes"])
        image = cv2.flip(image, flip_code)
        mask = cv2.flip(mask, flip_code)
        valid_mask = cv2.flip(valid_mask, flip_code)
        return image, mask, valid_mask

    def _noise(self, image, mask, valid_mask):
        noise = np.random.normal(0, random.uniform(*self.cfg["noise_std_range"]), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return image, mask, valid_mask

    def _blur(self, image, mask, valid_mask):
        kernel = random.choice(self.cfg["blur_kernel_choices"])
        image = cv2.GaussianBlur(image, kernel, sigmaX=random.uniform(*self.cfg["blur_sigma_range"]))
        return image, mask, valid_mask

    def _brightness(self, image, mask, valid_mask):
        beta = random.uniform(*self.cfg["brightness_beta_range"]) * 255.0
        image = np.clip(image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return image, mask, valid_mask

    def _contrast(self, image, mask, valid_mask):
        alpha = random.uniform(*self.cfg["contrast_alpha_range"])
        mean_val = float(image.mean())
        image = (image.astype(np.float32) - mean_val) * alpha + mean_val
        return np.clip(image, 0, 255).astype(np.uint8), mask, valid_mask

    def _illumination(self, image, mask, valid_mask):
        h, w = image.shape
        strength = random.uniform(*self.cfg["illumination_strength_range"])
        mode = random.choice(self.cfg["illumination_modes"])
        if mode == "x":
            grad = np.tile(np.linspace(-strength, strength, w, dtype=np.float32), (h, 1))
        elif mode == "y":
            grad = np.tile(np.linspace(-strength, strength, h, dtype=np.float32).reshape(h, 1), (1, w))
        else:
            gx = np.tile(np.linspace(-strength, strength, w, dtype=np.float32), (h, 1))
            gy = np.tile(np.linspace(strength, -strength, h, dtype=np.float32).reshape(h, 1), (1, w))
            grad = 0.5 * gx + 0.5 * gy
        image = np.clip(image.astype(np.float32) + grad, 0, 255).astype(np.uint8)
        return image, mask, valid_mask

    def _affine(self, image, mask, valid_mask):
        h, w = image.shape
        shear_min, shear_max = self.cfg["affine_shear_range"]
        shift_min, shift_max = self.cfg["affine_shift_range"]
        matrix = np.float32(
            [
                [1.0, random.uniform(shear_min, shear_max), random.uniform(shift_min, shift_max)],
                [random.uniform(shear_min, shear_max), 1.0, random.uniform(shift_min, shift_max)],
            ]
        )
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid_mask = cv2.warpAffine(
            valid_mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image, mask, valid_mask


class PatchSampler:
    def __init__(self, cfg):
        self.min_size_ratio_to_model_input = max(1e-3, float(cfg.get("min_size_ratio_to_model_input", 0.5)))
        self.positive_only = bool(cfg.get("positive_only", False))
        self.min_positive_pixels = max(1, int(cfg.get("min_positive_pixels", 1)))
        self.sampling_attempts = max(1, int(cfg.get("sampling_attempts", 8)))
        self.center_jitter = max(0, int(cfg.get("center_jitter", 48)))

    def build_sample_record(self, image_path, mask_path, valid_dir=None):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {mask_path}")
        valid_path = valid_dir / image_path.name
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing valid mask: {valid_path}")
        positive_coords = np.argwhere(mask > 127)
        return {
            "sample_name": image_path.stem,
            "image_path": image_path,
            "mask_path": mask_path,
            "valid_path": valid_path,
            "shape": image.shape,
            "positive_coords": positive_coords,
        }

    def sample_patch(self, image, mask, valid_mask, sample=None):
        crop_h, crop_w = self._choose_crop_size(image.shape)
        positive_coords = None if sample is None else sample.get("positive_coords")

        if self.positive_only:
            if positive_coords is None:
                raise ValueError("positive_only patch sampling requires positive_coords in sample metadata.")
            if len(positive_coords) == 0:
                sample_name = "unknown" if sample is None else sample.get("sample_name", "unknown")
                raise ValueError(f"positive_only patch sampling found no positive pixels in sample: {sample_name}")
            return self._sample_positive_patch(image, mask, valid_mask, positive_coords, crop_h, crop_w)

        max_top = max(0, image.shape[0] - crop_h)
        max_left = max(0, image.shape[1] - crop_w)
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0
        bottom = top + crop_h
        right = left + crop_w
        return {
            "image": image[top:bottom, left:right],
            "mask": mask[top:bottom, left:right],
            "valid_mask": valid_mask[top:bottom, left:right],
            "sample_center": (
                int(top + crop_h // 2),
                int(left + crop_w // 2),
            ),
            "crop_box": {
                "top": int(top),
                "left": int(left),
                "bottom": int(bottom),
                "right": int(right),
            },
            "crop_size": (int(crop_h), int(crop_w)),
        }

    def _choose_crop_size(self, image_shape):
        image_h, image_w = image_shape
        min_crop_h = min(image_h, max(1, int(round(REAL_ONLY_DEFAULTS["img_size"][0] * self.min_size_ratio_to_model_input))))
        min_crop_w = min(image_w, max(1, int(round(REAL_ONLY_DEFAULTS["img_size"][1] * self.min_size_ratio_to_model_input))))
        crop_h = random.randint(min_crop_h, image_h) if image_h > min_crop_h else image_h
        crop_w = random.randint(min_crop_w, image_w) if image_w > min_crop_w else image_w
        return int(crop_h), int(crop_w)

    def _sample_positive_patch(self, image, mask, valid_mask, positive_coords, crop_h, crop_w):
        max_top = max(0, image.shape[0] - crop_h)
        max_left = max(0, image.shape[1] - crop_w)

        for _ in range(self.sampling_attempts):
            center_y, center_x = positive_coords[random.randrange(len(positive_coords))]
            jitter_y = random.randint(-self.center_jitter, self.center_jitter) if self.center_jitter > 0 else 0
            jitter_x = random.randint(-self.center_jitter, self.center_jitter) if self.center_jitter > 0 else 0
            center_y = int(center_y) + jitter_y
            center_x = int(center_x) + jitter_x

            top = center_y - crop_h // 2
            left = center_x - crop_w // 2
            top = min(max(top, 0), max_top)
            left = min(max(left, 0), max_left)
            bottom = top + crop_h
            right = left + crop_w
            mask_patch = mask[top:bottom, left:right]
            if int((mask_patch > 127).sum()) >= self.min_positive_pixels:
                return {
                    "image": image[top:bottom, left:right],
                    "mask": mask_patch,
                    "valid_mask": valid_mask[top:bottom, left:right],
                    "sample_center": (int(center_y), int(center_x)),
                    "crop_box": {
                        "top": int(top),
                        "left": int(left),
                        "bottom": int(bottom),
                        "right": int(right),
                    },
                    "crop_size": (int(crop_h), int(crop_w)),
                }

        raise ValueError(
            "positive_only patch sampling could not find a crop with enough positive pixels "
            f"after {self.sampling_attempts} attempts."
        )


class SeamDataset(Dataset):
    def __init__(
        self,
        image_dir=None,
        mask_dir=None,
        valid_dir=None,
        img_size=REAL_ONLY_DEFAULTS["img_size"],
        augment=False,
        samples=None,
        length=None,
        use_patch_sampling=REAL_ONLY_DEFAULTS["use_patch_sampling"],
        patch_sampling_cfg=None,
    ):
        if image_dir is None or mask_dir is None or valid_dir is None:
            raise ValueError("image_dir, mask_dir, and valid_dir must be provided.")
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.valid_dir = Path(valid_dir)
        self.input_size = _normalize_hw(img_size)
        self.use_patch_sampling = bool(use_patch_sampling)
        self.patch_sampler = PatchSampler(patch_sampling_cfg) if self.use_patch_sampling else None
        self.augment = augment
        self.augmenter = SeamAugmenter()
        self.samples = self._discover_samples() if samples is None else list(samples)
        self.length = len(self.samples) if length is None else int(length)
        if not self.samples:
            raise ValueError(f"No paired training samples found under {self.image_dir} and {self.mask_dir}")
        if self.length <= 0:
            raise ValueError("Dataset length must be positive.")

    def _discover_samples(self):
        samples = []
        for image_path in sorted(self.image_dir.glob("*")):
            if not image_path.is_file():
                continue
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask: {mask_path}")
            if self.use_patch_sampling:
                samples.append(self.patch_sampler.build_sample_record(image_path, mask_path, self.valid_dir))
                continue
            valid_path = self.valid_dir / image_path.name
            if not valid_path.exists():
                raise FileNotFoundError(f"Missing valid mask: {valid_path}")
            samples.append(
                {
                    "sample_name": image_path.stem,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "valid_path": valid_path,
                }
            )
        return samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(sample["mask_path"]), cv2.IMREAD_GRAYSCALE)
        valid_mask = cv2.imread(str(sample["valid_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {sample['image_path']}")
        if mask is None:
            raise FileNotFoundError(f"Unable to read mask: {sample['mask_path']}")
        if valid_mask is None:
            raise FileNotFoundError(f"Unable to read valid mask: {sample['valid_path']}")

        if self.use_patch_sampling:
            sampled = self.patch_sampler.sample_patch(image, mask, valid_mask, sample=sample)
            image = sampled["image"]
            mask = sampled["mask"]
            valid_mask = sampled["valid_mask"]
            crop_box = sampled["crop_box"]
            crop_size = sampled["crop_size"]
            sample_mode = "patch"
        else:
            crop_box = {
                "top": 0,
                "left": 0,
                "bottom": int(image.shape[0]),
                "right": int(image.shape[1]),
            }
            crop_size = (int(image.shape[0]), int(image.shape[1]))
            sample_mode = "full_image"

        if self.augment:
            image, mask, valid_mask = self.augmenter.apply(image, mask, valid_mask)

        target_h, target_w = self.input_size
        if image.shape != (target_h, target_w):
            image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            valid_mask = cv2.resize(valid_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.tensor(image.astype(np.float32)[None, ...] / 255.0, dtype=torch.float32)
        mask_tensor = torch.tensor((mask > 127).astype(np.float32)[None, ...], dtype=torch.float32)
        valid_mask_tensor = torch.tensor((valid_mask > 127).astype(np.float32)[None, ...], dtype=torch.float32)
        meta = {
            "sample_name": sample["sample_name"],
            "image_path": str(sample["image_path"]),
            "mask_path": str(sample["mask_path"]),
            "valid_path": str(sample["valid_path"]),
            "sample_mode": sample_mode,
            "top": int(crop_box["top"]),
            "left": int(crop_box["left"]),
            "crop_height": int(crop_size[0]),
            "crop_width": int(crop_size[1]),
            "crop_box": crop_box,
            "sample_center_y": int(sampled["sample_center"][0]) if self.use_patch_sampling else int(image.shape[0] // 2),
            "sample_center_x": int(sampled["sample_center"][1]) if self.use_patch_sampling else int(image.shape[1] // 2),
        }
        return image_tensor, mask_tensor, valid_mask_tensor, meta


def build_datasets(cfg):
    sample_source = SeamDataset(
        image_dir=cfg["image_dir"],
        mask_dir=cfg["mask_dir"],
        valid_dir=cfg["valid_dir"],
        img_size=cfg["img_size"],
        augment=False,
        use_patch_sampling=cfg["use_patch_sampling"],
        patch_sampling_cfg=cfg["patch_sampling"],
    )

    preview_dataset = SeamDataset(
        image_dir=cfg["image_dir"],
        mask_dir=cfg["mask_dir"],
        valid_dir=cfg["valid_dir"],
        img_size=cfg["img_size"],
        augment=False,
        samples=sample_source.samples,
        length=min(cfg["preview_samples"], len(sample_source.samples)),
        use_patch_sampling=cfg["use_patch_sampling"],
        patch_sampling_cfg=cfg["patch_sampling"],
    )

    train_dataset = SeamDataset(
        image_dir=cfg["image_dir"],
        mask_dir=cfg["mask_dir"],
        valid_dir=cfg["valid_dir"],
        img_size=cfg["img_size"],
        augment=True,
        samples=sample_source.samples,
        length=cfg["batch_size"] * cfg["train_steps"],
        use_patch_sampling=cfg["use_patch_sampling"],
        patch_sampling_cfg=cfg["patch_sampling"],
    )
    return train_dataset, preview_dataset
