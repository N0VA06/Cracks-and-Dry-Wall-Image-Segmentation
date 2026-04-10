"""
preprocessing/transforms.py
Two albumentations pipelines:
  get_train_transforms()          — standard augmentation (used for cracks)
  get_minority_train_transforms() — stronger augmentation (used for drywall)
  get_val_transforms()            — resize + normalise only

Why a stronger pipeline for drywall:
  Taping areas are thin linear features (3–8 px wide at 512 resolution).
  Standard flip/rotate is insufficient to generate visually diverse training
  samples from only 655 images. Elastic deformation, CLAHE, and random crops
  simulate real-world variation in tape appearance and camera angle,
  effectively giving the minority class much more signal diversity.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from config import IMAGE_SIZE, MEAN, STD

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALB = True
except ImportError:
    _ALB = False


# ══════════════════════════════════════════════════════
#  Standard pipeline (cracks — majority class)
# ══════════════════════════════════════════════════════

def get_train_transforms():
    if _ALB:
        return A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-20, 20),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.5
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return _PytorchTransform(train=True, strong=False)


# ══════════════════════════════════════════════════════
#  Strong pipeline (drywall — minority class, 655 images)
# ══════════════════════════════════════════════════════

def get_minority_train_transforms():
    """
    Aggressive augmentation for the drywall dataset.
    Combined with MINORITY_REPEAT=6 in config, this effectively gives
    the model ~3900 visually distinct drywall training samples.

    Key additions over the standard pipeline:
      ElasticTransform : deforms thin tape lines → simulates uneven application
      GridDistortion   : perspective-like warp → different camera angles
      CLAHE            : local contrast normalisation → helps under dim lighting
      RandomShadow     : simulates shadows across tape joint
      RandomCrop + Pad : forces model to handle partially visible tape
    """
    if _ALB:
        return A.Compose([
            A.Resize(*IMAGE_SIZE),

            # Geometric: covers all rotation/flip variants
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                scale=(0.85, 1.15),
                rotate=(-30, 30),
                p=0.6,
            ),

            # Thin-feature deformation: critical for tape/crack augmentation
            A.ElasticTransform(
                alpha=80, sigma=10, p=0.4
            ),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

            # Photometric: simulates varying lighting conditions
            A.RandomBrightnessContrast(
                brightness_limit=0.35, contrast_limit=0.35, p=0.6
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),

            # Noise and blur
            A.GaussNoise(std_range=(0.01, 0.07), p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),

            # Spatial dropout: forces model to predict from partial context
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(20, 60),
                hole_width_range=(20, 60),
                fill=0,
                p=0.2,
            ),

            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return _PytorchTransform(train=True, strong=True)


# ══════════════════════════════════════════════════════
#  Val / test pipeline
# ══════════════════════════════════════════════════════

def get_val_transforms():
    if _ALB:
        return A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return _PytorchTransform(train=False, strong=False)


# ══════════════════════════════════════════════════════
#  Pure-PyTorch fallback (when albumentations not installed)
# ══════════════════════════════════════════════════════

class _PytorchTransform:
    """Matches albumentations dict interface: transform(image=arr, mask=arr)."""

    def __init__(self, train: bool = True, strong: bool = False):
        self.train  = train
        self.strong = strong
        self.h, self.w = IMAGE_SIZE
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std  = torch.tensor(STD).view(3, 1, 1)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict:
        img  = Image.fromarray(image.astype(np.uint8))
        msk  = Image.fromarray(mask.astype(np.uint8), mode="L")

        img  = img.resize((self.w, self.h), Image.BILINEAR)
        msk  = msk.resize((self.w, self.h), Image.NEAREST)

        if self.train:
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
            if self.strong and np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
            if self.strong and np.random.rand() > 0.5:
                img = img.transpose(Image.ROTATE_90)
                msk = msk.transpose(Image.ROTATE_90)

        img_t = TF.to_tensor(img)
        msk_t = torch.from_numpy(np.array(msk)).long()
        img_t = (img_t - self.mean) / self.std
        return {"image": img_t, "mask": msk_t}
