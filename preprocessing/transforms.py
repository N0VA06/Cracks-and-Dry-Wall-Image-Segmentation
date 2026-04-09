"""
preprocessing/transforms.py
Albumentations-based augmentation pipelines.
Falls back gracefully if albumentations is not installed.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from config import IMAGE_SIZE, MEAN, STD

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS = True
except ImportError:
    _ALBUMENTATIONS = False


# ══════════════════════════════════════════════════════
#  Albumentations pipelines
# ══════════════════════════════════════════════════════

def get_train_transforms():
    if _ALBUMENTATIONS:
        return A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            # ShiftScaleRotate is deprecated → use Affine
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            # var_limit renamed; use std_range (values in [0,1] scale after /255)
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return PytorchTransform(train=True)


def get_val_transforms():
    if _ALBUMENTATIONS:
        return A.Compose([
            A.Resize(*IMAGE_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return PytorchTransform(train=False)


# ══════════════════════════════════════════════════════
#  Pure-PyTorch fallback transform (dict interface)
# ══════════════════════════════════════════════════════

class PytorchTransform:
    """
    Minimal transform that matches the albumentations dict interface:
      transform(image=np.ndarray, mask=np.ndarray)
      → {"image": Tensor[3,H,W], "mask": Tensor[H,W]}
    """

    def __init__(self, train: bool = True):
        self.train   = train
        self.h, self.w = IMAGE_SIZE
        self.mean    = torch.tensor(MEAN).view(3, 1, 1)
        self.std     = torch.tensor(STD).view(3, 1, 1)

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> dict:
        # numpy → PIL
        img_pil  = Image.fromarray(image.astype(np.uint8))
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="L")

        # Resize
        img_pil  = img_pil.resize((self.w, self.h), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.w, self.h), Image.NEAREST)

        if self.train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img_pil  = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            # Random vertical flip
            if np.random.rand() > 0.7:
                img_pil  = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)

        # PIL → tensor
        img_t  = TF.to_tensor(img_pil)           # [3, H, W], float in [0,1]
        mask_t = torch.from_numpy(np.array(mask_pil)).long()  # [H, W]

        # Normalize image
        img_t = (img_t - self.mean) / self.std

        return {"image": img_t, "mask": mask_t}
