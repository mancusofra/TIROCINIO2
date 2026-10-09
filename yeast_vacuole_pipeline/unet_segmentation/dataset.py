"""
Image/mask pairing and the PyTorch Dataset used to train the segmentation model.

Masks live under `masks_dir` and their source images under `images_dir`, with
the same relative path (e.g. Mask/<class>/x.tif <-> Train_annotated/<class>/x.tif).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def find_pairs(masks_dir, images_dir, pattern="*/*.tif"):
    """
    Returns sorted (image_path, mask_path) pairs for every mask matching `pattern`.

    Raises:
        FileNotFoundError: If no masks are found, or a mask has no matching image.
    """
    masks_dir, images_dir = Path(masks_dir), Path(images_dir)
    masks = sorted(masks_dir.glob(pattern))
    if not masks:
        raise FileNotFoundError(f"No masks matching '{pattern}' in {masks_dir}")

    pairs = [(images_dir / m.relative_to(masks_dir), m) for m in masks]
    missing = [str(img) for img, _ in pairs if not img.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks have no image, e.g. {missing[0]}")
    return pairs


def split_pairs(pairs, seed=42):
    """Splits pairs into 80% train / 10% validation / 10% test."""
    train, rest = train_test_split(pairs, train_size=0.8, random_state=seed)
    val, test = train_test_split(rest, train_size=0.5, random_state=seed)
    return train, val, test


def read_image(path):
    """Reads an image as RGB uint8 (H, W, 3), the format the model is fed at inference."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path):
    """Reads a mask as a {0, 1} uint8 array (H, W)."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return (mask > 127).astype(np.uint8)


def to_tensor(img):
    """RGB uint8 (H, W, 3) -> float tensor (3, H, W) in [0, 1]."""
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class CellMaskDataset(Dataset):
    """
    Yields (image, mask) tensors: image (3, H, W) in [0, 1], mask (H, W) in {0, 1}.

    `augment` is an optional albumentations-style callable taking
    image=..., mask=... and returning a dict with the same keys, so geometric
    transforms are applied identically to both.
    """

    def __init__(self, pairs, augment=None):
        self.pairs = list(pairs)
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img, mask = read_image(img_path), read_mask(mask_path)
        if self.augment is not None:
            out = self.augment(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(mask)).float()
