"""
Segmentation metrics and loss for binary masks.

All functions take `probs` (sigmoid outputs in [0, 1]) and `target` (0/1 masks)
of shape (N, H, W) and work per image, so batch scores can be summed or averaged.
"""

import torch
import torch.nn.functional as F

EPS = 1e-7


def _binarize(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (probs > threshold).float()


def dice_score(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Dice coefficient 2|A∩B| / (|A| + |B|) per image, on thresholded predictions."""
    pred = _binarize(probs, threshold)
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2))
    total = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (2 * inter + EPS) / (total + EPS)


def iou_score(probs: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Intersection over Union |A∩B| / |A∪B| per image, on thresholded predictions."""
    pred = _binarize(probs, threshold)
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - inter
    return (inter + EPS) / (union + EPS)


def soft_dice_loss(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable 1 - Dice on raw probabilities, averaged over the batch."""
    target = target.float()
    inter = (probs * target).sum(dim=(1, 2))
    total = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1 - (2 * inter + EPS) / (total + EPS)).mean()


def bce_dice_loss(
    probs: torch.Tensor, target: torch.Tensor, dice_weight: float = 0.01
) -> torch.Tensor:
    """Binary cross-entropy plus a weighted soft-Dice term."""
    return F.binary_cross_entropy(probs, target.float()) + dice_weight * soft_dice_loss(
        probs, target
    )
