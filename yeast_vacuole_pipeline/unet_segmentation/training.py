"""
Training, evaluation and visualization of the U-Net segmentation model.

Run as a module:
    python -m yeast_vacuole_pipeline.unet_segmentation.training train
    python -m yeast_vacuole_pipeline.unet_segmentation.training evaluate
"""

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CellMaskDataset, find_pairs, split_pairs
from .metrics import bce_dice_loss, dice_score, iou_score
from .model import DEVICE, build_unet

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"


@dataclass
class TrainConfig:
    masks_dir: Path = DATA_DIR / "Mask"
    images_dir: Path = DATA_DIR / "Original_images" / "Train_annotated"
    model_dir: Path = DATA_DIR / "Model"
    epochs: int = 60
    batch_size: int = 16
    lr: float = 1e-3
    patience: int = 7  # epochs without val-loss improvement before stopping
    seed: int = 42

    @property
    def weights_path(self):
        return self.model_dir / "Weights.pt"

    @property
    def history_path(self):
        return self.model_dir / "history.pkl"


def training_augmentation():
    # Cells have no canonical orientation, so flips/rotations are label-preserving.
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ]
    )


def make_loaders(cfg):
    train, val, test = split_pairs(find_pairs(cfg.masks_dir, cfg.images_dir), seed=cfg.seed)
    return (
        DataLoader(
            CellMaskDataset(train, training_augmentation()), batch_size=cfg.batch_size, shuffle=True
        ),
        DataLoader(CellMaskDataset(val), batch_size=cfg.batch_size),
        DataLoader(CellMaskDataset(test), batch_size=cfg.batch_size),
    )


def run_epoch(model, loader, optimizer=None):
    """
    One pass over `loader`; trains when an optimizer is given, otherwise only evaluates.

    Returns:
        dict: Mean 'loss', 'dice' and 'iou' over all images in the loader.
    """
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0}

    with torch.set_grad_enabled(training):
        for img, mask in tqdm(loader, leave=False, disable=not training):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            probs = model(img).squeeze(1)
            loss = bce_dice_loss(probs, mask)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            totals["loss"] += loss.item() * len(img)
            totals["dice"] += dice_score(probs, mask).sum().item()
            totals["iou"] += iou_score(probs, mask).sum().item()

    return {k: v / len(loader.dataset) for k, v in totals.items()}


def train_model(cfg=None):
    """
    Trains a fresh U-Net, keeping the weights with the lowest validation loss.

    Saves the best weights and the per-epoch history to cfg.model_dir.

    Returns:
        dict: History with 'train_loss', 'val_loss', 'val_dice', 'val_iou' lists.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = make_loaders(cfg)
    model = build_unet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.2)

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_loss, stale_epochs = float("inf"), 0

    for epoch in range(1, cfg.epochs + 1):
        train = run_epoch(model, train_loader, optimizer)
        val = run_epoch(model, val_loader)
        scheduler.step(val["loss"])

        history["train_loss"].append(train["loss"])
        for k in ("loss", "dice", "iou"):
            history[f"val_{k}"].append(val[k])
        print(
            f"epoch {epoch:3d} | train loss {train['loss']:.4f} | val loss {val['loss']:.4f}"
            f" | val dice {val['dice']:.4f} | val IoU {val['iou']:.4f}"
        )

        if val["loss"] < best_loss:
            best_loss, stale_epochs = val["loss"], 0
            torch.save(model.state_dict(), cfg.weights_path)
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.patience:
                print(f"Early stopping: no improvement for {cfg.patience} epochs.")
                break

    with open(cfg.history_path, "wb") as f:
        pickle.dump(history, f)
    return history


def load_trained_model(weights_path):
    model = build_unet(encoder_weights=None)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


def plot_history(history):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history["train_loss"], label="train")
    ax_loss.plot(history["val_loss"], label="validation")
    ax_loss.set(title="BCE + Dice loss", xlabel="epoch")
    ax_loss.legend()
    ax_score.plot(history["val_dice"], label="Dice")
    # Histories saved before the training rewrite use the key "val_IoU".
    ax_score.plot(history.get("val_iou", history.get("val_IoU", [])), label="IoU")
    ax_score.set(title="Validation scores", xlabel="epoch", ylim=(0, 1))
    ax_score.legend()
    fig.tight_layout()


@torch.no_grad()
def plot_predictions(model, dataset, n=8):
    """Shows `n` test images with ground-truth (green) and predicted (red) mask contours."""
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(2.5 * ((n + 1) // 2), 5))
    for ax, idx in zip(axes.flat, np.linspace(0, len(dataset) - 1, n, dtype=int)):
        img, mask = dataset[idx]
        pred = model(img.unsqueeze(0).to(DEVICE))[0, 0].cpu() > 0.5
        ax.imshow(img.permute(1, 2, 0))
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=1)
        ax.contour(pred, levels=[0.5], colors="red", linewidths=1)
        ax.axis("off")
    fig.suptitle("Ground truth (green) vs prediction (red)")
    fig.tight_layout()


def evaluate(cfg=None):
    """Loads the trained model, prints test-set metrics and plots history and predictions."""
    cfg = cfg or TrainConfig()
    _, _, test_loader = make_loaders(cfg)
    model = load_trained_model(cfg.weights_path)

    scores = run_epoch(model, test_loader)
    print(f"test | loss {scores['loss']:.4f} | dice {scores['dice']:.4f} | IoU {scores['iou']:.4f}")

    if cfg.history_path.exists():
        with open(cfg.history_path, "rb") as f:
            plot_history(pickle.load(f))
    plot_predictions(model, test_loader.dataset)
    plt.show()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the U-Net segmentation model.")
    parser.add_argument("command", choices=["train", "evaluate"])
    args = parser.parse_args()
    if args.command == "train":
        train_model()
    else:
        evaluate()
