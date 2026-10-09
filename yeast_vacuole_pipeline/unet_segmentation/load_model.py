"""Inference helpers: load the trained U-Net and predict a binary mask for one image."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .dataset import read_image, to_tensor
from .model import DEVICE, build_unet


def load_model(model_path):
    """
    Loads the U-Net with the given fine-tuned weights, ready for inference.

    Args:
        model_path (str): Path to the saved state dict (Weights.pt).

    Returns:
        torch.nn.Module: The model, moved to DEVICE and set to eval mode.
    """
    model = build_unet(encoder_weights=None)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model, image_path, threshold=0.5):
    """
    Runs the segmentation model on a single image and returns its binary mask.

    Args:
        model (torch.nn.Module): A loaded U-Net model (see load_model).
        image_path (str): Path to the input image.
        threshold (float): Probability above which a pixel is foreground.

    Returns:
        np.ndarray: Binary (0/1) uint8 mask, same height/width as the input.

    Raises:
        FileNotFoundError: If image_path cannot be read.
    """
    img = to_tensor(read_image(image_path)).unsqueeze(0).to(DEVICE)
    probs = model(img)[0, 0].cpu()
    return (probs > threshold).byte().numpy()


def show_image(image):
    """Displays a single image (e.g. a predicted mask) full-frame, without axes."""
    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    # Demo: run inference on every annotated image and display the mask.
    data_dir = Path(__file__).resolve().parent.parent / "Data"
    model = load_model(data_dir / "Model" / "Weights.pt")
    for image_path in sorted((data_dir / "Original_images" / "Train_annotated").rglob("*.tif")):
        show_image(predict(model, image_path))
