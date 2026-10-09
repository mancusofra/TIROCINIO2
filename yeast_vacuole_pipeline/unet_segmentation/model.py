"""Single definition of the segmentation network, shared by training and inference."""

import segmentation_models_pytorch as smp
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_unet(encoder_weights="imagenet"):
    """
    U-Net with an EfficientNet-B7 encoder (segmentation_models_pytorch),
    3-channel input and a single sigmoid output channel (binary mask).

    Pass encoder_weights=None when loading saved weights, to skip the ImageNet download.
    """
    return smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
