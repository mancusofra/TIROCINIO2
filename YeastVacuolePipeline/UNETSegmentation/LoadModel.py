import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage.morphology import binary_dilation

from torchvision import transforms as T 
import segmentation_models_pytorch as smp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    """
    Loads the PyTorch U-Net segmentation model (EfficientNet-B7 encoder,
    ImageNet-pretrained) with the given fine-tuned weights, ready for inference.

    Args:
        model_path (str): Path to the saved state dict (Weights.pt).

    Returns:
        torch.nn.Module: The model, moved to `device` and set to eval mode.
    """
    model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Runs the segmentation model on a single image and returns its binary mask.

    Note: the `device` parameter shadows the module-level `device` variable —
    it is not read from the module-level one, so calling this before the
    model itself has been moved to the same device would fail.

    Args:
        model (torch.nn.Module): A loaded U-Net model (see load_model).
        image_path (str): Path to the input image.
        device (str): 'cuda' or 'cpu', the device to run inference on.

    Returns:
        np.ndarray: Binary (0/1) predicted mask, same height/width as the input.

    Raises:
        FileNotFoundError: If image_path cannot be read by OpenCV.
    """
    model.eval()
    transform = T.Compose([
        T.ToTensor(),
    ])
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img)
        prediction = torch.where(prediction > 0.5, 1, 0)
        prediction = prediction.cpu()

    prediction = prediction.to('cpu')[0][0]
    return prediction.squeeze().byte().numpy()

def show_image(image):
    """Displays a single image (e.g. a predicted mask) full-frame, without axes."""
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    # Data/ lives next to YeastVacuolePipeline/ (this file is two levels
    # down, in UNETSegmentation/), independent of machine/user.
    DATA_DIR = (Path(__file__).resolve().parent.parent / "Data").as_posix()

    model_path = f"{DATA_DIR}/Model/Weights.pt"
    model = load_model(model_path)

    # Demo: run inference on every annotated image and display the mask
    train_annotated_path = f"{DATA_DIR}/Original_images/Train_annotated/"
    train_annotated_list = []
    for root, dirs, files in os.walk(train_annotated_path):
        for file in files:
            if file.endswith(('.tif')):
                image_path = os.path.join(root, file)
                prediction = predict(model, image_path)
                show_image(prediction)
        






