import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation

from torchvision import transforms as T 
import segmentation_models_pytorch as smp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
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
    model.eval()
    transform = T.Compose([
        T.ToTensor(),
    ])
    #print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"Impossibile leggere l'immagine: {image_path}")

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
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    model_path = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Model/Weights.pt"
    model = load_model(model_path)

    train_annotated_path = '/home/francesco/TIROCINIO2/HybridPipeline2/Data/Train_annotated/'
    train_annotated_list = []
    for root, dirs, files in os.walk(train_annotated_path):
        for file in files:
            if file.endswith(('.tif')):
                image_path = os.path.join(root, file)
                prediction = predict(model, img)
                show_image(prediction)
        






