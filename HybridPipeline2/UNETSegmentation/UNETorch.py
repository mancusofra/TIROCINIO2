import os, time, sys, pickle
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import albumentations as A
from scipy.ndimage.morphology import binary_dilation
import segmentation_models_pytorch as smp
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


#Decide if to use GPU or CPU by checking if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStopping():
    """
    Stops training when loss stops decreasing in a PyTorch module.
    """
    def __init__(self, patience:int = 6, min_delta: float = 0, weights_path: str = 'Data/Model/Weights.pt'):
        """
        :param patience: number of epochs of non-decreasing loss before stopping
        :param min_delta: minimum difference between best and new loss that is considered
            an improvement
        :paran weights_path: Path to the file that should store the model's weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.weights_path = weights_path

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def load_weights(self, model: torch.nn.Module):
        """
        Loads weights of the best model.
        :param model: model to which the weigths should be loaded
        """
        return model.load_state_dict(torch.load(self.weights_path))

class MriDataset(Dataset):
    def __init__(self, df, transform=None, mean=0.5, std=0.25):
        super(MriDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.mean = mean
        self.std = std
        
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx, raw=False):
        row = self.df.iloc[idx]
        img = cv2.imread(row['images_paths'], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row['masks_paths'], cv2.IMREAD_GRAYSCALE)
        if raw:
            return img, mask
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        img = T.functional.to_tensor(img)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return img, mask

def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    
    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))
    
    iou = (intersection + e) / (union + e)
    return iou

def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Dice coefficient for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()
    
    intersection = (predictions & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e)

def BCE_dice(output, target, alpha=0.01):
    bce = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice

def create_df(data_dir):
    images_paths = []
    masks_paths = glob(f'{data_dir}/*/*.tif')
    print(len(masks_paths))

    for i in masks_paths:
        images_paths.append(i.replace('/Mask/', '/Train_annotated/'))

    print(len(images_paths))
    for img_path in images_paths:
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")

    df = pd.DataFrame(data= {'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

def split_df(df):
    # create train_df
    train_df, dummy_df = train_test_split(df, train_size= 0.8, random_state= 42)

    # create valid_df and test_df
    valid_df, test_df = train_test_split(dummy_df, train_size= 0.5, random_state= 42)

    return train_df, valid_df, test_df

def create_gens(df, aug_dict):
    img_size = (80, 80)
    batch_size = 16


    img_gen = ImageDataGenerator(**aug_dict)
    msk_gen = ImageDataGenerator(**aug_dict)

    # Create general generator
    image_gen = img_gen.flow_from_dataframe(df, x_col='images_paths', class_mode=None, color_mode='rgb', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix='image', seed=1)

    mask_gen = msk_gen.flow_from_dataframe(df, x_col='masks_paths', class_mode=None, color_mode='grayscale', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix= 'mask', seed=1)

    gen = zip(image_gen, mask_gen)

    for (img, msk) in gen:
        img = img / 255
        msk = msk / 255

        msk[msk > 0.5] = 1
        msk[msk <= 0.5] = 0


        yield (img, msk)

def training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler):
    history = {'train_loss': [], 'val_loss': [], 'val_IoU': [], 'val_dice': []}
    early_stopping = EarlyStopping(patience=7)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()       
        running_loss = 0
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        with torch.no_grad():
            running_IoU = 0
            running_dice = 0
            running_valid_loss = 0
            for i, data in enumerate(valid_loader):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                predictions = model(img)
                predictions = predictions.squeeze(1)
                running_dice += dice_pytorch(predictions, mask).sum().item()
                running_IoU += iou_pytorch(predictions, mask).sum().item()
                loss = loss_fn(predictions, mask)
                running_valid_loss += loss.item() * img.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        val_loss = running_valid_loss / len(valid_loader.dataset)
        val_dice = running_dice / len(valid_loader.dataset)
        val_IoU = running_IoU / len(valid_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_IoU'].append(val_IoU)
        history['val_dice'].append(val_dice)
        print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
         f'| Validation Dice coefficient: {val_dice}')
        
        lr_scheduler.step(val_loss)
        if early_stopping(val_loss, model):
            early_stopping.load_weights(model)
            break

    model.eval()
    return history

def umodel():
    """Returns a UNET model with EfficientNet-B7 as encoder"""
    model = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid',
    )
    model.to(device)
    return model

def exemple():
    n_examples = 4

    fig, axs = plt.subplots(n_examples, 3, figsize=(20, n_examples*7), constrained_layout=True)
    i = 0
    for ax in axs:
        while True:
            image, mask = train_dataset.__getitem__(i, raw=True)
            i += 1
            if np.any(mask): 
                ax[0].set_title("MRI images")
                ax[0].imshow(image)
                ax[1].set_title("Highlited abnormality")
                ax[1].imshow(image)
                ax[1].imshow(mask, alpha=0.2)
                ax[2].imshow(mask)
                ax[2].set_title("Abnormality mask")
                break
    plt.show()
    plt.close()

def plot_loss(history):
    plt.figure(figsize=(7, 7))
    plt.plot(history['train_loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(history['val_IoU'], label='Validation mean Jaccard index')
    plt.plot(history['val_dice'], label='Validation Dice coefficient')
    plt.legend()
    plt.show()
    plt.close()

def plot_test_evaluation(test_loader, test_dataset, model, loss_fn):
    with torch.no_grad():
        running_IoU = 0
        running_dice = 0
        running_loss = 0
        for i, data in enumerate(test_loader):
            img, mask = data
            img, mask = img.to(device), mask.to(device)
            predictions = model(img)
            predictions = predictions.squeeze(1)
            running_dice += dice_pytorch(predictions, mask).sum().item()
            running_IoU += iou_pytorch(predictions, mask).sum().item()
            loss = loss_fn(predictions, mask)
            running_loss += loss.item() * img.size(0)
        loss = running_loss / len(test_dataset)
        dice = running_dice / len(test_dataset)
        IoU = running_IoU / len(test_dataset)
        
        print(f'Tests: loss: {loss} | Mean IoU: {IoU} | Dice coefficient: {dice}')

def plot_test_images(test_loader, test_dataset, model):
    width = 10
    columns = 2
    n_examples = columns * width

    fig, axs = plt.subplots(columns, width, figsize=(15*width , 15*columns), constrained_layout=True)
    red_patch = mpatches.Patch(color='red', label='The red data')
    fig.legend(loc='upper right',handles=[
        mpatches.Patch(color='red', label='Ground truth'),
        mpatches.Patch(color='green', label='Predicted abnormality')])
    i = 0
    with torch.no_grad():
        for data in test_loader:
            image, mask = data
            mask = mask[0]
            if not mask.byte().any():
                continue

            image = image.to(device)
            prediction = model(image).to('cpu')[0][0]
            prediction = torch.where(prediction > 0.5, 1, 0)
            prediction_edges = prediction - binary_dilation(prediction)
            ground_truth = mask - binary_dilation(mask)
            image[0, 0, ground_truth.bool()] = 1
            image[0, 1, prediction_edges.bool()] = 1
            
            axs[i//width][i%width].imshow(image[0].to('cpu').permute(1, 2, 0))
            if n_examples == i + 1:
                break
            i += 1
    plt.show()

def train_model(data_dir):
    df = create_df(data_dir)
    train_df, valid_df, test_df = split_df(df)

    transform = A.Compose([
    A.ChannelDropout(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
    ])

    train_dataset = MriDataset(train_df, transform)
    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)


    model = umodel()
    loss_fn = BCE_dice
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 60
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2,factor=0.2)

    return training_loop(epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler)

def load_and_plot():
    df = create_df(data_dir)
    train_df, valid_df, test_df = split_df(df)

    transform = A.Compose([
    A.ChannelDropout(p=0.3),
    A.RandomBrightnessContrast(p=0.3),
    A.ColorJitter(p=0.3),
    ])

    train_dataset = MriDataset(train_df, transform)
    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)

    batch_size = 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation='sigmoid',
    )
    model.load_state_dict(torch.load("/home/francesco/TIROCINIO2/HybridPipeline2/Data/Model/Weights.pt"))
    model.to(device)
    model.eval()

    with open("/home/francesco/TIROCINIO2/HybridPipeline2/Data/Model/history.pkl", "rb") as f:
        history = pickle.load(f)


    plot_loss(history)
    plot_test_evaluation(test_loader, test_dataset, model, BCE_dice)
    plot_test_images(test_loader, test_dataset, model)


if __name__ == "__main__":

    data_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Mask/"

    if len(sys.argv) < 2:
        print("Please provide an argument: -t to train the model or -v to visualize the results.")
        sys.exit(1)

    elif sys.argv[1] == "-v":
        load_and_plot()

    elif sys.argv[1] == "-t":
        history = train_model(data_dir)
        with open("Data/Model/history.pkl", "wb") as f:
            pickle.dump(history, f)

    else:
        print("Invalid argument. Use -t to train the model or -v to visualize the results.")

    