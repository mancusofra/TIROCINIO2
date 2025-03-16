from tensorflow import keras
from sklearn.model_selection import train_test_split
from V1_train import load_images_from_directory, load_masks_from_directory
import kagglehub,os,sys
import numpy as np
import matplotlib.pyplot as plt


def display_results(X_test, y_test, predicted_masks, index=0):
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Test Image")
    plt.imshow(X_test[index])
    plt.axis('off')

    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(y_test[index].squeeze(), cmap='gray')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(predicted_masks[index].squeeze(), cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    dataset_path = "quadeer15sh/augmented-forest-segmentation"
    image_folder = "Forest Segmented/Forest Segmented/images/"
    mask_folder = "Forest Segmented/Forest Segmented/masks/"

    path = kagglehub.dataset_download(dataset_path)
    
    """print(f"Dataset downloaded to {path}")
    print(f"Image folder: {image_folder}")
    print(f"Number of files in image folder: {len(os.listdir(os.path.join(path, image_folder)))}")
    print(f"Mask folder: {len(os.listdir(os.path.join(path, mask_folder)))}")
    """

    image_dir = os.path.join(path, image_folder)
    mask_dir = os.path.join(path, mask_folder)
    target_size = (128, 128)

    images = load_images_from_directory(image_dir, target_size) / 255.0
    masks = load_masks_from_directory(mask_dir, target_size) / 255.0

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Caricare il modello
    model = keras.models.load_model("basic_unet_model.keras")
    predicted_masks = model.predict(X_test)
    predicted_masks = (predicted_masks > 0.5).astype(np.uint8)  # Threshold for binary masks
    """if sys.argv[0] == "1":
        # Stampare il riepilogo del modello
        model.summary()
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")"""

    # Display results for a few samples
    for i in range(10):  # Change 3 to any number of samples you want to visualize
        display_results(X_test, y_test, predicted_masks, index=i)
    
