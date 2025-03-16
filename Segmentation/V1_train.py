import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Forza l'uso di un backend GUI
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam

import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_images_from_directory(directory, target_size):
    images = []
    for filename in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, filename)
        img = load_img(img_path, target_size=target_size)
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

def load_masks_from_directory(directory, target_size):
    masks = []
    for filename in sorted(os.listdir(directory)):
        mask_path = os.path.join(directory, filename)
        mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")  # Load as grayscale
        mask_array = img_to_array(mask)
        masks.append(mask_array)
    return np.array(masks)

def display_sample(images, masks, index=0):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Sample Image")
    plt.imshow(images[index])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Sample Mask")
    plt.imshow(masks[index].squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

def unet(input_size=(128, 128, 3)):
    
    inputs = Input(input_size)

    # Downsampling
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)

    # Upsampling
    u1 = UpSampling2D(size=(2, 2))(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = UpSampling2D(size=(2, 2))(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def combine_generators(image_generator, mask_generator):
    for images, masks in zip(image_generator, mask_generator):
        yield images, masks

def estimate_model_size(model):

    total_params = model.count_params()
    param_size = 4 
    model_size_bytes = total_params * param_size
    model_size_mb = model_size_bytes / (1024 ** 2)
    print(f"Estimated model size: {model_size_mb:.2f} MB")

if __name__ == "__main__":
    dataset_path = "quadeer15sh/augmented-forest-segmentation"
    image_folder = "Forest Segmented/Forest Segmented/images/"
    mask_folder = "Forest Segmented/Forest Segmented/masks/"

    path = kagglehub.dataset_download(dataset_path)
    print(f"Dataset downloaded to {path}")
    print(f"Image folder: {image_folder}")
    print(f"Number of files in image folder: {len(os.listdir(os.path.join(path, image_folder)))}")
    print(f"Mask folder: {len(os.listdir(os.path.join(path, mask_folder)))}")

    image_dir = os.path.join(path, image_folder)
    mask_dir = os.path.join(path, mask_folder)
    target_size = (128, 128)

    images = load_images_from_directory(image_dir, target_size) / 255.0
    masks = load_masks_from_directory(mask_dir, target_size) / 255.0

    #display_sample(images, masks, index=0)

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    data_gen_args = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    seed = 42
    train_image_generator = image_datagen.flow(X_train, batch_size=16, seed=seed)
    train_mask_generator = mask_datagen.flow(y_train, batch_size=16, seed=seed)
    val_image_generator = image_datagen.flow(X_val, batch_size=16, seed=seed)
    val_mask_generator = mask_datagen.flow(y_val, batch_size=16, seed=seed)

    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    model = unet()

    train_generator = combine_generators(train_image_generator, train_mask_generator)
    val_generator = combine_generators(val_image_generator, val_mask_generator)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        steps_per_epoch=len(X_train) // 16,
        validation_steps=len(X_val) // 16,
    )

    model.summary()
    estimate_model_size(model)
    model.save('basic_unet_model.keras')


