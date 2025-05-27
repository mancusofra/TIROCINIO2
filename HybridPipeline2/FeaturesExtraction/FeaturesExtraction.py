import os, cv2, torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Features Extraction Modules
from .Extractors.GeometricFeatures import extract_geometric_features
from .Extractors.HuMomentsFeatures import extract_hu_moments
from .Extractors.ZernikeFeatures import extract_zernike_moments
from .Extractors.HaralickFeatures import extract_haralick_features
from .Extractors.LBPFeatures import extract_lbp_features
from .Extractors.GrayHistFeatures import extract_gray_hist_features

from HybridPipeline2.UNETSegmentation.LoadModel import load_model, predict, show_image
import glob

def file_extraction(input_dir, Verbose=False):
    image_dir = os.path.join(input_dir, "GrayImages")
    mask_dir = os.path.join(input_dir, "MaskedImages")
    image_files = []
    mask_files = []

    for root, dirs, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".tif"):
                mask_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".tif"):
                image_files.append(os.path.join(root, file))

    if len(mask_files) != len(image_files):
        raise ValueError("Masks and grayscale images do not match.")

    elif Verbose:
        print(f"Images found: {len(mask_files)}")

    return mask_files, image_files

def features_extraction(gray_images, masked_images, features_dir="./Data/Features", params = None):
    for gray_path, masked_path in tqdm(zip(gray_images, masked_images), total=len(gray_images), desc=f"Extracting {features_dir.split('/')[-2]}"):

        features = ""
        # Convert files to 8-bit format required for feature extraction operations
        gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        masked_image = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)

        # Define output path for CSV files
        dir_name = f"{features_dir}/{masked_path.split('/')[-2]}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + masked_path.split('/')[-1][0:-4] + ".csv"

        # This code extracts geometric features from the images to analyze the shape of the objects.
        # Since these features are based on object structure and shape, we use binary images (masks),
        # which provide a clear representation without color interference.

        # Extract geometric features based on object shape and structure.        
        geo_features = extract_geometric_features(gray_image)
        
        for val in geo_features.values():
            features += str(val) + "\n"

        # Hu invariant moments describe the shape of the object independently of rotation, scale, and translation.
        for val in extract_hu_moments(masked_image):
            features += str(val) + "\n"

        # Zernike moments capture symmetry and shape complexity.
        if params and ("zernike" in params):
            zernike_params = params["zernike"]
            if isinstance(zernike_params, dict):
                zernike_features = extract_zernike_moments(masked_image, **zernike_params)
            else:
                raise ValueError("Invalid parameters for Zernike features.")
        else:
            zernike_features = extract_zernike_moments(masked_image)


        for val in zernike_features:
            features += str(val) + "\n"

        # Extract texture-based features from the image.
        # Since these features depend on local pixel variation and surface patterns of the object,
        # we use grayscale images to better capture intensity differences without color influence.

        # Haralick features analyze pixel co-occurrence to describe image texture.
        for val in extract_haralick_features(gray_image):
            features += str(val) + "\n"

        # LBP (Local Binary Pattern) captures local texture patterns based on intensity differences among neighboring pixels.
        if params and ("lpb" in params):
            lpb_params = params["lpb"]
            if isinstance(lpb_params, dict):
                lpb_features = extract_lbp_features(gray_image, **lpb_params)
            else:
                raise ValueError("Invalid parameters for LBP features.")
        
        else:
            lbp_features = extract_lbp_features(gray_image)

        for val in extract_lbp_features(gray_image):
            features += str(val) + "\n"

        # Extract features based on grayscale histogram.
        for val in extract_gray_hist_features(gray_image):
            features += str(val) + "\n"

        with open(file_name, 'w', newline='') as file:
            file.write(features)

    return features_dir

def mask_dataset_maker(input_dir, output_dir, masked_dir, model_path, verbose=False, test = False):
    model = load_model(model_path)

    if test: 
        output_dir = output_dir + "_test"

    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for subdir in tqdm(subdirs):
        output_mask_path = f"{output_dir}/MaskedImages/{subdir}/"
        if not os.path.exists(output_mask_path):
            os.makedirs(output_mask_path)

        subdir_path = os.path.join(input_dir, subdir)
        masked_subdir = os.path.join(masked_dir, subdir)

        tif_files = [f for f in os.listdir(subdir_path) if f.endswith('.tif')]
        masked_tif_files = [f for f in os.listdir(masked_subdir) if f.endswith('.tif')]
        
        for f in tif_files:
            if f not in  masked_tif_files:
                full_path = os.path.join(subdir_path, f)
                predicted_f_mask = predict(model, full_path)
                #show_image(predicted_f_mask)
                
                
                cv2.imwrite(f"{output_mask_path}/{f}", predicted_f_mask*255)

            else:   
                manual_f_mask = cv2.imread(f"{masked_subdir}/{f}", cv2.IMREAD_COLOR)
                cv2.imwrite(f"{output_mask_path}/{f}", manual_f_mask)         
        
        #print(f"{count1} {count2}")
        num_files = sum([len(files) for _, _, files in os.walk(output_mask_path)])
        if verbose:
            print(f"Number of files in {output_mask_path}: {num_files}")

def gray_dataset_maker(input_dir, rgb_dir, verbose=False, test = False):

    if test: 
        output_dir = input_dir + "_test/GrayImages"
        input_dir = input_dir + "_test/MaskedImages"
    else:
        output_dir = input_dir + "/GrayImages"
        input_dir = input_dir + "/MaskedImages"

    
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    for subdir in subdirs:
        output_gray_dir = f"{output_dir}/{subdir}"
        input_gray_dir = f"{input_dir}/{subdir}"
        rgb_subdir = f"{rgb_dir}/{subdir}"

        if not os.path.exists(output_gray_dir):
            os.makedirs(output_gray_dir)

        tif_files = [f for f in os.listdir(input_gray_dir) if f.endswith('.tif')]
        rgb_tif_files = [f for f in os.listdir(rgb_subdir) if f.endswith('.tif')]

        # Assumiamo che i nomi corrispondano
        for mask_name, rgb_name in zip(sorted(tif_files), sorted(rgb_tif_files)):
            mask_path = os.path.join(input_gray_dir, mask_name)
            rgb_path = os.path.join(rgb_subdir, rgb_name)

            # Carica immagini
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            rgb = cv2.imread(rgb_path)

            # Assicura che le dimensioni corrispondano
            if mask.shape != rgb.shape[:2]:
                print(f"Dimension mismatch: {mask_name} vs {rgb_name}")
                continue

            # Crea maschera binaria (valori 0 e 1)
            binary_mask = (mask > 0).astype(np.uint8)

            # Espandi a 3 canali per mascherare RGB
            binary_mask_3c = cv2.merge([binary_mask]*3)

            # Applica la maschera all'immagine RGB
            masked_rgb = cv2.bitwise_and(rgb, rgb, mask=binary_mask)

            # Salva
            cv2.imwrite(f"{output_gray_dir}/{mask_name}", masked_rgb)

def full_dataset_maker(input_dir, output_dir, masked_dir, features_dir,  model_path, verbose=False, test = False):
    mask_dataset_maker(input_dir, output_dir, masked_dir, model_path, verbose=verbose, test=test) 
    gray_dataset_maker(output_dir, input_dir, verbose=verbose, test=test)

    if test:
        features_dir = features_dir + "_test"
        output_dir = output_dir + "_test"

    mask_files, image_files = file_extraction(output_dir, Verbose=verbose)
    features_dir = features_extraction(image_files, mask_files, features_dir=features_dir)
    return features_dir
    
if __name__ == "__main__":

    masked_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Mask"
    input_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Original_images/Train_annotated"
    input_dir_test = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Original_images/test"
    output_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/DataSet"
    model_path = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Model/Weights.pt"
    features_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Features"

    full_dataset_maker(input_dir, output_dir, masked_dir, features_dir, model_path, verbose = False, test = False)
    full_dataset_maker(input_dir_test, output_dir, masked_dir, features_dir, model_path, verbose = False, test = True)