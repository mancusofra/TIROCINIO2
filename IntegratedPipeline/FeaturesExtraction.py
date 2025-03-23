import cv2, os, json, sys, random, csv
import numpy as np
import pandas as pd

import mahotas
from skimage.feature import local_binary_pattern
from skimage.measure import moments, moments_hu
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature.texture import graycomatrix
import matplotlib.pyplot as plt

from VaculeMaskMaker import maskMaker

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def extract_geometric_features(gray, image, image_file):
    features = {
        "total_area": 0,
        "total_perimeter": 0,
    }
    eccentricities = []
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    if len(contours) > 0:
        cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            features["total_area"] += area
            features["total_perimeter"] += perimeter

            if len(cnt) >= 5:  # fitEllipse requires at least 5 points
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                major_axis = max(axes) / 2.0  # a
                minor_axis = min(axes) / 2.0  # b
                eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
                eccentricities.append(eccentricity)

        circularity = 4 * np.pi * (features["total_area"] / (features["total_perimeter"] ** 2)) if features["total_perimeter"] != 0 else 0
        features["circularity"] = circularity

        """if eccentricities:
            features["mean_eccentricity"] = np.mean(eccentricities)
        else:
            features["mean_eccentricity"] = 0"""

        
    else:
        print(f"⚠️ Nessun contorno trovato in {image_file}!")
    return features, image_contours

def extract_haralick_features(gray):
    return mahotas.features.haralick(gray).mean(axis=0)

def extract_hu_moments(gray):
    mom = moments(gray)
    return moments_hu(mom)

def extract_zernike_moments(gray, image_file):
    zernike_features = mahotas.features.zernike_moments(gray, radius=1000, degree=8)
    if np.all(zernike_features == 0):
        print(f"⚠️ Zernike momenti nulli per {image_file}, verifica l'immagine!")
        cv2.imwrite(f"error_{image_file}", gray)
    return zernike_features

def extract_lbp_features(gray):
    P, R = 52, 2
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return hist

def process_images_csv(gray_images, masked_images, contours = False):
    for gray_path, masked_path in zip (gray_images, masked_images):
        features = ""
        gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        masked_image = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)
        dir_name = f"./Features/{masked_path.split('/')[-2]}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + masked_path.split('/')[-1][0:-4] + ".csv"

        # Questo codice estrae feature geometriche dalle immagini per analizzare la forma degli oggetti.
        # Poiché queste feature si basano sulla struttura e sulla geometria dell'oggetto, utilizziamo immagini binarie (maschere),
        # che forniscono una rappresentazione chiara senza interferenze cromatiche.
                
        # Estrazione delle feature geometriche basate sulla forma e struttura dell'oggetto.
        """geo_features, image_contours = extract_geometric_features(masked_image, gray_image, masked_path)
        for val in geo_features.values():
            features += str(val) + "\n"
        """

        # I momenti invarianti di Hu descrivono la forma dell'oggetto indipendentemente da rotazione, scala e traslazione.
        for val in extract_hu_moments(masked_image):
            features += str(val) + "\n"

        # I momenti di Zernike catturano la simmetria e la complessità della forma.
        zernike_moments = extract_zernike_moments(masked_image, masked_path)
        for val in zernike_moments:
            features += str(val) + "\n"

        # Estrazione delle feature basate sulla texture dell'immagine.
        # Poiché queste caratteristiche dipendono dalla variazione locale dei pixel e dai pattern presenti nella superficie dell'oggetto,
        # utilizziamo immagini in scala di grigi per catturare meglio le differenze di intensità senza l'influenza del colore.
        
        # Le feature di Haralick analizzano la co-occorrenza dei pixel per descrivere la texture dell'immagine.
        for val in extract_haralick_features(gray_image):
            features += str(val) + "\n"
        
        # LBP (Local Binary Pattern) cattura pattern locali di texture basandosi sulle differenze di intensità tra pixel vicini.
        for val in extract_lbp_features(gray_image):
            features += str(val) + "\n"

        
        with open(file_name, 'w', newline='') as file:
            file.write(features)

        """if contours:
            if not os.path.exists("Images"):
                os.makedirs("Images")
            output_path = os.path.join("Images/", f"contours_{(gray_path.split('/')[-1])[0:-4]}.png")
            cv2.imwrite(output_path, image_contours)"""

def display_random_images(folder, num_images=9):
        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
        selected_images = random.sample(image_files, min(num_images, len(image_files)))

        plt.figure(figsize=(10, 10))
        for i, image_path in enumerate(selected_images):
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_rgb)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def create_dataset(images_dir):
    masked_directory = "./DataSet/MaskedImages"
    gray_directory = "./DataSet/GrayImages"
    maskMaker(images_dir, masked_directory)
    if not os.path.exists(gray_directory):
        os.makedirs(gray_directory)

    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith(".tif"):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    gray_image_path = os.path.join(gray_directory, os.path.relpath(image_path, images_dir))
                    gray_image_dir = os.path.dirname(gray_image_path)
                    if not os.path.exists(gray_image_dir):
                        os.makedirs(gray_image_dir)
                    cv2.imwrite(gray_image_path, image_8bit)

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        features_folder = "./Features"
        first_file = os.listdir(features_folder)[0]
        with open(os.path.join(features_folder, first_file), 'r') as file:
            features_list = [line.strip() for line in file]
        print(len(features_list))

    

        #display_random_images("Images")
    
    else:
        images_dir = "/home/francesco/Scaricati/Dataset/Images/train_clustershcf"
        dataset_path = "./DataSet"
        """create_dataset(images_dir)"""

        image_dir = os.path.join(dataset_path, "GrayImages")
        mask_dir = os.path.join(dataset_path, "MaskedImages")
        image_files = []
        mask_files = []
        for root, dirs, files in os.walk(mask_dir):
            for file in files:
                if file.endswith(".tif"):
                    mask_files.append(os.path.join(root, file))

        print(f"Immagini trovate: {len(mask_files)}")

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith(".tif"):
                    image_files.append(os.path.join(root, file))
            
        print(f"Immagini trovate: {len(image_files)}")
        process_images_csv(image_files, mask_files, contours=True)