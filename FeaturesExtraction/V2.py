import cv2, os, json
import numpy as np
import mahotas
from skimage.feature import local_binary_pattern
from skimage.measure import moments, moments_hu
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature.texture import graycomatrix, graycomatrix
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Converte automaticamente tutti gli array
        return super().default(obj)

def process_images(image_dir, image_files, output_folder):
    data = {}
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = {}
        
        # 📌 **Geometric features** (contorni, area, perimetro)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crea una copia dell'immagine per disegnare i contorni
        image_contours = image.copy()
        
        if len(contours) > 0:
            # Disegna i contorni in verde (BGR: 0, 255, 0) con spessore 1
            cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)
            
            # Calcolo area e perimetro del primo contorno
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            
            # Salva i dettagli geometrici
            #print(f"{image_file} -> Area: {area}, Perimetro: {perimeter}", end=' ')
            features["area"] = area
            features["perimeter"] = perimeter

        else:
            print(f"⚠️ Nessun contorno trovato in {image_file}!")
        
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        #print(f"LBP shape: {lbp.shape}")
        #features["lbp"] = lbp

        haralick_features = mahotas.features.haralick(gray).mean(axis=0)
        #print(f"Haralick features: {haralick_features}")
        features["haralick"] = haralick_features

        mom = moments(gray)
        hu_moments = moments_hu(mom)
        #print(f"Hu moments: {hu_moments}")
        features["hu_moments"] = hu_moments

        zernike_features = mahotas.features.zernike_moments(gray, radius=1000, degree=5)
        #print(f"Zernike moments: {zernike_features}")
        features["zernike_moments"] = zernike_features
        if np.all(zernike_features == 0):
            print(f"⚠️ Zernike momenti nulli per {image_file}, verifica l'immagine!")
            cv2.imwrite(f"error_{image_file}", gray)
        
        P = 8 # Numero di pixel nel vicinato
        R = 1  # Raggio
        lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")

        # Istogramma dei valori LBP
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 47), density=True)
        #print(len(hist))  # Output: 54

        features["lbp"] = hist

        # Salva l'immagine con contorni nella cartella Images
        output_path = os.path.join(output_folder, f"contours_{image_file}")
        cv2.imwrite(output_path, image_contours)

        data[image_file] = features
    # Nome del file JSON
    filename = "dati.json"

    # Scrivere il dizionario su file JSON
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, cls=NumpyEncoder)  # indent=4 rende il file più leggibile

    print(f"Dati salvati in {filename}")

dataset_path = "ipateam/nuinsseg"  # Nome del dataset Kaggle
image_folder = "human bladder/mask binary"  # Cartella contenente le immagini nel dataset

path = kagglehub.dataset_download(dataset_path)

image_dir = os.path.join(path, image_folder)

image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

output_folder = "Images"
os.makedirs(output_folder, exist_ok=True)
        
process_images(image_dir, image_files, output_folder)
