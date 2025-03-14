import cv2, os
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

dataset_path = "ipateam/nuinsseg"  # Nome del dataset Kaggle
image_folder = "human bladder/mask binary"  # Cartella contenente le immagini nel dataset

# 📌 Carica il dataset con KaggleHub
path = kagglehub.dataset_download(dataset_path)

# 📌 Costruisci il percorso completo alla cartella delle immagini
image_dir = os.path.join(path, image_folder)

# 📌 Lista tutti i file PNG
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Caricare immagine
#print(f"{image_dir}/{image_files[1]}")
image = cv2.imread(f"{image_dir}/{image_files[1]}")  # Cambia con il nome corretto del file
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 📌 **Geometric features** (contorni, area, perimetro)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crea una copia dell'immagine per non modificarla direttamente
image_contours = image.copy()  

if len(contours) > 0:
    # Disegna i contorni in rosso (BGR: 0, 0, 255) con spessore 5
    cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)
else:
    print("⚠️ Nessun contorno trovato!")

# Salva l'immagine con i contorni
cv2.imwrite("Contorni.png", image_contours) 
area = cv2.contourArea(contours[0])
perimeter = cv2.arcLength(contours[0], True)