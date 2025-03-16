import cv2, os, json
import numpy as np
import mahotas
from skimage.feature import local_binary_pattern
from skimage.measure import moments, moments_hu
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.feature.texture import graycomatrix
import matplotlib.pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def extract_geometric_features(gray, image, image_file):
    features = {}
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    if len(contours) > 0:
        cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)
        area = cv2.contourArea(contours[0])
        perimeter = cv2.arcLength(contours[0], True)
        features["area"] = area
        features["perimeter"] = perimeter
    else:
        print(f"⚠️ Nessun contorno trovato in {image_file}!")
    return features, image_contours

def extract_haralick_features(gray):
    return {"haralick": mahotas.features.haralick(gray).mean(axis=0)}

def extract_hu_moments(gray):
    mom = moments(gray)
    return {"hu_moments": moments_hu(mom)}

def extract_zernike_moments(gray, image_file):
    zernike_features = mahotas.features.zernike_moments(gray, radius=1000, degree=5)
    if np.all(zernike_features == 0):
        print(f"⚠️ Zernike momenti nulli per {image_file}, verifica l'immagine!")
        cv2.imwrite(f"error_{image_file}", gray)
    return {"zernike_moments": zernike_features}

def extract_lbp_features(gray):
    P, R = 16, 2
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return {"lbp": hist}

def process_images(image_dir, mask_dir, image_files, mask_files, output_folder):
    data = {}
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file) # Attenzione: stesso nome! trovare modo di scorrere lista nomi maschere insieme a quella delle immagini

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        features = {}

        # Questo codice estrae feature geometriche dalle immagini per analizzare la forma degli oggetti.
        # Poiché queste feature si basano sulla struttura e sulla geometria dell'oggetto, utilizziamo immagini binarie (maschere),
        # che forniscono una rappresentazione chiara senza interferenze cromatiche.
                
        # Estrazione delle feature geometriche basate sulla forma e struttura dell'oggetto.
        geo_features, image_contours = extract_geometric_features(mask_gray, mask, image_file)
        features.update(geo_features)

        # I momenti invarianti di Hu descrivono la forma dell'oggetto indipendentemente da rotazione, scala e traslazione.
        features.update(extract_hu_moments(mask_gray))

        # I momenti di Zernike catturano la simmetria e la complessità della forma.
        features.update(extract_zernike_moments(mask_gray, image_file))

        # Estrazione delle feature basate sulla texture dell'immagine.
        # Poiché queste caratteristiche dipendono dalla variazione locale dei pixel e dai pattern presenti nella superficie dell'oggetto,
        # utilizziamo immagini in scala di grigi per catturare meglio le differenze di intensità senza l'influenza del colore.

        # Le feature di Haralick analizzano la co-occorrenza dei pixel per descrivere la texture dell'immagine.
        features.update(extract_haralick_features(gray))

        # LBP (Local Binary Pattern) cattura pattern locali di texture basandosi sulle differenze di intensità tra pixel vicini.
        features.update(extract_lbp_features(gray))
        
        output_path = os.path.join(output_folder, f"contours_{image_file}")
        cv2.imwrite(output_path, image_contours)
        
        data[image_file] = features
    
    filename = "dati.json"
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, cls=NumpyEncoder)
    print(f"Dati salvati in {filename}")

if __name__ == "__main__":
    dataset_path = "abdallahwagih/retina-blood-vessel"
    image_folder = "Data/test/image"
    mask_folder = "Data/test/mask"
    path = kagglehub.dataset_download(dataset_path)
    image_dir = os.path.join(path, image_folder)
    mask_dir = os.path.join(path, mask_folder)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    
    output_folder = "Images"
    os.makedirs(output_folder, exist_ok=True)
    process_images(image_dir, mask_dir, image_files, mask_files, output_folder)
