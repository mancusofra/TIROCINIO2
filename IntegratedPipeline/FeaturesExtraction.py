import cv2, os, json, sys, random
import numpy as np
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
    features = {"total_area": 0, "total_perimeter": 0}
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_contours = image.copy()
    if len(contours) > 0:
        cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 1)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            features["total_area"] += area
            features["total_perimeter"] += perimeter
    else:
        print(f"⚠️ Nessun contorno trovato in {image_file}!")
    return features, image_contours

def extract_haralick_features(gray):
    return {"haralick": mahotas.features.haralick(gray).mean(axis=0)}

def extract_hu_moments(gray):
    mom = moments(gray)
    return {"hu_moments": moments_hu(mom)}

def extract_zernike_moments(gray, image_file):
    zernike_features = mahotas.features.zernike_moments(gray, radius=1000, degree=8)
    if np.all(zernike_features == 0):
        print(f"⚠️ Zernike momenti nulli per {image_file}, verifica l'immagine!")
        cv2.imwrite(f"error_{image_file}", gray)
    return {"zernike_moments": zernike_features}

def extract_lbp_features(gray):
    P, R = 52, 2
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), density=True)
    return {"lbp": hist}

def process_images(images, contours = False):
    data = {}
    for image_path in images:

        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ Errore nel caricamento di {image_path}!")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        features = {}

        # Questo codice estrae feature geometriche dalle immagini per analizzare la forma degli oggetti.
        # Poiché queste feature si basano sulla struttura e sulla geometria dell'oggetto, utilizziamo immagini binarie (maschere),
        # che forniscono una rappresentazione chiara senza interferenze cromatiche.
                
        # Estrazione delle feature geometriche basate sulla forma e struttura dell'oggetto.
        geo_features, image_contours = extract_geometric_features(mask_gray, image, image_path)
        features.update(geo_features)

        # I momenti invarianti di Hu descrivono la forma dell'oggetto indipendentemente da rotazione, scala e traslazione.
        features.update(extract_hu_moments(mask_gray))

        # I momenti di Zernike catturano la simmetria e la complessità della forma.
        features.update(extract_zernike_moments(mask_gray, image_path))

        # Estrazione delle feature basate sulla texture dell'immagine.
        # Poiché queste caratteristiche dipendono dalla variazione locale dei pixel e dai pattern presenti nella superficie dell'oggetto,
        # utilizziamo immagini in scala di grigi per catturare meglio le differenze di intensità senza l'influenza del colore.

        # Le feature di Haralick analizzano la co-occorrenza dei pixel per descrivere la texture dell'immagine.
        features.update(extract_haralick_features(gray))

        # LBP (Local Binary Pattern) cattura pattern locali di texture basandosi sulle differenze di intensità tra pixel vicini.
        features.update(extract_lbp_features(gray))
        
        if contours:
            if not os.path.exists("Images"):
                os.makedirs("Images")
            output_path = os.path.join("Images/", f"contours_{(image_path.split('/')[-1])[0:-4]}.png")
            cv2.imwrite(output_path, image_contours)
            
            data[image_path.split('/')[-1]] = features
    
    filename = "dati.json"
    with open(filename, "w") as file:
        json.dump(data, file, indent=4, cls=NumpyEncoder)
    print(f"Dati salvati in {filename}")

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
        with open("dati.json") as file:
            data = json.load(file)
            campione = data[list(data.keys())[0]]
            total = 0
            for key, value in campione.items():
                try:
                    numero_feature = len(list(value))
                except:
                    numero_feature = 1

                print(f"{key}: {numero_feature}")
                total += numero_feature
            print(f"Totale: {total}")    
    

        display_random_images("Images")
    
    else:
        images_dir = "/home/francesco/Scaricati/Dataset/Images/test"
        dataset_path = "./DataSet"
        create_dataset(images_dir)

        image_dir = os.path.join(dataset_path, "")
        mask_dir = os.path.join(dataset_path, "")
        image_files = []
        mask_files = []
        for root, dirs, files in os.walk(dataset_path+"/MaskedImages"):
            for file in files:
                if file.endswith(".tif"):
                    image_files.append(os.path.join(root, file))
            
        print(f"Immagini trovate: {len(image_files)}")
        process_images(image_files)