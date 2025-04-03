import os, cv2

#Features Extraction Modules
from .Extractors.GeometricFeatures import extract_geometric_features
from .Extractors.HuMomentsFeatures import extract_hu_moments
from .Extractors.ZernikeFeatures import extract_zernike_moments
from .Extractors.HaralickFeatures import extract_haralick_features
from .Extractors.LBPFeatures import extract_lbp_features
from .Extractors.GrayHistFeatures import extract_gray_hist_features


def file_extraction(input_dir, Verbose = False):
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
        raise ValueError("Maschere e Immagini in scala grigi non corrispondono.")

    elif Verbose:
        print(f"Immagini trovate: {len(mask_files)}")

    return mask_files, image_files

def features_extraction(gray_images, masked_images, features_dir = "./Data/Features/"):
    for gray_path, masked_path in zip (gray_images, masked_images):

        features = ""
        #Conversione file in formato 8bit necessario per le operazioni di estrazione delle features
        gray_image = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        masked_image = cv2.imread(masked_path, cv2.IMREAD_GRAYSCALE)

        #definizione path cartelle e file csv
        dir_name = f"{features_dir}{masked_path.split('/')[-2]}/"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_name = dir_name + masked_path.split('/')[-1][0:-4] + ".csv"

        # Questo codice estrae feature geometriche dalle immagini per analizzare la forma degli oggetti.
        # Poiché queste feature si basano sulla struttura e sulla geometria dell'oggetto, utilizziamo immagini binarie (maschere),
        # che forniscono una rappresentazione chiara senza interferenze cromatiche.
        
        # Estrazione delle feature geometriche basate sulla forma e struttura dell'oggetto.       
        geo_features = extract_geometric_features(gray_image)
        for val in geo_features.values():
            features += str(val) + "\n"

        # I momenti invarianti di Hu descrivono la forma dell'oggetto indipendentemente da rotazione, scala e traslazione.
        for val in extract_hu_moments(masked_image):
            features += str(val) + "\n"

        # I momenti di Zernike catturano la simmetria e la complessità della forma.
        for val in extract_zernike_moments(masked_image):
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

        # Estrazione delle feature basate sull'istogramma dei livelli di grigio.
        for val in extract_gray_hist_features(gray_image):
            features += str(val) + "\n"

        with open(file_name, 'w', newline='') as file:
            file.write(features)

    return features_dir

if __name__ == "__main__":
    dataset_path = "../DataSet"
    mask_files, image_files = file_extraction(dataset_path, Verbose = True)

    features_dir = features_extraction(image_files, mask_files)
    print(features_dir)
    



