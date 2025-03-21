import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
import os,sys
import random

def auto_local_threshold_median(image_path, block_size=21, display=False):
    # Carica l'immagine in scala di grigi
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Immagine non trovata o errore nel caricamento.")
    
    # Applica il metodo di soglia locale Median
    local_thresh = threshold_local(image, block_size, method='median')
    binary_image = image > local_thresh  # Binarizzazione
    
    if display:
        # Visualizza i risultati
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Immagine Originale')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(binary_image, cmap='gray')
        plt.title('Auto Local Threshold (Median)')
        plt.axis('off')
        
        plt.show()
    return binary_image

# Esempio di utilizzo
# Funzione per raccogliere tutti i percorsi delle immagini in una cartella, inclusi quelli nidificati
def collect_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif'):
                folder_name = os.path.basename(root)
                image_paths.append((file, os.path.join(root, file), folder_name))
    return image_paths

def maskMaker(input_directory, output_directory):
    # Raccogli tutti i percorsi delle immagini
    image_paths = collect_image_paths(input_directory)

    # Seleziona casualmente 15 immagini dall'array
    selected_images = random.sample(image_paths, min(250, len(image_paths)))

    # Applica la funzione auto_local_threshold_median a ciascuna immagine selezionata
    for image_path in image_paths:
        image = auto_local_threshold_median(image_path[1])
        new_name = image_path[0][:-4] + "-BG" + image_path[0][-4:]

        # Controlla e crea la directory di salvataggio se non esiste
        output_dir = f"{output_directory}/{image_path[2]}"
        os.makedirs(output_dir, exist_ok=True)

        # Salva l'immagine convertita in uint8
        cv2.imwrite(os.path.join(output_dir, new_name), (image * 255).astype(np.uint8))

        #print(f"{image_path[2]}: {new_name} salvato con successo!")
    
    return os.path.relpath(output_directory, start=os.getcwd())
    

    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-v":
        plt.figure(figsize=(10, 10))

        # Raccolta delle immagini
        image_paths = collect_image_paths("MaskedImages/")
        folder_images = {}

        # Organizza le immagini per cartella
        for file, path, folder_name in image_paths:
            if folder_name not in folder_images:
                folder_images[folder_name] = []
            folder_images[folder_name].append((file, path))

        n = 0
        total_folders = len(folder_images)
        rows = total_folders if total_folders <= 4 else 4  # Max 4 righe
        cols = 3  # 3 colonne fisse

        # Seleziona casualmente 3 immagini per cartella
        for folder, images in folder_images.items():
            selected_images = random.sample(images, min(3, len(images)))  # Scegli 3 immagini casuali
            for img in selected_images:
                n += 1
                image = cv2.imread(img[1], cv2.IMREAD_GRAYSCALE)
                plt.subplot(rows, cols, n)
                plt.imshow(image, cmap='gray')
                plt.title(folder)
                plt.axis('off')

        plt.tight_layout()  # Migliora la disposizione dei subplot
        plt.show()
        
    else: 
        # Percorso della cartella contenente le immagini
        input_directory = "/home/francesco/Scaricati/Dataset/Images/test"
        output_directory = "./MaskedImages"

        print(maskMaker(input_directory, output_directory))