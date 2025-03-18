from PIL import Image
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def remove_background(image_path):
    """Rimuove lo sfondo dell'immagine sostituendo i pixel collegati con il nero e mantenendo gli altri."""
    image = Image.open(image_path).convert("RGBA")
    pixels = np.array(image)
    h, w, _ = pixels.shape
    
    target_color = pixels[0, 0, :3]  # Colore del primo pixel (RGB)
    
    mask = np.zeros((h, w), dtype=bool)
    to_visit = [(0, 0)]
    
    while to_visit:
        x, y = to_visit.pop()
        if not mask[x, y] and np.array_equal(pixels[x, y, :3], target_color):
            mask[x, y] = True
            if x > 0: to_visit.append((x-1, y))
            if x < h-1: to_visit.append((x+1, y))
            if y > 0: to_visit.append((x, y-1))
            if y < w-1: to_visit.append((x, y+1))
    
    # Crea l'output: i pixel di sfondo diventano neri, gli altri rimangono invariati
    new_pixels = pixels.copy()
    new_pixels[mask] = [0, 0, 0, 255]  # Nero con opacità piena
    
    cv2_image = cv2.cvtColor(new_pixels, cv2.COLOR_RGBA2BGRA)
    return cv2_image

def segment_image(image_path, saturation_thresh=50, value_thresh=50):
    # Caricare l'immagine
    
    #image = remove_background(image_path)
    image = cv2.imread(image_path)


    # Convertire l'immagine da BGR a RGB per la visualizzazione corretta
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convertire l'immagine nello spazio colore HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Estrarre le componenti HSV
    hue, saturation, value = cv2.split(image_hsv)

    # Creare una maschera basata sui valori di Saturation e Value
    mask = (saturation > saturation_thresh) & (value > value_thresh)
    mask = mask.astype(np.uint8) * 255  # Convertire in immagine binaria

    # Applicare operazioni morfologiche per riempire i buchi
    kernel = np.ones((5,5), np.uint8)  # Kernel per operazioni morfologiche
    #mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Chiusura per riempire buchi
    #mask = cv2.dilate(mask, kernel, iterations=1)  # Leggera dilatazione per uniformare

    # Applicare la maschera all'immagine originale
    segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    return image_rgb, mask, segmented_image

# Percorso della directory delle immagini
base_dir = "/home/francesco/Scaricati/Dataset/Images/"

# Scorrere tutte le cartelle in Images/
for root, dirs, files in os.walk(base_dir):
    # Prendere le prime 3 immagini della cartella
    image_files = [f for f in files if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))][:3]
    print(f"Lunghezza immagini: {len(image_files)}")
    for image_file in image_files:
        image_path = os.path.join(root, image_file)
        
        # Segmentare l'immagine
        image_rgb, mask, segmented_image = segment_image(image_path)
        
        # Mostrare i risultati
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        axs[0].imshow(image_rgb)
        axs[0].set_title("Immagine Originale")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Maschera Segmentata")
        axs[1].axis("off")

        axs[2].imshow(segmented_image)
        axs[2].set_title("Immagine Segmentata")
        axs[2].axis("off")

        plt.show()
