import cv2
import numpy as np
import matplotlib.pyplot as plt

def segment_vacuoles(image_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    
    # Controlla se l'immagine è stata caricata
    if image is None:
        print("Errore: Immagine non trovata. Controlla il percorso!")
        return

    # Se l'immagine è grayscale, convertila in BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Per la visualizzazione corretta con Matplotlib

    # Stampa la dimensione dell'immagine per debug
    print(f"Dimensione immagine: {image.shape}")

    # Definisci un range BGR adeguato (modifica con i tuoi valori)
    lower_vacuole = np.array([65, 0, 0])  # Es. blu scuro
    upper_vacuole = np.array([255, 255, 255])  # Es. blu chiaro

    # Crea la maschera basata sul colore BGR
    mask = cv2.inRange(image, lower_vacuole, upper_vacuole)

    # Pulizia della maschera: Rimozione del rumore con operazioni morfologiche
    kernel = np.ones((3, 3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Trova TUTTI i contorni (sia esterni che interni)
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Crea una copia dell'immagine originale per disegnare i contorni
    image_with_contours = image.copy()

    # Disegna tutti i contorni in rosso (BGR: (0, 0, 255)) con spessore 2
    cv2.drawContours(image_with_contours, contours, -1, (0, 0, 255), 1)

    # Converti in RGB per Matplotlib
    image_with_contours_rgb = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)

    # Visualizzazione dell'immagine originale con contorni sovrapposti
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_rgb)
    ax[0].set_title("Immagine Originale")
    ax[0].axis("off")

    ax[1].imshow(image_with_contours_rgb)
    ax[1].set_title("Tutti i Contorni Sovrapposti")
    ax[1].axis("off")

    plt.show()

# Percorso immagine
image_path = "/home/francesco/Scaricati/Dataset/Images/test/multiple/A-01_cell_fld-3_000547.tif"
segment_vacuoles(image_path)
