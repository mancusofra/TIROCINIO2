import cv2
import numpy as np
import matplotlib.pyplot as plt

# Percorso dell'immagine (assicurati che il file esista)
image_path = "/home/francesco/Scaricati/Dataset/Images/test/multiple/A-01_cell_fld-3_000508.tif"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verifica se l'immagine è stata caricata correttamente
if image is None:
    print("Errore: l'immagine non è stata caricata. Controlla il percorso del file.")
else:
    # Applica la segmentazione con thresholding
    _, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Mostra l'immagine usando Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(thresholded, cmap='gray')  # Usa 'gray' per immagini in scala di grigi
    plt.title("Segmented Image")
    plt.axis("off")  # Nasconde gli assi
    plt.show()
