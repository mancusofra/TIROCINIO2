import cv2
import numpy as np

def nothing(x):
    pass  # Funzione vuota per le trackbar

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

    # Crea una finestra per i controlli della trackbar
    cv2.namedWindow("Segmentazione Vacuoli", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmentazione Vacuoli", 400, 150)  # Dimensione più grande per i controlli

    # Crea le finestre delle immagini con dimensioni maggiori
    cv2.namedWindow("Maschera", cv2.WINDOW_NORMAL)

    cv2.namedWindow("Contorni Sovrapposti", cv2.WINDOW_NORMAL)

    # Crea trackbar per il valore inferiore del range BGR
    cv2.createTrackbar("Lower B", "Segmentazione Vacuoli", 0, 255, nothing)
    cv2.createTrackbar("Lower G", "Segmentazione Vacuoli", 0, 255, nothing)
    cv2.createTrackbar("Lower R", "Segmentazione Vacuoli", 0, 255, nothing)

    while True:
        # Legge i valori della trackbar
        lb = cv2.getTrackbarPos("Lower B", "Segmentazione Vacuoli")
        lg = cv2.getTrackbarPos("Lower G", "Segmentazione Vacuoli")
        lr = cv2.getTrackbarPos("Lower R", "Segmentazione Vacuoli")

        # Definisci i range BGR basati sulle trackbar
        lower_vacuole = np.array([lb, lg, lr])
        upper_vacuole = np.array([255, 255, 255])  # Fisso il valore massimo

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

        # Disegna tutti i contorni in verde (BGR: (0, 255, 0)) con spessore 1
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)

        # Mostra la maschera e l'immagine con contorni in finestre separate (ingrandite)
        cv2.imshow("Maschera", mask_cleaned)
        cv2.imshow("Contorni Sovrapposti", image_with_contours)

        # Esci premendo 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Chiudi tutte le finestre
    cv2.destroyAllWindows()

# Percorso immagine
image_path = "/home/francesco/Scaricati/Dataset/Images/test/multiple/A-01_cell_fld-3_000508.tif"
segment_vacuoles(image_path)
