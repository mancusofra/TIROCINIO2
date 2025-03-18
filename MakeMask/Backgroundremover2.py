import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def process_image(input_path):
    # Carica l'immagine a colori
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    # Converti l'immagine nello spazio colore Lab per migliorare la separazione tra sfondo e cellula
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    # Equalizzazione dell'istogramma per migliorare il contrasto
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2BGR)
    
    # Converti l'immagine in scala di grigi
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    
    # Applica la soglia per creare la maschera binaria
    _, mask = cv2.threshold(gray, 54, 255, cv2.THRESH_BINARY_INV)
    
    # Migliora la separazione usando il filtro Canny per evidenziare i bordi
    #edges = cv2.Canny(gray, 50, 150)
    #mask = cv2.bitwise_or(mask, edges)
    
    # Rimuove piccole macchie con operazioni morfologiche
    #kernel = np.ones((5, 5), np.uint8)
    #mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Chiude eventuali buchi
    #mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)  # Aumenta leggermente la maschera
    
    # Trova i contorni e conserva solo il più grande
    #contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #mask_final = mask_cleaned.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = mask.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Assicura che il contorno sia chiuso utilizzando il convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Disegna il contorno convesso sulla maschera senza rimuovere i pixel esistenti
        cv2.drawContours(mask_final, [hull], -1, (255), thickness=cv2.FILLED)
    
    # Crea una copia dell'immagine originale a colori per disegnare i contorni
    img_contours = img.copy()
    if contours:
        print("Contorni trovati:", len(contours))
        cv2.drawContours(img_contours, [hull], -1, (0, 255, 0), thickness=1)
    
    # Mostra l'immagine originale con contorni e la maschera elaborata a confronto
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title("Immagine Originale con Contorni")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(mask_final, cmap='gray')
    plt.title("Maschera Elaborata")
    plt.axis('off')
    
    plt.show()




# Esempio di utilizzo
if __name__ == "__main__":
    image_folder = "/home/francesco/Scaricati/Dataset/Images/test/positive"
    for filename in os.listdir(image_folder):
        if filename.endswith(".tif"):
            image_path = os.path.join(image_folder, filename)
            process_image(image_path)
