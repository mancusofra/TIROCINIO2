import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def process_image(input_path, threshold_value):
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
    _, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Migliora la separazione usando il filtro Canny per evidenziare i bordi
    edges = cv2.Canny(gray, 50, 150)
    mask = cv2.bitwise_or(mask, edges)
    
    # Rimuove piccole macchie con operazioni morfologiche
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Chiude eventuali buchi
    mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)  # Aumenta leggermente la maschera
    
    # Trova i contorni e conserva solo il più grande
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = mask_cleaned.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Assicura che il contorno sia chiuso utilizzando il convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Disegna il contorno convesso sulla maschera senza rimuovere i pixel esistenti
        cv2.drawContours(mask_final, [hull], -1, (255), thickness=cv2.FILLED)
    
    return img, mask_final

def on_trackbar(val):
    global image_path
    threshold_value = val
    img, mask_final = process_image(image_path, threshold_value)
    
    combined = np.hstack((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask_final))
    cv2.imshow("Threshold Adjustment", combined)

if __name__ == "__main__":
    image_folder = "/home/francesco/Scaricati/Dataset/Images/test/positive"
    for filename in os.listdir(image_folder):
        if filename.endswith(".tif"):
            image_path = os.path.join(image_folder, filename)
            img, mask_final = process_image(image_path, 54)
            
            cv2.namedWindow("Threshold Adjustment")
            cv2.createTrackbar("Threshold", "Threshold Adjustment", 54, 255, on_trackbar)
            
            combined = np.hstack((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask_final))
            cv2.imshow("Threshold Adjustment", combined)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
