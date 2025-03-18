from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_top_left_pixel_color(image_path):
    """Legge il valore RGB del primo pixel in alto a sinistra."""
    image = Image.open(image_path).convert("RGB")
    pixels = image.load()
    return pixels[0, 0]  # Restituisce il valore RGB del primo pixel

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



if __name__ == "__main__":
    # Esempio di utilizzo
    image_path = "/home/francesco/Scaricati/Dataset/Images/test/multiple/A-01_cell_fld-3_000508.tif"
    print("Colore del primo pixel:", get_top_left_pixel_color(image_path))
    result_image = remove_background(image_path)

    # Mostra l'immagine risultante con OpenCV
    cv2.imshow("Risultato", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
