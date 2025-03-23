import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carica immagine
img = cv2.imread('/home/francesco/Scaricati/Dataset/Images/test/condensed/A-01_cell_fld-3_000569.tif')  # Sostituisci con il tuo path
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Per visualizzare bene con matplotlib

# 1. Converti in grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Rimozione del rumore
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. (Opzionale) Equalizzazione dell'istogramma
equalized = cv2.equalizeHist(blurred)

# 4. Thresholding (Otsu)
_, mask = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Operazioni morfologiche
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

# Visualizzazione
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Immagine Originale')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Grayscale Equalized')
plt.imshow(equalized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Maschera Binaria')
plt.imshow(cleaned, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
