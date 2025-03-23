import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Carica tutti i CSV da sottocartelle (ognuna rappresenta una classe)
data_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Features"  # <-- Sostituisci con il percorso della tua cartella principale
all_data = []

for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    if os.path.isdir(class_path):
        for filename in os.listdir(class_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(class_path, filename)
                # Leggi il CSV come vettore riga
                df = pd.read_csv(file_path, header=None)
                feature_vector = df.values.flatten()  # Ottieni un array 1D
                sample_df = pd.DataFrame([feature_vector])  # Metti in DataFrame come riga
                sample_df["object_name"] = os.path.splitext(filename)[0]
                sample_df["class"] = class_name
                all_data.append(sample_df)


      
# Combina tutti i dati
full_df = pd.concat(all_data, ignore_index=True)

# 2. Seleziona solo le feature numeriche
X = full_df.select_dtypes(include='number')

# 3. Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if "--check" in sys.argv:
    print("Ci sono NaN?", pd.DataFrame(X_scaled).isna().any().any())
    print("Quante colonne con NaN:", pd.DataFrame(X_scaled).isna().sum().sum())
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(class_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    if df.isna().any().any():
                        print(f"❗ File con NaN: {file_path}")

# 4. PCA con 3 componenti
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# 5. Visualizzazione 3D con colori per classe
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Colori per classe
classes = full_df["class"].unique()
colors = plt.cm.get_cmap("tab10", len(classes))

for i, class_name in enumerate(classes):
    indices = full_df["class"] == class_name
    ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2],
               label=class_name, c=[colors(i)], s=40)

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.title('Visualizzazione PCA (3D)')
plt.legend()
plt.show()
