import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import mode
from sklearn.feature_selection import VarianceThreshold

def load_data(data_dir):
    all_data = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(class_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature_vector = df.values.flatten()
                    sample_df = pd.DataFrame([feature_vector])
                    sample_df["object_name"] = os.path.splitext(filename)[0]
                    sample_df["class"] = class_name
                    all_data.append(sample_df)
    return pd.concat(all_data, ignore_index=True)

def run_kmeans_pipeline(data_dir, plot_confusion=True, var_threshold=0.01):
    # 1. Carica i dati
    full_df = load_data(data_dir)
    X = full_df.select_dtypes(include='number')
    y = full_df["class"]

    # 2. Preprocessing: scaling e rimozione feature a bassa varianza
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = VarianceThreshold(threshold=var_threshold)
    X_selected = selector.fit_transform(X_scaled)

    # 3. Encoding delle etichette
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y)

    # 4. KMeans
    n_clusters = len(np.unique(y_true))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
    y_pred = kmeans.fit_predict(X_selected)

    # 5. Mappatura cluster → label vera tramite majority voting
    labels = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            print(f"Cluster {i} is empty")
            continue
        labels[mask] = mode(y_true[mask], keepdims=True).mode[0]


    # 6. Accuracy
    accuracy = accuracy_score(y_true, labels)

    # 7. Matrice di confusione (opzionale)
    if plot_confusion:
        cm = confusion_matrix(y_true, labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    return accuracy

if __name__ == "__main__":
    data_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet"
    print(run_kmeans_pipeline(data_dir, plot_confusion=True, var_threshold=0.01))