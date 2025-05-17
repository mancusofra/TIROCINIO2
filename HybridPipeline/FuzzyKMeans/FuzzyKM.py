import os, sys
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import mode

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

def accuracy_calculator(y_true, y_pred, n_clusters):
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_true)
    labels = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            print(f"Cluster {i} is empty")
            continue
        labels[mask] = mode(y_true[mask], keepdims=True).mode[0]

    return accuracy_score(y_true, labels)

def run_fuzzy_kmeans(full_df, n_clusters):
    X = full_df.select_dtypes(include='number')
    y = full_df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X = X_scaled.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X, n_clusters, m=1.53, error=0.005, maxiter=1000, init=None
    )
    return u, y

def get_valid_elements(u, y_true, y_pred, n_clusters, threshold):

    encoder = LabelEncoder()
    y_true_encoded = encoder.fit_transform(y_true)

    # Mappa i cluster alla classe più frequente
    cluster_to_class = {}
    for i in range(n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            continue
        cluster_to_class[i] = mode(y_true_encoded[mask], keepdims=True).mode[0]

    # Costruisci y_pred_class_label (basata sui cluster assegnati)
    y_pred_class = np.array([cluster_to_class[cluster] for cluster in y_pred])

    # Certezza = max(u) per ogni sample
    certainties = np.max(u, axis=0)
    mask_confident = certainties < threshold

    # Maschere
    correct_mask = (y_pred_class == y_true_encoded)

    combined_mask = correct_mask | mask_confident

    return correct_mask


if __name__ == "__main__":
    data_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_annotated"
    full_df = load_data(data_dir)
    n_clusters = 4
    u, y_true = run_fuzzy_kmeans(full_df, n_clusters)
    
    y_pred = np.argmax(u, axis=0)
    accuracy = accuracy_calculator(y_true, y_pred, n_clusters)
    print(f"Accuracy: {accuracy}")