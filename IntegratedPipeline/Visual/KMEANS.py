import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score


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

def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def check_nans(data_dir, X_scaled):
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

def apply_and_plot_pca(X_scaled, labels, n_components=3):
    plt.close('all')
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    classes = labels.unique()
    colors = plt.cm.get_cmap("tab10", len(classes))

    for i, class_name in enumerate(classes):
        indices = labels == class_name
        if n_components == 3:
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2],
                       label=class_name, c=[colors(i)], s=40)
        elif n_components == 2:
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1],
                       label=class_name, c=[colors(i)], s=40)

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    if n_components == 3:
        ax.set_zlabel('PC 3')
        plt.title('Visualizzazione PCA (3D)')
        plt.legend()
        plt.show()
    if n_components == 2:
        plt.title('Visualizzazione PCA (2D)')
        plt.legend()
        plt.show()

    return X_pca

def apply_and_plot_kmeans(X_scaled, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA per visualizzazione 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='viridis', s=40)
    plt.legend(*scatter.legend_elements(), title="Cluster")

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.title(f'Clustering K-Means con k={n_clusters} (PCA 3D)')
    plt.show()

    return cluster_labels

def elbow_method(X_scaled, max_k=10):
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k+1), sse, marker='o')
    plt.title('Metodo del gomito per determinare il numero ottimale di cluster')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Somma degli errori quadratici (SSE)')
    plt.grid(True)
    plt.show()

def silhouette_analysis(X_scaled, max_k=10):
    scores = []
    best_score = -1
    best_k = 2
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k+1), scores, marker='o')
    plt.title('Analisi del coefficiente di Silhouette')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    return best_k
if __name__ == "__main__":
    data_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Features"
    #data_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_clustershcf/"
 
    full_df = load_data(data_dir)
    X = full_df.select_dtypes(include='number')
    X_scaled = standardize_data(X)

    if "--check" in sys.argv:
        check_nans(data_dir, X_scaled)

    #apply_and_plot_kmeans(X_scaled, n_clusters=4)
    #elbow_method(X_scaled, max_k=10)
    print(silhouette_analysis(X_scaled, max_k=10))

