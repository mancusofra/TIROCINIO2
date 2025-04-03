import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def apply_and_plot_kmeans(X_scaled, n_clusters=3):
    """Apply K-Means clustering and plot the 3D PCA projection."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # PCA for 3D visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=cluster_labels, cmap='viridis', s=40)
    plt.legend(*scatter.legend_elements(), title="Cluster")

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    plt.title(f'K-Means Clustering with k={n_clusters} (3D PCA)')
    plt.show(block=False)

    return cluster_labels

def elbow_method(X_scaled, max_k=10):
    """Use the Elbow Method to find the optimal number of clusters."""
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.title('Elbow Method for Determining Optimal Number of Clusters')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    plt.show()

def silhouette_analysis(X_scaled, max_k=10):
    """Use the Silhouette Score to evaluate the best number of clusters."""
    scores = []
    best_score = -1
    best_k = 2
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, max_k + 1), scores, marker='o')
    plt.title('Silhouette Score Analysis')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()

    return best_k

if __name__ == "__main__":
    pass
