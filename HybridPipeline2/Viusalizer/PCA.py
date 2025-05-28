import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def apply_and_plot_pca(X_scaled, labels, n_components=3):
    """
    Applies PCA to the scaled dataset and plots the first 2 or 3 principal components.

    Parameters:
    - X_scaled: Scaled feature matrix (e.g., result of StandardScaler)
    - labels: Labels for coloring points by class
    - n_components: Number of principal components to compute (2 or 3)

    Returns:
    - X_pca: Transformed dataset in the PCA space
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d' if n_components == 3 else None)

    classes = labels.unique()
    colors = plt.cm.get_cmap("tab10", len(classes))

    for i, class_name in enumerate(classes):
        indices = labels == class_name
        if n_components == 3:
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2],
                       label=class_name, c=[colors(i)], s=40)
        else:  # 2D case
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1],
                       label=class_name, c=[colors(i)], s=40)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    if n_components == 3:
        ax.set_zlabel('Principal Component 3')
        plt.title('PCA Visualization (3D)')
    else:
        plt.title('PCA Visualization (2D)')

    plt.legend()
    plt.show(block=False)

    return X_pca
