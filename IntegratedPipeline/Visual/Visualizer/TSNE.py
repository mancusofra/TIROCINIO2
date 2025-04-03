import os, sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def apply_and_plot_tsne(X_scaled, labels, n_components=2, perplexity=5, learning_rate=200, n_iter=1000):
    plt.close('all')
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)

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
            ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1], X_tsne[indices, 2],
                       label=class_name, c=[colors(i)], s=40)
        elif n_components == 2:
            ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                       label=class_name, c=[colors(i)], s=40)
        else:
            raise ValueError("Il numero di componenti deve essere 2 o 3 per la visualizzazione.")

    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    if n_components == 3:
        ax.set_zlabel('Dim 3')
        plt.title('Visualizzazione t-SNE (3D)')
    else:
        plt.title('Visualizzazione t-SNE (2D)')

    plt.legend()
    plt.show(block=False)

    return X_tsne

if __name__ == "__main__":
    #data_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_clustershcf/"
    data_dir = "Features/"
    full_df = load_data(data_dir)
    X = full_df.select_dtypes(include='number')
    X_scaled = standardize_data(X)

    if "--check" in sys.argv:
        check_nans(data_dir, X_scaled)
    apply_and_plot_tsne(X_scaled, full_df["class"], n_components=3)

