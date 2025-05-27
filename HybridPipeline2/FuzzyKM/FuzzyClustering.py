import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from . import AccuracyCalculator

def fuzzy_kmeans(full_df, n_clusters, verbose=False):
    X = full_df.select_dtypes(include='number')
    y = full_df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled).T  # importante: trasposto

    X = X_scaled.T
    best_m = None
    best_acc = -1
    best_u = None

    for m in np.arange(1.2, 2.71, 0.1):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled.T, n_clusters, m=m, error=0.001, maxiter=1000, init=None
        )
        acc = AccuracyCalculator.accuracy_calculator(y, u.argmax(axis=0), n_clusters)
        if acc > best_acc:
            best_acc = acc
            best_m = m
            best_u = u
        if verbose:
            print(f"Tested m={m:.2f}, Accuracy={acc}")

    m = best_m
    u = best_u

    if verbose:
        print(f"Fuzziness parameter: {m}")
        print(f"Accuracy: {AccuracyCalculator.accuracy_calculator(y, u.argmax(axis=0), n_clusters)}")
        print(f"Hungarian Accuracy: {AccuracyCalculator.hungarian_accuracy(y, u.argmax(axis=0), n_clusters)}\n\n")


    return u, y
