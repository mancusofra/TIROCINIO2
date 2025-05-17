import numpy as np
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler

def fuzzy_kmeans(full_df, n_clusters):
    X = full_df.select_dtypes(include='number')
    y = full_df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X = X_scaled.T

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X, n_clusters, m=1.53, error=0.005, maxiter=1000, init=None
    )
    return u, y
