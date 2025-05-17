import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode

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
