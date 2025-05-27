import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from scipy.optimize import linear_sum_assignment


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

def hungarian_accuracy(y_true, y_pred, n_clusters):
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_true)

    # Costruisce matrice di confusione
    cm = confusion_matrix(y_true, y_pred)

    # Hungarian method per assegnamento ottimale
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))

    # Riassegna y_pred in base alla mappatura ottimale
    y_pred_mapped = np.array([mapping.get(label, -1) for label in y_pred])

    return accuracy_score(y_true, y_pred_mapped)