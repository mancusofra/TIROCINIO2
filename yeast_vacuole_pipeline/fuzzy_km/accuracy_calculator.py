import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def accuracy_calculator(y_true, y_pred, n_clusters):
    """
    Scores a clustering against ground-truth labels via majority-vote mapping.

    Each cluster is assigned the true label most frequent among its members
    (majority vote), then accuracy is computed by comparing every sample's
    true label against its cluster's assigned label.

    Args:
        y_true (array-like): Ground-truth class labels (any type, encoded internally).
        y_pred (array-like): Cluster index assigned to each sample.
        n_clusters (int): Number of clusters in y_pred.

    Returns:
        float: Accuracy of the majority-vote-mapped predictions.
    """
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_true)
    labels = np.zeros_like(y_pred)
    for i in range(n_clusters):
        mask = y_pred == i
        if np.sum(mask) == 0:
            print(f"Cluster {i} is empty")
            continue
        labels[mask] = mode(y_true[mask], keepdims=True).mode[0]

    return accuracy_score(y_true, labels)


def hungarian_accuracy(y_true, y_pred, n_clusters):
    """
    Scores a clustering against ground-truth labels via optimal cluster-to-class assignment.

    Unlike accuracy_calculator's per-cluster majority vote, this finds the
    global one-to-one cluster-to-class assignment that maximizes total
    agreement, using the Hungarian algorithm on the confusion matrix.

    Args:
        y_true (array-like): Ground-truth class labels (any type, encoded internally).
        y_pred (array-like): Cluster index assigned to each sample.
        n_clusters (int): Number of clusters in y_pred (unused directly, kept
            for interface consistency with accuracy_calculator).

    Returns:
        float: Accuracy of the optimally-mapped predictions.
    """
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(y_true)

    # Build the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Hungarian method for the optimal assignment (maximize agreement, hence -cm)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = dict(zip(col_ind, row_ind))

    # Remap y_pred according to the optimal cluster-to-class mapping
    y_pred_mapped = np.array([mapping.get(label, -1) for label in y_pred])

    return accuracy_score(y_true, y_pred_mapped)
