import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


def get_valid_elements(u, y_true, y_pred, n_clusters, threshold):
    """
    Flags samples to keep, dropping those the fuzzy clustering confidently
    disagrees with the expert (ground-truth) label on.

    Each cluster is mapped to its most frequent true class (majority vote).
    A sample is flagged invalid (dropped) only when both conditions hold:
    the unsupervised cluster-based label differs from its true label, AND
    the clustering's confidence for that sample (max membership value)
    exceeds `threshold`. This is meant to catch likely-mislabeled samples
    without discarding cases where the clustering itself is just uncertain.

    Args:
        u (np.ndarray): Fuzzy membership matrix, shape (n_clusters, n_samples).
        y_true (array-like): Ground-truth class labels (any type, encoded internally).
        y_pred (array-like): Cluster index assigned to each sample (e.g. u.argmax(axis=0)).
        n_clusters (int): Number of clusters.
        threshold (float): Minimum membership confidence for a disagreement
            to count as a confident misclassification.

    Returns:
        np.ndarray: Boolean mask, True for samples to keep.
    """
    encoder = LabelEncoder()
    y_true_encoded = encoder.fit_transform(y_true)

    # Map each cluster to its most frequent true class
    cluster_to_class = {}
    for i in range(n_clusters):
        mask = y_pred == i
        if np.sum(mask) == 0:
            continue
        cluster_to_class[i] = mode(y_true_encoded[mask], keepdims=True).mode[0]

    # Per-sample predicted class, based on its assigned cluster's mapping
    y_pred_class = np.array([cluster_to_class[cluster] for cluster in y_pred])

    # Confidence = max membership value for each sample
    certainties = np.max(u, axis=0)

    # True for samples the fuzzy clustering classifies with confidence above threshold
    mask_confident = certainties > threshold

    # True for samples where the unsupervised and expert classifications disagree
    not_agree_mask = y_pred_class != y_true_encoded

    # Bitwise AND of the two masks: True where the classifications DISAGREE
    # and the unsupervised classification is confident about it (per threshold)
    combined_mask = not_agree_mask & mask_confident
    valid_mask = ~combined_mask

    return valid_mask


def get_valid_df(valid_mask, df):
    """
    Returns a DataFrame with only the rows matching the boolean mask.

    Args:
        valid_mask (array-like): Boolean mask (True to keep the row).
        df (pd.DataFrame): Original DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame, with a reset index.
    """
    return df[valid_mask].reset_index(drop=True)
