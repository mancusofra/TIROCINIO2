import numpy as np
import pandas as pd

from yeast_vacuole_pipeline.fuzzy_km.accuracy_calculator import (
    accuracy_calculator,
    hungarian_accuracy,
)
from yeast_vacuole_pipeline.fuzzy_km.unsupervised_assisted_filtering import (
    get_valid_df,
    get_valid_elements,
)


def test_accuracy_perfect_clustering():
    y_true = ["a", "a", "b", "b", "c", "c"]
    y_pred = np.array([2, 2, 0, 0, 1, 1])
    assert accuracy_calculator(y_true, y_pred, 3) == 1.0
    assert hungarian_accuracy(y_true, y_pred, 3) == 1.0


def test_accuracy_one_error():
    y_true = ["a", "a", "a", "b", "b", "b"]
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    assert accuracy_calculator(y_true, y_pred, 2) == 5 / 6


def test_filter_drops_only_confident_disagreements():
    y_true = np.array(["a", "a", "a", "b", "b", "b"])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    u = np.array(
        [
            [0.9, 0.9, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9, 0.9, 0.9, 0.9],
        ]
    )
    # Sample 5 sits confidently in cluster 1 (majority "b") but is labeled "a".
    y_true[5] = "a"
    keep = get_valid_elements(u, y_true, y_pred, n_clusters=2, threshold=0.8)
    assert keep.tolist() == [True, True, True, True, True, False]


def test_filter_keeps_unconfident_disagreements():
    y_true = np.array(["a", "a", "b", "b", "a"])
    y_pred = np.array([0, 0, 1, 1, 1])
    u = np.array([[0.9, 0.9, 0.1, 0.1, 0.45], [0.1, 0.1, 0.9, 0.9, 0.55]])
    keep = get_valid_elements(u, y_true, y_pred, n_clusters=2, threshold=0.8)
    assert keep.all()


def test_get_valid_df_resets_index():
    df = pd.DataFrame({"x": [1, 2, 3]})
    out = get_valid_df(np.array([True, False, True]), df)
    assert out["x"].tolist() == [1, 3]
    assert out.index.tolist() == [0, 1]
