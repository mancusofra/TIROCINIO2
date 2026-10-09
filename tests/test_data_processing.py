import pandas as pd

from yeast_vacuole_pipeline.data_processing.df_compare import (
    find_class_mismatches,
    get_differences,
)
from yeast_vacuole_pipeline.data_processing.shuffle_data import shuffle_data


def make_df(n=100):
    classes = ["multiple", "condensed", "positive", "negative"]
    return pd.DataFrame({"f": range(n), "class": [classes[i % 4] for i in range(n)]})


def test_shuffle_changes_expected_fraction_of_labels():
    df = make_df(100)
    shuffled = shuffle_data(df, p=0.2)
    assert len(find_class_mismatches(df, shuffled)) == 20


def test_shuffle_does_not_modify_input():
    df = make_df(40)
    original = df.copy()
    shuffle_data(df, p=0.5)
    pd.testing.assert_frame_equal(df, original)


def test_shuffle_with_p_zero_is_identity():
    df = make_df(40)
    assert find_class_mismatches(df, shuffle_data(df, p=0.0)) == []


def test_get_differences_returns_removed_rows():
    df = make_df(10)
    assert sorted(get_differences(df, df.drop(index=[2, 5]))) == [2, 5]
