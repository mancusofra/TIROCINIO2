import pandas as pd

def find_class_mismatches(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """
    Compares df1 and df2 (same index) and returns the indices of the rows
    where the 'class' column value differs between the two DataFrames.

    Args:
        df1 (pd.DataFrame): Reference DataFrame (e.g. original data).
        df2 (pd.DataFrame): DataFrame to compare against (e.g. shuffled data).

    Returns:
        list: Indices of the rows whose class differs between df1 and df2.
    """
    mismatches = []
    for index, row in df1.iterrows():
        if index in df2.index and row['class'] != df2.loc[index, 'class']:
            mismatches.append(index)
    return mismatches

def get_differences(df_big: pd.DataFrame, df_small: pd.DataFrame) -> list:
    """
    Returns the indices of the rows present in df_big but not in df_small.

    Args:
        df_big (pd.DataFrame): Main DataFrame.
        df_small (pd.DataFrame): Subset of df_big.

    Returns:
        list: Indices of the rows present only in df_big.
    """
    df_diff = pd.concat([df_big, df_small]).drop_duplicates(keep=False)
    return df_diff.index.tolist()