import numpy as np


def shuffle_data(df, p=0.2):
    """
    Returns a copy of the DataFrame with a percentage p of rows where the
    'class' column value has been changed to a different, random class.

    Args:
        df (pd.DataFrame): Original DataFrame.
        p (float): Percentage of rows to modify (between 0 and 1).

    Returns:
        pd.DataFrame: New DataFrame with the injected class mismatches.
    """
    # Copy the DataFrame so the original isn't modified
    df_copy = df.copy()

    # Compute how many rows to modify
    num_da_modificare = int(len(df_copy) * p)

    # Randomly select rows to modify
    indici_modificare = df_copy.sample(n=num_da_modificare, random_state=42).index

    # Get all available classes
    classi_possibili = df_copy["class"].unique()

    # Change the selected rows' class to a different random class
    for idx in indici_modificare:
        classe_originale = df_copy.at[idx, "class"]
        nuove_classi = [c for c in classi_possibili if c != classe_originale]
        nuova_classe = np.random.choice(nuove_classi)
        df_copy.at[idx, "class"] = nuova_classe

    return df_copy
