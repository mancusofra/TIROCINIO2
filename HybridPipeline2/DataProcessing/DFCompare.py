import pandas as pd

def find_class_mismatches(df1: pd.DataFrame, df2: pd.DataFrame) -> list:

    mismatches = []
    for index, row in df1.iterrows():
        if index in df2.index and row['class'] != df2.loc[index, 'class']:
            mismatches.append(index)
    return mismatches

def get_differences(df_big: pd.DataFrame, df_small: pd.DataFrame) -> list:
    
    # Restituisce gli indici delle righe che sono in df_big ma non in df_small.
    
    # Args:
    #     df_big (pd.DataFrame): DataFrame principale.
    #     df_small (pd.DataFrame): Sottoinsieme di df_big.
    
    # Returns:
    #     list: Indici delle righe presenti solo in df_big.
    
    df_diff = pd.concat([df_big, df_small]).drop_duplicates(keep=False)
    return df_diff.index.tolist()