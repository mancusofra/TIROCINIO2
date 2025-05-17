import numpy as np
import pandas as pd

def shuffle_data(df, p=0.2):
    # Restituisce una copia del DataFrame con una percentuale p di righe
    # dove il valore della colonna 'class' è modificato.
    
    # Args:
    #     df (pd.DataFrame): DataFrame originale.
    #     p (float): Percentuale di righe da modificare (tra 0 e 1).
        
    # Returns:
    #     pd.DataFrame: Nuovo DataFrame modificato.
    
    # Crea una copia del DataFrame per non modificare l'originale
    df_copy = df.copy()

    # Calcola quante righe modificare
    num_da_modificare = int(len(df_copy) * p)

    # Seleziona righe a caso da modificare
    indici_modificare = df_copy.sample(n=num_da_modificare, random_state=42).index

    # Ottieni tutte le classi disponibili
    classi_possibili = df_copy['class'].unique()

    # Cambia le classi selezionate in un'altra classe a caso diversa dall'originale
    for idx in indici_modificare:
        classe_originale = df_copy.at[idx, 'class']
        nuove_classi = [c for c in classi_possibili if c != classe_originale]
        nuova_classe = np.random.choice(nuove_classi)
        df_copy.at[idx, 'class'] = nuova_classe

    return df_copy

