import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder

def get_valid_elements(u, y_true, y_pred, n_clusters, threshold):

    encoder = LabelEncoder()
    y_true_encoded = encoder.fit_transform(y_true)

    # Mappa i cluster alla classe più frequente
    cluster_to_class = {}
    for i in range(n_clusters):
        mask = (y_pred == i)
        if np.sum(mask) == 0:
            continue
        cluster_to_class[i] = mode(y_true_encoded[mask], keepdims=True).mode[0]

    # Costruisci y_pred_class_label (basata sui cluster assegnati
    y_pred_class = np.array([cluster_to_class[cluster] for cluster in y_pred])

    # Certezza = max(u) per ogni sample
    certainties = np.max(u, axis=0)

    # Questa maschera restituisce true per ogni elemento che il fuzzy kemeans riesce a classificare con una certezza maggiore del threshold
    mask_confident = certainties > threshold

    # Questa maschera resituisce true per ogni elemento in cui la classificcazione unsupervised e quella da perte del esperto non sono daccordo
    not_agree_mask = (y_pred_class != y_true_encoded)

    # AND bit a bit delle due maschere --> otteniamo una maschera in cui i true indicano elementi in cui le due classificazioni sono in DISACCORDO e in cui 
    # la unsupervised classification risulta sicura della sua classificazione (sicurezza basata su threshold)
    combined_mask = not_agree_mask & mask_confident
    valid_mask = ~combined_mask

    return valid_mask

def get_valid_df(valid_mask, df):

    # Restituisce un DataFrame con le righe corrispondenti alla maschera booleana.
    
    # Args:
    #     valid_mask (array-like): Maschera booleana (True per tenere la riga).
    #     df (pd.DataFrame): DataFrame originale.
    
    # Returns:
    #     pd.DataFrame: DataFrame filtrato.

    return df[valid_mask].reset_index(drop=True)
