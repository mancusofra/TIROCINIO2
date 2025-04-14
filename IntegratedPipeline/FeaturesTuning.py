import os, cv2
from time import sleep
from tqdm import tqdm

# Iteration tools for generating combinations
from itertools import combinations, product


# Features Extraction Modules
from FeaturesExtraction.FeaturesExtraction import file_extraction, features_extraction

from KMeans.PrecisionCalculatore import run_kmeans_pipeline



def feature_tuning():

    files_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet"
    mask_files, image_files = file_extraction(files_dir, Verbose = False)

    geometric_features = ["total_area", "total_perimeter", "eccentricity", "circularity", "solidity", "extent", "mean_radius"]
    geometric_all_combinations = [list(combo) for r in range(1, len(geometric_features)+1) for combo in combinations(geometric_features, r)]

    gray_hist_features = ["mean", "std", "kurtosis", "skewness", "entropy"]
    gray_hist_all_combinations = [list(combo) for r in range(1, len(gray_hist_features)+1) for combo in combinations(gray_hist_features, r)]
    
    lpb_all_combinations = [
        {'P': 8, 'R': 1}, {'P': 16, 'R': 1}, {'P': 24, 'R': 1}, {'P': 54, 'R': 1},
        {'P': 8, 'R': 2}, {'P': 16, 'R': 2}, {'P': 24, 'R': 2}, {'P': 54, 'R': 2},
        {'P': 8, 'R': 3}, {'P': 16, 'R': 3}, {'P': 24, 'R': 3}, {'P': 54, 'R': 3},
        {'P': 8, 'R': 4}, {'P': 16, 'R': 4}, {'P': 24, 'R': 4}, {'P': 54, 'R': 4},
        {'P': 8, 'R': 5}, {'P': 16, 'R': 5}, {'P': 24, 'R': 5}, {'P': 54, 'R': 5},

        ]
    
    # Definiamo i range
    radius_values = list(range(20, 101, 10))   # [20, 30, ..., 100]
    degree_values = list(range(6, 19, 2))      # [6, 8, ..., 18]


    zernike_all_combinations = [{"radius": r, "degree": d} for r, d in product(radius_values, degree_values)]

    params = {
    "zernike": zernike_all_combinations,
    }   

    # Ottieni le chiavi e le liste dei valori
    keys = params.keys()
    values = params.values()

    # Combina tutte le possibili combinazioni
    diz = [dict(zip(keys, combo)) for combo in product(*values)]

    # Stampa le combinazioni
    print(len(diz))
    max = 0
    for combo in tqdm(diz, desc="Processing combinations"):
        features_dir = features_extraction(image_files, mask_files, params=combo)
        val = run_kmeans_pipeline(features_dir, plot_confusion=False, var_threshold=0.01)
        if max < val:
            max = val
            print(f"Max: {max}")
            print(combo["zernike"])

if __name__ == "__main__":
    feature_tuning()