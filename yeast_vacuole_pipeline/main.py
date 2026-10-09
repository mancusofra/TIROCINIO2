"""
Interactive entry point for the yeast vacuole classification pipeline (run
as `python -m yeast_vacuole_pipeline.main`). The menu() options are:
    1. Train the U-Net segmentation model on Data/Mask (slow; overwrites weights).
    2. Load the trained U-Net model and plot training/test results.
    3. Extract hand-crafted features from the dataset (masks + grayscale images).
    4. Visualize extracted features with PCA.
    5. Train a Random Forest on full/shuffled/filtered data and compare accuracy,
       to measure the effect of fuzzy-clustering-assisted mislabel filtering.

plot_different_th, try_different_th and get_filtered_df are standalone
analysis helpers for tuning the filtering threshold — not wired into the menu.
"""

from .data_processing import load_data, shuffle_data, df_compare
from .fuzzy_km import fuzzy_clustering, accuracy_calculator, unsupervised_assisted_filtering
from .random_forest import fit_random_forest, model_accuracy
from .unet_segmentation import unet_torch
from .features_extraction import features_extraction
from .visualizer import pca_plot

import matplotlib.pyplot as plt
import os, pickle, platform
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Data/ lives next to this file; legacy/ is the repo-level folder one level up
# from yeast_vacuole_pipeline/. Computed from __file__ so this works on any
# machine/user, regardless of where the repo is checked out.
PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = (PIPELINE_DIR / "Data").as_posix()
LEGACY_DIR = (PIPELINE_DIR.parent / "legacy").as_posix()

def clear_terminal():
    """Clears the terminal (cross-platform: 'cls' on Windows, 'clear' otherwise)."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def confirm_long_operation():
    """
    Warns the user that the requested operation is time-expensive and will
    overwrite the existing trained model, then asks for confirmation.

    Returns:
        bool: True if the user confirmed (typed 'y'/'yes'), False otherwise.
    """
    clear_terminal()
    print("WARNING!!: You are about to perform an operation that may take a long time")
    print("and will **overwrite the existing model**.")
    response = input("Are you sure you want to continue? [y/N]: ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Operation aborted.")
        return False
    return True

def filter_accuracy(shuffled_elements, removed_elements, common_elements):
    """
    Scores how well a filtering pass targeted the actually-mislabeled rows.

    Args:
        shuffled_elements (Sized): The full set of rows with injected label
            mismatches (see shuffle_data.shuffle_data).
        removed_elements (Sized): The rows the filtering pass removed.
        common_elements (Sized): The rows that are both mislabeled and removed.

    Returns:
        tuple: (accuracy_correctelements, accuracy_incorrectelements) —
        fraction of the mislabeled rows that got removed, and fraction of
        the removed rows that were actually mislabeled.
    """
    accuracy_correctelements =  len(common_elements) / len(shuffled_elements)
    accuracy_incorrectelements = len(common_elements) / len(removed_elements) if len(removed_elements) > 0 else 0
    return accuracy_correctelements, accuracy_incorrectelements

def try_different_th(shuffled_df, lista_elementi_cambiati):
    """
    Sweeps the unsupervised-filtering confidence threshold from 0.1 to 0.9
    and, for each value, measures how well filtering at that threshold
    targets the rows in `lista_elementi_cambiati` (the known mislabeled
    rows) — see filter_accuracy.

    Args:
        shuffled_df (pd.DataFrame): Feature DataFrame with injected label
            mismatches (see shuffle_data.shuffle_data).
        lista_elementi_cambiati (list): Indices of rows whose label was
            actually changed (see df_compare.find_class_mismatches).

    Returns:
        list: (threshold, accuracy_correctelements, accuracy_incorrectelements)
        tuples, one per tested threshold.
    """
    lis = []
    u, y = fuzzy_clustering.fuzzy_kmeans(shuffled_df, n_clusters=4)
    y_pred = u.argmax(axis=0)
    y_true = y
    n_clusters = 4

    for th in range(1, 10):
        th = th/10
        valid_elements = unsupervised_assisted_filtering.get_valid_elements(u, y_true, y_pred, n_clusters, threshold=th)
        reduced_df = unsupervised_assisted_filtering.get_valid_df(valid_elements, shuffled_df)
        lista_elementi_tolti = df_compare.get_differences(shuffled_df, reduced_df)
        common_elements = set(lista_elementi_cambiati).intersection(set(lista_elementi_tolti))

        accuracy_correctelements, accuracy_incorrectelements = filter_accuracy(lista_elementi_cambiati, lista_elementi_tolti, common_elements)
        lis.append((th, accuracy_correctelements, accuracy_incorrectelements))
    return lis

def plot_different_th(shuffle_p = 0.2):
    """
    Standalone analysis helper (not wired into the menu): loads the features,
    injects `shuffle_p` mislabeled rows, then plots how many correctly- vs.
    incorrectly-flagged rows the unsupervised filtering removes across
    thresholds 0.1-0.9 (see try_different_th).

    Args:
        shuffle_p (float): Fraction of rows to mislabel before filtering.
    """
    Feature_dir_train = f"{DATA_DIR}/Features/"
    Feature_dir_test = f"{DATA_DIR}/Features_test/"
    full_df = load_data.load_data(Feature_dir_train)
    shuffled_df = shuffle_data.shuffle_data(full_df, p=shuffle_p)
    lista_elementi_cambiati = df_compare.find_class_mismatches(full_df, shuffled_df)

    results = try_different_th(shuffled_df, lista_elementi_cambiati)
    thresholds = [x[0] for x in results]
    accuracy_correct = [x[1] for x in results]
    accuracy_incorrect = [x[2] for x in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracy_correct, label="Number of correct elements removed", marker='o')
    plt.plot(thresholds, accuracy_incorrect, label="Number of incorrect elements removed", marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Number of elements")
    plt.title(f"Percentage of shuffled elements: {shuffle_p}")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
        
def get_filtered_df(shuffled_df, threshold):
    """
    Runs fuzzy c-means on shuffled_df and returns only the rows that pass
    unsupervised-assisted filtering at the given confidence threshold.

    Args:
        shuffled_df (pd.DataFrame): Feature DataFrame with a 'class' column.
        threshold (float): Minimum clustering confidence for a disagreement
            with the labeled class to count as a confident misclassification
            (see unsupervised_assisted_filtering.get_valid_elements).

    Returns:
        pd.DataFrame: shuffled_df filtered down to the rows kept as valid.
    """
    u, y = fuzzy_clustering.fuzzy_kmeans(shuffled_df, n_clusters=4)
    y_pred = u.argmax(axis=0)
    y_true = y
    n_clusters = 4

    valid_elements = unsupervised_assisted_filtering.get_valid_elements(u, y_true, y_pred, n_clusters, threshold=threshold)
    reduced_df = unsupervised_assisted_filtering.get_valid_df(valid_elements, shuffled_df)
    return reduced_df

def menu():
    """Interactive CLI entry point for the pipeline; see main.py's module docstring for what each option does."""
    while True:
        clear_terminal()
        print("1. Train U-Net segmentation model (slow, overwrites the existing weights)")
        print("2. Load trained U-Net model and plot training/test results")
        print("3. Extract features from the dataset (masks + grayscale images)")
        print("4. Visualize extracted features with PCA")
        print("5. Train Random Forest (full vs. shuffled vs. filtered data) and compare accuracy")
        print("0. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            if confirm_long_operation():
                data_dir = f"{DATA_DIR}/Mask/"
                history = unet_torch.train_model(data_dir)
                with open(f"{DATA_DIR}/Model/history.pkl", "wb") as f:
                    pickle.dump(history, f)

        elif choice == '2':
            unet_torch.load_and_plot()

        elif choice == '3':
            masked_dir = f"{DATA_DIR}/Mask"
            input_dir = f"{DATA_DIR}/Original_images/Train_annotated"
            input_dir_test = f"{DATA_DIR}/Original_images/test"
            output_dir = f"{DATA_DIR}/DataSet"
            model_path = f"{DATA_DIR}/Model/Weights.pt"
            features_dir = f"{DATA_DIR}/Features"

            features_extraction.full_dataset_maker(input_dir, output_dir, masked_dir, features_dir, model_path, verbose = False, test = False)
            features_extraction.full_dataset_maker(input_dir_test, output_dir, masked_dir, features_dir, model_path, verbose = False, test = True)
        
        elif choice == '4':
            extracted_dir = f"{LEGACY_DIR}/IntegratedPipeline/Data/Features"
            full_df = load_data.load_data(extracted_dir)
            X = full_df.select_dtypes(include='number')
            X_scaled = StandardScaler().fit_transform(X)
            labels = full_df["class"]

            pca_plot.apply_and_plot_pca(X_scaled, labels)

        elif choice == '5':
            Feature_dir_train = f"{DATA_DIR}/Features/"
            Feature_dir_test = f"{DATA_DIR}/Features_test/"
            full_df = load_data.load_data(Feature_dir_train)
            fuzzy_clustering.fuzzy_kmeans(full_df, n_clusters=4, verbose=False)
            shuffled_df = shuffle_data.shuffle_data(full_df, p=0.3)
            filtrered_df = get_filtered_df(shuffled_df, threshold=0.85)

            rfc_full = fit_random_forest.fit_random_forest(full_df, n_trees=10)
            rfc_shuffled = fit_random_forest.fit_random_forest(shuffled_df, n_trees=100)
            rfc_filtered = fit_random_forest.fit_random_forest(filtrered_df, n_trees=100)

            df_test = load_data.load_data(Feature_dir_test)
            accuracy_full = model_accuracy.accuracy_calculator(df_test, rfc_full)
            accuracy_shuffled = model_accuracy.accuracy_calculator(df_test, rfc_shuffled)
            accuracy_filtered = model_accuracy.accuracy_calculator(df_test, rfc_filtered)

            print(f"Accuracy with full data: {accuracy_full}")
            print(f"Accuracy with shuffled data: {accuracy_shuffled}")
            print(f"Accuracy with filtered data: {accuracy_filtered}\n\n")
            print(f"Delta accuracy filtered - unfiletered: {accuracy_filtered - accuracy_shuffled}")
            input("Enter to continue . . .")

        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":

    menu()


