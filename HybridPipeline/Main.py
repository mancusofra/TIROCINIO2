from FeaturesExtraction.FeaturesExtraction import *
from FuzzyKMeans import FuzzyKM
from RForest import RandomForest

import os, sys
import numpy as np
import pandas as pd
#from Visual.VisualizaztionClustering import *

def load_data(data_dir):
    all_data = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(".csv"):
                    file_path = os.path.join(class_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature_vector = df.values.flatten()
                    sample_df = pd.DataFrame([feature_vector])
                    sample_df["object_name"] = os.path.splitext(filename)[0]
                    sample_df["class"] = class_name
                    all_data.append(sample_df)
    return pd.concat(all_data, ignore_index=True)

def rf_pipeline(train_dir, test_dir, n_trees=10):
    fitted_model = fit_random_forest(train_dir, n_trees)
    accuracy = accuracy_calculator(test_dir, fitted_model)
    return accuracy

def shuffle_data(df, p=0.2, output_file='cambiati.txt'):
    # Calcola quante righe modificare
    num_da_modificare = int(len(df) * p)
    print(f"\nModificando {num_da_modificare} righe.")

    # Seleziona righe a caso da modificare
    indici_modificare = df.sample(n=num_da_modificare, random_state=42).index

    # Copia la colonna 'class' originale per il confronto
    classi_originali = df['class'].copy()

    # Ottieni tutte le classi disponibili
    classi_possibili = df['class'].unique()

    # Cambia le classi selezionate in un'altra classe a caso diversa dall'originale
    for idx in indici_modificare:
        classe_originale = df.at[idx, 'class']
        nuove_classi = [c for c in classi_possibili if c != classe_originale]
        nuova_classe = np.random.choice(nuove_classi)
        df.at[idx, 'class'] = nuova_classe

    # Trova gli indici dove la classe è cambiata
    cambiati = df[df['class'] != classi_originali]

    # Salva solo i valori di 'object_name' in un file di testo, uno per riga
    if 'object_name' in df.columns:
        with open(output_file, 'w') as f:
            for nome in sorted(cambiati['object_name']):
                f.write(f"{nome}\n")
    else:
        print("\nColonna 'object_name' non trovata nel DataFrame!")

    return df


def compare_methods(full_df, full_df_test, n_trees):
    n_clusters = 4
    u, y_true = FuzzyKM.run_fuzzy_kmeans(full_df, n_clusters)
    y_pred = np.argmax(u, axis=0)

    valid = FuzzyKM.get_valid_elements(u, y_true, y_pred, n_clusters, threshold=0.01)
    filtered_df = []
    for oggetto in full_df['object_name'].values[valid]:
        filtered_df.append(full_df[full_df['object_name'] == oggetto])
    filtered_df = pd.concat(filtered_df, ignore_index=True)  # Combine into a single DataFrame

    # Trova gli oggetti NON presenti nel DataFrame filtrato
    oggetti_tutti = set(full_df['object_name'].values)
    oggetti_filtrati = set(filtered_df['object_name'].values)
    oggetti_esclusi = oggetti_tutti - oggetti_filtrati

    # Salva in un file di testo
    with open("filtrati.txt", "w") as f:
        for nome in sorted(oggetti_esclusi):
            f.write(f"{nome}\n")
    print(f"{len(oggetti_esclusi)} oggetti esclusi salvati in filtrati.txt")

    # Addestra Random Forest
    rfc = RandomForest.fit_random_forest(full_df, n_trees=n_trees)
    rfc2 = RandomForest.fit_random_forest(filtered_df, n_trees=n_trees)

    # Calcola accuracy
    accuracy = RandomForest.accuracy_calculator(full_df_test, rfc)
    accuracy2 = RandomForest.accuracy_calculator(full_df_test, rfc2)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Accuracy2: {accuracy2:.2f}")

def confronta_file(filtrati_path='filtrati.txt', cambiati_path='cambiati.txt'):
    # Leggi i nomi dal file filtrati.txt
    with open(filtrati_path, 'r') as f:
        filtrati = set(line.strip() for line in f if line.strip())

    # Leggi i nomi dal file cambiati.txt
    with open(cambiati_path, 'r') as f:
        cambiati = set(line.strip() for line in f if line.strip())

    # Trova l'intersezione
    in_entrambi = filtrati & cambiati

    # Output
    print(f"Oggetti presenti in entrambi i file: {len(in_entrambi)}")

    return in_entrambi


def menu():
    features_dir = "/home/francesco/TIROCINIO2/HybridPipeline/Data/Features/"
    features_dir_test = "/home/francesco/TIROCINIO2/HybridPipeline/Data/Features_test/"

    #dataset_path = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_annotated"
    #dataset_path_test = "/home/francesco/Scaricati/Dataset/hand_crafted_features/test"

    if not (os.path.exists(features_dir) or os.path.exists(features_dir_test)):
        features_dir = None
        features_dir_test = None

    while True:
        os.system('clear')  # Use 'cls' on Windows instead of 'clear'
        print("1. Features extraction")
        print("2. Shuffle data")
        print("0. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            os.system('clear')
            dataset_path = "/home/francesco/TIROCINIO2/HybridPipeline/Data/DataSet/"
            dataset_path_test = "/home/francesco/TIROCINIO2/HybridPipeline/Data/DataSet_test/"
            print("Extracting features, please wait...")
            
            mask_files, image_files = file_extraction(dataset_path, Verbose=False)
            features_dir = features_extraction(image_files, mask_files, features_dir="/home/francesco/TIROCINIO2/HybridPipeline/Data/Features/")
            
            mask_files_test, image_files_test = file_extraction(dataset_path_test, Verbose=False)
            features_dir_test = features_extraction(image_files_test, mask_files_test, features_dir="/home/francesco/TIROCINIO2/HybridPipeline/Data/Features_test/")
            
            input("File extraction completed.")
        
        elif choice == '2':
            shuffled_df = shuffle_data(load_data(features_dir), p=0.2)
            full_df = load_data(features_dir)

            full_df_test = load_data(features_dir_test)

            rfc = RandomForest.fit_random_forest(full_df, n_trees=10)
            rfc2 = RandomForest.fit_random_forest(shuffled_df, n_trees=10)

            accuracy = RandomForest.accuracy_calculator(full_df_test, rfc)
            accuracy2 = RandomForest.accuracy_calculator(full_df_test, rfc2)
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Accuracy2: {accuracy2:.2f}")
            u, y_true = FuzzyKM.run_fuzzy_kmeans(shuffled_df, n_clusters=4)
            y_pred = np.argmax(u, axis=0)

            valid = FuzzyKM.get_valid_elements(u, y_true, y_pred, n_clusters=4, threshold=0.70)
            filtered_df = []
            count = 0
            for oggetto in shuffled_df['object_name'].values[valid]:
                count += 1
                filtered_df.append(shuffled_df[shuffled_df['object_name'] == oggetto])
            filtered_df = pd.concat(filtered_df, ignore_index=True)  # Combine into a single DataFrame

            # Save the object_name of filtered elements to a file
            with open("filtrati.txt", "w") as f:
                for nome in sorted(filtered_df['object_name']):
                    f.write(f"{nome}\n")

            rfc3 = RandomForest.fit_random_forest(filtered_df, n_trees=10)
            accuracy3 = RandomForest.accuracy_calculator(full_df_test, rfc3)
            print(f"Accuracy3: {accuracy3:.2f}")
            print(f"Filtered elements: {len(shuffled_df) - count}")

            # Confronta i file
            in_entrambi = confronta_file(filtrati_path='filtrati.txt', cambiati_path='cambiati.txt')

            input("Press Enter to continue...")


        elif choice == '3':
            dataset_path = "/home/francesco/TIROCINIO2/HybridPipeline/Data/Features/"
            dataset_path_test = "/home/francesco/TIROCINIO2/HybridPipeline/Data/Features_test/"

            #dataset_path = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_annotated"
            #dataset_path_test = "/home/francesco/Scaricati/Dataset/hand_crafted_features/test"

            full_df = FuzzyKM.load_data(dataset_path)
            n_clusters = 4
            u, y_true =FuzzyKM.run_fuzzy_kmeans(full_df, n_clusters)
            y_pred = np.argmax(u, axis=0)

            valid = FuzzyKM.get_valid_elements(u, y_true, y_pred, n_clusters)
            filtered_df = []
            for oggetto in full_df['object_name'].values[valid]:
                filtered_df.append(full_df[full_df['object_name'] == oggetto])
            filtered_df = pd.concat(filtered_df, ignore_index=True)  # Combine into a single DataFrame

            print(full_df.shape)
            print(filtered_df.shape)
            rfc = RandomForest.fit_random_forest(dataset_path, full_df, n_trees=10)
            rfc2 = RandomForest.fit_random_forest(dataset_path, filtered_df, n_trees=10)

            full_df_test = FuzzyKM.load_data(dataset_path_test)
            accuracy = RandomForest.accuracy_calculator(dataset_path_test, full_df_test, rfc)
            accuracy2 = RandomForest.accuracy_calculator(dataset_path_test, full_df_test, rfc2)

            print(f"Accuracy: {accuracy:.2f}")
            print(f"Accuracy2: {accuracy2:.2f}")

        

        elif choice == '0':
            sys.exit()
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    os.system('clear')

    features_dir = "/home/francesco/TIROCINIO2/HybridPipeline/Data/Features/"
    features_dir_test = "/home/francesco/TIROCINIO2/HybridPipeline/Data/Features_test/"

    shuffled_df = shuffle_data(load_data(features_dir), p=0.2)
    full_df = load_data(features_dir)

    full_df_test = load_data(features_dir_test)

    rfc = RandomForest.fit_random_forest(full_df, n_trees=10)
    rfc2 = RandomForest.fit_random_forest(shuffled_df, n_trees=10)

    accuracy = RandomForest.accuracy_calculator(full_df_test, rfc)
    accuracy2 = RandomForest.accuracy_calculator(full_df_test, rfc2)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Accuracy2: {accuracy2:.2f}")
    u, y_true = FuzzyKM.run_fuzzy_kmeans(shuffled_df, n_clusters=4)
    y_pred = np.argmax(u, axis=0)

    valid = FuzzyKM.get_valid_elements(u, y_true, y_pred, n_clusters=4, threshold=0.99)
    filtered_df = []
    count = 0
    for oggetto in shuffled_df['object_name'].values[valid]:
        count += 1
        filtered_df.append(shuffled_df[shuffled_df['object_name'] == oggetto])
    filtered_df = pd.concat(filtered_df, ignore_index=True)  # Combine into a single DataFrame

    # Save the object_name of filtered elements to a file
    with open("filtrati.txt", "w") as f:
        for nome in sorted(filtered_df['object_name']):
            f.write(f"{nome}\n")

    rfc3 = RandomForest.fit_random_forest(filtered_df, n_trees=10)
    accuracy3 = RandomForest.accuracy_calculator(full_df_test, rfc3)
    print(f"Accuracy3: {accuracy3:.2f}")
    print(f"Filtered elements: {len(shuffled_df) - count}")

    # Confronta i file
    in_entrambi = confronta_file(filtrati_path='filtrati.txt', cambiati_path='cambiati.txt')
    