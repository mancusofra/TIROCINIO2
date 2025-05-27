from .DataProcessing import LoadData, ShuffleData, DFCompare
from .FuzzyKM import FuzzyClustering, AccuracyCalculator, UnsupervisedAssistedFiltering
from .RandomForestClassifier import FitRandomForest, ModelAccuracy
from .UNETSegmentation import UNETorch
from .FeaturesExtraction import FeaturesExtraction

import matplotlib.pyplot as plt
import os, platform

# Function for clearing terminal
def clear_terminal():
    os.system('cls' if platform.system() == 'Windows' else 'clear')

# Funzione che avverte l'user che la funzione che sta per eseguire e time expensive
def confirm_long_operation():
    clear_terminal()
    print("WARNING!!: You are about to perform an operation that may take a long time")
    print("and will **overwrite the existing model**.")
    response = input("Are you sure you want to continue? [y/N]: ").strip().lower()
    
    if response not in ['y', 'yes']:
        print("Operation aborted.")
        return False
    return True

def filter_accuracy(shuffled_elements, removed_elements, common_elements):
    accuracy_correctelements =  len(common_elements) / len(shuffled_elements)
    accuracy_incorrectelements = len(common_elements) / len(removed_elements) if len(removed_elements) > 0 else 0
    return accuracy_correctelements, accuracy_incorrectelements

def try_different_th(shuffled_df, lista_elementi_cambiati):
    lis = []
    u, y = FuzzyClustering.fuzzy_kmeans(shuffled_df, n_clusters=4)
    y_pred = u.argmax(axis=0)
    y_true = y
    n_clusters = 4

    for th in range(1, 10):
        th = th/10
        valid_elements = UnsupervisedAssistedFiltering.get_valid_elements(u, y_true, y_pred, n_clusters, threshold=th)
        reduced_df = UnsupervisedAssistedFiltering.get_valid_df(valid_elements, shuffled_df)
        lista_elementi_tolti = DFCompare.get_differences(shuffled_df, reduced_df)
        common_elements = set(lista_elementi_cambiati).intersection(set(lista_elementi_tolti))

        accuracy_correctelements, accuracy_incorrectelements = FilterAccuracy(lista_elementi_cambiati, lista_elementi_tolti, common_elements)
        lis.append((th, accuracy_correctelements, accuracy_incorrectelements))
    return lis

def plot_different_th(shuffle_p = 0.2):
    Feature_dir_train = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Features/"
    Feature_dir_test = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Features_test/"
    full_df = LoadData.load_data(Feature_dir_train)
    shuffled_df = ShuffleData.shuffle_data(full_df, p=shuffle_p)
    lista_elementi_cambiati = DFCompare.find_class_mismatches(full_df, shuffled_df)

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
    u, y = FuzzyClustering.fuzzy_kmeans(shuffled_df, n_clusters=4)
    y_pred = u.argmax(axis=0)
    y_true = y
    n_clusters = 4

    valid_elements = UnsupervisedAssistedFiltering.get_valid_elements(u, y_true, y_pred, n_clusters, threshold=threshold)
    reduced_df = UnsupervisedAssistedFiltering.get_valid_df(valid_elements, shuffled_df)
    return reduced_df

def menu():
    while True:
        clear_terminal()
        choice = input("Enter your choice: ")
        if choice == '1':
            if confirm_long_operation():
                history = UNETorch.train_model(data_dir)
                with open("Data/Model/history.pkl", "wb") as f:
                    pickle.dump(history, f)
        

            pass
        
        elif choice == '2':
            UNETorch.load_and_plot()

        elif choice == '3':
            masked_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Mask"
            input_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Original_images/Train_annotated"
            input_dir_test = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Original_images/test"
            output_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/DataSet"
            model_path = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Model/Weights.pt"
            features_dir = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Features"

            FeaturesExtraction.full_dataset_maker(input_dir, output_dir, masked_dir, features_dir, model_path, verbose = False, test = False)
            FeaturesExtraction.full_dataset_maker(input_dir_test, output_dir, masked_dir, features_dir, model_path, verbose = False, test = True)
        
        elif choice == '0':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":

    menu()
    # for i in range(1, 5):
    #     plot_different_th(i/10)
    # input("Press Enter to continue...")

    # Feature_dir_train = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Features/"
    # Feature_dir_test = "/home/francesco/TIROCINIO2/HybridPipeline2/Data/Features_test/"
    # full_df = LoadData.load_data(Feature_dir_train)
    # FuzzyClustering.fuzzy_kmeans(full_df, n_clusters=4, verbose=True)
    # shuffled_df = ShuffleData.shuffle_data(full_df, p=0.2)
    # filtrered_df = get_filtered_df(shuffled_df, threshold=0.85)

    # rfc_full = FitRandomForest.fit_random_forest(full_df, n_trees=10)
    # rfc_shuffled = FitRandomForest.fit_random_forest(shuffled_df, n_trees=10)
    # rfc_filtered = FitRandomForest.fit_random_forest(filtrered_df, n_trees=10)

    # df_test = LoadData.load_data(Feature_dir_test)
    # accuracy_full = ModelAccuracy.accuracy_calculator(df_test, rfc_full)
    # accuracy_shuffled = ModelAccuracy.accuracy_calculator(df_test, rfc_shuffled)
    # accuracy_filtered = ModelAccuracy.accuracy_calculator(df_test, rfc_filtered)

    # print(f"Accuracy with full data: {accuracy_full}")
    # print(f"Accuracy with shuffled data: {accuracy_shuffled}")
    # print(f"Accuracy with filtered data: {accuracy_filtered}")
    # print(f"\n\nDelta accuracy filtered - unfiletered: {accuracy_filtered - accuracy_shuffled}")

