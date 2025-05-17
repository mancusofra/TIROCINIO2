from FeaturesExtraction.FeaturesExtraction import *
from KMeans.PrecisionCalculatore import *
from RForest.RandomForest import *
from Visual import VisualizationClustering
import os,sys
#from Visual.VisualizaztionClustering import *

def rf_pipeline(train_dir, test_dir, n_trees=10):
    fitted_model = fit_random_forest(train_dir, n_trees)
    accuracy = accuracy_calculator(test_dir, fitted_model)
    return accuracy

def menu():
    features_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features/"
    features_dir_test = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features_test/"
    if not (os.path.exists(features_dir) or os.path.exists(features_dir_test)):
        features_dir = None
        features_dir_test = None

    while True:
        os.system('clear')  # Use 'cls' on Windows instead of 'clear'
        print("1. Features extraction")
        print("2. Run KMeans Pipeline")
        print("3. Random Forest Fitting pipeline (10 trees)")  
        print("4. Cluster visualization")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            os.system('clear')
            dataset_path = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet/"
            dataset_path_test = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet_test/"
            print("Extracting features, please wait...")
            
            mask_files, image_files = file_extraction(dataset_path, Verbose=False)
            features_dir = features_extraction(image_files, mask_files, features_dir="/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features/")
            
            mask_files_test, image_files_test = file_extraction(dataset_path_test, Verbose=False)
            features_dir_test = features_extraction(image_files_test, mask_files_test, features_dir="/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features_test/")
            
            input("File extraction completed.")

        elif choice == '2':
            os.system('clear')
            if features_dir is None:
                input("Please run Feature Extraction first.")
                continue
            
            print("Running KMeans pipeline, please wait...")
            print(run_kmeans_pipeline(features_dir, plot_confusion=True, var_threshold=0.01))
            input("KMeans pipeline completed.")

        elif choice == '3':
            os.system('clear')
            if features_dir is None:
                input("Please run Feature Extraction first.")
                continue
            
            print("Fitting Random Forest, please wait...")
            accuracy = rf_pipeline(features_dir, features_dir_test, n_trees=10)
            print(f"Random Forest accuracy: {accuracy}")
            input("Random Forest fitting completed.")

        elif choice == '4':
            os.system('clear')
            if features_dir is None:
                input("Please run Feature Extraction first.")
                continue
            gold_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_clustershcf"

            VisualizationClustering.menu(features_dir, gold_dir)

        elif choice == '5':
            sys.exit()
        else:
            print("Invalid choice. Please try again.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    menu()
