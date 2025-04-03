import sys
import os

from sklearn.preprocessing import StandardScaler

from .Visualizer.PCA import *
from .Visualizer.TSNE import *
from .Visualizer.KMEANS import *

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

def menu(extracted_dir, gold_dir):
    choice = input("Do you want to use extracted features or database features?: ").strip().lower()

    if choice == "1":
        data_dir = extracted_dir

    elif choice == "2":
        data_dir = gold_dir

    elif choice == "0":
        print("Exiting...")
        sys.exit(0)

    else:
        print("Invalid choice.")
        return

    full_df = load_data(data_dir)
    X = full_df.select_dtypes(include='number')
    X_scaled = StandardScaler().fit_transform(X)

    print("Choose a visualization method:")
    print("1. PCA")
    print("2. t-SNE")
    print("3. K-Means")

    vis_choice = input("Enter the number of your choice: ").strip()

    if vis_choice == "1":
        apply_and_plot_pca(X_scaled, full_df["class"], n_components=3)

    elif vis_choice == "2":
        apply_and_plot_tsne(X_scaled, full_df["class"], n_components=3)

    elif vis_choice == "3":
        apply_and_plot_kmeans(X_scaled, n_clusters=4)

    else:
        print("Invalid choice.")
        return
        
if __name__ == "__main__":

    extracted_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features"
    gold_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_clustershcf"
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        menu(extracted_dir, gold_dir)




