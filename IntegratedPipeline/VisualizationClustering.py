from Visual.PCA import *
from Visual.TSNE import *
from Visual.KMEANS import *
import sys
        
if __name__ == "__main__":
    data_dir = "Features/"
    data_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_clustershcf/"

    full_df = load_data(data_dir)
    X = full_df.select_dtypes(include='number')
    X_scaled = standardize_data(X)

    if "--check" in sys.argv:
        check_nans(data_dir, X_scaled)

    choice = input("Do you want to use extracted features or database features?: ").strip().lower()
    if ch oice == "1":
        data_dir = "Features/"
    elif choice == "2":
        data_dir = "/home/francesco/Scaricati/Dataset/hand_crafted_features/train_clustershcf/"
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    full_df = load_data(data_dir)
    X = full_df.select_dtypes(include='number')
    X_scaled = standardize_data(X)

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
        print("Invalid choice. Exiting.")
        sys.exit(1)



