from FeaturesExtraction.FeaturesExtraction import *
from KMeans.PrecisionCalculatore import *

if __name__ == "__main__":
    dataset_path = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/DataSet"
    mask_files, image_files = file_extraction(dataset_path, Verbose = True)

    features_dir = features_extraction(image_files, mask_files)
    print(run_kmeans_pipeline(features_dir, plot_confusion=False, var_threshold=0.01))
