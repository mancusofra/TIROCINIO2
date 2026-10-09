import os
import pandas as pd
import numpy as np

def load_data(data_dir):
    """
    Loads all extracted feature CSVs under data_dir into a single DataFrame.

    Expects data_dir to contain one subfolder per class, each holding one CSV
    per sample (a single-column feature vector, no header). Each row of the
    result is one sample's flattened feature vector, plus 'object_name' (the
    CSV filename without extension) and 'class' (the subfolder name).

    Args:
        data_dir (str): Path to the features directory (one subfolder per class).

    Returns:
        pd.DataFrame: All samples, with 'object_name' and 'class' columns
        added to the flattened feature vectors.
    """
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
