import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler



def load_data(data_dir, valid = None, check_validity=False):
    all_data = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith(".csv") and ( (not check_validity) or (filename[:-4] in valid) ):
                    file_path = os.path.join(class_path, filename)
                    df = pd.read_csv(file_path, header=None)
                    feature_vector = df.values.flatten()
                    sample_df = pd.DataFrame([feature_vector])
                    sample_df["object_name"] = os.path.splitext(filename)[0]
                    sample_df["class"] = class_name
                    all_data.append(sample_df)
    return pd.concat(all_data, ignore_index=True)

def fit_random_forest(full_df_train , n_trees= 10):
    scaler = StandardScaler()
    X_train = full_df_train.select_dtypes(include='number')
    X_train_norm = scaler.fit_transform(X_train)
    y_train = full_df_train["class"]

    rfc = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    # fit the model
    rfc.fit(X_train_norm, y_train)
    return rfc

def accuracy_calculator(full_df_test, rf_calculator):
    scaler = StandardScaler()
    X_test = full_df_test.select_dtypes(include='number')
    X_test_norm = scaler.fit_transform(X_test)
    y_test = full_df_test["class"]

    y_pred = rf_calculator.predict(X_test_norm)
    return accuracy_score(y_test, y_pred)

    