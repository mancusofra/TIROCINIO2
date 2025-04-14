import warnings, os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#import seaborn as sns # statistical data visualization


warnings.filterwarnings('ignore')

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

def fit_random_forest(train_dir, n_trees= 10):
    scaler = StandardScaler()
    full_df_train = load_data(train_dir)
    X_train = full_df_train.select_dtypes(include='number')
    X_train_norm = scaler.fit_transform(X_train)
    y_train = full_df_train["class"]

    rfc = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    # fit the model
    rfc.fit(X_train_norm, y_train)
    return rfc

def accuracy_calculator(test_dir, rf_calculator):
    scaler = StandardScaler()
    full_df_test = load_data(test_dir)
    X_test = full_df_test.select_dtypes(include='number')
    X_test_norm = scaler.fit_transform(X_test)
    y_test = full_df_test["class"]

    y_pred = rf_calculator.predict(X_test_norm)
    return accuracy_score(y_test, y_pred)

def plot_accuracy(train_dir, test_dir):
    accuracy_arr = []
    tree_counts = range(5, 1005, 100)
    for number in tqdm(tree_counts, desc="Calculating accuracy"):
        rfc = fit_random_forest(train_dir, n_trees=number)
        accuracy_arr.append(accuracy_calculator(test_dir, rfc))
    
    plt.figure(figsize=(10, 6))
    plt.plot(tree_counts, accuracy_arr, marker='o', linestyle='-', color='b')
    plt.title('Accuracy vs Number of Trees in Random Forest')
    plt.xlabel('Number of Trees')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(train_dir, test_dir, n_trees = 10):
    rfc = fit_random_forest(train_dir, n_trees=n_trees)
    scaler = StandardScaler()
    full_df_test = load_data(test_dir)
    X_test = full_df_test.select_dtypes(include='number')
    X_test_norm = scaler.fit_transform(X_test)
    y_test = full_df_test["class"]

    y_pred = rfc.predict(X_test_norm)
    cm = confusion_matrix(y_test, y_pred, labels=rfc.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (n_trees={n_trees})')
    plt.show()
    
    


if __name__ == "__main__":
    
    train_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features/"
    test_dir = "/home/francesco/TIROCINIO2/IntegratedPipeline/Data/Features_test/"

    # rfc = fit_random_forest(train_dir)

    # print(f'Model accuracy score : {accuracy_calculator(test_dir, rfc)}')

    #plot_accuracy(train_dir, test_dir)
    #plot_confusion_matrix(train_dir, test_dir)

    
