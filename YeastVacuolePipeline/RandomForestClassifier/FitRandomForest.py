from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def fit_random_forest(full_df_train , n_trees= 10):
    """
    Trains a Random Forest classifier on the given feature DataFrame.

    Note: the scaler is fit here on the training data only, and
    ModelAccuracy.accuracy_calculator fits a separate scaler on the test
    data — features are standardized independently for train and test
    rather than reusing this scaler's fitted mean/std.

    Args:
        full_df_train (pd.DataFrame): Feature DataFrame with a 'class'
            column (numeric columns are used as features).
        n_trees (int): Number of trees in the forest (n_estimators).

    Returns:
        RandomForestClassifier: The fitted model.
    """
    scaler = StandardScaler()
    X_train = full_df_train.select_dtypes(include='number')
    X_train_norm = scaler.fit_transform(X_train)
    y_train = full_df_train["class"]

    rfc = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rfc.fit(X_train_norm, y_train)
    return rfc