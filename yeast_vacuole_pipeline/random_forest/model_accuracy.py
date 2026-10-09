from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def accuracy_calculator(full_df_test, rf_calculator):
    """
    Evaluates a fitted Random Forest on a held-out feature DataFrame.

    Note: standardizes the test features with a scaler fit on the test set
    itself, independently from the scaler used in fit_random_forest.fit_random_forest.

    Args:
        full_df_test (pd.DataFrame): Feature DataFrame with a 'class' column
            (numeric columns are used as features).
        rf_calculator (RandomForestClassifier): Model fitted by fit_random_forest.

    Returns:
        float: Accuracy of the model's predictions on full_df_test.
    """
    scaler = StandardScaler()
    X_test = full_df_test.select_dtypes(include="number")
    X_test_norm = scaler.fit_transform(X_test)
    y_test = full_df_test["class"]

    y_pred = rf_calculator.predict(X_test_norm)
    return accuracy_score(y_test, y_pred)
