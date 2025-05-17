from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def accuracy_calculator(full_df_test, rf_calculator):
    scaler = StandardScaler()
    X_test = full_df_test.select_dtypes(include='number')
    X_test_norm = scaler.fit_transform(X_test)
    y_test = full_df_test["class"]

    y_pred = rf_calculator.predict(X_test_norm)
    return accuracy_score(y_test, y_pred)