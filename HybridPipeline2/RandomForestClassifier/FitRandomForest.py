from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def fit_random_forest(full_df_train , n_trees= 10):
    scaler = StandardScaler()
    X_train = full_df_train.select_dtypes(include='number')
    X_train_norm = scaler.fit_transform(X_train)
    y_train = full_df_train["class"]

    rfc = RandomForestClassifier(n_estimators=n_trees, random_state=42)

    # fit the model
    rfc.fit(X_train_norm, y_train)
    return rfc