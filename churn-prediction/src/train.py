import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from src.preprocess import preprocess_features


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "churn_cleaning.csv")
    model_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # target
    if "Churn_binary" not in df.columns:
        raise ValueError("Expected target column 'Churn_binary' in processed data.")

    y = df["Churn_binary"].copy()
    X = preprocess_features(df)

    # Keep stable feature list for inference
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

    joblib.dump(model, os.path.join(model_dir, "churn_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(model_dir, "feature_columns.pkl"))

    print("Saved model artifacts:")
    print("- churn_model.pkl")
    print("- scaler.pkl")
    print("- feature_columns.pkl")


if __name__ == "__main__":
    main()