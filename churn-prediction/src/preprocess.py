import pandas as pd


# Columns expected from raw Telco dataset / user input form
RAW_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw customer input into model-ready numeric features.
    This function must be used in BOTH training and inference.
    """
    X = df.copy()

    # Optional column cleanup
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])
    if "Churn" in X.columns:
        X = X.drop(columns=["Churn"])
    if "Churn_binary" in X.columns:
        X = X.drop(columns=["Churn_binary"])

    # Numeric handling
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    # Binary mappings
    yes_no_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in yes_no_cols:
        if col in X.columns:
            X[col] = (X[col] == "Yes").astype(int)

    if "gender" in X.columns:
        # Male -> 1, Female -> 0
        X["gender"] = (X["gender"] == "Male").astype(int)

    # Ensure senior citizen numeric
    if "SeniorCitizen" in X.columns:
        X["SeniorCitizen"] = pd.to_numeric(X["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # One-hot encode non-binary categoricals
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Feature engineering
    if "MonthlyCharges" in X.columns and "tenure" in X.columns:
        X["charges_per_tenure"] = X["MonthlyCharges"] / (X["tenure"] + 1)

    service_cols = [
        "PhoneService",
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
        "StreamingTV_Yes",
        "StreamingMovies_Yes",
    ]
    existing_service_cols = [c for c in service_cols if c in X.columns]
    if existing_service_cols:
        X["num_services"] = X[existing_service_cols].sum(axis=1)

    # Fill numeric NaNs
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = X[num_cols].fillna(0)

    return X