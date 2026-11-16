from pathlib import Path

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

def required_columns():
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES

def model_path(base_dir):
    return Path(base_dir) / "ml_models" / "churn_pipeline.joblib"
