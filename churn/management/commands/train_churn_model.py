import json
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.ml_utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, required_columns, model_path

class Command(BaseCommand):
    help = "Train and save the customer churn prediction model using the bundled demo dataset."

    def handle(self, *args, **options):
        data_file = settings.BASE_DIR / "data" / "sample_customers.csv"
        if not data_file.exists():
            raise CommandError(f"Demo dataset not found: {data_file}")
        df = pd.read_csv(data_file)
        if "Churn" not in df.columns:
            raise CommandError("Training CSV must contain a Churn target column.")
        X = df[required_columns()].copy()
        y = df["Churn"].map({"No": 0, "Yes": 1})
        if y.isna().any():
            raise CommandError("Churn column must contain only Yes or No.")
        for col in NUMERIC_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer([
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ])
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=250, class_weight="balanced",
                random_state=42, min_samples_leaf=2
            )),
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, predicted)) * 100, 2),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
            "training_rows": int(len(X_train)),
            "testing_rows": int(len(X_test)),
            "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        }
        target = model_path(settings.BASE_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target)
        (settings.BASE_DIR / "ml_models" / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        self.stdout.write(self.style.SUCCESS(
            f"Model trained successfully. Accuracy: {metrics['accuracy']}%, ROC-AUC: {metrics['roc_auc']}"
        ))
        self.stdout.write(f"Saved model to: {target}")
        self.stdout.write(f"Training rows: {metrics['training_rows']}; test rows: {metrics['testing_rows']}")
