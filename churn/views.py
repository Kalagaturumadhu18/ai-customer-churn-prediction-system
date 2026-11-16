import json
import uuid
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings
from django.contrib import messages
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.urls import reverse

from .ml_utils import required_columns, model_path

def dashboard(request):
    metrics_file = settings.BASE_DIR / "ml_models" / "metrics.json"
    metrics = None
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    latest_count = request.session.get("latest_prediction_count", 0)
    high_risk = request.session.get("latest_high_risk_count", 0)
    return render(request, "churn/dashboard.html", {
        "metrics": metrics, "latest_count": latest_count, "high_risk": high_risk,
        "model_ready": model_path(settings.BASE_DIR).exists(),
    })

def upload_csv(request):
    if request.method == "POST":
        uploaded = request.FILES.get("customer_file")
        if not uploaded:
            messages.error(request, "Please choose a CSV file.")
            return redirect("upload_csv")
        if not uploaded.name.lower().endswith(".csv"):
            messages.error(request, "Only CSV files are supported.")
            return redirect("upload_csv")
        if uploaded.size > 10 * 1024 * 1024:
            messages.error(request, "File is too large. Maximum size is 10 MB.")
            return redirect("upload_csv")
        try:
            df = pd.read_csv(uploaded)
            missing = [c for c in required_columns() if c not in df.columns]
            if missing:
                messages.error(request, "Missing required columns: " + ", ".join(missing))
                return redirect("upload_csv")
            if df.empty:
                messages.error(request, "The CSV contains no customer rows.")
                return redirect("upload_csv")
            for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(0)
            for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
                if df[col].isna().any():
                    messages.error(request, f"Column {col} contains invalid numeric values.")
                    return redirect("upload_csv")
            if not model_path(settings.BASE_DIR).exists():
                messages.error(request, "The ML model is not trained yet. Run: python manage.py train_churn_model")
                return redirect("upload_csv")
            pipeline = joblib.load(model_path(settings.BASE_DIR))
            X = df[required_columns()].copy()
            probabilities = pipeline.predict_proba(X)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            output = df.copy()
            output["Churn_Prediction"] = ["Yes" if p else "No" for p in predictions]
            output["Churn_Probability_Percent"] = (probabilities * 100).round(2)
            output["Risk_Level"] = pd.cut(
                probabilities, bins=[-0.001, 0.30, 0.60, 1.001],
                labels=["Low", "Medium", "High"]
            ).astype(str)
            filename = f"predictions_{uuid.uuid4().hex[:10]}.csv"
            output_path = settings.MEDIA_ROOT / "predictions" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(output_path, index=False)
            request.session["latest_prediction_file"] = filename
            request.session["latest_prediction_count"] = int(len(output))
            request.session["latest_high_risk_count"] = int((output["Risk_Level"] == "High").sum())
            messages.success(request, f"Successfully analysed {len(output)} customers.")
            return redirect("results")
        except Exception as exc:
            messages.error(request, f"Could not process the CSV: {exc}")
            return redirect("upload_csv")
    return render(request, "churn/upload.html", {"columns": required_columns()})

def _latest_output(request):
    filename = request.session.get("latest_prediction_file")
    if not filename or Path(filename).name != filename:
        return None
    path = settings.MEDIA_ROOT / "predictions" / filename
    return path if path.exists() else None

def results(request):
    path = _latest_output(request)
    if not path:
        messages.info(request, "Upload a customer CSV to view prediction results.")
        return redirect("upload_csv")
    df = pd.read_csv(path).fillna("")
    rows = df.head(50).to_dict(orient="records")
    return render(request, "churn/results.html", {
        "rows": rows, "total": len(df),
        "high_risk": int((df["Risk_Level"] == "High").sum()),
        "medium_risk": int((df["Risk_Level"] == "Medium").sum()),
        "low_risk": int((df["Risk_Level"] == "Low").sum()),
    })

def download_results(request):
    path = _latest_output(request)
    if not path:
        raise Http404("No prediction file found. Upload a CSV first.")
    return FileResponse(open(path, "rb"), as_attachment=True, filename="customer_churn_predictions.csv")
