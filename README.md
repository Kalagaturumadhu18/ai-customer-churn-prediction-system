
# AI Customer Churn Prediction System

## Live Demo
[Click Here to View Live Project](https://ai-customer-churn-prediction-system.onrender.com)

# AI Customer Churn Prediction System

A Django + Machine Learning web application that estimates which telecom customers may leave a service. It supports CSV uploads, churn-risk predictions, a results table, and CSV downloads.

## Features
- Upload customer data in CSV format
- Predict churn probability using a scikit-learn Random Forest pipeline
- Classify customers as High, Medium, or Low risk
- View model evaluation metrics
- Download prediction results as CSV
- Includes a clearly labelled **synthetic demo dataset** for training and testing

## Technology
Python, Django, Pandas, NumPy, Scikit-learn, SQLite, HTML, CSS.

## Setup (Windows PowerShell)

```powershell
cd C:\projects_github\ai-customer-churn-prediction-system
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py train_churn_model
python manage.py runserver
```

Open http://127.0.0.1:8000/

If PowerShell blocks virtual-environment activation, run this in the same PowerShell window:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Try the sample file
Use `data/sample_customers.csv` on the Upload CSV page. It is synthetic demonstration data, not real customer information.

## CSV columns
The uploaded CSV must contain these columns (column names are case-sensitive):
`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

An optional `customerID` column is retained in the output. Do not include the actual `Churn` answer column in an upload; the model predicts it.

## Important
- The included sample dataset is synthetically generated for learning and demonstration. It is not a real-world benchmark and model scores should not be presented as evidence of business performance.
- Predictions are estimates, not guarantees. Validate any model on representative, properly consented data before using it for real decisions.
- `db.sqlite3`, virtual environments, uploaded prediction files, and the trained model are excluded from Git by `.gitignore`. The model is recreated by `python manage.py train_churn_model`.

## GitHub
Suggested repository name: `ai-customer-churn-prediction-system`
