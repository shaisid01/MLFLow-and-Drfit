from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource, fields
import pandas as pd
import mlflow.sklearn
import joblib
import csv
from datetime import datetime
import os
from evidently import Report
from evidently.presets import DataDriftPreset

# Import feature engineering
from src.preprocess import feature_engineering

app = Flask(__name__)
api = Api(
    app,
    doc="/docs",
    title="Spaceship Titanic API",
    description="Predict if passengers are transported"
)

# ----------------- Load model & preprocessor -----------------
MODEL_NAME = "SpaceshipTitanicModel"
MODEL_STAGE = "Production"

# Load sklearn model registered in MLflow
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Load fitted preprocessor saved during training
preprocessor = joblib.load("models/preprocessor.pkl")

# ----------------- Swagger Input Schema -----------------
input_model = api.model('Input', {
    "PassengerId": fields.String(required=True),
    "Name": fields.String(required=True),
    "HomePlanet": fields.String(required=True),
    "CryoSleep": fields.Boolean(required=True),
    "Cabin": fields.String(required=True),
    "Destination": fields.String(required=True),
    "Age": fields.Float(required=True),
    "VIP": fields.Boolean(required=True),
    "RoomService": fields.Float(required=True),
    "FoodCourt": fields.Float(required=True),
    "ShoppingMall": fields.Float(required=True),
    "Spa": fields.Float(required=True),
    "VRDeck": fields.Float(required=True)
})

# ----------------- Prediction Endpoint -----------------
@api.route("/predict")
class Predict(Resource):
    @api.expect(input_model)
    def post(self):
        data = request.get_json()
        df = pd.DataFrame([data])

        # Apply feature engineering (Deck, GroupSize)
        df = feature_engineering(df)

        # Transform using fitted preprocessor
        X_processed = preprocessor.transform(df)

        # Predict
        prediction = model.predict(X_processed)

        # Log prediction for monitoring
        os.makedirs("logs", exist_ok=True)
        log_file = "logs/prediction_log.csv"
        log_row = {**data, "Transported": int(prediction[0]), "timestamp": datetime.utcnow().isoformat()}
        with open(log_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_row.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(log_row)

        return {"Transported": bool(prediction[0])}

# ----------------- Home & Monitor -----------------
@app.route("/")
def home():
    return "Spaceship Titanic API is running! Go to /docs for Swagger UI."

@app.route("/monitor", methods=["GET"])
def monitor():
    dashboard_path = "evidently_dashboard.html"
    if os.path.exists(dashboard_path):
        return send_file(dashboard_path)
    return "Evidently dashboard not found", 404

# ----------------- Generate Drift Dashboard -----------------
@app.route("/generate_dashboard", methods=["GET"])
def generate_dashboard():
    reference_path = "data/train.csv"
    production_path = "logs/prediction_log.csv"
    html_path = "evidently_dashboard.html"

    if not os.path.exists(reference_path) or not os.path.exists(production_path):
        return "Reference or production data not found", 404

    reference = pd.read_csv(reference_path)
    production = pd.read_csv(production_path)

    if production.empty:
        return "Production log is empty — cannot generate dashboard", 400

    # Drop non-feature columns
    non_feature = ["timestamp", "Name", "PassengerId", "Cabin", "Group", "Transported"]
    ref_features = reference.drop(columns=[c for c in non_feature if c in reference.columns])
    prod_features = production.drop(columns=[c for c in non_feature if c in production.columns])

    # Convert categorical columns to string
    cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
    for col in cat_cols:
        if col in ref_features.columns:
            ref_features[col] = ref_features[col].astype(str)
        if col in prod_features.columns:
            prod_features[col] = prod_features[col].astype(str)

    try:
        # Run data drift report
        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=ref_features, current_data=prod_features)

        # Parse JSON and extract ValueDrift metrics
        import json
        report_data = json.loads(snapshot.json())
        drift_metrics = []
        for m in report_data.get("metrics", []):
            if "ValueDrift" in m.get("metric_name", ""):
                value = m.get("value")
                if isinstance(value, float):
                    drift_metrics.append({"feature": m["metric_name"], "drift": value})

        # Sort and prepare for Chart.js
        drift_metrics = sorted(drift_metrics, key=lambda x: x["drift"], reverse=True)
        features = [m["feature"] for m in drift_metrics]
        drift_values = [m["drift"] for m in drift_metrics]

        # Build interactive HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spaceship Titanic — Data Drift Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                h2 {{ text-align: center; }}
                .container {{ width: 90%; margin: 20px auto; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 30px; }}
                th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Spaceship Titanic — Data Drift Dashboard</h2>
            <div class="container">
                <canvas id="driftChart"></canvas>
            </div>
            <div class="container">
                <table>
                    <tr><th>Feature</th><th>Drift Value</th></tr>
                    {''.join([f"<tr style='background-color:{'#f8d7da' if v['drift']>0.5 else 'transparent'}'><td>{v['feature']}</td><td>{v['drift']:.3f}</td></tr>" for v in drift_metrics])}
                </table>
            </div>
            <script>
                const ctx = document.getElementById('driftChart').getContext('2d');
                const chart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {features},
                        datasets: [{{
                            label: 'Drift (Jensen-Shannon distance)',
                            data: {drift_values},
                            backgroundColor: {['rgba(255, 99, 132, 0.7)' if v['drift']>0.5 else 'rgba(54, 162, 235, 0.7)' for v in drift_metrics]},
                            borderColor: {['rgba(255, 99, 132, 1)' if v['drift']>0.5 else 'rgba(54, 162, 235, 1)' for v in drift_metrics]},
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 1
                            }}
                        }},
                        plugins: {{
                            legend: {{ display: false }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    except Exception as e:
        return f"Failed to generate dashboard: {e}", 500

    return f"Dashboard generated! Open {html_path} in your browser."

@app.route("/dashboard")
def dashboard():
    return send_file("evidently_dashboard.html")

# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)