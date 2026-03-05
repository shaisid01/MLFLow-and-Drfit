import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import preprocess_and_split  # your preprocessing function

# --- Step 1: Load and preprocess data ---
X_train, X_test, y_train, y_test = preprocess_and_split("data/train.csv")

# --- Step 2: Train model ---
model = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Evaluate ---
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Validation Accuracy: {acc:.4f}")

# --- Step 4: Log model to MLflow ---
mlflow.set_experiment("Spaceship_Titanic_Academic")
with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", 7)
    mlflow.log_metric("accuracy", acc)

    # Log the sklearn model
    mlflow.sklearn.log_model(model, artifact_path="model")

    run_id = run.info.run_id
    print("MLflow Run ID:", run_id)

# --- Step 5: Register & Promote ---
client = MlflowClient()
model_uri = f"runs:/{run_id}/model"

# Register model (new version automatically created)
registered_model_name = "SpaceshipTitanicModel"
mlflow.register_model(model_uri, registered_model_name)

# Fetch the version just created
latest_version_info = client.get_latest_versions(registered_model_name, stages=["None"])
latest_version = max([int(v.version) for v in latest_version_info])
print(f"New version registered: {latest_version}")

# Optional: promote automatically if validation metric >= threshold
accuracy_threshold = 0.7
if acc >= accuracy_threshold:
    client.transition_model_version_stage(
        name=registered_model_name,
        version=latest_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Version {latest_version} promoted to Production")
else:
    print(f"Version {latest_version} not promoted, accuracy below threshold ({accuracy_threshold})")