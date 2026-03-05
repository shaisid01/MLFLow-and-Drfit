# 🚀 Spaceship Titanic ML Pipeline

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.9-lightgrey)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)

Predict whether passengers are transported in the **Spaceship Titanic** dataset using a full ML pipeline, including data preprocessing, model training, MLflow tracking, Flask API deployment, and data drift monitoring.

---

## 📂 Project Structure


```
├── data/
│ └── train.csv # Input dataset
├── models/ # Saved preprocessors and MLflow models
├── logs/ # Prediction logs
├── src/
│ └── preprocess.py # Feature engineering
├── preprocessor.py # Data preprocessing wrapper
├── train_and_promote.py # Model training, evaluation, and MLflow registration
├── app.py # Flask API for inference and monitoring
├── README.md
```

---

## 🛠 Features

- **Data Preprocessing:** Handle missing values, scaling, one-hot encoding, feature engineering (`Deck`, `GroupSize`)  
- **Model Training & Promotion:** RandomForestClassifier, MLflow logging, automated versioning & promotion  
- **API Deployment:** REST API with Swagger UI for predictions  
- **Monitoring:** Prediction logging and data drift dashboard using Evidently  

---

## ⚡ Preprocessing (`preprocessor.py`)

### Usage

```python
from preprocessor import preprocess_and_split

X_train, X_test, y_train, y_test = preprocess_and_split("data/train.csv")
```
* Saves the preprocessor to models/preprocessor.pkl for inference.

## 🎯 Model Training (train_and_promote.py)
**Steps**

1. Preprocess the data
2. Train RandomForestClassifier
3. Evaluate with accuracy
4. Log parameters, metrics, and model to MLflow

Register and optionally promote model to Production

**Run**

```
python train_and_promote.py
```
##🌐 API(app.py)
**Endpoints**

| Endpoint              | Method | Description                                  |
| --------------------- | ------ | -------------------------------------------- |
| `/predict`            | POST   | Predict `Transported` for a single passenger |
| `/docs`               | GET    | Swagger UI for API documentation             |
| `/generate_dashboard` | GET    | Generate data drift dashboard                |
| `/dashboard`          | GET    | View previously generated dashboard          |

**Example Request**
```
POST /predict
Content-Type: application/json

{
  "PassengerId": "0001_01",
  "Name": "John Doe",
  "HomePlanet": "Earth",
  "CryoSleep": false,
  "Cabin": "B/101",
  "Destination": "TRAPPIST-1e",
  "Age": 30,
  "VIP": false,
  "RoomService": 0.0,
  "FoodCourt": 100.0,
  "ShoppingMall": 50.0,
  "Spa": 0.0,
  "VRDeck": 10.0
}
```
**Example Response**
```
{
  "Transported": true
}
```
**Run API**
```
python app.py
```
**📦 Dependencies**
```
pip install pandas scikit-learn joblib mlflow flask flask-restx evidently
```

##📈 Monitoring & Data Drift ##
* Logs predictions to logs/prediction_log.csv
* Generates interactive HTML dashboard via /generate_dashboard
* Detects drift in features and highlights high-drift features (>0.5)

##🔧 Next Steps ##
* Add batch predictions endpoint
* Improve feature engineering for better model performance
* Automate dashboard generation with scheduled jobs

**License**
MIT License -see LICENSE 