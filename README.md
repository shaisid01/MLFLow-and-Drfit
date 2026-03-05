# Spaceship Titanic ML Project

This is a production-ready ML project for the **Spaceship Titanic Kaggle dataset**, including:

- Data preprocessing with Scikit-learn pipelines
- Random Forest model training
- Flask API for predictions
- MLflow experiment tracking
- Docker containerization
- AWS EC2 deployment

---

## 1. Project Structure


```
spaceship-titanic-ml/
├── data/ # Raw dataset CSVs
├── src/
│ ├── preprocess.py # Preprocessing pipeline
│ └── train.py # Model training + preprocessor saving
├── models/
│ ├── model.pkl
│ └── preprocessor.pkl
├── app.py # Flask API
├── requirements.txt
├── Dockerfile
├── input.json # Sample input for API testing
└── README.md
```
## 2. Setup (Local)

Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```
## 3. Train Model Locally


```
python src/train.py
```

* Trains Random Forest model
* Preprocesses data using pipeline
* Saves .pkl files in models/
* Logs experiment metrics to MLflow

## 4. Run Flask API Locally

Start API - python app.py
```
python app.py
```
OR via Docker
```
docker build -t spaceship-titanic-app .
docker run -d -p 5000:5000 spaceship-titanic-app
```


## 5. Test API with curl

```
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d "{\"HomePlanet\":\"Earth\",\"CryoSleep\":false,\"Destination\":\"TRAPPIST-1e\",\"Age\":27,\"VIP\":false,\"RoomService\":0,\"FoodCourt\":50,\"ShoppingMall\":0,\"Spa\":10,\"VRDeck\":5,\"Cabin\":\"B/0/P\",\"PassengerId\":\"0001_01\",\"Name\":\"John Doe\"}"
```



Or using JSON file:
```
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @input.json
```


Example response: {"Transported": false}

## 6. View MLflow Experiments
```
mlflow ui
```


Open browser: http://127.0.0.1:5000

View experiment metrics, parameters, accuracy, etc.

## 7. Docker Instructions
Build Docker Image :
```
 docker build -t spaceship-titanic-app .
```

Run Docker Container :
```
docker run -d -p 5000:5000 spaceship-titanic-app
```
 
Stop and Remove Container : 
```
docker ps
docker stop <CONTAINER_ID>
docker rm <CONTAINER_ID>
```


## 8. AWS Deployment
