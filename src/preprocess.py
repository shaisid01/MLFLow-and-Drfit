import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os


def load_data(path):
    return pd.read_csv(path)


def feature_engineering(df):
    # Extract Deck from Cabin
    df["Deck"] = df["Cabin"].str.split("/").str[0]

    # Extract group size from PassengerId
    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("Group")["Group"].transform("count")

    # Drop unnecessary columns
    df.drop(["PassengerId", "Name", "Cabin", "Group"], axis=1, inplace=True)

    return df


def build_preprocessor(df):

    categorical_cols = [
        "HomePlanet",
        "Destination",
        "Deck"
    ]

    numeric_cols = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "GroupSize"
    ]

    # Pipelines
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def preprocess_and_split(path):

    df = load_data(path)
    df = feature_engineering(df)

    y = df["Transported"].astype(int)
    X = df.drop("Transported", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(df)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor for inference
     # Create models folder if not exists
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_train_processed, X_test_processed, y_train, y_test