# src/train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import os

from sklearn.neighbors import KNeighborsRegressor


def train_model(train_file="data/train_data.csv", models_dir="models/"):
    # Завантаження тренувальних даних
    df = pd.read_csv(train_file)
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    # Ініціалізація моделей
    kn_model = KNeighborsRegressor()
    xgb_model = xgb.XGBRegressor(random_state=42)

    # Тренування Random Forest
    print("Training K Neighbors...")
    kn_model.fit(X, y)
    os.makedirs(models_dir, exist_ok=True)
    rf_model_path = os.path.join(models_dir, "k_neighbors_model.pkl")
    joblib.dump(kn_model, rf_model_path)
    print(f"K Neighbors model saved to {rf_model_path}")

    # Тренування XGBoost
    print("Training XGBoost...")
    xgb_model.fit(X, y)
    xgb_model_path = os.path.join(models_dir, "xgboost_model.pkl")
    joblib.dump(xgb_model, xgb_model_path)
    print(f"XGBoost model saved to {xgb_model_path}")
