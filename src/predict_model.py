# src/predict_model.py
import pandas as pd
import joblib
import os


def predict_new_data(
    input_file="data/test_data.csv", models_dir="models/", output_dir="results/"
):
    # Завантаження тестових даних
    df = pd.read_csv(input_file)
    X = df.drop(columns=["popularity"])

    os.makedirs(output_dir, exist_ok=True)

    # Прогнозування Random Forest
    rf_model_path = os.path.join(models_dir, "k_neighbors_model.pkl")
    rf_model = joblib.load(rf_model_path)
    rf_predictions = rf_model.predict(X)
    df["KN_Predicted popularity"] = rf_predictions

    # Прогнозування XGBoost
    xgb_model_path = os.path.join(models_dir, "xgboost_model.pkl")
    xgb_model = joblib.load(xgb_model_path)
    xgb_predictions = xgb_model.predict(X)
    df["XGB_Predicted popularity"] = xgb_predictions

    # Збереження результатів з обома прогнозами
    result_output_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(result_output_path, index=False)
    print(f"Combined predictions saved to {result_output_path}")

    # Створення окремого датасету для порівняння цільової змінної та прогнозів
    comparison_df = df[
        ["popularity", "KN_Predicted popularity", "XGB_Predicted popularity"]
    ]
    comparison_output_path = os.path.join(output_dir, "comparison_with_actual.csv")
    comparison_df.to_csv(comparison_output_path, index=False)
    print(f"Comparison with actual values saved to {comparison_output_path}")
