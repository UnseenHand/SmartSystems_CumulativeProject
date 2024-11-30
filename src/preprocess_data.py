# src/preprocess_data.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import numpy as np


def preprocess_data(
    input_file="data/spotify_songs_dataset.csv", output_file="data/preprocessed_data.csv"
):
    df = pd.read_csv(input_file)

    # Очищення пропущених значень
    df.dropna(inplace=True)

    # Вибір колонок для обробки
    categorical_features = [
        "genre",
        "language",
        "label",
        "explicit_content"
    ]
    numerical_features = ["duration", "stream"]

    # Переконаємося, що цільова змінна `Price (Euro)` наявна і є числовою
    if "popularity" not in df.columns:
        raise ValueError("Цільова змінна 'popularity' не знайдена у датасеті.")
    if not np.issubdtype(df["popularity"].dtype, np.number):
        raise ValueError("Цільова змінна 'popularity' повинна бути числовою.")

    # Визначення трансформера для колонок
    transformer = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",  # Видалити інші колонки, якщо вони є
    )

    # Застосування трансформацій
    processed_data = transformer.fit_transform(df)
    transformed_columns = (
        numerical_features
        + transformer.named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
        .tolist()
    )

    # Створення обробленого DataFrame
    processed_df = pd.DataFrame(processed_data, columns=transformed_columns)

    # Додавання цільової змінної
    processed_df["popularity"] = df["popularity"].values

    # Збереження
    os.makedirs("data", exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")
