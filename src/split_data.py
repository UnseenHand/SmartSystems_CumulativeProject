import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataset(
    input_file="data/preprocessed_data.csv",
    train_file="data/train_data.csv",
    test_file="data/test_data.csv",
):
    df = pd.read_csv(input_file)
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs("data", exist_ok=True)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"Train data saved to {train_file}")
    print(f"Test data saved to {test_file}")
