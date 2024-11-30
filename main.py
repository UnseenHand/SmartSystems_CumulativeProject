# main.py
from src.preprocess_data import preprocess_data
from src.split_data import split_dataset
from src.train_model import train_model
from src.predict_model import predict_new_data


def run():
    preprocess_data()
    split_dataset()
    train_model()
    predict_new_data()


if __name__ == "__main__":
    run()
