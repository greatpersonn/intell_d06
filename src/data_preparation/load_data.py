import pandas as pd
import os

def load_train(data_dir: str) -> pd.DataFrame:
    """
    Завантажує train.csv із папки data/raw.
    """
    path = os.path.join(data_dir, "train.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

def load_test(data_dir: str) -> pd.DataFrame:
    """
    Завантажує test.csv із папки data/raw.
    """
    path = os.path.join(data_dir, "test.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df

def load_store(data_dir: str) -> pd.DataFrame:
    """
    Завантажує store.csv із папки data/raw.
    """
    path = os.path.join(data_dir, "store.csv")
    df = pd.read_csv(path)
    return df
