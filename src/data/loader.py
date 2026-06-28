import pandas as pd


def load_raw(csv_path):
    df = pd.read_csv(csv_path)
    return df
