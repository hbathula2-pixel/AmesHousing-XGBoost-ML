import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def split_features_target(df):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    return train_test_split(X, y, test_size=0.2, random_state=42)
