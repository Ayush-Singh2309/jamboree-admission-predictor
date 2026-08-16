import joblib
import pandas as pd

def load_model(path):
    return joblib.load(path)

def predict(model, df: pd.DataFrame):
    return model.predict(df)