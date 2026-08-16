from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def preprocess():
    """Returns a preprocessing pipeline"""
    preprocess = Pipeline([
        ('scaler', StandardScaler())
    ])
    return preprocess

