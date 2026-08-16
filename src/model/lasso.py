from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from src.features import preprocessor
from src.config import MODEL_CONFIG

def build_model():
    """Builds and returns the lasso regression model."""
    preprocess = preprocessor.preprocess()
    model = Pipeline([
            ('preprocess', preprocess),
            ('model', Lasso(**MODEL_CONFIG['lasso']))
        ])
    return model