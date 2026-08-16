from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from src.features import preprocessor
from src.config import MODEL_CONFIG

def build_model():
    """Builds and returns the ridge regression model."""
    preprocess = preprocessor.preprocess()
    model = Pipeline([
            ('preprocess', preprocess),
            ('model', Ridge(**MODEL_CONFIG['ridge']))
        ])
    return model