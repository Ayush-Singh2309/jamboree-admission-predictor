from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from src.features import preprocessor

def build_model():
    """Builds and returns the linear regression model."""
    preprocess = preprocessor.preprocess()
    model = Pipeline([
            ('preprocess', preprocess),
            ('model', LinearRegression())
        ])
    return model