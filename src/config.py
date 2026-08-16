from pathlib import Path

NUM_FEATS = 7
TEST_SIZE = 0.2
RANDOM_STATE = 42


API_URL = "https://jamboree-admission-predictor-rfjw.onrender.com"
UVICORN_URL = "http://127.0.0.1:8000"


MLFLOW_EXPERIMENT_NAME = "Jamboree Admission Predictor"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_FILE_PATH = PROJECT_ROOT / "data" / "raw" / "Jamboree_Admission.csv"


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
COEFFICIENTS_DIR = ARTIFACTS_DIR / "coefficients"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


MODEL_NAME = 'ridge'
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}_model.pkl"

MODEL_CONFIG = {
    'linear': {},
    'ridge': {
        'alpha': 5.5127
    },
    'lasso': {
        'alpha': 0.000818
    }
}

MODEL_INFO = {
    "linear": {
        "algorithm": "Linear Regression",
        "framework": "scikit-learn",
        "version": "1.0.0"
    },
    "ridge": {
        "algorithm": "Ridge Regression",
        "framework": "scikit-learn",
        "version": "1.0.0"
    },
    "lasso": {
        "algorithm": "Lasso Regression",
        "framework": "scikit-learn",
        "version": "1.0.0"
    }
}


if __name__ == '__main__':
    print(MODEL_INFO[MODEL_NAME])