from src.data import loader, cleaner, train_test
import src.config
from src.evaluations import evaluator
from src.model.factory import get_model
from src.mlflow_logger import MlflowLogger
from artifacts.artifact_manager import ArtifactManager
from src.evaluations import visualizer

logger = MlflowLogger()
artifact_manager = ArtifactManager()

def train_model():
    """Trains the model, evaluates it and returns model, train and test metrics"""
    df = loader.load_data(path=src.config.RAW_DATA_FILE_PATH)
    train, test = train_test.split_train_test(df)
    train = cleaner.clean_data(train)
    test = cleaner.clean_data(test)
    X_train, y_train = train_test.split_X_y(train)
    X_test, y_test = train_test.split_X_y(test)
    model = get_model(src.config.MODEL_NAME)
    model.fit(X_train, y_train)

    train_pred, train_metrics = evaluator.evaluate(model, X_train, y_train)
    test_pred, test_metrics = evaluator.evaluate(model, X_test, y_test)

    tags = {
        "project": src.config.MLFLOW_EXPERIMENT_NAME,
        "framework": "scikit-learn",
        "algorithm": src.config.MODEL_NAME,
        "author": "Ayush Singh Songara"
    }        

    params = {
        "model": src.config.MODEL_NAME,
        **src.config.MODEL_CONFIG[src.config.MODEL_NAME],
        "test_size": src.config.TEST_SIZE,
        "random_state": src.config.RANDOM_STATE
    }

    metrics = {
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()}
    }    

    figures = visualizer.visualize_results(y_true=y_test, y_pred=test_pred)

    artifacts = artifact_manager.generate_training_artifacts(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        metrics=metrics,
        figures=figures
    )

    logger.log_runs(
        model=model,
        params=params,
        metrics=metrics,
        tags=tags,
        artifacts=artifacts
    )

    return model, {
        'train': train_metrics,
        'test': test_metrics
    }
