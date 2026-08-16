from src.model.factory import get_model
from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.data.train_test import split_X_y
from src.evaluations.evaluator import evaluate

import src.config as config


def test_evaluator():

    df = clean_data(load_data(config.RAW_DATA_FILE_PATH))

    X, y = split_X_y(df)

    model = get_model("linear")

    model.fit(X, y)

    predictions, metrics = evaluate(model, X, y)

    expected = {
        "R2",
        "Adjusted_R2",
        "RMSE",
        "MAE",
    }

    assert expected == set(metrics.keys())