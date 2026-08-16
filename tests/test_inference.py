import numpy as np

from src.model.factory import get_model
from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.data.train_test import split_X_y
from src.model.inference import predict

import src.config as config


def test_predict():

    df = clean_data(load_data(config.RAW_DATA_FILE_PATH))

    X, y = split_X_y(df)

    model = get_model("linear")

    model.fit(X, y)

    predictions = predict(model, X)

    assert isinstance(predictions, np.ndarray)

    assert predictions.shape[0] == X.shape[0]