from src.data.loader import load_data
from src.data.cleaner import clean_data
from src.data.train_test import split_X_y
import src.config as config


def test_split_X_y():

    df = clean_data(load_data(config.RAW_DATA_FILE_PATH))

    X, y = split_X_y(df)

    assert "Chance_of_Admit" not in X.columns

    assert y.name == "Chance_of_Admit"

    assert len(X) == len(y)