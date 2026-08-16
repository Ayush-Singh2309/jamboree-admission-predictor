import pandas as pd
import src.config as config
from src.data.loader import load_data


def test_load_data_returns_dataframe():
    df = load_data(config.RAW_DATA_FILE_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty