from src.data.loader import load_data
from src.data.cleaner import clean_data
import src.config as config


def test_clean_data():

    df = load_data(config.RAW_DATA_FILE_PATH)

    clean_df = clean_data(df)

    assert "Serial No." not in clean_df.columns

    expected_cols = {
        "GRE_Score",
        "TOEFL_Score",
        "University_Rating",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
        "Chance_of_Admit",
    }

    assert set(clean_df.columns) == expected_cols

    assert clean_df.isna().sum().sum() == 0

    assert clean_df.duplicated().sum() == 0