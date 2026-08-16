import pandas as pd

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning the data by dropping unnecessary columns and renaming columns
    """
    data = df.copy()
    data.drop(columns=["Serial No."], inplace=True)
    data.rename({
    "GRE Score": "GRE_Score",
    "TOEFL Score": "TOEFL_Score",
    "University Rating": "University_Rating",
    "Chance of Admit ": "Chance_of_Admit",
    "LOR ": "LOR"
    }, axis=1, inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    return data