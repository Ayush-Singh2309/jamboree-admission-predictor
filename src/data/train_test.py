from sklearn.model_selection import train_test_split
import src.config
import pandas as pd

def split_train_test(df:pd.DataFrame):
    """
    Splitting the data into train and test
    """
    train, test = train_test_split(df, test_size=src.config.TEST_SIZE, random_state=src.config.RANDOM_STATE)
    return train, test

def split_X_y(df:pd.DataFrame):
    """
    Splitting the data into features(X) and target(y)
    """
    X = df.drop(columns=["Chance_of_Admit"])
    y = df["Chance_of_Admit"]
    return X, y