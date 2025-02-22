import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import Config

class DataManager:
    def __init__(self, data_path: str = Config.DATA_PATH):
        self.data_path = data_path

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load the dataset from a CSV file."""
        data = pd.read_csv(f'{self.data_path}/{filename}')
        return data

    def split_data(self, data: pd.DataFrame):
        """Split the data into train and test datasets."""
        X = data.drop('cnt', axis=1) 
        y = data['cnt']
        return train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)