import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#defining class DataLoader
class DataLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def load_data(self):
        try :
            data_path = self.config['data']['raw_data_path']
            logger.info(f"Loading data from {data_path}")
            return pd.read_csv(data_path)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def split_data(self, df):
        try:
            X = df.drop(columns=['Exited'])
            y = df['Exited']

            test_size = self.config['data']['test_size']
            val_size = self.config['data']['val_size']
            random_state = self.config['data']['random_state']

            X_temp, X_test, y_temp, y_test = train_test_split(
                X, 
                y, 
                test_size=test_size, 
                random_state=random_state)
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size/(1-test_size),  # Adjust val_size to account for the reduced dataset
                random_state=random_state
            )
            
            logger.info(f"Data split into train, validation, and test sets with test size {self.config['data']['test_size']} and validation size {self.config['data']['val_size']}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise
