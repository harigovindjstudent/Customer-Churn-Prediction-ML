import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import yaml

#set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.label_encoders ={}
        self.scaler = StandardScaler()

    def process_features(self, df, is_training=True):

        try:
            df_processed = df.copy()

            categorical_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()

            # Encode categorical features
            for col in categorical_columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
                else:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])

            # Scale numerical features
            if is_training:
                df_processed[numerical_columns] = self.scaler.fit_transform(df_processed[numerical_columns])
            else:
                df_processed[numerical_columns] = self.scaler.transform(df_processed[numerical_columns])

        except Exception as e:
            logger.error(f"Error processing features: {e}")
            raise

        