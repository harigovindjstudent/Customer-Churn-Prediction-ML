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

            logger.info(f"Feature preprocessing completed successfully")
            logger.info(f"Categorical features processed: {categorical_columns}")
            logger.info(f"Numerical features processed: {numerical_columns}")
            return df_processed
        
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            raise

    def create_features(self, df):
        try:
            df_new = df.copy()
            
            # Example feature: Balance per product
            df_new['BalancePerProduct'] = df_new['Balance'] / (df_new['NumOfProducts'] + 1)
            
            # Example feature: Customer engagement score
            df_new['EngagementScore'] = df_new['IsActiveMember'] * 0.5 + df_new['HasCrCard'] * 0.3 + \
                                      (df_new['NumOfProducts'] / 4) * 0.2
            
            logger.info("Feature creation completed successfully")
            return df_new
            
        except Exception as e:
            logger.error(f"Error in feature creation: {str(e)}")
            raise

        