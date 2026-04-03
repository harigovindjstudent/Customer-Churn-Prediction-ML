import logging
import yaml
import mlflow
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineering
from src.models.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        config_path = 'config/config.yaml'

        DataLoader_instance = DataLoader(config_path)
        FeatureEngineering_instance = FeatureEngineering(config_path)
        ModelTrainer_instance = ModelTrainer(config_path)

        #load data
        df = DataLoader_instance.load_data()
        #split data
        X_train, X_test, y_train, y_test = DataLoader_instance.split_data(df)