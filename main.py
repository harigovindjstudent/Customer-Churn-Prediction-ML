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

        #feature engineering
        X_train = FeatureEngineering_instance.create_features(X_train)
        X_test = FeatureEngineering_instance.create_features(X_test)

        #feature processing
        X_train_processed = FeatureEngineering_instance.process_features(X_train, is_training=True)
        X_test_processed = FeatureEngineering_instance.process_features(X_test, is_training=False)

        #model training
        best_model, best_score = ModelTrainer_instance.train_model(X_train_processed, X_test_processed, y_train, y_test)

        logger.info(f"Best model trained with F1 Score: {best_score}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise    