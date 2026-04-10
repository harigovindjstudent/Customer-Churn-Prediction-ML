import logging
import yaml
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.features.feature_engineering import FeatureEngineering
from src.models.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        config_path = 'config/config.yaml'
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # load test data
        try:
            X_test = pd.read_csv('data/processed/X_test.csv')
            y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
        except FileNotFoundError:
            logger.error("Test data files not found in 'data/processed'. Run train.py first.")
            return
        
        FeatureEngineering_instance = FeatureEngineering(config_path)
    
        #load preprocessing pipeline and selector
        FeatureEngineering_instance.load_pipeline('models/preprocessing_pipeline.joblib', 'models/selector.joblib')

        #feature engineering
        X_test = FeatureEngineering_instance.create_features(X_test)

        #feature processing
        X_test_processed = FeatureEngineering_instance.process_features(X_test, is_training=False)

        #feature selection
        X_test_processed = FeatureEngineering_instance.select_k_features(
            X_test_processed, y_test, is_training=False
        )

        #load best model from mlflow
        model_path = config['model']['best_model_path']
        best_model = joblib.load(model_path)
        y_pred = best_model.predict(X_test_processed)

        # evaluate model and log metrics/curves via ModelTrainer
        ModelTrainer_instance = ModelTrainer(config_path)
        ModelTrainer_instance.evaluate_model(best_model, X_test_processed, y_test)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error("="*50)
        raise

if __name__ == "__main__":
    main()