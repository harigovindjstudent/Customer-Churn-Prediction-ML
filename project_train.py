import logging
import yaml
import mlflow
import os
import joblib
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
        X_train, X_val, X_test, y_train, y_val, y_test = DataLoader_instance.split_data(df)

        os.makedirs('data/processed', exist_ok=True)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        y_test.to_csv('data/processed/y_test.csv', index=False)
        logger.info("Test set saved for evaluate.py")        

        #feature creation
        X_train = FeatureEngineering_instance.create_features(X_train)
        X_val = FeatureEngineering_instance.create_features(X_val)

        #feature processing
        X_train_processed = FeatureEngineering_instance.process_features(X_train, is_training=True)
        X_val_processed = FeatureEngineering_instance.process_features(X_val, is_training=False)
        
        #feature selection
        X_train_processed = FeatureEngineering_instance.select_k_features(
            X_train_processed, y_train, is_training=True, k=10
        )
        X_val_processed = FeatureEngineering_instance.select_k_features(
            X_val_processed, is_training=False
        )

        #handle class imbalance on train data
        X_train_processed, y_train = FeatureEngineering_instance.smote(X_train_processed, y_train)

        #saving preprocessing pipeline
        FeatureEngineering_instance.save_pipeline('models/preprocessing_pipeline.joblib', 'models/selector.joblib')

        #model training
        best_model, best_score, best_threshold = ModelTrainer_instance.train_model(X_train_processed, X_val_processed, y_train, y_val)

        # Save optimal threshold for evaluation
        joblib.dump(best_threshold, 'models/best_threshold.joblib')

        logger.info(f"Best model trained with F1 Score: {best_score}")
        logger.info("="*50)

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error("="*50)
        raise    

if __name__ == "__main__":
    main()