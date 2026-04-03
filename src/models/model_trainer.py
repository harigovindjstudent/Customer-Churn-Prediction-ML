import yaml
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def train_model(self, X_train, X_test, y_train, y_test):

        try:
            #parent experiment
            mlflow.set_experiment("customer_churn_prediction")
            
            best_model = None
            best_score = 0

            for algorithm in self.config['model']['algorithms']:
                #nested = True allows us to track each algorithm's performance separately under the same parent run
                with mlflow.start_run(run_name=algorithm['name'], nested=True):
                    if algorithm['name'] == 'logistic_regression':
                        model = LogisticRegression(**algorithm['parameters'])
                    elif algorithm['name'] == 'random_forest':
                        model = RandomForestClassifier(**algorithm['parameters'])
                    elif algorithm['name'] == 'xgboost':
                        model = xgb.XGBClassifier(**algorithm['parameters'])
                    else:
                        logger.warning(f"Unsupported algorithm: {algorithm['name']}")
                        continue

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                    mlflow.log_metric(f"{algorithm['name']}_accuracy", accuracy)
                    mlflow.log_metric(f"{algorithm['name']}_precision", precision)
                    mlflow.log_metric(f"{algorithm['name']}_recall", recall)
                    mlflow.log_metric(f"{algorithm['name']}_f1_score", f1)
                    mlflow.log_metric(f"{algorithm['name']}_roc_auc", roc_auc)

                    logger.info(f"{algorithm['name']} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")

                if f1 > best_score:
                    best_score = f1
                    best_model = model

            self._save_model(best_model)
            return best_model, best_score    

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def _save_model(self, model):
        import os
        model_path = self.config['model']['best_model_path']
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")