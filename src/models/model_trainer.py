import yaml
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import logging
import optuna
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def train_model(self, X_train, X_val, y_train, y_val):

        try:
            #parent experiment
            mlflow.set_experiment("customer_churn_prediction")
            
            best_model = None
            best_score = 0

            for algorithm in self.config['model']['algorithm']:
                #nested = True allows us to track each algorithm's performance separately under the same parent run
                with mlflow.start_run(run_name=algorithm['name'], nested=True):

                    def objective(trial):
                        if algorithm['name'] == 'logistic_regression':
                            C = trial.suggest_float("C", 0.01, 10.0, log=True)
                            model = LogisticRegression(C=C, max_iter=1000)
                        elif algorithm['name'] == 'random_forest':
                            n_estimators = trial.suggest_int("n_estimators", 50, 200)
                            max_depth = trial.suggest_int("max_depth", 3, 15)
                            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        elif algorithm['name'] == 'xgboost':
                            n_estimators = trial.suggest_int("n_estimators", 50, 200)
                            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
                            max_depth = trial.suggest_int("max_depth", 3, 10)
                            model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
                        else:
                            return 0.0
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        f1 = f1_score(y_val, y_pred)
                        return f1

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=15)

                    best_params = study.best_params
                    logger.info(f"Best hyperparameters for {algorithm['name']}: {best_params}")

                    # Train model with best hyperparameters
                    if algorithm['name'] == 'logistic_regression':
                        model = LogisticRegression(**best_params, max_iter=1000)
                    elif algorithm['name'] == 'random_forest':
                        model = RandomForestClassifier(**best_params, random_state=42)
                    elif algorithm['name'] == 'xgboost':
                        model = xgb.XGBClassifier(**best_params, random_state=42)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)

                    accuracy = accuracy_score(y_val, y_pred)
                    precision = precision_score(y_val, y_pred)
                    recall = recall_score(y_val, y_pred)
                    f1 = f1_score(y_val, y_pred)
                    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

                    mlflow.log_metric(f"{algorithm['name']}_accuracy", accuracy)
                    mlflow.log_metric(f"{algorithm['name']}_precision", precision)
                    mlflow.log_metric(f"{algorithm['name']}_recall", recall)
                    mlflow.log_metric(f"{algorithm['name']}_f1_score", f1)
                    mlflow.log_metric(f"{algorithm['name']}_roc_auc", roc_auc)

                    logger.info(f"{algorithm['name']} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
                    logger.info("="*50)

                if f1 > best_score:
                    best_score = f1
                    best_model = model

            self._save_model(best_model)
            return best_model, best_score    

        except Exception as e:
            logger.error(f"Error training model: {e}")
            logger.error("="*50)
            raise

    def evaluate_model(self, model, X_test, y_test):
        try:
            mlflow.set_experiment("customer_churn_prediction")
            with mlflow.start_run(run_name="final_model_evaluation", nested=True):
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1_score", f1)
                mlflow.log_metric("test_roc_auc", roc_auc)

                logger.info(f"Final Model Performance on Test Set:")
                logger.info(f"Accuracy: {accuracy}")
                logger.info(f"Precision: {precision}")
                logger.info(f"Recall: {recall}")
                logger.info(f"F1-Score: {f1}")
                logger.info(f"AUC-ROC: {roc_auc}")
                logger.info("="*50)

                # Generate and log ROC Curve
                fig_roc, ax_roc = plt.subplots()
                RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
                ax_roc.set_title("ROC Curve")
                mlflow.log_figure(fig_roc, "roc_curve.png")
                plt.close(fig_roc)
                
                # Generate and log Precision-Recall Curve
                fig_pr, ax_pr = plt.subplots()
                PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr)
                ax_pr.set_title("Precision-Recall Curve")
                mlflow.log_figure(fig_pr, "pr_curve.png")
                plt.close(fig_pr)

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            logger.error("="*50)
            raise

    def _save_model(self, model):
        import os
        model_path = self.config['model']['best_model_path']
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        logger.info("="*50)