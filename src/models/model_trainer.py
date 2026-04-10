import yaml
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
            
            trained_models = {}

            for algorithm in self.config['model']['algorithm']:
                #nested = True allows us to track each algorithm's performance separately under the same parent run
                with mlflow.start_run(run_name=algorithm['name'], nested=True):

                    def objective(trial):
                        if algorithm['name'] == 'logistic_regression':
                            C = trial.suggest_float("C", 1e-3, 10.0, log=True)
                            penalty = trial.suggest_categorical("penalty", ["l2"])
                            solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])

                            model = LogisticRegression(
                                C=C,
                                penalty=penalty,
                                solver=solver,
                                max_iter=1000
                            )

                        elif algorithm['name'] == 'random_forest':
                            n_estimators = trial.suggest_int("n_estimators", 10, 300)
                            max_depth = trial.suggest_int("max_depth", 3, 20)
                            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
                            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
                            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
                            bootstrap = trial.suggest_categorical("bootstrap", [True, False])

                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                max_features=max_features,
                                bootstrap=bootstrap,
                                random_state=42,
                                n_jobs=-1
                            )

                        elif algorithm['name'] == 'xgboost':
                            n_estimators = trial.suggest_int("n_estimators", 100, 300)
                            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                            max_depth = trial.suggest_int("max_depth", 3, 10)
                            subsample = trial.suggest_float("subsample", 0.6, 1.0)
                            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
                            gamma = trial.suggest_float("gamma", 0, 5)
                            reg_alpha = trial.suggest_float("reg_alpha", 0, 5)
                            reg_lambda = trial.suggest_float("reg_lambda", 0, 5)

                            model = xgb.XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                subsample=subsample,
                                colsample_bytree=colsample_bytree,
                                gamma=gamma,
                                reg_alpha=reg_alpha,
                                reg_lambda=reg_lambda,
                                random_state=42,
                                use_label_encoder=False,
                                eval_metric="logloss",
                                n_jobs=-1
                            )

                        else:
                            return 0.0
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        f1 = f1_score(y_val, y_pred)
                        return f1

                    study = optuna.create_study(direction="maximize")
                    study.optimize(objective, n_trials=300)

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
                    
                    trained_models[algorithm['name']] = model

            # Train the ensemble model
            with mlflow.start_run(run_name='ensemble', nested=True):
                ensemble_model = VotingClassifier(
                    estimators=[
                        ('xgboost', trained_models['xgboost']),
                        ('random_forest', trained_models['random_forest']),
                        ('logistic_regression', trained_models['logistic_regression'])
                    ],
                    voting='soft'
                )
                
                ensemble_model.fit(X_train, y_train)
                y_pred = ensemble_model.predict(X_val)

                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)
                roc_auc = roc_auc_score(y_val, ensemble_model.predict_proba(X_val)[:, 1])

                mlflow.log_metric("ensemble_accuracy", accuracy)
                mlflow.log_metric("ensemble_precision", precision)
                mlflow.log_metric("ensemble_recall", recall)
                mlflow.log_metric("ensemble_f1_score", f1)
                mlflow.log_metric("ensemble_roc_auc", roc_auc)

                logger.info(f"Ensemble Model - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, ROC AUC: {roc_auc}")
                logger.info("="*50)

                best_model = ensemble_model
                best_score = f1

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