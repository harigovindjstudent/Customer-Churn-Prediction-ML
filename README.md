# Customer Churn Prediction
The project predicts customer churn for a bank. By analyzing customer demographics and account information, the system predicts whether a customer is likely to close their account.

## What Was Built
A structured machine learning pipeline. The project features:
- Data preprocessing, synthetic feature creation, and SMOTE for class imbalance.
- Feature Selection
- Hyperparameter tuning using Optuna.
- Experiment tracking and metric logging using MLflow. 
- An ensemble model (Voting Classifier) combining Logistic Regression, Random Forest, and XGBoost.
- A REST API built with FastAPI to give predictions based on the trained model.

## How to Run it

1. Install the required dependencies (pandas, scikit-learn, xgboost, imblearn, mlflow, optuna, fastapi, uvicorn, pyyaml).
2. To train the models and find the best ensemble: **python project_train.py**

3. To evaluate the saved model on the test dataset: **python project_evaluate.py**

## API Usage
The project includes a FastAPI web server to make model predictions via HTTP POST requests.

Endpoint: `POST /predict`

Example request body:
```
[
    {
    "RowNumber": 6253,
    "CustomerId": 15687492,
    "Surname": "Anderson",
    "CreditScore": 596,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 32,
    "Tenure": 3,
    "Balance": 96709.07,
    "NumOfProducts": 2,
    "HasCrCard": 0,
    "IsActiveMember": 0,
    "EstimatedSalary": 41788.37
    }
]
```

Example Response:
```
{
  "predictions": [0]
}
```
A prediction of `1` indicates the customer is likely to churn, while `0` indicates they will remain.
