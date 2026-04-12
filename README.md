# Customer Churn Prediction

## Problem
The goal of this project is to predict customer churn for a bank. By analyzing customer demographics and account information, the system predicts whether a customer is likely to close their account.

## What Was Built
A structured machine learning pipeline. The project features:
- Data preprocessing, synthetic feature creation, and SMOTE for class imbalance.
- Automated hyperparameter tuning using Optuna.
- Experiment tracking and metric logging using MLflow.
- An ensemble model (Voting Classifier) combining Logistic Regression, Random Forest, and XGBoost.
- A REST API built with FastAPI to serve real-time predictions based on the trained model.

## How to Run it

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. To train the models and find the best ensemble:
   ```bash
   python project_train.py
   ```
3. To evaluate the saved model on the test dataset:
   ```bash
   python project_evaluate.py
   ```
4. To start the REST API server:
   ```bash
   python app.py
   ```

## API Usage
The project includes a FastAPI web server to make model predictions via HTTP POST requests.

Endpoint: `POST /predict`

Example cURL Request:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '[
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
        ]'
```

Example Response:
```json
{
  "predictions": [0]
}
```
A prediction of `1` indicates the customer is likely to churn, while `0` indicates they will remain.
