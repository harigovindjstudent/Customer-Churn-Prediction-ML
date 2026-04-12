import os
import pandas as pd
import joblib
import yaml
import uvicorn
from src.features.feature_engineering import FeatureEngineering
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, HTTPException



app = FastAPI(title="Customer Churn Prediction API")

config_path = 'config/config.yaml'
try:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    raise RuntimeError(f"config file not found at: {config_path}")

try:
    FeatureEngineering_intance = FeatureEngineering(config_path)
    FeatureEngineering_intance.load_pipeline('models/preprocessing_pipeline.joblib', 
                                             'models/selector.joblib')
    best_model = joblib.load("models/best_model.pkl")
except Exception as e:
    print(f"Could not load model or pipeline: {e}")
    best_model = None
    FeatureEngineering_intance = None


class InputFeatures(BaseModel):
    RowNumber: Optional[int] = 0
    CustomerId: Optional[int] = 0
    Surname: Optional[str] = "Unknown"
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

class Output(BaseModel):
    predictions: List[int]


@app.post("/predict", response_model=Output)
def predict_churn(features_list: List[InputFeatures]):
    """
    Endpoint to predict customer churn.
    Accepts a list of JSON objects containing customer features.
    Returns a list of predictions (1 for churn, 0 for no churn).
    """
    if best_model is None or FeatureEngineering_intance is None:
        raise HTTPException(status_code=500, detail="Models not properly loaded.")
    
    try:
        #the model_dump thing converts input to python dict
        data = [feat.model_dump() for feat in features_list]
        df = pd.DataFrame(data)

        #prediction
        df_engineered = FeatureEngineering_intance.create_features(df)
        df_processed = FeatureEngineering_intance.process_features(df_engineered, 
                                                                   is_training=False)
        df_selected = FeatureEngineering_intance.select_k_features(df_processed, 
                                                                   is_training=False)
        
        predictions = best_model.predict(df_selected)
        
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API. Use the /predict endpoint to get scores."}


if __name__ == "__main__":
    # You can run this file directly with `python app.py`
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
