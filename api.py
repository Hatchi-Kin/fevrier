# Import des librairies uvicorn, pickle, FastAPI, File, UploadFile, BaseModel
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from pydantic import BaseModel
import pickle
import pandas as pd

import mlflow
import os
import boto3


# Création des tags
tags = [
    {
        "name": "Hello",
        "description": "Greeting service",
    },
    {
        "name": "Predict V1",
        "description": "Prediction service for model  1",
    },
    {
        "name": "Predict V2",
        "description": "Prediction service for model  2",
    }
]

# Création de l'application
app = FastAPI(
    title="API de prediction",
    description="Predictions",
    version="1.0.0",
    openapi_tags=tags,
)


# Point de terminaison avec paramètre
@app.get("/hello", tags=["Hello name V1"])
def hello(name: str = "World"):
    return {"message": f"Hello {name}"}


# Création du modèle de données pour le modéle 1 ('Gender', 'Age', 'Physical Activity Level', 'Heart Rate', 'Daily Steps', 'BloodPressure_high', 'BloodPressure_low', 'Sleep Disorder'])
class Credit(BaseModel):
    Gender: int
    Age: int
    Physical_Activity_Level: int
    Heart_Rate: int
    Daily_Steps: int
    BloodPressure_high: int
    BloodPressure_low: int


# # Point de terminaison : Prédiction 1
@app.post("/predict", tags=["Predict V1"])
def predict(credit: Credit):
    try:
        with open("model_1.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return {"error": "Model file not found"}
# commentaire
    try:
        # No need to unpack 'credit' here
        data = pd.DataFrame([credit.model_dump()])
        rename_dict = {
            "Daily_Steps": "Daily Steps",
            "Heart_Rate": "Heart Rate",
            "Physical_Activity_Level": "Physical Activity Level",
        }
        data.rename(columns=rename_dict, inplace=True)
        prediction = model.predict(data)

        # Ensure prediction is a JSON serializable type
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


# Création du modèle de données pour le modéle 2 ('Physical Activity Level', 'Heart Rate', 'Daily Steps', 'Sleep Disorder')
class Credit2(BaseModel):
    Physical_Activity_Level: int
    Heart_Rate: int
    Daily_Steps: int


# Point de terminaison : Prédiction 2
@app.post("/predict2", tags=["Predict V2"])
def predict_2(credit: Credit2):
    try:
        with open("model_2.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return {"error": "Model file not found"}

    try:
        # No need to unpack 'credit' here
        data = pd.DataFrame([credit.dict()])
        rename_dict = {
            "Daily_Steps": "Daily Steps",
            "Heart_Rate": "Heart Rate",
            "Physical_Activity_Level": "Physical Activity Level",
        }
        data.rename(columns=rename_dict, inplace=True)
        prediction = model.predict(data)

        # Ensure prediction is a JSON serializable type
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


# Démarage de l'application
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
