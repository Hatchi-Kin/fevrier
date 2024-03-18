import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel, Field

# import boto3


tags = [
    {"name": "help", "description": "Loupiotte Verte !"},
    {"name": "RandomForest", "description": "Utilise Random Forest le plus à jour"},
    {
        "name": "GradientBoosting",
        "description": "Utilise Gradient Boosting le plus à jour",
    },
]


class QueryFormat(BaseModel):
    PassengerId: int = Field(..., example=892)
    Pclass: int = Field(..., example=3)
    Sex: int = Field(..., example=1)
    Age: float = Field(..., example=34.5)
    SibSp: int = Field(..., example=0)
    Parch: int = Field(..., example=0)
    Fare: float = Field(..., example=7.8292)
    Embarked: int = Field(..., example=1)


app = FastAPI(
    title="API de prediction Titanic / Kaggle",
    description="Predictions",
    version="1.0.0",
    openapi_tags=tags,
)


@app.get("/", tags=["help"])
def get_help():
    """
    Endpoint returning a welcome message.
    """
    return {"message": "Welcome to the API"}


@app.post("/predict", tags=["Predict with RandomForest"])
def predict(credit: QueryFormat):
    """
    Endpoint for making predictions with the first model.
    """
    try:
        with open("random_forest.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return {"error": "Model file not found"}

    try:
        data = pd.DataFrame([credit.model_dump()])
        prediction = model.predict(data)

        # Ensure prediction is a JSON serializable type
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict2", tags=["Predict with GradientBoosting"])
def predict_2(credit: QueryFormat):
    """
    Endpoint for making predictions with the second model.
    """
    try:
        with open("gradient_boosting.pkl", "rb") as file:
            model = pickle.load(file)
    except FileNotFoundError:
        return {"error": "Model file not found"}

    try:
        data = pd.DataFrame([credit.dict()])
        prediction = model.predict(data)

        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()

        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
