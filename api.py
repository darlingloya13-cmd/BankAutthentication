from typing import List
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Bank Note Authentication API",
    description="API for authenticating bank notes using a trained machine learning model",
    version="1.0.0"
)

# Load the model directly using pickle
with open("model.pkl", "rb") as rf:
    model = pickle.load(rf)

# Request schema for multiple rows of data
class PredictAuthenticationRequest(BaseModel):
    data: List[List[float]]

# Response schema to return the predictions
class PredictAuthenticationResponse(BaseModel):
    data: List[int]

@app.post("/predict", response_model=PredictAuthenticationResponse, tags=['prediction'])
async def predict_authentication(request: PredictAuthenticationRequest):
    x = np.array(request.data)
    y_pred = model.predict(x)
    return PredictAuthenticationResponse(data=y_pred.tolist())

@app.get('/healthcheck', status_code=200, tags=['healthcheck'])
async def healthcheck():
    return {"message": "Bank Note Authentication API is running successfully!"}
