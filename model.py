import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model safely
model_path = "model.pkl"
with open(model_path, "rb") as rf:
    model = pickle.load(rf)

# Define input structure (matches BankNote Authentication features)
class ModelInput(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

@app.get("/")
def home():
    return {"message": "Model API is running successfully!"}

@app.post("/predict")
def predict(data: ModelInput):
    input_features = [[data.variance, data.skewness, data.curtosis, data.entropy]]
    prediction = model.predict(input_features)
    return {"prediction": int(prediction[0])}
