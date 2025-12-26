from pyexpat import features
from fastapi import FastAPI
from typing import Union 
from pydantic import BaseModel
import joblib


app = FastAPI()

model = joblib.load("finale_into_final.model.pkl")
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float
    feature6: float
    feature7: float
    feature8: float
    feature9: float
    feature10: float
    feature11: float
    feature12: float
    feature13: float
    feature14: float
    feature15: float
    feature16: float
    feature17: float
    feature18: float
    feature19: float
    feature20: float
    feature21: float
    feature22: float
    feature23: float
    feature24: float
    feature25: float
    feature26: float
    feature27: float
    feature28: float
    feature29: float
    feature30: float
    feature31: float
    feature32: float
    feature33: float
    feature34: float
    feature35: float
    feature36: float
    feature37: float
    feature38: float
    feature39: float
    feature40: float
    feature41: float
    feature42: float
    feature43: float
    feature44: float
    feature45: float
    feature46: float
    feature47: float
    feature48: float
    feature49: float
    feature50: float
    feature51: float
    feature52: float
    feature53: float
    feature54: float


@app.get("/")
def read_root():
    return {"message": "Welcome to My Model"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.feature1]])[0]
    return{"prediction": prediction}
    