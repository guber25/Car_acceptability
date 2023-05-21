import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Car CLassification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('C:/Users/gugli/Desktop/Università/Hackathon/Lab/notebooks/Miei/model.pkl', 'rb')
)


@app.get("/")
def read_root(text: str = ""):
    if not text:
        return f"Try to append ?text=something in the URL!"
    else:
        return text


class Car(BaseModel):
    Buying_Price: int
    Maintenance_Price: int
    No_of_Doors: int
    Person_Capacity: int
    Size_of_Luggage: int
    Safety: int


@app.post("/predict/")
def predict(cars: List[Car]) -> List[str]:
    X = pd.DataFrame([dict(car) for car in cars])
    y_pred = model.predict(X)
    return list(y_pred)
