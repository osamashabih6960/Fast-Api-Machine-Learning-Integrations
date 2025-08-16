from fastapi import FastAPI
from schemas import InputSchema, OutputSchema
from predict import make_prediction, make_batch_prediction
from typing import List


app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Welcome to the Model Prediciton API'}


@app.post('/prediction', response_model=OutputSchema)
def predict(user_input, InputSchema):
    prediciton = make_prediction(user_input.model_dump())
    return OutputSchema(predicted_price=prediciton)


@app.post('/batch_prediciton', response_model=List[OutputSchema])
def batch_predict(user_inputs: List[InputSchema]):
    predicition = make_batch_prediction([X.model_dump() for X in user_inputs])
    return [OutputSchema(predicted_price=(predicition,2)) for prediction in predicition]