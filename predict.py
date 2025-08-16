import joblib
import numpy as np
from typing import List, saved_model

save_model = joblib.load('model.joblib')
print('Loaded the Model')

def make_prediction(data: dict) -> float:
    features = np.array([ 
        [
            data['longitude'],
            data['latitude'],
            data['housing_median_age'],
            data['total_rooms'],
            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income']
        ]
    ])
    return save_model.predict(features)[0]

def make_batch_prediction(data: List[dict]) -> np.array:
    X  =  np.array([ 
        [
            X['longitude'],
            X['latitude'],
            X['housing_median_age'],
            X['total_rooms'],
            X['total_bedrooms'],
            X['population'],
            X['households'],
            X['median_income']
        ]
        for  X in data
    ])
    return saved_model.predict(X)
    
    