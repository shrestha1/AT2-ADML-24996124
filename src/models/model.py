import pandas as pd
import numpy as np

class NullRegressor:
    """
    Class used as baseline model for regression problem
    ...
    Args:
        
        y : Numpy Array-like
            Target variable
        pred_value : Float
            Value to be used for prediction
        preds : Numpy Array
            Predicted array

    Methods:
        
        fit(y)
            Store the input target variable and calculate the predicted value to be used
        predict(y)
            Generate the predictions
        fit_predict(y)
            Perform a fit followed by predict
    """


    def __init__(self):
        self.y = None
        self.pred_value = None
        self.preds = None

    def fit(self, y):
        self.y = y
        self.pred_value = y.mean()

    def predict(self, y):
        self.preds = np.full((len(y), 1), self.pred_value)
        return self.preds

    def fit_predict(self, y):
        self.fit(y)
        return self.predict(self.y)

class BaseLineRegressor:
    """
    To define any desired model to act as baseline for the evaluation process
    Args:
        
        model: any machine learning model object

    Methods:
        
        fit(y)
            train the model

        predict(y)
            Generate the predictions
        
        fit_predict(y)
            Perform a train followed by predict

    """
    def __init__(self, model) -> None:
        self.model = model

    def fit(self, X, y):
        self.model.fit(X,y)

    def predict(self, y):
        return self.model.predict(y)
        

    