from .base_model import BaseModel
import numpy as np


class LinearModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config, 'linear_regression')

    def prepare_data(self, df, target):
        # TODO: Select relevant features and create feature matrix
        pass

    def build_model(self, input_shape):
        # TODO: Initialize sklearn LinearRegression or similar
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Fit the model
        pass

    def predict(self, X):
        # TODO: Make predictions
        pass
