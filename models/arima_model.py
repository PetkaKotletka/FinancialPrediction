from .base_model import BaseModel


class ARIMAModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config, 'arima')

    def prepare_data(self, df, target):
        # TODO: Extract price series for ARIMA
        pass

    def build_model(self, input_shape):
        # TODO: Initialize ARIMA with parameters
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Fit ARIMA model
        pass

    def predict(self, X):
        # TODO: Forecast using ARIMA
        pass
