from .base_model import BaseModel


class CNNModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config, 'cnn')

    def prepare_data(self, df, target):
        # TODO: Transform time series to 2D representation
        pass

    def build_model(self, input_shape):
        # TODO: Build CNN architecture
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Train CNN
        pass

    def predict(self, X):
        # TODO: CNN predictions
        pass
