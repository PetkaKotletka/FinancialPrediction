from .base_model import BaseModel


class LSTMModel(BaseModel):
    def build_model(self, input_shape):
        # TODO: Build LSTM architecture
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Train with Keras/PyTorch
        pass

    def predict(self, X):
        # TODO: LSTM predictions
        pass


class GRUModel(BaseModel):
    def build_model(self, input_shape):
        # TODO: Build GRU architecture
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Train GRU
        pass

    def predict(self, X):
        # TODO: GRU predictions
        pass
