from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config, 'decision_tree')

    def prepare_data(self, df, target):
        # TODO: Prepare tabular features
        pass

    def build_model(self, input_shape):
        # TODO: Initialize sklearn DecisionTreeRegressor/Classifier
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Fit decision tree
        pass

    def predict(self, X):
        # TODO: Tree predictions
        pass


class XGBoostModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__(config, 'xgboost')

    def prepare_data(self, df, target):
        # TODO: Prepare tabular features
        pass

    def build_model(self, input_shape):
        # TODO: Initialize XGBoost
        pass

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # TODO: Train with early stopping
        pass

    def predict(self, X):
        # TODO: XGBoost predictions
        pass
