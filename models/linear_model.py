from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


class LinearModel(BaseModel):
    is_implemented = True

    def __init__(self, model_name: str, data_config: dict, models_config: dict, target_config: dict):
        super().__init__(model_name, data_config, models_config, target_config)
        self.scaler = StandardScaler()

    def build_model(self):
        """Initialize sklearn linear model based on task type"""
        if self.target_config['type'] == 'regression':
            self.model = LinearRegression()
        else:  # classification
            self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> dict:
        """Train the linear model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Fit model
        self.model.fit(X_train_scaled, y_train)

        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)

        history = {}
        if self.target_config['type'] == 'regression':
            history['train_rmse'] = np.sqrt(np.mean((y_train - train_pred) ** 2))
            history['val_rmse'] = np.sqrt(np.mean((y_val - val_pred) ** 2))
        else:
            history['train_accuracy'] = np.mean(y_train == train_pred)
            history['val_accuracy'] = np.mean(y_val == val_pred)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with scaling"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
