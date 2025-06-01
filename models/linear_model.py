from .base_model import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler


class LinearModel(BaseModel):
    def __init__(self, model_name: str, target_config: dict):
        super().__init__(model_name, target_config)
        self.model_type = 'tabular'
        self.scaler = StandardScaler()
        self.target_column = target_config['name']

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data using feature engineer"""

        self.set_model_type_features(df)

        # Combine with target and remove NaN rows
        data_with_target = df[self.feature_columns + [self.target_column]].copy()
        data_clean = data_with_target.dropna()

        # Extract X and y
        X = data_clean[self.feature_columns].values
        y = data_clean[self.target_column].values

        return X, y

    def build_model(self):
        """Initialize sklearn linear model based on task type"""
        if self.target_config['type'] == 'regression':
            self.model = LinearRegression()
        else:  # classification
            self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Train the linear model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit model
        self.model.fit(X_train_scaled, y_train)

        # Calculate training metrics
        train_pred = self.model.predict(X_train_scaled)

        history = {}
        if self.target_config['type'] == 'regression':
            history['train_rmse'] = np.sqrt(np.mean((y_train - train_pred) ** 2))
        else:
            history['train_accuracy'] = np.mean(y_train == train_pred)

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)

            if self.target_config['type'] == 'regression':
                history['val_rmse'] = np.sqrt(np.mean((y_val - val_pred) ** 2))
            else:
                history['val_accuracy'] = np.mean(y_val == val_pred)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with scaling"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
