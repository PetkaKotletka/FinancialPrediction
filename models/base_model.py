from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

from data import FeatureEngineer


class BaseModel(ABC):
    is_implemented = False

    def __init__(self, model_name: str, models_config: dict, target_config: dict):
        self.model_name = model_name
        self.models_config = models_config
        self.target_config = target_config
        self.model = None
        self.target_column = target_config['name']
        self.feature_columns = FeatureEngineer.get_feature_columns(self.get_model_type())

    def get_model_type(self):
        """Get model type (tabular, sequence, cnn, arima)"""
        return 'tabular'

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform dataframe into model-specific format"""

        # Combine with target and remove NaN rows
        data_with_target = df[self.feature_columns + [self.target_column]].copy()
        data_clean = data_with_target.dropna()

        # Extract X and y
        X = data_clean[self.feature_columns].values
        y = data_clean[self.target_column].values

        return X, y

    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Train the model and return training history"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def split_data(self, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
        """Time-based train/val/test split"""
        n = len(X)
        train_idx = int(n * self.models_config['train_split'])
        val_idx = int(n * (self.models_config['train_split'] + self.models_config['val_split']))

        return {
            'X_train': X[:train_idx],
            'y_train': y[:train_idx],
            'X_val': X[train_idx:val_idx],
            'y_val': y[train_idx:val_idx],
            'X_test': X[val_idx:],
            'y_test': y[val_idx:],
            'dates_train': dates[:train_idx],
            'dates_val': dates[train_idx:val_idx],
            'dates_test': dates[val_idx:],
            'idx_test': np.arange(val_idx, n)
        }
