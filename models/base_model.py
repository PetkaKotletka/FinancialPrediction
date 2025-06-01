from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any


class BaseModel(ABC):
    def __init__(self, model_name: str, target_config: Dict[str, Any]):
        self.model_type = 'tabular'  # tabular, sequence, cnn
        self.model_name = model_name
        self.target_config = target_config
        self.model = None
        self.feature_columns = None
        self.target_column = None

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Transform dataframe into model-specific format"""
        pass

    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train the model and return training history"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def split_data(self, models_config: Dict[str, float], X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
        """Time-based train/val/test split"""
        n = len(X)
        train_idx = int(n * models_config['train_split'])
        val_idx = int(n * (models_config['train_split'] + models_config['val_split']))

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

    def set_model_type_features(self, df: pd.DataFrame):
        """Prepare features for specific model types"""
        if self.model_type == 'tabular':
            # Extract feature columns for tabular models
            technical_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                                  'BB_upper', 'BB_middle', 'BB_lower',
                                  'SMA_short', 'SMA_long', 'OBV']

            price_features = ['returns', 'log_returns', 'volume_ratio']

            economic_features = ['VIXCLS', 'GDP', 'T10YIE']

            # Get sector columns (one-hot encoded)
            sector_features = [col for col in df.columns if col.startswith('sector_')]

            self.feature_columns = (
                technical_features + price_features +
                economic_features + sector_features
            )

        elif self.model_type == 'sequence':
            # TODO: Implement sequence preparation for RNNs
            pass

        elif self.model_type == 'cnn':
            # TODO: Implement 2D representation for CNN
            pass

        elif self.model_type == 'arima':
            # TODO: Implement price series extraction for ARIMA
            pass

        else:
            raise ValueError(f"Unknown model type: {model_type}")
