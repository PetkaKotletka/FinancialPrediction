from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional


class BaseModel(ABC):
    def __init__(self, config: dict, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model = None
        self.feature_columns = None
        self.target_column = None
        self.is_pretrained = False

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """Transform dataframe into model-specific format"""
        pass

    @abstractmethod
    def build_model(self, input_shape: tuple):
        """Build the model architecture"""
        pass

    def load_pretrained(self, weights_path: str, fine_tune: bool = True):
        """Load pre-trained weights"""
        self.is_pretrained = True
        # TODO: Implementation depends on model type
        # For Keras: model.load_weights()
        # For sklearn: joblib.load()
        # Set fine_tune flag to freeze/unfreeze layers
        pass

    def load_pretrained_from_web(self):
        """Load pre-trained weights from internet (override in subclasses if available)"""
        raise NotImplementedError(f"No pre-trained weights available for {self.model_name}")

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, float]:
        """Train the model and return training history"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass

    def fine_tune(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None,
                  learning_rate: float = 0.0001) -> Dict[str, float]:
        """Fine-tune pre-trained model with lower learning rate"""
        # TODO: Implement fine-tuning logic
        # Lower learning rate, fewer epochs, freeze early layers
        pass

    def split_data(self, X: np.ndarray, y: np.ndarray, dates: pd.DatetimeIndex) -> Dict:
        """Time-based train/val/test split"""
        n = len(X)
        train_idx = int(n * self.config['models']['train_split'])
        val_idx = int(n * (self.config['models']['train_split'] + self.config['models']['val_split']))

        return {
            'X_train': X[:train_idx],
            'y_train': y[:train_idx],
            'X_val': X[train_idx:val_idx],
            'y_val': y[train_idx:val_idx],
            'X_test': X[val_idx:],
            'y_test': y[val_idx:],
            'dates_train': dates[:train_idx],
            'dates_val': dates[train_idx:val_idx],
            'dates_test': dates[val_idx:]
        }
