from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple, Any

from data import FeatureEngineer, DataSplitter


class BaseModel(ABC):
    is_implemented = False

    def __init__(self, model_name: str, data_config: dict, models_config: dict, target_config: dict):
        self.model_name = model_name
        self.model_class_name = re.match(r'^([^_]+)', model_name).group(1)
        self.models_config = models_config
        self.target_config = target_config
        self.model = None
        self.target_column = target_config['name']
        self.feature_columns = FeatureEngineer.get_feature_columns(self.get_model_type())
        self.data_splitter = DataSplitter(data_config)

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

    def prepare_data_for_dates(self, df: pd.DataFrame, start_date: str, end_date: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for specific date range - override in models with constraints"""
        # Filter data to date range first
        filtered_df = self.data_splitter.filter_data_by_dates(df, start_date, end_date)

        # Then prepare data normally
        return self.prepare_data(filtered_df)

    def get_available_dates(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DatetimeIndex:
        """Return dates this model can actually predict for in given range"""
        # Default: all dates in range (override in sequence models)
        filtered_df = self.data_splitter.filter_data_by_dates(df, start_date, end_date)
        X, y = self.prepare_data(filtered_df)

        # Get the dates that have valid data after prepare_data cleaning
        data_with_target = filtered_df[self.feature_columns + [self.target_column]].copy()
        clean_data = data_with_target.dropna()

        return clean_data.index

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
        """Date-based train/val/test split using fixed boundaries"""
        masks = self.data_splitter.get_date_masks(dates)

        return {
            'X_train': X[masks['train']],
            'y_train': y[masks['train']],
            'X_val': X[masks['val']],
            'y_val': y[masks['val']],
            'X_test': X[masks['test']],
            'y_test': y[masks['test']],
            'dates_train': dates[masks['train']],
            'dates_val': dates[masks['val']],
            'dates_test': dates[masks['test']],
            'idx_test': np.where(masks['test'])[0]
        }
