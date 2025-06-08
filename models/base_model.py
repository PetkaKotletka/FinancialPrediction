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
        self.data_config = data_config
        self.models_config = models_config
        self.target_config = target_config
        self.model = None
        self.target_column = target_config['name']
        self.all_sectors = list(set(data_config['tickers'].values()))
        self.feature_columns = FeatureEngineer.get_feature_columns(
            self.get_model_type(),
            self.all_sectors
        )
        self.data_splitter = DataSplitter(data_config)

    def get_model_type(self):
        """Get model type (tabular, sequence, cnn, arima)"""
        return 'tabular'

    def prepare_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        """Universal data preparation - all models get same date coverage"""
        if self.get_model_type() == 'arima':
            # ARIMA: store per-ticker data
            self.ticker_data = {}
            for ticker, df in data_dict.items():
                clean_data = df[['Close', self.target_column]].dropna()
                self.ticker_data[ticker] = {
                    'prices': clean_data['Close'].values,
                    'returns': clean_data[self.target_column].values
                }
            # Return dummy data for interface compatibility
            return np.array([]), np.array([])

        elif self.get_model_type() in ['sequence', 'cnn']:
            # Sequence models: create windows per ticker
            sequences, targets = [], []
            for ticker, df in data_dict.items():
                clean_data = df[self.feature_columns + [self.target_column]].dropna()
                if len(clean_data) >= self.window_size:
                    # Create sequences - targets cover ALL available dates
                    for i in range(self.window_size, len(clean_data)):
                        seq = clean_data.iloc[i-self.window_size:i][self.feature_columns].values
                        target = clean_data.iloc[i][self.target_column]
                        sequences.append(seq)
                        targets.append(target)
            return np.array(sequences), np.array(targets)

        else:
            # Tabular models: concatenate with sector encoding
            all_X, all_y = [], []
            for ticker, df in data_dict.items():
                df_copy = df.copy()
                df_copy['sector'] = self.data_config['tickers'].get(ticker, 'Unknown')

                # Create all sector columns for this ticker
                for sector in self.all_sectors:
                    df_copy[f'sector_{sector}'] = 1 if df_copy['sector'].iloc[0] == sector else 0

                feature_data = df_copy[self.feature_columns]
                clean_data = pd.concat([feature_data, df_copy[[self.target_column]]], axis=1).dropna()

                if len(clean_data) > 0:
                    all_X.append(clean_data.iloc[:, :-1].values)
                    all_y.append(clean_data[self.target_column].values)

            return np.vstack(all_X) if all_X else np.empty((0, len(self.feature_columns))), \
                   np.concatenate(all_y) if all_y else np.empty((0,))

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
