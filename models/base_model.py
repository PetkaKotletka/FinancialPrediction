from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from typing import Dict, List, Tuple, Any


class BaseModel(ABC):
    is_implemented = False

    def __init__(self, model_name: str, config: dict, target_config: dict):
        self.model_name = model_name
        self.model_class_name = re.match(r'^([^_]+)', model_name).group(1)
        self.data_config = config['data']
        self.models_config = config['models']
        self.target_config = target_config
        self.target_column = target_config['name']
        self.all_sectors = list(set(self.data_config['tickers'].values()))
        # Calculated in prepare data
        self.categorical_vocab_sizes = None
        self.numerical_features = None
        self.categorical_features = None
        # Calculated in build model
        self.model = None


    def get_model_type(self):
        """Get model type (tabular, sequence, cnn, arima)"""
        return 'tabular'

    def prepare_data(self, data_dict: Dict[str, pd.DataFrame]):
        """Process and store data in model-specific format with consistent test dates"""

        # Get split dates from config
        train_start = pd.to_datetime(self.data_config['train_start'])
        train_end = pd.to_datetime(self.data_config['train_end'])
        val_start = pd.to_datetime(self.data_config['val_start'])
        val_end = pd.to_datetime(self.data_config['val_end'])
        test_start = pd.to_datetime(self.data_config['test_start'])
        test_end = pd.to_datetime(self.data_config['test_end'])

        if self.get_model_type() == 'arima':
            # ARIMA: store per-ticker price series
            self.ticker_data = {}
            multiindex_data = []
            all_targets = []

            for ticker, df in data_dict.items():
                # Filter by date periods
                train_data = df[train_start:train_end]
                test_data = df[test_start:test_end]

                self.ticker_data[ticker] = {
                    'train_prices': train_data['Adj Close'].values,
                    'test_prices': test_data['Adj Close'].values,
                    'train_targets': train_data[self.target_column].values,
                    'test_targets': test_data[self.target_column].values,
                    'test_dates': test_data.index
                }

                # Build MultiIndex data in same loop
                test_targets = test_data[self.target_column].values
                test_dates = test_data.index

                all_targets.extend(test_targets)
                multiindex_data.extend([(date, ticker) for date in test_dates])

            # Create MultiIndex Series
            multiindex = pd.MultiIndex.from_tuples(multiindex_data, names=['date', 'ticker'])
            self.y_test = pd.Series(all_targets, index=multiindex).sort_index()
        else:
            # Auto-select features
            sample_df = next(iter(data_dict.values()))
            exclude_cols = ['ticker', 'return_1d', 'direction_1d', 'return_5d']
            available_cols = [col for col in sample_df.columns if col not in exclude_cols]

            numerical_cols = [col for col in available_cols if sample_df[col].dtype != 'object']
            categorical_cols = [col for col in available_cols if sample_df[col].dtype == 'object'] # sector, volatility_regime

            # Process based on model type
            if self.get_model_type() in ['sequence', 'cnn']:
                self._prepare_sequence_data(data_dict, numerical_cols, categorical_cols)
            else:
                self._prepare_tabular_data(data_dict, numerical_cols, categorical_cols)

    def _prepare_tabular_data(self, data_dict, numerical_cols, categorical_cols):
        """Prepare tabular data with one-hot encoding"""

        # Get split dates from config
        train_start = pd.to_datetime(self.data_config['train_start'])
        train_end = pd.to_datetime(self.data_config['train_end'])
        val_start = pd.to_datetime(self.data_config['val_start'])
        val_end = pd.to_datetime(self.data_config['val_end'])
        test_start = pd.to_datetime(self.data_config['test_start'])
        test_end = pd.to_datetime(self.data_config['test_end'])

        # Collect all unique values for each categorical column
        categorical_mappings = {}
        for cat_col in categorical_cols:
            unique_values = set()
            for ticker, df in data_dict.items():
                unique_values.update(df[cat_col].dropna().unique())
            categorical_mappings[cat_col] = sorted(list(unique_values))

        all_train_X, all_train_y = [], []
        all_val_X, all_val_y = [], []
        all_test_X, all_test_y = [], []
        multiindex_data = []

        for ticker, df in data_dict.items():
            df_work = df.copy()

            # One-hot encode all categorical columns systematically
            for cat_col in categorical_cols:
                # Create dummy columns for all possible values
                for value in categorical_mappings[cat_col]:
                    df_work[f'{cat_col}_{value}'] = (df_work[cat_col] == value).astype(int)

            # Final features: numerical + all encoded categoricals
            encoded_cols = [f'{cat}_{val}' for cat in categorical_cols for val in categorical_mappings[cat]]
            final_features = numerical_cols + encoded_cols

            # Split by dates
            train_data = df_work[train_start:train_end][final_features + [self.target_column]]
            val_data = df_work[val_start:val_end][final_features + [self.target_column]]
            test_data = df_work[test_start:test_end][final_features + [self.target_column]]

            all_train_X.append(train_data[final_features].values)
            all_train_y.append(train_data[self.target_column].values)

            all_val_X.append(val_data[final_features].values)
            all_val_y.append(val_data[self.target_column].values)

            all_test_X.append(test_data[final_features].values)
            all_test_y.append(test_data[self.target_column].values)

            # Build MultiIndex data explicitly
            multiindex_data.extend([(date, ticker) for date in test_data.index])

        # Store processed data
        self.X_train = np.vstack(all_train_X)
        self.y_train = np.concatenate(all_train_y)
        self.X_val = np.vstack(all_val_X)
        self.y_val = np.concatenate(all_val_y)

        # Create MultiIndex
        multiindex = pd.MultiIndex.from_tuples(multiindex_data, names=['date', 'ticker'])

        self.X_test = np.vstack(all_test_X)
        self.y_test = pd.Series(np.concatenate(all_test_y), index=multiindex).sort_index()
        self.X_test_index = multiindex

    def _prepare_sequence_data(self, data_dict, numerical_cols, categorical_cols):
        """Prepare sequence data with label encoding for categorical features"""

        # Get split dates from config
        train_start = pd.to_datetime(self.data_config['train_start'])
        train_end = pd.to_datetime(self.data_config['train_end'])
        val_start = pd.to_datetime(self.data_config['val_start'])
        val_end = pd.to_datetime(self.data_config['val_end'])
        test_start = pd.to_datetime(self.data_config['test_start'])
        test_end = pd.to_datetime(self.data_config['test_end'])

        # Create label encodings for all categorical features
        categorical_mappings = {}

        for cat_col in categorical_cols:
            unique_values = set()
            for ticker, df in data_dict.items():
                unique_values.update(df[cat_col].dropna().unique())

            # Create label mapping (0-indexed for embeddings)
            sorted_values = sorted(list(unique_values))
            categorical_mappings[cat_col] = {val: idx for idx, val in enumerate(sorted_values)}

        # Store vocab sizes for use in model building
        self.categorical_vocab_sizes = {col: len(mapping) for col, mapping in categorical_mappings.items()}

        train_sequences, train_targets = [], []
        val_sequences, val_targets = [], []
        test_sequences, test_targets = [], []
        test_multiindex_data = []

        for ticker, df in data_dict.items():
            df_work = df.copy()

            # Label encode all categorical features
            for cat_col in categorical_cols:
                df_work[f'{cat_col}_encoded'] = df_work[cat_col].map(categorical_mappings[cat_col])

            # Final features: numerical + encoded categoricals
            encoded_categorical_cols = [f'{cat}_encoded' for cat in categorical_cols]
            sequence_features = numerical_cols + encoded_categorical_cols

            # Add buffer for windowing (based on window_size)
            buffer = pd.Timedelta(days=self.window_size * 1.5) # ~30 calendar days = ~22 trading days

            # For training: can't add buffer before start, use exact dates
            train_data = df_work[train_start:train_end]
            # For val/test: add buffer for windowing
            val_data = df_work[val_start - buffer:val_end]
            test_data = df_work[test_start - buffer:test_end]

            # Create sequences for each period
            for period_data, sequences_list, targets_list, start_date, end_date, is_test in [
                (train_data, train_sequences, train_targets, train_start, train_end, False),
                (val_data, val_sequences, val_targets, val_start, val_end, False),
                (test_data, test_sequences, test_targets, test_start, test_end, True)
            ]:
                clean_data = period_data[sequence_features + [self.target_column]]

                if len(clean_data) >= self.window_size:
                    for i in range(self.window_size, len(clean_data)):
                        target_date = clean_data.index[i]

                        # Only include if target is in actual period (not buffer)
                        if start_date <= target_date <= end_date:
                            seq = clean_data.iloc[i-self.window_size:i][sequence_features].values
                            sequences_list.append(seq)
                            targets_list.append(clean_data.iloc[i][self.target_column])
                            if is_test:
                                test_multiindex_data.append((target_date, ticker))

        # Store processed sequences
        self.X_train = np.array(train_sequences)
        self.y_train = np.array(train_targets)
        self.X_val = np.array(val_sequences)
        self.y_val = np.array(val_targets)

        multiindex = pd.MultiIndex.from_tuples(test_multiindex_data, names=['date', 'ticker'])
        self.X_test = np.array(test_sequences)
        self.y_test = pd.Series(test_targets, index=multiindex).sort_index()
        self.X_test_index = multiindex

        # Store features for model building
        self.numerical_features = numerical_cols
        self.categorical_features = encoded_categorical_cols

    @classmethod
    def compile_keras_model(cls, keras_model, target_type: str):
        """Compile the model"""
        if target_type == 'regression':
            keras_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.5),
                loss='mse',
                metrics=['mae']
            )
        else:  # classification
            keras_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=0.5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass

    @abstractmethod
    def train(self) -> Dict[str, float]:
        """Train the model and return training history"""
        pass

    @abstractmethod
    def predict(self, return_probabilities=False) -> pd.Series:
        """Make predictions"""
        pass
